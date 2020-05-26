/*
 * Title: cell_xgfm
 * Description: xgfm solver at cell-centers
 * Author: Raphael Egan
 * Date Created: 10-05-2018
 */

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_xgfm_cells.h>
#else
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_xgfm_cells.h>
#endif

#include <src/Parser.h>
#include <examples/scalar_jump_tests/scalar_tests.h>

using namespace std;
#undef MIN
#undef MAX

const static string main_description =
    string("In this example, we test the xGFM technique to solve scalar Poisson equations with discontinuities \n")
    + string("across an irregular interface. \n")
    + string("The user can choose from several test cases (described in the list of possible 'test'), set various \n")
    + string("Boundary conditions, min/max levels of refinement, number of grid splitting(s) for accuracy analysis,\n")
    + string("the number of trees long every Cartesian direction, in the macromesh. Results and illustrative data \n")
    + string("can be saved in vtk format as well and the order of accuracy for the localization of interface points\n")
    + string("based on the levelset values can be chosen between 1 and 2. \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), Summer 2018.\n");

const int default_lmin = 3;
const int default_lmax = 4;

const int default_ngrids  = 4;
const int default_ntree   = 2;

const BoundaryConditionType default_bc_wtype = DIRICHLET; // NEUMANN;
const bool default_use_second_order_theta = false; // true;
const bool default_get_integral = false;
const bool default_print_summary = false;
const int default_test_number = 3;

const bool track_residuals_and_corrections = false;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/cell_xgfm";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/cell_xgfm";
#else
const string default_work_folder = "/home/regan/workspace/projects/cell_center_xgfm";
#endif


class BCWALLTYPE : public WallBCDIM
{
  BoundaryConditionType bc_walltype;
public:
  BCWALLTYPE(BoundaryConditionType bc_walltype_): bc_walltype(bc_walltype_) {}
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return bc_walltype;
  }
};

class BCWALLVAL : public CF_DIM
{
  const test_case_for_scalar_jump_problem_t *test_problem;
  const BCWALLTYPE *bc_wall_type;
public:
  BCWALLVAL(const test_case_for_scalar_jump_problem_t *test_problem_, const BCWALLTYPE *bc_wall_type_)
    : test_problem(test_problem_), bc_wall_type(bc_wall_type_) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    if((*bc_wall_type)(DIM(x, y, z)) == DIRICHLET)
      return (test_problem->levelset(DIM(x, y, z)) > 0.0 ? test_problem->solution_plus(DIM(x, y, z)) : test_problem->solution_minus(DIM(x, y, z)));
    else
    {
      P4EST_ASSERT((*bc_wall_type)(DIM(x, y, z)) == NEUMANN);
      const domain_t &domain= test_problem->get_domain();
      const double xyz[P4EST_DIM] = {DIM(x, y, z)};
      double to_return = 0.0;
      const double ls_value = test_problem->levelset(xyz);
      for(u_char dim = 0; dim < P4EST_DIM; ++dim)
      {
        if(fabs(xyz[dim] - domain.xyz_min[dim]) < EPS*(domain.xyz_max[dim] - domain.xyz_min[dim]))
          to_return -= (ls_value > 0.0 ? test_problem->first_derivative_solution_plus(dim, DIM(x, y, z)) : test_problem->first_derivative_solution_minus(dim, DIM(x, y, z)));
        else if(fabs(xyz[dim] - domain.xyz_max[dim]) < EPS*(domain.xyz_max[dim] - domain.xyz_min[dim]))
          to_return += (ls_value > 0.0 ? test_problem->first_derivative_solution_plus(dim, DIM(x, y, z)) : test_problem->first_derivative_solution_minus(dim, DIM(x, y, z)));
      }
      return to_return;
    }
  }
};

p4est_bool_t
refine_levelset_cf_finest_in_negative (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*)p4est->user_pointer;

  if (quad->level < data->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    const double *tree_xyz_min =  p4est->connectivity->vertices + 3*v_m;
    const double *tree_xyz_max =  p4est->connectivity->vertices + 3*v_p;
    const double tree_size[P4EST_DIM] = {DIM(tree_xyz_max[0] - tree_xyz_min[0], tree_xyz_max[1] - tree_xyz_min[1], tree_xyz_max[2] - tree_xyz_min[2])};

    const double quad_size = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    const double dxyz[P4EST_DIM] = {DIM(tree_size[0]*quad_size, tree_size[1]*quad_size, tree_size[2]*quad_size)};
    const double quad_diag = sqrt(SUMD(SQR(dxyz[0]), SQR(dxyz[1]), SQR(dxyz[2])));
    const double xyz[P4EST_DIM] = {DIM(
                                   tree_xyz_min[0] + tree_size[0]*(double) quad->x/(double) P4EST_ROOT_LEN,
                                   tree_xyz_min[1] + tree_size[1]*(double) quad->y/(double) P4EST_ROOT_LEN,
                                   tree_xyz_min[2] + tree_size[2]*(double) quad->z/(double) P4EST_ROOT_LEN)};

    const CF_DIM& phi = *(data->phi);
    const double& lip = data->lip;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for(u_char ck = 0; ck < 2; ++ck)
#endif
      for(u_char cj = 0; cj < 2; ++cj)
        for(u_char ci = 0; ci < 2; ++ci){
          f[SUMD(ci, 2*cj, 4*ck)] = phi(DIM(xyz[0] + ci*dxyz[0], xyz[1] + cj*dxyz[1], xyz[2] + ck*dxyz[2]));
          if (f[SUMD(ci, 2*cj, 4*ck)] <= 0.5*lip*quad_diag)
            return P4EST_TRUE;
        }

    if (f[0]*f[1] < 0.0 || f[0]*f[2] < 0.0 || f[1]*f[3] < 0.0 || f[2]*f[3] < 0.0
        ONLY3D(|| f[3]*f[4] < 0.0 || f[4]*f[5] < 0.0 || f[5]*f[6] < 0.0 || f[6]*f[7] < 0.0))
      return P4EST_TRUE;

    return P4EST_FALSE;
  }
}


void save_VTK(const string out_dir, const int &iter, my_p4est_xgfm_cells_t& GFM_solver, my_p4est_xgfm_cells_t& xGFM_solver,
              my_p4est_brick_t *brick, Vec GFM_error, Vec xGFM_error, Vec exact_msol_at_nodes, Vec exact_psol_at_nodes, Vec phi_on_computational_grid_nodes)
{
  PetscErrorCode ierr;

  splitting_criteria_t* data = (splitting_criteria_t*) GFM_solver.get_computational_p4est()->user_pointer;

  ostringstream command;
  command << "mkdir -p " << out_dir.c_str();
  int uu = system(command.str().c_str()); (void) uu;

  ostringstream oss_coarse;
  oss_coarse << out_dir << "/computational_grid_macromesh_" << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] ONLY3D(<< "x" << brick->nxyztrees[2])
      << "_lmin_" << data->min_lvl - iter << "_lmax_" << data->max_lvl - iter << "_iter_" << iter;

  const p4est_t* p4est = xGFM_solver.get_computational_p4est();
  const p4est_nodes_t* nodes = xGFM_solver.get_computational_nodes();
  const p4est_ghost_t* ghost = xGFM_solver.get_computational_ghost();

#ifndef SUBREFINED
  const double *xGFM_node_extension_p;
#endif
  const double *exact_msol_at_nodes_p, *exact_psol_at_nodes_p, *phi_p;
  const double *GFM_solution_p, *GFM_error_p;
  const double *xGFM_solution_p, *xGFM_error_p, *xGFM_cell_extension_p;
  ierr = VecGetArrayRead(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_on_computational_grid_nodes, &phi_p); CHKERRXX(ierr);
#ifndef SUBREFINED
  ierr = VecGetArrayRead(xGFM_solver.get_extended_interface_values_interpolated_on_nodes(), &xGFM_node_extension_p); CHKERRXX(ierr);
#endif
  ierr = VecGetArrayRead(GFM_solver.get_solution(), &GFM_solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_solver.get_solution(), &xGFM_solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(GFM_error, &GFM_error_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_error, &xGFM_error_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_solver.get_extended_interface_values(), &xGFM_cell_extension_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all_general(p4est, nodes, ghost,
                                 P4EST_TRUE, P4EST_TRUE,
                               #ifdef SUBREFINED
                                 3, 0, 0,
                               #else
                                 4, 0, 0,
                               #endif
                                 5, 0, 0, oss_coarse.str().c_str(),
                                 VTK_NODE_SCALAR, "exact_sol_m", exact_msol_at_nodes_p,
                                 VTK_NODE_SCALAR, "exact_sol_p", exact_psol_at_nodes_p,
                                 VTK_NODE_SCALAR, "phi", phi_p,
                               #ifndef SUBREFINED
                                 VTK_NODE_SCALAR, "extension_xgfm_nodes", xGFM_node_extension_p,
                               #endif
                                 VTK_CELL_SCALAR, "sol_gfm", GFM_solution_p,
                                 VTK_CELL_SCALAR, "sol_xgfm", xGFM_solution_p,
                                 VTK_CELL_SCALAR, "err_gfm", GFM_error_p,
                                 VTK_CELL_SCALAR, "err_xgfm", xGFM_error_p,
                                 VTK_CELL_SCALAR, "extension_xgfm", xGFM_cell_extension_p);

  ierr = VecRestoreArrayRead(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
#ifndef SUBREFINED
  ierr = VecRestoreArrayRead(xGFM_solver.get_extended_interface_values_interpolated_on_nodes(), &xGFM_node_extension_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArrayRead(phi_on_computational_grid_nodes, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(GFM_solver.get_solution(), &GFM_solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_solver.get_solution(), &xGFM_solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(GFM_error, &GFM_error_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_error, &xGFM_error_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_solver.get_extended_interface_values(), &xGFM_cell_extension_p); CHKERRXX(ierr);

#ifdef SUBREFINED
  std::ostringstream oss_fine;
  oss_fine << out_dir << "/interface_capturing_grid_macromesh_" << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] ONLY3D(<< "x" << brick->nxyztrees[2])
      << "_lmin_" << data->min_lvl - iter << "_lmax_" << data->max_lvl - iter << "_iter_" << iter;

  const p4est_t* fine_p4est = xGFM_solver.get_subrefined_p4est();
  const p4est_nodes_t* fine_nodes = xGFM_solver.get_subrefined_nodes();
  const p4est_ghost_t* fine_ghost = xGFM_solver.get_subrefined_ghost();

  Vec correction_jump_mu_grad;
  ierr = VecDuplicate(xGFM_solver.get_jump_in_flux(), &correction_jump_mu_grad);      CHKERRXX(ierr);
  ierr = VecCopyGhost(xGFM_solver.get_jump_in_flux(), correction_jump_mu_grad);       CHKERRXX(ierr);
  ierr = VecAXPYGhost(correction_jump_mu_grad, -1.0, GFM_solver.get_jump_in_flux());  CHKERRXX(ierr);
  const double *jump_u_p, *jump_normal_flux_p, *extended_field_fine_nodes_xgfm_p;
  const double *normals_p, *jump_flux_GFM_p, *jump_flux_xGFM_p, *correction_jump_mu_grad_p;
  ierr = VecGetArrayRead(xGFM_solver.get_subrefined_jump(), &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_solver.get_subrefined_jump_in_normal_flux(), &jump_normal_flux_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_solver.get_subrefined_phi(), &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_solver.get_extended_interface_values_interpolated_on_nodes(), &extended_field_fine_nodes_xgfm_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_solver.get_subrefined_normals(), &normals_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(GFM_solver.get_jump_in_flux(), &jump_flux_GFM_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(xGFM_solver.get_jump_in_flux(), &jump_flux_xGFM_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(correction_jump_mu_grad, &correction_jump_mu_grad_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all_general(fine_p4est, fine_nodes, fine_ghost,
                                 P4EST_TRUE, P4EST_TRUE,
                                 4, 0, 4, 0, 0, 0, oss_fine.str().c_str(),
                                 VTK_NODE_SCALAR, "phi", phi_p,
                                 VTK_NODE_SCALAR, "jump", jump_u_p,
                                 VTK_NODE_SCALAR, "jump_normal_flux", jump_normal_flux_p,
                                 VTK_NODE_VECTOR_BLOCK, "normal", normals_p,
                                 VTK_NODE_VECTOR_BLOCK, "gfm_jump_flux", jump_flux_GFM_p,
                                 VTK_NODE_VECTOR_BLOCK, "xgfm_jump_flux", jump_flux_xGFM_p,
                                 VTK_NODE_VECTOR_BLOCK, "corr_jump_mu_du", correction_jump_mu_grad_p,
                                 VTK_NODE_SCALAR, "extension_xgfm_nodes", extended_field_fine_nodes_xgfm_p);

  ierr = VecRestoreArrayRead(xGFM_solver.get_subrefined_jump(), &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_solver.get_subrefined_jump_in_normal_flux(), &jump_normal_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_solver.get_subrefined_phi(), &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_solver.get_extended_interface_values_interpolated_on_nodes(), &extended_field_fine_nodes_xgfm_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_solver.get_subrefined_normals(), &normals_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(GFM_solver.get_jump_in_flux(), &jump_flux_GFM_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(xGFM_solver.get_jump_in_flux(), &jump_flux_xGFM_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(correction_jump_mu_grad, &correction_jump_mu_grad_p); CHKERRXX(ierr);
  ierr = VecDestroy(correction_jump_mu_grad); CHKERRXX(ierr);

  PetscPrintf(fine_p4est->mpicomm, "VTK saved in %s\n", out_dir.c_str());
#endif
  return;
}

void get_normals_and_flattened_jumps(p4est_t* p4est_fine, p4est_nodes_t* nodes_fine, my_p4est_node_neighbors_t *ngbd_n_fine, Vec phi_fine, const bool &use_second_order_theta, const test_case_for_scalar_jump_problem_t *test_problem, //input
                                     Vec& jump_u, Vec& jump_normal_flux, Vec normals, Vec phi_xxyyzz) // output
{
  PetscErrorCode ierr;
  my_p4est_level_set_t ls(ngbd_n_fine);

  if(use_second_order_theta)
    ngbd_n_fine->second_derivatives_central(phi_fine, phi_xxyyzz);
  compute_normals(*ngbd_n_fine, phi_fine, normals);

  double *jump_u_p, *jump_normal_flux_p;
  const double *normals_p;
  ierr = VecGetArray(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArray(jump_normal_flux, &jump_normal_flux_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normals, &normals_p); CHKERRXX(ierr);

  double node_xyz[P4EST_DIM];
  for(size_t k = 0; k < ngbd_n_fine->get_layer_size(); ++k) {
    p4est_locidx_t node_idx = ngbd_n_fine->get_layer_node(k);
    node_xyz_fr_n(node_idx, p4est_fine, nodes_fine, node_xyz);
    jump_u_p[node_idx] = test_problem->jump_in_solution(DIM(node_xyz[0], node_xyz[1], node_xyz[2]));
    jump_normal_flux_p[node_idx] = test_problem->jump_in_normal_flux(normals_p + P4EST_DIM*node_idx, DIM(node_xyz[0], node_xyz[1], node_xyz[2]));
  }
  ierr = VecGhostUpdateBegin(jump_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(jump_normal_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t k = 0; k < ngbd_n_fine->get_local_size(); ++k) {
    p4est_locidx_t node_idx = ngbd_n_fine->get_local_node(k);
    node_xyz_fr_n(node_idx, p4est_fine, nodes_fine, node_xyz);
    jump_u_p[node_idx] = test_problem->jump_in_solution(DIM(node_xyz[0], node_xyz[1], node_xyz[2]));
    jump_normal_flux_p[node_idx] = test_problem->jump_in_normal_flux(normals_p + P4EST_DIM*node_idx, DIM(node_xyz[0], node_xyz[1], node_xyz[2]));
  }
  ierr = VecGhostUpdateEnd(jump_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(jump_normal_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(normals, &normals_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(jump_normal_flux, &jump_normal_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(jump_u, &jump_u_p); CHKERRXX(ierr);

  Vec jump_u_flattened, jump_normal_flux_flattened;
  ierr = VecDuplicate(jump_u, &jump_u_flattened); CHKERRXX(ierr);
  ierr = VecDuplicate(jump_normal_flux, &jump_normal_flux_flattened); CHKERRXX(ierr);
  ls.extend_from_interface_to_whole_domain_TVD(phi_fine, jump_u, jump_u_flattened);
  ls.extend_from_interface_to_whole_domain_TVD(phi_fine, jump_normal_flux, jump_normal_flux_flattened);
  ierr = VecDestroy(jump_u); CHKERRXX(ierr); jump_u = jump_u_flattened; jump_u_flattened = NULL;
  ierr = VecDestroy(jump_normal_flux); CHKERRXX(ierr); jump_normal_flux = jump_normal_flux_flattened; jump_normal_flux_flattened = NULL;
}

void get_sharp_rhs(const p4est_t* p4est, p4est_ghost_t* ghost, my_p4est_node_neighbors_t *ngbd_n_fine, Vec phi, const test_case_for_scalar_jump_problem_t *test_problem, // inputs
                   Vec rhs) // output
{
  PetscErrorCode ierr;
  my_p4est_interpolation_nodes_t interp_phi(ngbd_n_fine);
  interp_phi.set_input(phi, linear);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  double xyz_quad[P4EST_DIM];
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      p4est_locidx_t q_idx = quad_idx + tree->quadrants_offset;
      quad_xyz_fr_q(q_idx, tree_idx, p4est, ghost, xyz_quad);
      if(interp_phi(xyz_quad) > 0.0 )
        rhs_p[q_idx] = -test_problem->get_mu_plus()*test_problem->laplacian_u_plus(DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2]));
      else
        rhs_p[q_idx] = -test_problem->get_mu_minus()*test_problem->laplacian_u_minus(DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2]));
    }
  }
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
}

void measure_errors(my_p4est_xgfm_cells_t& solver, my_p4est_faces_t* faces,
                    const test_case_for_scalar_jump_problem_t *test_problem,
                    Vec err_cells, double &err_n, double err_flux_components[P4EST_DIM], double err_derivatives_components[P4EST_DIM])
{
  PetscErrorCode ierr;

  const p4est_t* p4est                  = solver.get_computational_p4est();
  const p4est_ghost_t* ghost            = solver.get_computational_ghost();
  const my_p4est_hierarchy_t* hierarchy = solver.get_computational_hierarchy();
#ifdef WITH_SUBREFINEMENT
  my_p4est_interpolation_nodes_t interp_phi(solver.get_subrefined_node_neighbors());
  interp_phi.set_input(solver.get_subrefined_phi(), linear);
#else
#endif
  Vec flux[P4EST_DIM];
  for(u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecCreateGhostFaces(p4est, faces, &flux[dim], dim); CHKERRXX(ierr); }
  solver.get_flux_components(flux, faces);

  const double *sol_p, *flux_components_p[P4EST_DIM];
  ierr = VecGetArrayRead(solver.get_solution(), &sol_p); CHKERRXX(ierr);
  for(u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArrayRead(flux[dim], &flux_components_p[dim]); CHKERRXX(ierr);}

  double *err_p;
  ierr = VecGetArray(err_cells, &err_p); CHKERRXX(ierr);

  err_n = 0.0;
  for(size_t k = 0; k < hierarchy->get_layer_size(); ++k)
  {
    const p4est_topidx_t tree_idx = hierarchy->get_tree_index_of_layer_quadrant(k);
    const p4est_locidx_t q_idx    = hierarchy->get_local_index_of_layer_quadrant(k);
    double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(q_idx, tree_idx, p4est, ghost, xyz_quad);
    if(interp_phi(xyz_quad) > 0.0)
      err_p[q_idx] = fabs(sol_p[q_idx] - test_problem->solution_plus(DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])));
    else
      err_p[q_idx] = fabs(sol_p[q_idx] - test_problem->solution_minus(DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])));
    err_n = MAX(err_n, err_p[q_idx]);
  }
  ierr = VecGhostUpdateBegin(err_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t k = 0; k < hierarchy->get_inner_size(); ++k)
  {
    const p4est_topidx_t tree_idx = hierarchy->get_tree_index_of_inner_quadrant(k);
    const p4est_locidx_t q_idx    = hierarchy->get_local_index_of_inner_quadrant(k);
    double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(q_idx, tree_idx, p4est, ghost, xyz_quad);
    if(interp_phi(xyz_quad) > 0.0)
      err_p[q_idx] = fabs(sol_p[q_idx] - test_problem->solution_plus(DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])));
    else
      err_p[q_idx] = fabs(sol_p[q_idx] - test_problem->solution_minus(DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])));
    err_n = MAX(err_n, err_p[q_idx]);
  }
  ierr = VecGhostUpdateEnd  (err_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


  for(u_char dim = 0; dim < P4EST_DIM; ++dim) {
    err_flux_components[dim] = 0.0;
    err_derivatives_components[dim] = 0.0;
    for(p4est_locidx_t face_idx = 0; face_idx < faces->num_local[dim]; ++face_idx) {
      double xyz_face[P4EST_DIM]; faces->xyz_fr_f(face_idx, dim, xyz_face);
      const double phi_face = interp_phi(xyz_face);
      if(phi_face > 0.0)
      {
        const double mu_ = test_problem->get_mu_plus();
        err_flux_components[dim]        = MAX(err_flux_components[dim], fabs(flux_components_p[dim][face_idx] - mu_*test_problem->first_derivative_solution_plus(dim, DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        err_derivatives_components[dim] = MAX(err_derivatives_components[dim], fabs(flux_components_p[dim][face_idx]/mu_ - test_problem->first_derivative_solution_plus(dim, DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
      }
      else
      {
        const double mu_ = test_problem->get_mu_minus();
        err_flux_components[dim]        = MAX(err_flux_components[dim], fabs(flux_components_p[dim][face_idx] - mu_*test_problem->first_derivative_solution_minus(dim, DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        err_derivatives_components[dim] = MAX(err_derivatives_components[dim], fabs(flux_components_p[dim][face_idx]/mu_ - test_problem->first_derivative_solution_minus(dim, DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
      }
    }
  }

  ierr = VecRestoreArrayRead(solver.get_solution(), &sol_p); CHKERRXX(ierr);
  for(u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(flux[dim], &flux_components_p[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(flux[dim]); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(err_cells, &err_p); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_flux_components[0], P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_derivatives_components[0], P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}

void build_computational_grid_data(const mpi_environment_t &mpi, my_p4est_brick_t* brick, p4est_connectivity_t *connectivity, const splitting_criteria_cf_t &data, const test_case_for_scalar_jump_problem_t *test_problem,
                                   p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, Vec &phi_comp,
                                   my_p4est_hierarchy_t* &hierarchy, my_p4est_node_neighbors_t* &ngbd_n, my_p4est_cell_neighbors_t* &ngbd_c, my_p4est_faces_t* &faces)
{
  if(p4est != NULL)
    p4est_destroy(p4est);
  p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*)(&data);

  for(int i = 0; i < data.max_lvl; ++i) {
    if(!test_problem->requires_fine_cells_in_negative_domain())
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    else
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf_finest_in_negative, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  if(ghost != NULL)
    p4est_ghost_destroy(ghost);
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est, ghost);

  if(nodes != NULL)
    p4est_nodes_destroy(nodes);
  nodes = my_p4est_nodes_new(p4est, ghost);

  if(hierarchy != NULL)
    delete hierarchy;
  hierarchy = new my_p4est_hierarchy_t(p4est, ghost, brick);

  if(ngbd_n != NULL)
    delete ngbd_n;

  ngbd_n = new my_p4est_node_neighbors_t(hierarchy, nodes); ngbd_n->init_neighbors();

  PetscErrorCode ierr;
  if(phi_comp != NULL){
    ierr = VecDestroy(phi_comp); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_comp); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *test_problem->get_levelset_cf(), phi_comp);
  my_p4est_level_set_t ls_coarse(ngbd_n);
  ls_coarse.reinitialize_2nd_order(phi_comp);

  const double *phi_comp_p;
  ierr = VecGetArrayRead(phi_comp, &phi_comp_p); CHKERRXX(ierr);
  splitting_criteria_tag_t data_tag(data.min_lvl, data.max_lvl, 1.2);
  p4est_t* new_p4est = p4est_copy(p4est, P4EST_FALSE);

  while(data_tag.refine_and_coarsen(new_p4est, nodes, phi_comp_p, test_problem->requires_fine_cells_in_negative_domain()))
  {
    my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
    interp_nodes.set_input(phi_comp, linear);

    my_p4est_partition(new_p4est, P4EST_FALSE, NULL);
    p4est_ghost_t *new_ghost  = my_p4est_ghost_new(new_p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(new_p4est, new_ghost);
    p4est_nodes_t *new_nodes  = my_p4est_nodes_new(new_p4est, new_ghost);
    Vec new_coarse_phi;
    ierr = VecCreateGhostNodes(new_p4est, new_nodes, &new_coarse_phi); CHKERRXX(ierr);
    for(size_t nn=0; nn<new_nodes->indep_nodes.elem_count; ++nn)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(nn, new_p4est, new_nodes, xyz);
      interp_nodes.add_point(nn, xyz);
    }
    interp_nodes.interpolate(new_coarse_phi);

    p4est_destroy(p4est); p4est = new_p4est; new_p4est = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_destroy(ghost); ghost = new_ghost;
    hierarchy->update(p4est, ghost);
    p4est_nodes_destroy(nodes); nodes = new_nodes;
    ngbd_n->update(hierarchy, nodes);

    ierr = VecRestoreArrayRead(phi_comp, &phi_comp_p); CHKERRXX(ierr);
    ierr = VecDestroy(phi_comp); CHKERRXX(ierr); phi_comp = new_coarse_phi;
    ierr = VecGetArrayRead(phi_comp, &phi_comp_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(phi_comp, &phi_comp_p); CHKERRXX(ierr);
  p4est_destroy(new_p4est);
  if(ngbd_c != NULL)
    delete ngbd_c;
  ngbd_c = new my_p4est_cell_neighbors_t(hierarchy);

  if(faces != NULL)
    delete faces;
  faces = new my_p4est_faces_t(p4est, ghost, brick, ngbd_c);
}

void build_interface_capturing_grid_data(p4est_t* p4est_comp, my_p4est_brick_t *brick,  const splitting_criteria_cf_t &data_fine, const test_case_for_scalar_jump_problem_t *test_problem,
                                         p4est_t* &p4est_fine, p4est_ghost_t* &ghost_fine, p4est_nodes_t* &nodes_fine, Vec &phi_fine,
                                         my_p4est_hierarchy_t* &hierarchy_fine, my_p4est_node_neighbors_t* &ngbd_n_fine)
{
  if(p4est_fine != NULL)
    p4est_destroy(p4est_fine);
  p4est_fine = p4est_copy(p4est_comp, P4EST_FALSE);
  p4est_fine->user_pointer = (void*)(&data_fine);
  my_p4est_refine(p4est_fine, P4EST_FALSE, refine_levelset_cf, NULL);

  if(ghost_fine != NULL)
    p4est_ghost_destroy(ghost_fine);
  ghost_fine = my_p4est_ghost_new(p4est_fine, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_fine, ghost_fine);

  if(nodes_fine != NULL)
    p4est_nodes_destroy(nodes_fine);
  nodes_fine = my_p4est_nodes_new(p4est_fine, ghost_fine);

  if(hierarchy_fine != NULL)
    delete hierarchy_fine;
  hierarchy_fine = new my_p4est_hierarchy_t(p4est_fine, ghost_fine, brick);

  if(ngbd_n_fine != NULL)
    delete ngbd_n_fine;
  ngbd_n_fine = new my_p4est_node_neighbors_t(hierarchy_fine, nodes_fine); ngbd_n_fine->init_neighbors();

  PetscErrorCode ierr;
  if (phi_fine != NULL){
    ierr = VecDestroy(phi_fine); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &phi_fine); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_fine, nodes_fine, *test_problem->get_levelset_cf(), phi_fine);
  my_p4est_level_set_t ls(ngbd_n_fine);
  ls.reinitialize_2nd_order(phi_fine);

  const double *phi_read_p;
  ierr = VecGetArrayRead(phi_fine, &phi_read_p); CHKERRXX(ierr);
  splitting_criteria_tag_t data_tag_fine(data_fine.min_lvl, data_fine.max_lvl, 1.2);
  p4est_t *new_p4est_fine = p4est_copy(p4est_fine, P4EST_FALSE);

  while(data_tag_fine.refine(new_p4est_fine, nodes_fine, phi_read_p)) // not refine_and_coarsen, because we need the fine grid to be everywhere finer or as coarse as the coarse grid!
  {
    ierr = VecRestoreArrayRead(phi_fine, &phi_read_p); CHKERRXX(ierr);
    my_p4est_interpolation_nodes_t interp_nodes_fine(ngbd_n_fine);
    interp_nodes_fine.set_input(phi_fine, linear);

    p4est_ghost_t *new_ghost_fine = my_p4est_ghost_new(new_p4est_fine, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(new_p4est_fine, new_ghost_fine);
    p4est_nodes_t *new_nodes_fine  = my_p4est_nodes_new(new_p4est_fine, new_ghost_fine);
    Vec new_phi;
    ierr = VecCreateGhostNodes(new_p4est_fine, new_nodes_fine, &new_phi); CHKERRXX(ierr);
    for(size_t nn=0; nn<new_nodes_fine->indep_nodes.elem_count; ++nn)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(nn, new_p4est_fine, new_nodes_fine, xyz);
      interp_nodes_fine.add_point(nn, xyz);
    }
    interp_nodes_fine.interpolate(new_phi);


    p4est_destroy(p4est_fine); p4est_fine = new_p4est_fine; new_p4est_fine = p4est_copy(p4est_fine, P4EST_FALSE);
    p4est_ghost_destroy(ghost_fine); ghost_fine = new_ghost_fine;
    hierarchy_fine->update(p4est_fine, ghost_fine);
    p4est_nodes_destroy(nodes_fine); nodes_fine = new_nodes_fine;
    ngbd_n_fine->update(hierarchy_fine, nodes_fine);
    ls.update(ngbd_n_fine);

    ierr = VecDestroy(phi_fine); CHKERRXX(ierr); phi_fine = new_phi;

    ierr = VecGetArrayRead(phi_fine, &phi_read_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArrayRead(phi_fine, &phi_read_p); CHKERRXX(ierr);
  p4est_destroy(new_p4est_fine);
}

void shift_solution_to_match_exact_integral(my_p4est_xgfm_cells_t& solver, const test_case_for_scalar_jump_problem_t* test_problem)
{
  if(ISNAN(test_problem->get_integral_of_solution()) && solver.get_computational_p4est()->mpirank == 0)
  {
    std::cerr << "The average of the exact solution is unknown and would be required to check the accuracy " << endl << "of the solution (for shifting the solution and matching average value)" << endl;
    std::cerr << "Disregard the errors on the solution fields and consider derivatives/fluxes only!" << std::endl << std::endl;
    return;
  }
  const double shift = (test_problem->get_integral_of_solution() - solver.get_sharp_integral_solution())/MULTD(test_problem->get_domain().length(), test_problem->get_domain().height(), test_problem->get_domain().width());

  PetscErrorCode ierr = VecShiftGhost(solver.get_solution(), shift); CHKERRXX(ierr);
  return;
}

void print_iteration_info(const mpi_environment_t &mpi, const my_p4est_xgfm_cells_t& solver)
{
  vector<double> max_corr = solver.get_max_corrections();
  vector<double> rel_res = solver.get_relative_residuals();
  vector<PetscInt> nb_iter = solver.get_numbers_of_ksp_iterations();

  if(mpi.rank() == 0)
  {
    PetscInt total_nb_iterations = nb_iter.at(0);
    for(size_t tt = 1; tt < max_corr.size(); ++tt){
      total_nb_iterations += nb_iter.at(tt);
      if(track_residuals_and_corrections)
        cout << "max corr " << tt << " = " << max_corr[tt] << " and rel residual " << tt << " = " << rel_res[tt] << " after " << nb_iter[tt] << " iterations." << endl;
    }
    cout << "The solver converged after a total of " << total_nb_iterations << " iterations." << std::endl;
  }
}

void print_errors_and_orders(const mpi_environment_t &mpi, const int &iter, const double err[][2], const double err_flux_components[][2][P4EST_DIM], const double err_derivatives_components[][2][P4EST_DIM])
{
  PetscErrorCode ierr;
  if(iter > 0){
    ierr = PetscPrintf(mpi.comm(), "Error on cells  for  gfm: %.5e, order = %g\n", err[iter][0], log(err[iter - 1][0]/err[iter][0])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on cells  for xgfm: %.5e, order = %g\n", err[iter][1], log(err[iter - 1][1]/err[iter][1])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-x for  gfm: %.5e, order = %g\n", err_flux_components[iter][0][0], log(err_flux_components[iter - 1][0][0]/err_flux_components[iter][0][0])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-x for xgfm: %.5e, order = %g\n", err_flux_components[iter][1][0], log(err_flux_components[iter - 1][1][0]/err_flux_components[iter][1][0])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-y for  gfm: %.5e, order = %g\n", err_flux_components[iter][0][1], log(err_flux_components[iter - 1][0][1]/err_flux_components[iter][0][1])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-y for xgfm: %.5e, order = %g\n", err_flux_components[iter][1][1], log(err_flux_components[iter - 1][1][1]/err_flux_components[iter][1][1])/log(2)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscPrintf(mpi.comm(), "Error on flux-z for  gfm: %.5e, order = %g\n", err_flux_components[iter][0][2], log(err_flux_components[iter - 1][0][2]/err_flux_components[iter][0][2])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-z for xgfm: %.5e, order = %g\n", err_flux_components[iter][1][2], log(err_flux_components[iter - 1][1][2]/err_flux_components[iter][1][2])/log(2)); CHKERRXX(ierr);
#endif
    ierr = PetscPrintf(mpi.comm(), "Error on x-der  for  gfm: %.5e, order = %g\n", err_derivatives_components[iter][0][0], log(err_derivatives_components[iter - 1][0][0]/err_derivatives_components[iter][0][0])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on x-der  for xgfm: %.5e, order = %g\n", err_derivatives_components[iter][1][0], log(err_derivatives_components[iter - 1][1][0]/err_derivatives_components[iter][1][0])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on y-der  for  gfm: %.5e, order = %g\n", err_derivatives_components[iter][0][1], log(err_derivatives_components[iter - 1][0][1]/err_derivatives_components[iter][0][1])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on y-der  for xgfm: %.5e, order = %g\n", err_derivatives_components[iter][1][1], log(err_derivatives_components[iter - 1][1][1]/err_derivatives_components[iter][1][1])/log(2)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscPrintf(mpi.comm(), "Error on z-der  for  gfm: %.5e, order = %g\n", err_derivatives_components[iter][0][2], log(err_derivatives_components[iter - 1][0][2]/err_derivatives_components[iter][0][2])/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on z-der  for xgfm: %.5e, order = %g\n", err_derivatives_components[iter][1][2], log(err_derivatives_components[iter - 1][1][2]/err_derivatives_components[iter][1][2])/log(2)); CHKERRXX(ierr);
#endif
  }
  else {
    ierr = PetscPrintf(mpi.comm(), "Error on cells  for  gfm: %.5e\n", err[iter][0]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on cells  for xgfm: %.5e\n", err[iter][1]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-x for  gfm: %.5e\n", err_flux_components[iter][0][0]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-x for xgfm: %.5e\n", err_flux_components[iter][1][0]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-y for  gfm: %.5e\n", err_flux_components[iter][0][1]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-y for xgfm: %.5e\n", err_flux_components[iter][1][1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscPrintf(mpi.comm(), "Error on flux-z for  gfm: %.5e\n", err_flux_components[iter][0][2]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on flux-z for xgfm: %.5e\n", err_flux_components[iter][1][2]); CHKERRXX(ierr);
#endif
    ierr = PetscPrintf(mpi.comm(), "Error on x-der  for  gfm: %.5e\n", err_derivatives_components[iter][0][0]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on x-der  for xgfm: %.5e\n", err_derivatives_components[iter][1][0]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on y-der  for  gfm: %.5e\n", err_derivatives_components[iter][0][1]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on y-der  for xgfm: %.5e\n", err_derivatives_components[iter][1][1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscPrintf(mpi.comm(), "Error on z-der  for  gfm: %.5e\n", err_derivatives_components[iter][0][2]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on z-der  for xgfm: %.5e\n", err_derivatives_components[iter][1][2]); CHKERRXX(ierr);
#endif
  }
}

void get_sampled_exact_solution(Vec exact_msol_at_nodes, Vec exact_psol_at_nodes,
                                const p4est_t* p4est, const p4est_nodes_t* nodes, const test_case_for_scalar_jump_problem_t *test_problem)
{
  PetscErrorCode ierr;

  double *exact_msol_at_nodes_p, *exact_psol_at_nodes_p;
  ierr = VecGetArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
  for(size_t node_idx = 0; node_idx < nodes->indep_nodes.elem_count; ++node_idx) {
    double xyz_node[P4EST_DIM];
    node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
    exact_msol_at_nodes_p[node_idx] = test_problem->solution_minus(DIM(xyz_node[0], xyz_node[1], xyz_node[2]));
    exact_psol_at_nodes_p[node_idx] = test_problem->solution_plus(DIM(xyz_node[0], xyz_node[1], xyz_node[2]));
  }
  ierr = VecRestoreArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
}

void print_integral_of_exact_solution(Vec exact_msol_at_nodes, Vec exact_psol_at_nodes, Vec phi_comp, const p4est_t* p4est, const p4est_nodes_t* nodes)
{
  PetscErrorCode ierr;
  P4EST_ASSERT(exact_msol_at_nodes != NULL && exact_psol_at_nodes != NULL);

  double integral_of_exact = 0.0;
  integral_of_exact += integrate_over_negative_domain(p4est, nodes, phi_comp, exact_msol_at_nodes);
  if(ISNAN(integral_of_exact))
    std::cout << "the first integral part is nan" << std::endl;
  ierr = VecScaleGhost(phi_comp, -1.0); CHKERRXX(ierr);
  integral_of_exact += integrate_over_negative_domain(p4est, nodes, phi_comp, exact_psol_at_nodes);
  ierr = VecScaleGhost(phi_comp, -1.0); CHKERRXX(ierr);
  ierr = PetscPrintf(p4est->mpicomm, "The integral calculated with exact fields is %.12e \n", integral_of_exact); CHKERRXX(ierr);
}

void print_convergence_summary_in_file(const string& out_folder, const string& test_name, const int &lmin, const int &lmax, const int &ntree, const int &ngrids, const bool &use_second_order_theta, const BoundaryConditionType bc_wtype,
                                       const double err[][2], const double err_derivatives_components[][2][P4EST_DIM], const double err_flux_components[][2][P4EST_DIM])
{
  string summary_folder = out_folder + "/summaries";
  ostringstream command;
  command << "mkdir -p " << summary_folder.c_str();
  int uu = system(command.str().c_str()); (void) uu; // create the summary folder
  string summary_file = summary_folder + "/convergence_summary_" + test_name
      + "_lmin" + to_string(lmin) + "_lmax" + to_string(lmax) + "_ngrids" + to_string(ngrids) + "_ntree" + to_string(ntree)
      + "_accuracyls" + to_string(use_second_order_theta ? 2 : 1) + "_" + (bc_wtype == DIRICHLET ? "dirichlet": "neumann") + ".dat";

  FILE *fid = fopen(summary_file.c_str(), "w");
  fprintf(fid, "=================================================================\n");
  fprintf(fid, "========================= SUMMARY ===============================\n");
  fprintf(fid, "=================================================================\n");
  fprintf(fid, "Test case: %s (%d-D)\n", test_name.c_str(), P4EST_DIM);
  fprintf(fid, "lmin: %d\n", lmin);
  fprintf(fid, "lmax: %d\n", lmax);
  fprintf(fid, "Number of grids: %d\n", ngrids);
  fprintf(fid, "Number of trees along minimum dimension of domain in macromesh: %d\n", ntree);
  fprintf(fid, "Order of accuracy for interface localization: %d\n", (use_second_order_theta? 2 : 1));
  fprintf(fid, "Wall boundary condition: %s\n", (bc_wtype == DIRICHLET ? "dirichlet" : "neumann"));
  fprintf(fid, "Resolution: " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%d/%d ", ntree*(1<<(lmin+k)), ntree*(1<<(lmax+k)));
    else
      fprintf(fid, "%d/%d\n", ntree*(1<<(lmin+k)), ntree*(1<<(lmax+k)));
  }
  fprintf(fid, "Error on solution (gfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err[k][0]);
    else
      fprintf(fid, "%.5e\n", err[k][0]);
  }
  fprintf(fid, "Error on solution (xgfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err[k][1]);
    else
      fprintf(fid, "%.5e\n", err[k][1]);
  }

  fprintf(fid, "Error on x-derivative (gfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_derivatives_components[k][0][0]);
    else
      fprintf(fid, "%.5e\n", err_derivatives_components[k][0][0]);
  }
  fprintf(fid, "Error on x-derivative (xgfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_derivatives_components[k][1][0]);
    else
      fprintf(fid, "%.5e\n", err_derivatives_components[k][1][0]);
  }

  fprintf(fid, "Error on y-derivative (gfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_derivatives_components[k][0][1]);
    else
      fprintf(fid, "%.5e\n", err_derivatives_components[k][0][1]);
  }
  fprintf(fid, "Error on y-derivative (xgfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_derivatives_components[k][1][1]);
    else
      fprintf(fid, "%.5e\n", err_derivatives_components[k][1][1]);
  }
#ifdef P4_TO_P8
  fprintf(fid, "Error on z-derivative (gfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_derivatives_components[k][0][2]);
    else
      fprintf(fid, "%.5e\n", err_derivatives_components[k][0][2]);
  }
  fprintf(fid, "Error on z-derivative (xgfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_derivatives_components[k][1][2]);
    else
      fprintf(fid, "%.5e\n", err_derivatives_components[k][1][2]);
  }
#endif

  fprintf(fid, "Error on x-flux (gfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_flux_components[k][0][0]);
    else
      fprintf(fid, "%.5e\n", err_flux_components[k][0][0]);
  }
  fprintf(fid, "Error on x-flux (xgfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_flux_components[k][1][0]);
    else
      fprintf(fid, "%.5e\n", err_flux_components[k][1][0]);
  }

  fprintf(fid, "Error on y-flux (gfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_flux_components[k][0][1]);
    else
      fprintf(fid, "%.5e\n", err_flux_components[k][0][1]);
  }
  fprintf(fid, "Error on y-flux (xgfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_flux_components[k][1][1]);
    else
      fprintf(fid, "%.5e\n", err_flux_components[k][1][1]);
  }
#ifdef P4_TO_P8
  fprintf(fid, "Error on z-flux (gfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_flux_components[k][0][2]);
    else
      fprintf(fid, "%.5e\n", err_flux_components[k][0][2]);
  }
  fprintf(fid, "Error on z-flux (xgfm): " );
  for(int k = 0; k < ngrids; ++k)
  {
    if(k!=ngrids-1)
      fprintf(fid, "%.5e ", err_flux_components[k][1][2]);
    else
      fprintf(fid, "%.5e\n", err_flux_components[k][1][2]);
  }
#endif

  fprintf(fid, "=================================================================\n");
  fprintf(fid, "===================== END OF SUMMARY ============================\n");
  fprintf(fid, "=================================================================");
  fclose(fid);
  printf("Summary file printed in %s\n", summary_file.c_str());
}

#ifdef SUBREFINED
Vec create_phi_on_computational_nodes(const my_p4est_xgfm_cells_t& solver)
{
  PetscErrorCode ierr;
  my_p4est_interpolation_nodes_t interp_phi(solver.get_subrefined_node_neighbors()); interp_phi.set_input(solver.get_subrefined_phi(), linear);
  Vec phi_comp; double *phi_comp_p;
  ierr = VecCreateGhostNodes(solver.get_computational_p4est(), solver.get_computational_nodes(), &phi_comp); CHKERRXX(ierr);
  ierr = VecGetArray(phi_comp, &phi_comp_p); CHKERRXX(ierr);
  const my_p4est_node_neighbors_t* node_ngbd_comp = solver.get_computational_node_neighbors();
  for (size_t k = 0; k < node_ngbd_comp->get_layer_size(); ++k) {
    const p4est_locidx_t node_idx = node_ngbd_comp->get_layer_node(k);
    double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, node_ngbd_comp->get_p4est(), node_ngbd_comp->get_nodes(), xyz_node);
    phi_comp_p[node_idx] = interp_phi(xyz_node);
  }
  ierr = VecGhostUpdateBegin(phi_comp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < node_ngbd_comp->get_local_size(); ++k) {
    const p4est_locidx_t node_idx = node_ngbd_comp->get_local_node(k);
    double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, node_ngbd_comp->get_p4est(), node_ngbd_comp->get_nodes(), xyz_node);
    phi_comp_p[node_idx] = interp_phi(xyz_node);
  }
  ierr = VecGhostUpdateEnd(phi_comp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_comp, &phi_comp_p); CHKERRXX(ierr);

  return phi_comp;
}
#endif

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree in the computational grid, default is " + to_string(default_lmin));
  cmd.add_option("lmax", "max level of the tree in the computational grid, default is " + to_string(default_lmax));
  cmd.add_option("ngrids", "number of computational grids for accuracy analysis, default is " + to_string(default_ngrids));
  ostringstream oss; oss << default_bc_wtype;
  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall ('Dirichlet' or 'Neumann'), default is " + oss.str());
  cmd.add_option("save_vtk", "saves the p4est's (computational and interface-capturing grids) in vtk format");
  cmd.add_option("work_dir", "exportation directory, if not defined otherwise in the environment variable OUT_DIR. \n\
\tThis is required if saving vtk or summary files: work_dir/vtu for vtk files work_dir/summaries for summary files. Default is " + default_work_folder);
  cmd.add_option("second_order_ls", "activate second order interface localization if present. Default is " + string(default_use_second_order_theta ? "true" : "false"));
  cmd.add_option("ntree", "number of trees in the macromesh along every dimension of the computational domain. Default value is " + to_string(default_ntree));
  cmd.add_option("test", "Test problem to choose. Available choices are (default test number is " + to_string(default_test_number) +"): \n" + list_of_test_problems_for_scalar_jump_problems.get_description_of_tests() + "\n");
  cmd.add_option("get_integral", "Calculates the integral of the solution if present. Default is " + string(default_get_integral ? "true" : "false"));
  cmd.add_option("summary", "Prints a summary of the convergence results in a file on disk if present. Default is " + string(default_print_summary ? "true" : "false"));
  if(cmd.parse(argc, argv, main_description))
    return 0;

  const int lmin = cmd.get<int>("lmin", default_lmin);
  const int lmax = cmd.get<int>("lmax", default_lmax);
  const int ngrids = cmd.get<int>("ngrids", default_ngrids);
  const int test_number = cmd.get<int>("test", default_test_number);
  const BoundaryConditionType bc_wtype = cmd.get<BoundaryConditionType>("bc_wtype", default_bc_wtype);
  const int ntree = cmd.get<int>("ntree", default_ntree);
  const int n_xyz [P4EST_DIM] = {DIM(ntree, ntree, ntree)};
  const bool use_second_order_theta = default_use_second_order_theta || cmd.contains("second_order_ls");
  const bool get_integral = default_get_integral || cmd.contains("get_integral");
  const bool print_summary = default_print_summary || cmd.contains("summary");
  const bool save_vtk = cmd.contains("save_vtk");

  parStopWatch watch, watch_global;
  watch_global.start("Total run time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const test_case_for_scalar_jump_problem_t *test_problem = list_of_test_problems_for_scalar_jump_problems[test_number];
  const string out_folder = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR"))) + "/" + test_problem->get_name();
  const string vtk_out = out_folder + "/vtu";

  connectivity = my_p4est_brick_new(n_xyz, test_problem->get_xyz_min(), test_problem->get_xyz_max(), &brick, test_problem->get_periodicity());

  double err[ngrids][2], err_flux_components[ngrids][2][P4EST_DIM], err_derivatives_components[ngrids][2][P4EST_DIM];
  for(int iter = 0; iter < ngrids; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin + iter, lmax + iter); CHKERRXX(ierr);

    /* build the computational grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods, its cell neighborhoods
     * the REINITIALIZED levelset on the computational grid
     */
    p4est_t       *p4est = NULL;
    p4est_nodes_t *nodes = NULL;
    p4est_ghost_t *ghost = NULL;
    my_p4est_hierarchy_t* hierarchy = NULL;
    my_p4est_node_neighbors_t* ngbd_n = NULL;
    my_p4est_cell_neighbors_t* ngbd_c = NULL;
    my_p4est_faces_t* faces = NULL;
    Vec phi = NULL;
    splitting_criteria_cf_t data(lmin + iter, lmax + iter, test_problem->get_levelset_cf(), 1.2);
    build_computational_grid_data(mpi, &brick, connectivity, data, test_problem,
                                  p4est, ghost, nodes, phi, hierarchy, ngbd_n, ngbd_c, faces);

    /* build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
     * the REINITIALIZED levelset on the interface-capturing grid
     */
    p4est_t       *p4est_fine = NULL;
    p4est_nodes_t *nodes_fine = NULL;
    p4est_ghost_t *ghost_fine = NULL;
    my_p4est_hierarchy_t* hierarchy_fine = NULL;
    my_p4est_node_neighbors_t* ngbd_n_fine = NULL;
    splitting_criteria_cf_t data_fine(data.min_lvl, data.max_lvl + 1, test_problem->get_levelset_cf(), 1.2);
    build_interface_capturing_grid_data(p4est, &brick, data_fine, test_problem,
                                        p4est_fine, ghost_fine, nodes_fine, phi, hierarchy_fine, ngbd_n_fine);

    /* Get the normals, the second derivatives of the levelset (if required) and the relevant flattened jumps
     */
    Vec jump_u, jump_normal_flux;
    Vec normals     = NULL;
    Vec phi_xxyyzz  = NULL;
    ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &jump_u); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &jump_normal_flux); CHKERRXX(ierr);
    ierr = VecCreateGhostNodesBlock(p4est_fine, nodes_fine, P4EST_DIM, &normals); CHKERRXX(ierr);
    if(use_second_order_theta){
      ierr = VecCreateGhostNodesBlock(p4est_fine, nodes_fine, P4EST_DIM, &phi_xxyyzz); CHKERRXX(ierr); }

    get_normals_and_flattened_jumps(p4est_fine, nodes_fine, ngbd_n_fine, phi, use_second_order_theta, test_problem, //input
                                    jump_u, jump_normal_flux, normals, phi_xxyyzz); // output

    /* TEST THE JUMP SOLVER AND COMPARE TO ORIGINAL GFM */
    BoundaryConditionsDIM bc;
    BCWALLTYPE bc_wall_type(bc_wtype);
    bc.setWallTypes(bc_wall_type);
    BCWALLVAL bc_wall_val(test_problem, &bc_wall_type);
    bc.setWallValues(bc_wall_val);

    Vec sharp_rhs;
    ierr = VecCreateNoGhostCells(p4est, &sharp_rhs); CHKERRXX(ierr);
    get_sharp_rhs(p4est, ghost, ngbd_n_fine, phi, test_problem, sharp_rhs);

    my_p4est_xgfm_cells_t *GFM_solver = new my_p4est_xgfm_cells_t(ngbd_c, ngbd_n, ngbd_n_fine, false); // --> standard GFM solver ("A Boundary Condition Capturing Method for Poisson's Equation on Irregular Domains", JCP, 160(1):151-178, Liu, Fedkiw, Kand, 2000);
    my_p4est_xgfm_cells_t *xGFM_solver = new my_p4est_xgfm_cells_t(ngbd_c, ngbd_n, ngbd_n_fine, true);  // --> xGFM solver ("xGFM: Recovering Convergence of Fluxes in the Ghost Fluid Method", JCP, Volume 409, 15 May 2020, 19351, R. Egan, F. Gibou);
    for(u_char xgfm_flag = 0; xgfm_flag < 2; ++xgfm_flag) {
      my_p4est_xgfm_cells_t& jump_solver = (xgfm_flag == 0 ? *GFM_solver : *xGFM_solver);
      jump_solver.set_phi(phi, phi_xxyyzz);
      jump_solver.set_normals(normals);
      jump_solver.set_mus(test_problem->get_mu_minus(), test_problem->get_mu_plus());
      jump_solver.set_jumps(jump_u, jump_normal_flux);
      jump_solver.set_diagonals(0.0, 0.0);
      jump_solver.set_bc(bc);
      jump_solver.set_rhs(sharp_rhs);

      watch.start("Total time:");
      jump_solver.solve();
      watch.stop(); watch.read_duration();

      print_iteration_info(mpi, jump_solver);

      /* if null space, shift solution */
      if(jump_solver.get_matrix_has_nullspace())
        shift_solution_to_match_exact_integral(jump_solver, test_problem);
    }
    MPI_Barrier(mpi.comm());

    /* measure the error(s) */
    Vec err_GFM, err_xGFM;
    ierr = VecCreateGhostCells(p4est, ghost, &err_GFM); CHKERRXX(ierr);
    ierr = VecCreateGhostCells(p4est, ghost, &err_xGFM); CHKERRXX(ierr);
    measure_errors(*GFM_solver, faces, test_problem, err_GFM, err[iter][0], err_flux_components[iter][0], err_derivatives_components[iter][0]);
    measure_errors(*xGFM_solver, faces, test_problem, err_xGFM, err[iter][1], err_flux_components[iter][1], err_derivatives_components[iter][1]);

    print_errors_and_orders(mpi, iter, err, err_flux_components, err_derivatives_components);

    if(save_vtk || get_integral)
    {
      Vec exact_msol_at_nodes = NULL, exact_psol_at_nodes  = NULL; // to enable illustration of exact solution with wrap-by-scalar in paraview or to calculate the integral of the exact solution, numerically
      ierr = VecCreateGhostNodes(p4est, nodes, &exact_msol_at_nodes); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &exact_psol_at_nodes); CHKERRXX(ierr);
      get_sampled_exact_solution(exact_msol_at_nodes, exact_psol_at_nodes, p4est, nodes, test_problem);
#ifdef SUBREFINED
      Vec phi_on_computational_grid_nodes = create_phi_on_computational_nodes(*xGFM_solver);
#else
      Vec phi_on_computational_grid_nodes = phi;
#endif

      if(get_integral)
        print_integral_of_exact_solution(exact_msol_at_nodes, exact_psol_at_nodes, phi_on_computational_grid_nodes, p4est, nodes);

      if(save_vtk)
        save_VTK(vtk_out, iter, *GFM_solver, *xGFM_solver, &brick,
                 err_GFM, err_xGFM, exact_msol_at_nodes, exact_psol_at_nodes, phi_on_computational_grid_nodes);
      ierr = VecDestroy(exact_msol_at_nodes); CHKERRXX(ierr);
      ierr = VecDestroy(exact_psol_at_nodes); CHKERRXX(ierr);
#ifdef SUBREFINED
    ierr = VecDestroy(phi_on_computational_grid_nodes); CHKERRXX(ierr);
#endif
    }

    // destroy data created for this iteration
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    if(use_second_order_theta) {
      ierr = VecDestroy(phi_xxyyzz); CHKERRXX(ierr); }
    ierr = VecDestroy(jump_u); CHKERRXX(ierr);
    ierr = VecDestroy(jump_normal_flux); CHKERRXX(ierr);
    ierr = VecDestroy(normals); CHKERRXX(ierr);
    ierr = VecDestroy(err_GFM); CHKERRXX(ierr);
    ierr = VecDestroy(err_xGFM); CHKERRXX(ierr);


    p4est_nodes_destroy(nodes); p4est_nodes_destroy(nodes_fine);
    p4est_ghost_destroy(ghost); p4est_ghost_destroy(ghost_fine);
    p4est_destroy      (p4est); p4est_destroy(p4est_fine);
    delete hierarchy; delete hierarchy_fine;
    delete ngbd_n; delete ngbd_n_fine;
    delete ngbd_c;
    delete faces;
    delete GFM_solver;
    delete xGFM_solver;
  }

  if(mpi.rank() == 0 && print_summary)
    print_convergence_summary_in_file(out_folder, test_problem->get_name(), lmin, lmax, ntree, ngrids, use_second_order_theta, bc_wtype,
                                      err, err_derivatives_components, err_flux_components);

  my_p4est_brick_destroy(connectivity, &brick);

  watch_global.stop(); watch_global.read_duration();

  return 0;
}
