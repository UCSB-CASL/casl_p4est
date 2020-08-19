// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#else
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#endif

#include <src/Parser.h>
#include <examples/scalar_jump_tests/scalar_tests.h>

using namespace std;
#undef MIN
#undef MAX

const static string main_description =
    string("In this example, we test the mls Poisson solvers for node-sampled scalar Poisson problems with \n")
    + string("discontinuities across an irregular interface. \n")
    + string("The user can choose from several test cases (described in the list of possible 'test'), set various \n")
    + string("Boundary conditions, min/max levels of refinement, number of grid splitting(s) for accuracy analysis,\n")
    + string("the number of trees long every Cartesian direction, in the macromesh and, most importantly.\n")
    + string("Results and illustration data can be saved in vtk format as well. \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), Summer 2020 (solver implemented by Daniil Bochkov)\n");

const int default_lmin = 3;
const int default_lmax = 4;

const int default_ngrids  = 4;
const int default_ntree   = 2;

const BoundaryConditionType default_bc_wtype = DIRICHLET; // NEUMANN;
const interpolation_method default_interp_method_phi = linear;
const bool default_print_summary  = false;
const int default_test_number = 3;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/poisson_jump_nodes_mls";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/poisson_jump_nodes_mls";
#else
const string default_work_folder = "/home/regan/workspace/projects/poisson_jump_nodes_mls";
#endif

struct convergence_analyzer_for_jump_node_solver_t {
  my_p4est_poisson_nodes_mls_t* jump_solver;
  std::vector<double> errors_in_solution;
  std::vector<double> errors_in_extrapolated_solution_minus;
  std::vector<double> errors_in_extrapolated_solution_plus;
  Vec node_sampled_error, node_sampled_ghost_error_minus, node_sampled_ghost_error_plus;

  void delete_and_nullify_node_sampled_errors_if_needed()
  {
    PetscErrorCode ierr;
    ierr = delete_and_nullify_vector(node_sampled_error); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(node_sampled_ghost_error_minus); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(node_sampled_ghost_error_plus); CHKERRXX(ierr);
  }

  convergence_analyzer_for_jump_node_solver_t() : node_sampled_error(NULL),
    node_sampled_ghost_error_minus(NULL), node_sampled_ghost_error_plus(NULL) { }

  void measure_errors(Vec ghosted_solution_minus, Vec ghosted_solution_plus, const test_case_for_scalar_jump_problem_t *test_problem)
  {
    PetscErrorCode ierr;
    const p4est_t* p4est        = jump_solver->get_p4est();
    const p4est_nodes_t* nodes  = jump_solver->get_nodes();
    Vec phi = jump_solver->get_interface_phi_eff();

    const double *ghosted_solution_minus_p, *ghosted_solution_plus_p, *phi_p;
    ierr = VecGetArrayRead(ghosted_solution_minus, &ghosted_solution_minus_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(ghosted_solution_plus, &ghosted_solution_plus_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    delete_and_nullify_node_sampled_errors_if_needed();
    ierr = VecCreateGhostNodes(p4est, nodes, &node_sampled_error); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &node_sampled_ghost_error_minus); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &node_sampled_ghost_error_plus); CHKERRXX(ierr);
    double *node_sampled_error_p, *node_sampled_ghost_error_minus_p, *node_sampled_ghost_error_plus_p;
    ierr = VecGetArray(node_sampled_error, &node_sampled_error_p); CHKERRXX(ierr);
    ierr = VecGetArray(node_sampled_ghost_error_minus, &node_sampled_ghost_error_minus_p); CHKERRXX(ierr);
    ierr = VecGetArray(node_sampled_ghost_error_plus, &node_sampled_ghost_error_plus_p); CHKERRXX(ierr);

    double err_n = 0.0;
    for(size_t k = 0; k < jump_solver->get_ngbd()->get_layer_size(); ++k)
    {
      const p4est_locidx_t node_idx  = jump_solver->get_ngbd()->get_layer_node(k);
      double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      const double phi_node = phi_p[node_idx];
      node_sampled_ghost_error_minus_p[node_idx] = node_sampled_ghost_error_plus_p[node_idx] = 0.0;
      if(!ISNAN(ghosted_solution_minus_p[node_idx]))
        node_sampled_ghost_error_minus_p[node_idx] = fabs(ghosted_solution_minus_p[node_idx] - test_problem->solution_minus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      if(!ISNAN(ghosted_solution_plus_p[node_idx]))
        node_sampled_ghost_error_plus_p[node_idx] = fabs(ghosted_solution_plus_p[node_idx] - test_problem->solution_plus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      if(phi_node >= 0.0)
        node_sampled_error_p[node_idx] = fabs(ghosted_solution_plus_p[node_idx] - test_problem->solution_plus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      else
        node_sampled_error_p[node_idx] = fabs(ghosted_solution_minus_p[node_idx] - test_problem->solution_minus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      err_n = MAX(err_n, node_sampled_error_p[node_idx]);
    }
    ierr = VecGhostUpdateBegin(node_sampled_error, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(node_sampled_ghost_error_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(node_sampled_ghost_error_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t k = 0; k < jump_solver->get_ngbd()->get_local_size(); ++k)
    {
      const p4est_locidx_t node_idx  = jump_solver->get_ngbd()->get_local_node(k);
      double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      const double phi_node = phi_p[node_idx];
      node_sampled_ghost_error_minus_p[node_idx] = node_sampled_ghost_error_plus_p[node_idx] = 0.0;
      if(!ISNAN(ghosted_solution_minus_p[node_idx]))
        node_sampled_ghost_error_minus_p[node_idx] = fabs(ghosted_solution_minus_p[node_idx] - test_problem->solution_minus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      if(!ISNAN(ghosted_solution_plus_p[node_idx]))
        node_sampled_ghost_error_plus_p[node_idx] = fabs(ghosted_solution_plus_p[node_idx] - test_problem->solution_plus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      if(phi_node >= 0.0)
        node_sampled_error_p[node_idx] = fabs(ghosted_solution_plus_p[node_idx] - test_problem->solution_plus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      else
        node_sampled_error_p[node_idx] = fabs(ghosted_solution_minus_p[node_idx] - test_problem->solution_minus(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
      err_n = MAX(err_n, node_sampled_error_p[node_idx]);
    }
    ierr = VecGhostUpdateEnd  (node_sampled_error, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (node_sampled_ghost_error_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (node_sampled_ghost_error_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    ierr = VecRestoreArrayRead(ghosted_solution_minus, &ghosted_solution_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(ghosted_solution_plus, &ghosted_solution_plus_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(node_sampled_error, &node_sampled_error_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(node_sampled_ghost_error_minus, &node_sampled_ghost_error_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(node_sampled_ghost_error_plus, &node_sampled_ghost_error_plus_p); CHKERRXX(ierr);

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    errors_in_solution.push_back(err_n);

    char convergence_order_info[BUFSIZ] = "\0";
    const size_t iter_idx = errors_in_solution.size() - 1;

    if(iter_idx > 0)
      sprintf(convergence_order_info, ", order = %g", -log(errors_in_solution[iter_idx]/errors_in_solution[iter_idx - 1])/log(2.0));
    ierr = PetscPrintf(p4est->mpicomm, "\nFor FV solver: \n"); CHKERRXX(ierr); // some spacing
    ierr = PetscPrintf(p4est->mpicomm, "Error on nodes:\t\t%.5e%s \n", errors_in_solution.back(), convergence_order_info); CHKERRXX(ierr);
  }

  ~convergence_analyzer_for_jump_node_solver_t()
  {
    delete_and_nullify_node_sampled_errors_if_needed();
  }
};

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

p4est_bool_t refine_levelset_cf_finest_in_negative (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
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

struct Vec_for_vtk_export_t {
  Vec vector;
  const double* ptr;
  string name;
  Vec_for_vtk_export_t(Vec to_export, const string& name_tag)
  {
    vector = to_export;
    PetscErrorCode ierr = VecGetArrayRead(vector, &ptr); CHKERRXX(ierr);
    name = name_tag;
  }
  ~Vec_for_vtk_export_t()
  {
    PetscErrorCode ierr = VecRestoreArrayRead(vector, &ptr); CHKERRXX(ierr);
  }
};

void add_vtk_export_to_list(const Vec_for_vtk_export_t& to_export, std::vector<const double *>& list_of_data_pointers, std::vector<string>& list_of_data_name_tags)
{
  list_of_data_pointers.push_back(to_export.ptr);
  list_of_data_name_tags.push_back(to_export.name);
}

void save_VTK(const string out_dir, const int &iter, Vec exact_solution_minus, Vec exact_solution_plus,
              Vec phi, Vec sharp_solution, Vec ghosted_solution_minus, Vec ghosted_solution_plus,
              const convergence_analyzer_for_jump_node_solver_t& convergence_analyzer, const my_p4est_brick_t *brick)
{
  splitting_criteria_t* data = (splitting_criteria_t*) convergence_analyzer.jump_solver->get_p4est()->user_pointer;

  ostringstream command;
  command << "mkdir -p " << out_dir.c_str();
  int system_return = system(command.str().c_str()); (void) system_return;

  ostringstream oss_computational;
  oss_computational << out_dir << "/computational_grid_macromesh_" << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] ONLY3D(<< "x" << brick->nxyztrees[2])
      << "_lmin_" << data->min_lvl - iter << "_lmax_" << data->max_lvl - iter << "_iter_" << iter;

  const p4est_t* p4est = convergence_analyzer.jump_solver->get_p4est();
  const p4est_nodes_t* nodes = convergence_analyzer.jump_solver->get_nodes();
  const p4est_ghost_t* ghost = convergence_analyzer.jump_solver->get_ghost();

  std::vector<const double*> comp_node_scalar_fields_pointers;
  std::vector<string> comp_node_scalar_fields_names;
  std::vector<const double*> comp_node_vector_fields_block_pointers;
  std::vector<string> comp_node_vector_fields_block_names;
  std::vector<const double*> comp_cell_scalar_fields_pointers;
  std::vector<string> comp_cell_scalar_fields_names;

  std::vector<Vec_for_vtk_export_t> list_of_vtk_vectors_to_export;
  // on computational grid nodes
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(exact_solution_minus, "exact_solution_minus"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(exact_solution_plus, "exact_solution_plus"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(phi, "phi"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(sharp_solution, "solution"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(ghosted_solution_minus, "minus_solution_with_ghost"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(ghosted_solution_plus, "plus_solution_with_ghost"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);

  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(convergence_analyzer.node_sampled_error, "error_FV"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(convergence_analyzer.node_sampled_ghost_error_minus, "error_minus_with_FV_ghost"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);
  list_of_vtk_vectors_to_export.push_back(Vec_for_vtk_export_t(convergence_analyzer.node_sampled_ghost_error_plus, "error_plus_with_FV_ghost"));
  add_vtk_export_to_list(list_of_vtk_vectors_to_export.back(), comp_node_scalar_fields_pointers, comp_node_scalar_fields_names);

  my_p4est_vtk_write_all_general_lists(p4est, nodes, ghost, P4EST_TRUE, P4EST_TRUE, oss_computational.str().c_str(),
                                       &comp_node_scalar_fields_pointers, &comp_node_scalar_fields_names,
                                       NULL, NULL,
                                       &comp_node_vector_fields_block_pointers, &comp_node_vector_fields_block_names,
                                       &comp_cell_scalar_fields_pointers, &comp_cell_scalar_fields_names,
                                       NULL, NULL, NULL, NULL);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", out_dir.c_str());
  return;
}

// JUMP CONDITIONS
class jc_value_cf_t : public CF_DIM
{
  const test_case_for_scalar_jump_problem_t* test_problem;
public:
  jc_value_cf_t(const test_case_for_scalar_jump_problem_t *test_problem_) : test_problem(test_problem_) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return test_problem->jump_in_solution(DIM(x, y, z));
  }
};

class jc_flux_value_cf_t : public CF_DIM
{
  const test_case_for_scalar_jump_problem_t* test_problem;
  const my_p4est_interpolation_nodes_t* interp_grad_phi;
public:
  jc_flux_value_cf_t(const test_case_for_scalar_jump_problem_t *test_problem_, const my_p4est_interpolation_nodes_t* interp_grad_phi_) : test_problem(test_problem_), interp_grad_phi(interp_grad_phi_) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    double local_grad_phi[P4EST_DIM];
    (*interp_grad_phi)(DIM(x, y, z), local_grad_phi);
    return test_problem->jump_in_normal_flux(local_grad_phi, DIM(x, y, z));
  }
};

void get_sharp_rhs(const p4est_t* p4est, const p4est_nodes_t* nodes, const test_case_for_scalar_jump_problem_t *test_problem, // inputs
                   Vec rhs_minus, Vec rhs_plus) // output
{
  PetscErrorCode ierr;

  double *rhs_minus_p, *rhs_plus_p;
  ierr = VecGetArray(rhs_minus, &rhs_minus_p);  CHKERRXX(ierr);
  ierr = VecGetArray(rhs_plus,  &rhs_plus_p);   CHKERRXX(ierr);

  double xyz_node[P4EST_DIM];
  for(size_t node_idx = 0; node_idx < nodes->indep_nodes.elem_count; ++node_idx)
  {
    node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
    rhs_minus_p[node_idx]  = -test_problem->get_mu_minus()*test_problem->laplacian_u_minus(DIM(xyz_node[0], xyz_node[1], xyz_node[2]));
    rhs_plus_p[node_idx]   = -test_problem->get_mu_plus()*test_problem->laplacian_u_plus(DIM(xyz_node[0], xyz_node[1], xyz_node[2]));
  }
  ierr = VecRestoreArray(rhs_minus, &rhs_minus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_plus,  &rhs_plus_p);   CHKERRXX(ierr);
}

void set_computational_grid_data(const mpi_environment_t &mpi, my_p4est_brick_t* brick, p4est_connectivity_t *connectivity, const splitting_criteria_cf_t* data, const test_case_for_scalar_jump_problem_t *test_problem,
                                 p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, Vec &phi,
                                 my_p4est_hierarchy_t* &hierarchy, my_p4est_node_neighbors_t* &ngbd_n)
{
  if(p4est == NULL)
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*) data;

  for(int i = find_max_level(p4est); i < data->max_lvl; ++i) {
    if(!test_problem->requires_fine_cells_in_negative_domain())
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    else
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf_finest_in_negative, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }

  if(ghost != NULL)
    p4est_ghost_destroy(ghost);
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est, ghost);

  if(nodes != NULL)
    p4est_nodes_destroy(nodes);
  nodes = my_p4est_nodes_new(p4est, ghost);

  if(hierarchy != NULL)
    hierarchy->update(p4est, ghost);
  else
    hierarchy = new my_p4est_hierarchy_t(p4est, ghost, brick);

  if(ngbd_n != NULL)
    ngbd_n->update(hierarchy, nodes);
  else
  {
    ngbd_n = new my_p4est_node_neighbors_t(hierarchy, nodes);
    ngbd_n->init_neighbors();
  }

  PetscErrorCode ierr;
  if(phi != NULL){
    ierr = VecDestroy(phi); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *test_problem->get_levelset_cf(), phi);
  my_p4est_level_set_t ls_coarse(ngbd_n);
  ls_coarse.reinitialize_2nd_order(phi);

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  splitting_criteria_tag_t data_tag(data->min_lvl, data->max_lvl);
  p4est_t* new_p4est = p4est_copy(p4est, P4EST_FALSE);

  while(data_tag.refine_and_coarsen(new_p4est, nodes, phi_p, test_problem->requires_fine_cells_in_negative_domain()))
  {
    my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
    interp_nodes.set_input(phi, linear);

    my_p4est_partition(new_p4est, P4EST_FALSE, NULL);
    p4est_ghost_t *new_ghost  = my_p4est_ghost_new(new_p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(new_p4est, new_ghost);
    p4est_nodes_t *new_nodes  = my_p4est_nodes_new(new_p4est, new_ghost);
    Vec new_phi;
    ierr = VecCreateGhostNodes(new_p4est, new_nodes, &new_phi); CHKERRXX(ierr);
    for(size_t nn = 0; nn < new_nodes->indep_nodes.elem_count; ++nn)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(nn, new_p4est, new_nodes, xyz);
      interp_nodes.add_point(nn, xyz);
    }
    interp_nodes.interpolate(new_phi);

    p4est_destroy(p4est); p4est = new_p4est; new_p4est = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_destroy(ghost); ghost = new_ghost;
    hierarchy->update(p4est, ghost);
    p4est_nodes_destroy(nodes); nodes = new_nodes;
    ngbd_n->update(hierarchy, nodes);

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecDestroy(phi); CHKERRXX(ierr); phi = new_phi;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  p4est_destroy(new_p4est);

  ls_coarse.update(ngbd_n);
  ls_coarse.reinitialize_2nd_order(phi);
  return;
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

void shift_solution_to_match_exact_integral(my_p4est_node_neighbors_t *ngbd_n, Vec phi, Vec sharp_solution, const test_case_for_scalar_jump_problem_t* test_problem)
{
  PetscErrorCode ierr;
  if(ISNAN(test_problem->get_integral_of_solution()))
  {
    ierr = PetscPrintf(MPI_COMM_WORLD, "The average of the exact solution is unknown and would be required to check the accuracy of the solution."); CHKERRXX(ierr);
    return;
  }

  my_p4est_level_set_t ls(ngbd_n);

  Vec solution_minus,solution_plus;
  ierr = VecCreateGhostNodes(ngbd_n->get_p4est(), ngbd_n->get_nodes(), &solution_minus);  ierr = VecGhostCopy(sharp_solution, solution_minus); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(ngbd_n->get_p4est(), ngbd_n->get_nodes(), &solution_plus);   ierr = VecGhostCopy(sharp_solution, solution_plus); CHKERRXX(ierr);

  ls.extend_Over_Interface_TVD(phi, solution_minus);
  double integral_sharp_solution = integrate_over_negative_domain(ngbd_n->get_p4est(), ngbd_n->get_nodes(), phi, solution_minus);
  VecScaleGhost(phi, -1.0);
  ls.extend_Over_Interface_TVD(phi, solution_plus);
  integral_sharp_solution += integrate_over_negative_domain(ngbd_n->get_p4est(), ngbd_n->get_nodes(), phi, solution_plus);
  VecScaleGhost(phi, -1.0);

  ierr = VecDestroy(solution_minus); CHKERRXX(ierr);
  ierr = VecDestroy(solution_plus); CHKERRXX(ierr);

  const double shift = (test_problem->get_integral_of_solution() - integral_sharp_solution)/MULTD(test_problem->get_domain().length(), test_problem->get_domain().height(), test_problem->get_domain().width());

  ierr = VecShiftGhost(sharp_solution, shift); CHKERRXX(ierr);
  return;
}

void print_convergence_summary_in_file(const string& out_folder, const string& test_name, const int &lmin, const int &lmax, const int &ntree, const int &ngrids,
                                       const BoundaryConditionType bc_wtype, const convergence_analyzer_for_jump_node_solver_t& analysis)
{
  string summary_folder = out_folder + "/summaries";
  ostringstream command;
  command << "mkdir -p " << summary_folder.c_str();
  int uu = system(command.str().c_str()); (void) uu; // create the summary folder
  string summary_file = summary_folder + "/convergence_summary_" + test_name
      + "_lmin" + to_string(lmin) + "_lmax" + to_string(lmax) + "_ngrids" + to_string(ngrids) + "_ntree" + to_string(ntree)
      + "_" + (bc_wtype == DIRICHLET ? "dirichlet": "neumann") + ".dat";

  FILE *fid = fopen(summary_file.c_str(), "w");
  fprintf(fid, "=================================================================\n");
  fprintf(fid, "========================= SUMMARY ===============================\n");
  fprintf(fid, "=================================================================\n");
  fprintf(fid, "Test case: %s (%d-D)\n", test_name.c_str(), P4EST_DIM);
  fprintf(fid, "lmin: %d\n", lmin);
  fprintf(fid, "lmax: %d\n", lmax);
  fprintf(fid, "Number of grids: %d\n", ngrids);
  fprintf(fid, "Number of trees along minimum dimension of domain in macromesh: %d\n", ntree);
  fprintf(fid, "Wall boundary condition: %s\n", (bc_wtype == DIRICHLET ? "dirichlet" : "neumann"));
  fprintf(fid, "Resolution: " );
  for(int idx = 0; idx < ngrids; ++idx)
    fprintf(fid, "%d/%d%s", ntree*(1 << (lmin + idx)), ntree*(1 << (lmax + idx)), (idx == ngrids - 1 ? "\n" : " "));

  fprintf(fid, "Error on solution (FV): ");
  P4EST_ASSERT(analysis.errors_in_solution.size() == (size_t) ngrids);
  for(int idx = 0; idx < ngrids; ++idx)
    fprintf(fid, "%.5e%s", analysis.errors_in_solution[idx], (idx == ngrids - 1 ? "\n" : " "));

  fprintf(fid, "=================================================================\n");
  fprintf(fid, "===================== END OF SUMMARY ============================\n");
  fprintf(fid, "=================================================================");
  fclose(fid);
  printf("Summary file printed in %s\n", summary_file.c_str());
}

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin",            "min level of the tree in the computational grid, default is " + to_string(default_lmin));
  cmd.add_option("lmax",            "max level of the tree in the computational grid, default is " + to_string(default_lmax));
  cmd.add_option("ngrids",          "number of computational grids for accuracy analysis, default is " + to_string(default_ngrids));
  ostringstream oss; oss << default_bc_wtype;
  cmd.add_option("bc_wtype",        "type of boundary condition to use on the wall ('Dirichlet' or 'Neumann'), default is " + oss.str());
  cmd.add_option("save_vtk",        "saves the p4est's (computational and interface-capturing grids) in vtk format");
  cmd.add_option("work_dir",        "exportation directory, if not defined otherwise in the environment variable OUT_DIR. \n\
\tThis is required if saving vtk or summary files: work_dir/vtu for vtk files work_dir/summaries for summary files. Default is " + default_work_folder);
  cmd.add_option("ntree",           "number of trees in the macromesh along every dimension of the computational domain. Default value is " + to_string(default_ntree));
  cmd.add_option("test",            "Test problem to choose. Available choices are (default test number is " + to_string(default_test_number) +"): \n" + list_of_test_problems_for_scalar_jump_problems.get_description_of_tests() + "\n");
  cmd.add_option("summary",         "Prints a summary of the convergence results in a file on disk if present. Default is " + string(default_print_summary ? "true" : "false"));
  oss.str("");
  oss << default_interp_method_phi;
  cmd.add_option("phi_interp",      "interpolation method for the node-sampled levelset function. Default is " + oss.str());
  if(cmd.parse(argc, argv, main_description))
    return 0;

  const int lmin                        = cmd.get<int>("lmin", default_lmin);
  const int lmax                        = cmd.get<int>("lmax", default_lmax);
  const int ngrids                      = cmd.get<int>("ngrids", default_ngrids);
  const int test_number                 = cmd.get<int>("test", default_test_number);
  const BoundaryConditionType bc_wtype  = cmd.get<BoundaryConditionType>("bc_wtype", default_bc_wtype);
  const int ntree                       = cmd.get<int>("ntree", default_ntree);
  const int n_xyz[P4EST_DIM]            = {DIM(ntree, ntree, ntree)};
  const bool save_vtk                   = cmd.contains("save_vtk");
  convergence_analyzer_for_jump_node_solver_t analysis;

  parStopWatch watch, watch_global;
  watch_global.start("Total run time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const test_case_for_scalar_jump_problem_t *test_problem = list_of_test_problems_for_scalar_jump_problems[test_number];
  BoundaryConditionsDIM bc;
  BCWALLTYPE bc_wall_type(bc_wtype); bc.setWallTypes(bc_wall_type);
  BCWALLVAL bc_wall_val(test_problem, &bc_wall_type); bc.setWallValues(bc_wall_val);

  const string out_folder = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR"))) + "/" + test_problem->get_name();
  const string vtk_out = out_folder + "/vtu";

  connectivity = my_p4est_brick_new(n_xyz, test_problem->get_xyz_min(), test_problem->get_xyz_max(), &brick, test_problem->get_periodicity());
  splitting_criteria_cf_t       *data       = NULL;
  p4est_t                       *p4est      = NULL;
  p4est_nodes_t                 *nodes      = NULL;
  p4est_ghost_t                 *ghost      = NULL;
  my_p4est_hierarchy_t          *hierarchy  = NULL;
  my_p4est_node_neighbors_t     *ngbd_n     = NULL;
  Vec                            phi        = NULL;
  Vec                            phi_xxyyzz[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  Vec                            grad_phi   = NULL;
  my_p4est_interpolation_nodes_t *interp_grad_phi = NULL;

  for(int iter = 0; iter < ngrids; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin + iter, lmax + iter); CHKERRXX(ierr);

    /* build/updates the computational grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods, its cell neighborhoods
     * the REINITIALIZED levelset on the computational grid
     */
    data = new splitting_criteria_cf_t(lmin + iter, lmax + iter, test_problem->get_levelset_cf());
    set_computational_grid_data(mpi, &brick, connectivity, data, test_problem,
                                p4est, ghost, nodes, phi, hierarchy, ngbd_n);
    if(grad_phi != NULL){
      ierr = VecDestroy(grad_phi); CHKERRXX(ierr); }
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(phi_xxyyzz[dim] != NULL){
        ierr = VecDestroy(phi_xxyyzz[dim]); CHKERRXX(ierr); }
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_xxyyzz[dim]); CHKERRXX(ierr);
    }
    ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &grad_phi); CHKERRXX(ierr);
    ngbd_n->first_derivatives_central(phi, grad_phi);
    ngbd_n->second_derivatives_central(phi, phi_xxyyzz);


    if(interp_grad_phi != NULL)
      delete interp_grad_phi;
    interp_grad_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
    interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);

    jc_value_cf_t jump_solution(test_problem);
    jc_flux_value_cf_t jump_normal_flux(test_problem, interp_grad_phi);

    /* TEST THE JUMP SOLVER AND COMPARE TO ORIGINAL GFM */
    Vec sharp_rhs_minus, sharp_rhs_plus;
    ierr = VecCreateGhostNodes(p4est, nodes, &sharp_rhs_minus); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &sharp_rhs_plus); CHKERRXX(ierr);
    get_sharp_rhs(p4est, nodes, test_problem, sharp_rhs_minus, sharp_rhs_plus);

    my_p4est_poisson_nodes_mls_t *solver = new my_p4est_poisson_nodes_mls_t(ngbd_n);
    analysis.jump_solver = solver;
    solver->set_use_centroid_always(true);
    solver->set_store_finite_volumes(true);
    solver->set_jump_scheme(0);
    solver->set_use_sc_scheme(true);
    solver->set_integration_order(1);
    solver->set_lip(2.0);
    solver->set_wc(bc_wall_type, bc_wall_val);

    solver->add_interface(MLS_INTERSECTION, phi, DIM(phi_xxyyzz[0], phi_xxyyzz[1],phi_xxyyzz[2]), jump_solution, jump_normal_flux);
    solver->set_mu(test_problem->get_mu_minus(), test_problem->get_mu_plus());

    solver->set_rhs(sharp_rhs_minus, sharp_rhs_plus);
    solver->set_diag(0.0, 0.0);

    Vec sharp_solution, ghosted_solution_minus, ghosted_solution_plus;
    ierr = VecCreateGhostNodes(p4est, nodes, &sharp_solution); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &ghosted_solution_minus); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &ghosted_solution_plus); CHKERRXX(ierr);
    watch.start("Total time:");
    solver->solve(sharp_solution);
    watch.stop(); watch.read_duration();

    /* if null space, shift solution */
    if(solver->get_matrix_has_nullspace())
      shift_solution_to_match_exact_integral(ngbd_n, phi, sharp_solution, test_problem);

    solver->get_ghosted_solutions(sharp_solution, ghosted_solution_minus, ghosted_solution_plus);

    /* measure the error(s) */
    analysis.measure_errors(ghosted_solution_minus, ghosted_solution_plus, test_problem);

    if(save_vtk)
    {
      Vec exact_solution_minus = NULL, exact_solution_plus = NULL; // to enable illustration of exact solution with wrap-by-scalar in paraview or to calculate the integral of the exact solution, numerically
      ierr = VecCreateGhostNodes(p4est, nodes, &exact_solution_minus);  CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &exact_solution_plus);   CHKERRXX(ierr);
      get_sampled_exact_solution(exact_solution_minus, exact_solution_plus, p4est, nodes, test_problem);

      if(save_vtk)
        save_VTK(vtk_out, iter, exact_solution_minus, exact_solution_plus, phi, sharp_solution, ghosted_solution_minus, ghosted_solution_plus, analysis, &brick);
      ierr = VecDestroy(exact_solution_minus); CHKERRXX(ierr);
      ierr = VecDestroy(exact_solution_plus); CHKERRXX(ierr);
    }

    // destroy data created for this iteration
    ierr = VecDestroy(phi);       CHKERRXX(ierr); phi = NULL;
    ierr = VecDestroy(grad_phi);  CHKERRXX(ierr); grad_phi = NULL;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecDestroy(phi_xxyyzz[dim]); CHKERRXX(ierr); phi_xxyyzz[dim] = NULL;
    }
    ierr = VecDestroy(sharp_rhs_minus); CHKERRXX(ierr); sharp_rhs_minus = NULL;
    ierr = VecDestroy(sharp_rhs_plus);  CHKERRXX(ierr); sharp_rhs_plus = NULL;
    ierr = VecDestroy(sharp_solution);  CHKERRXX(ierr); sharp_solution = NULL;
    ierr = VecDestroy(ghosted_solution_minus);  CHKERRXX(ierr); ghosted_solution_minus = NULL;
    ierr = VecDestroy(ghosted_solution_plus);  CHKERRXX(ierr); ghosted_solution_plus = NULL;

    delete data;
    delete analysis.jump_solver;
  }

  delete ngbd_n;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);
  watch_global.stop(); watch_global.read_duration();

  return 0;
}
