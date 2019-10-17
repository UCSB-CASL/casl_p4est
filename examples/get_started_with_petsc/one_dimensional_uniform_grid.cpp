#include "one_dimensional_uniform_grid.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>

void one_dimensional_uniform_grid::set_partition_and_ghosts(const PetscInt &n_local, const PetscInt &ghost_layer_size_)
{
#ifdef DEBUG
  PetscErrorCode ierr;
  ierr = (n_local < 0); CHKERRXX(ierr);
  std::vector<PetscInt> ghost_layer_on_proc(mpi.size(), 0);
  MPI_Allgather(&ghost_layer_size_, 1, MPIU_INT, ghost_layer_on_proc.data(), 1, MPIU_INT, mpi.comm());
  bool check_if_const_ghost_layer = true;
  for (int r = 0; check_if_const_ghost_layer && r < mpi.size(); ++r)
    check_if_const_ghost_layer = check_if_const_ghost_layer && (ghost_layer_on_proc[r] == ghost_layer_size_);
  ierr = !check_if_const_ghost_layer; CHKERRXX(ierr);
#endif
  ghost_layer_size = ghost_layer_size_;
  // initialize number of local nodes, offsets and number of global nodes
  n_owned = n_local;
  offset_on_rank.resize(mpi.size(), 0);
  MPI_Allgather(&n_local, 1, MPIU_INT, offset_on_rank.data(), 1, MPIU_INT, mpi.comm());
  PetscInt offset = 0;
  for (int r = 0; r < mpi.size(); ++r) {
    PetscInt owned_local = offset_on_rank[r];
    offset_on_rank[r] = offset;
    offset += owned_local;
  }
  n_global = offset;
  // initialize layer nodes, inner nodes and ghost nodes
  const bool has_left_neighbor  = periodic || (mpi.rank() > 0);
  const bool has_right_neighbor = periodic || (mpi.rank() < mpi.size()-1);
  const PetscInt n_layer_nodes = ghost_layer_size_*((has_left_neighbor?1:0)+(has_right_neighbor?1:0));
  // check if valid
#ifdef DEBUG
  ierr =  n_layer_nodes > n_owned; CHKERRXX(ierr);
#endif
  // initialize layer nodes
  layer_nodes.resize(n_layer_nodes);
  PetscInt layer_idx = 0;
  if(has_left_neighbor)
    for (PetscInt k = 0; k < ghost_layer_size_; ++k)
      layer_nodes[layer_idx++] = k;
  if(has_right_neighbor)
    for (PetscInt k = 0; k < ghost_layer_size_; ++k)
      layer_nodes[layer_idx++] = n_owned-ghost_layer_size_+k;
#ifdef DEBUG
  ierr = layer_idx !=n_layer_nodes; CHKERRXX(ierr);
#endif
  //initialize inner nodes
  inner_nodes.resize(n_owned-n_layer_nodes);
  for (size_t k = 0; k < inner_nodes.size(); ++k)
    inner_nodes[k] = (has_left_neighbor? ghost_layer_size_ : 0) + k;
  // initialize the ghost nodes
  // For the ghost nodes, the corresponding local index is either negative (neighbor to the left of
  // the boundary of the process' subdomain) or greater than or equal to n_owned (neighbor to the
  // right of the boundary of the process' subdomain)
  const PetscInt n_ghost_nodes = ghost_layer_size_*((has_left_neighbor?1:0)+(has_right_neighbor?1:0));
  ghost_nodes.resize(n_ghost_nodes);
  PetscInt ghost_idx = 0; // index of the ghost node in the ghost_nodes standard vector
  if(has_left_neighbor){
    for (PetscInt k = 0; k < ghost_layer_size_; ++k){
      ghost_nodes[ghost_idx] = ghost_node(n_owned+ghost_idx, -ghost_layer_size_+k);
      ghost_idx++;
    }
  }
  if(has_right_neighbor){
    for (PetscInt k = 0; k < ghost_layer_size_; ++k){
      ghost_nodes[ghost_idx] = ghost_node(n_owned+ghost_idx, n_owned+k);
      ghost_idx++;
    }
  }
#ifdef DEBUG
  ierr = ghost_idx != n_ghost_nodes; CHKERRXX(ierr);
#endif
  loc_idx_min = (has_left_neighbor? -ghost_layer_size_:0);
  loc_idx_max = n_owned + (has_right_neighbor? ghost_layer_size_ : 0);
}

PetscInt one_dimensional_uniform_grid::global_idx_of_local_node(const PetscInt j_loc) const
{
  PetscErrorCode ierr = (j_loc < loc_idx_min || j_loc >= loc_idx_max); CHKERRXX(ierr);
  PetscInt glo_idx = offset_on_rank[mpi.rank()] + j_loc;
  if(periodic)
    glo_idx = (glo_idx%n_global+n_global)%n_global;
  return glo_idx;
}

double one_dimensional_uniform_grid::get_x_of_node(const PetscInt j_loc) const
{
  PetscInt glo_idx = global_idx_of_local_node(j_loc);
  return xmin + ((double) glo_idx)*get_delta_x();
}

double one_dimensional_uniform_grid::get_delta_x() const
{
  return (xmax-xmin)/((double) (n_global-(periodic?0:1)));
}

PetscInt one_dimensional_uniform_grid::array_idx_of_node(const PetscInt &loc_idx) const
{
  PetscErrorCode ierr = (loc_idx < loc_idx_min || loc_idx >= loc_idx_max); CHKERRXX(ierr);
  if(loc_idx >= 0 && loc_idx < n_owned) // same index for owned nodes
    return  loc_idx;

  // for ghost nodes, the array index was stored at construction (in set_partition_and_ghosts),
  // just find the right node

  // quite bad practice, a binary serach would be better but it's not the focus of the current example
  bool is_found = false;
  size_t ghost_idx = 0; // index of the ghost node to be found in the ghost_nodes standard vector
  for ( ghost_idx = 0; !is_found && (ghost_idx < ghost_nodes.size()); ++ghost_idx){
    is_found = (ghost_nodes[ghost_idx].local_idx==loc_idx);
    if(is_found)
      break;
  }
  CHKERRXX(!is_found);
  return ghost_nodes[ghost_idx].array_idx;
}


void one_dimensional_uniform_grid::calculate_second_derivative_of_field(Vec node_sampled_field, Vec second_derivative) const
{
  const double *node_sample_field_p;
  double *second_derivative_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(node_sampled_field, &node_sample_field_p); CHKERRXX(ierr);
  ierr = VecGetArray(second_derivative, &second_derivative_p); CHKERRXX(ierr);
  // calculation on layer nodes first
  for (size_t k = 0; k < layer_nodes.size(); ++k)
    second_derivative_p[layer_nodes[k]] = second_derivative_of_field_at_node(layer_nodes[k], node_sample_field_p);
  // scatter the results so that ghost values are updated on remote processes
  ierr = VecGhostUpdateBegin(second_derivative, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // calculation on inner nodes
  for (size_t k = 0; k < inner_nodes.size(); ++k)
    second_derivative_p[inner_nodes[k]] = second_derivative_of_field_at_node(inner_nodes[k], node_sample_field_p);
  // complete communications
  ierr = VecGhostUpdateEnd(second_derivative, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(second_derivative, &second_derivative_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(node_sampled_field, &node_sample_field_p); CHKERRXX(ierr);
}


double one_dimensional_uniform_grid::second_derivative_of_field_at_node(PetscInt idx_loc, const double *node_sampled_field_p) const
{
  PetscErrorCode ierr = idx_loc < 0 || idx_loc >= number_of_locally_owned_nodes(); CHKERRXX(ierr); // this function is meant for owned nodes only!
  ierr = idx_loc-1 < loc_idx_min && idx_loc+1 >= loc_idx_max; CHKERRXX(ierr);
  const double h = get_delta_x();
  if(idx_loc-1 >= loc_idx_min && idx_loc+1 < loc_idx_max)
    return (node_sampled_field_p[array_idx_of_node(idx_loc+1)] - 2.0*node_sampled_field_p[array_idx_of_node(idx_loc)] + node_sampled_field_p[array_idx_of_node(idx_loc-1)])/(h*h);
  // if not enough neighbor, slide the stencil within the domain (order h error, still kinda ok)
  if(idx_loc-1 < loc_idx_min && idx_loc+1 < loc_idx_max)
    return (node_sampled_field_p[array_idx_of_node(idx_loc+2)] - 2.0*node_sampled_field_p[array_idx_of_node(idx_loc+1)] + node_sampled_field_p[array_idx_of_node(idx_loc)])/(h*h);
  return (node_sampled_field_p[array_idx_of_node(idx_loc-2)] - 2.0*node_sampled_field_p[array_idx_of_node(idx_loc-1)] + node_sampled_field_p[array_idx_of_node(idx_loc)])/(h*h);
}

// this following function has a layout very similar to create_compact_finite_differences_operators
// except that it does not fill values in, it simply preallocate memory space for the relevant values
// Go through the create_compact_finite_differences_operators functon first, understand it and then
// compare it to the following one!
void one_dimensional_uniform_grid::create_and_preallocate_compact_finite_difference_matrices(Mat &lhs_matrix, Mat &rhs_matrix, const unsigned int &OOA) const
{
  PetscErrorCode ierr;
  if(lhs_matrix!=NULL){
    ierr = MatDestroy(lhs_matrix); CHKERRXX(ierr); }
  if(rhs_matrix!=NULL){
    ierr = MatDestroy(rhs_matrix); CHKERRXX(ierr); }

  // create the matrices
  ierr = MatCreate(mpi.comm(), &lhs_matrix); CHKERRXX(ierr);
  ierr = MatCreate(mpi.comm(), &rhs_matrix); CHKERRXX(ierr);

  /* set up the matrix */
  ierr = MatSetType(lhs_matrix, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetType(rhs_matrix, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(lhs_matrix, n_owned, n_owned, n_global, n_global); CHKERRXX(ierr);
  ierr = MatSetSizes(rhs_matrix, n_owned, n_owned, n_global, n_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(lhs_matrix); CHKERRXX(ierr);
  ierr = MatSetFromOptions(rhs_matrix); CHKERRXX(ierr);
  /*
   * For preallocation and creation purposes, we need to make a distinction between matrix entries
   * that are fully local and and the ones that involve ghost values. Consider a square matrix M
   * distributed over 3 processes for instance, the global (full) matrix is divided in 3x3 blocks as
   *     ________________________________________
   *     |            |             |            |   /|\
   *     |  M_{0,0}   |   M_{0,1}   |   M_{0,2}  |    |  n_loc_{0}
   *     |____________|_____________|____________|   \|/
   *     |            |             |            |   /|\
   * M = |  M_{1,0}   |   M_{1,1}   |   M_{1,2}  |    |  n_loc_{1}
   *     |____________|_____________|____________|   \|/
   *     |            |             |            |   /|\
   *     |  M_{2,0}   |   M_{2,1}   |   M_{2,2}  |    |  n_loc_{2}
   *     |____________|_____________|____________|   \|/
   *
   *     <------------><------------><----------->
   *        n_loc_{0}     n_loc_{1}    n_loc_{2}
   *
   * where n_loc_{r} is number of values locally owned on processor r.
   *
   * Any value belonging to a diagonal block M_{r,r} is fully local, while values belonging to blocks M_{i, j} where i!=j
   * involve nonlocal values (possibly ghost values)
   *
   * */
  std::vector<PetscInt> lhs_number_of_nonzero_in_diagonal_block(n_owned, 0), rhs_number_of_nonzero_in_diagonal_block(n_owned, 0);
  std::vector<PetscInt> lhs_number_of_nonzero_in_offdiagonal_block(n_owned, 0), rhs_number_of_nonzero_in_offdiagonal_block(n_owned, 0);

  for (PetscInt node_idx = 0; node_idx < n_owned; ++node_idx) {
    if(node_exists(node_idx-1) && node_exists(node_idx+1)) {
      lhs_number_of_nonzero_in_diagonal_block[node_idx]++; // diagonal term
      const bool right_neighbor_is_ghost = node_is_ghost(node_idx+1);
      const bool left_neighbor_is_ghost = node_is_ghost(node_idx-1);
      if(right_neighbor_is_ghost){
        lhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
        rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
      } else {
        lhs_number_of_nonzero_in_diagonal_block[node_idx]++;
        rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
      }
      if(left_neighbor_is_ghost){
        lhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
        rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
      } else {
        lhs_number_of_nonzero_in_diagonal_block[node_idx]++;
        rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
      }
      if(OOA==6)
      {
        if(node_exists(node_idx-2) && node_exists(node_idx+2))
        {
          if(node_is_ghost(node_idx+2))
            rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
          else
            rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
          if(node_is_ghost(node_idx-2))
            rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
          else
            rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
        }
      }
    }
    else { // wall
      PetscInt wall_normal = (!node_exists(node_idx-1)? + 1: -1);
      lhs_number_of_nonzero_in_diagonal_block[node_idx]++; // diagonal term
      rhs_number_of_nonzero_in_diagonal_block[node_idx]++; // diagonal term
      ierr = !node_exists(node_idx+wall_normal); CHKERRXX(ierr);
      if(node_is_ghost(node_idx+wall_normal)){
        lhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
        rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
      } else {
        lhs_number_of_nonzero_in_diagonal_block[node_idx]++;
        rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
      }
      ierr = !node_exists(node_idx+2*wall_normal); CHKERRXX(ierr);
      if(node_is_ghost(node_idx+2*wall_normal))
        rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
      else
        rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
      ierr = !node_exists(node_idx+3*wall_normal); CHKERRXX(ierr);
      if(node_is_ghost(node_idx+3*wall_normal))
        rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
      else
        rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
      ierr = !node_exists(node_idx+4*wall_normal); CHKERRXX(ierr);
      if(node_is_ghost(node_idx+4*wall_normal))
        rhs_number_of_nonzero_in_offdiagonal_block[node_idx]++;
      else
        rhs_number_of_nonzero_in_diagonal_block[node_idx]++;
    }
  }

  ierr = MatSeqAIJSetPreallocation(lhs_matrix, 0, (const PetscInt*) lhs_number_of_nonzero_in_diagonal_block.data()); CHKERRXX(ierr);
  ierr = MatSeqAIJSetPreallocation(rhs_matrix, 0, (const PetscInt*) rhs_number_of_nonzero_in_diagonal_block.data()); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(lhs_matrix, 0, (const PetscInt*) lhs_number_of_nonzero_in_diagonal_block.data(),
                                   0, (const PetscInt*) lhs_number_of_nonzero_in_offdiagonal_block.data()); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(rhs_matrix, 0, (const PetscInt*) rhs_number_of_nonzero_in_diagonal_block.data(),
                                   0, (const PetscInt*) rhs_number_of_nonzero_in_offdiagonal_block.data()); CHKERRXX(ierr);
}

// The following function creates two matrix for the respective lhs and rhs operators required in the
// compact finite difference approach: for inner points, one wants to solve equations of the type
// alpha*f'_{i-1} + f'_{i} + alpha*f'_{i+1} = b*f_{i-2} + a*f_{i-1} + f_{i} + a*f_{i+1} + b*f_{i+2}
// where alpha, a and b depend on the desired order of accuracy OOA (in particular b = 0 is OOA==4
// so that the required number of ghost neighbors is only one in that case)
// (The treatment for (close-to) wall nodes differs in case of nonperiodic boundary conditions, see
// the original paper cited in the main file for more information)
PetscErrorCode one_dimensional_uniform_grid::create_compact_finite_differences_operators(Mat &lhs_matrix, Mat &rhs_matrix, const unsigned int &OOA) const
{
  PetscErrorCode ierr;
  create_and_preallocate_compact_finite_difference_matrices(lhs_matrix, rhs_matrix, OOA);

  ierr = !((OOA==4) || (OOA==6)); CHKERRQ(ierr);

  const double alpha  = (OOA==4) ? 0.25 : 1.0/3.0;
  const double a      = 2.0*(alpha+2.0)/3.0;
  const double b      = (4.0*alpha-1)/3.0;

  for (PetscInt node_idx = 0; node_idx < n_owned; ++node_idx) {
    if(node_exists(node_idx-1) && node_exists(node_idx+1)) {
      PetscInt glo_idx    = global_idx_of_local_node(node_idx);
      PetscInt glo_idx_r  = global_idx_of_local_node(node_idx+1);
      PetscInt glo_idx_l  = global_idx_of_local_node(node_idx-1);
      if(OOA==4 || (OOA==6 && node_exists(node_idx-2) && node_exists(node_idx+2)))
      {
        // lhs matrix
        ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx, 1.0, ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx_r, alpha, ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx_l, alpha, ADD_VALUES); CHKERRQ(ierr);
        //rhs matrix
        ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx_r, 0.5*a/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx_l, -0.5*a/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
        if(OOA==6)
        {
          ierr = MatSetValue(rhs_matrix, glo_idx, global_idx_of_local_node(node_idx+2), 0.25*b/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
          ierr = MatSetValue(rhs_matrix, glo_idx, global_idx_of_local_node(node_idx-2), -0.25*b/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
        }
      }
      else // OOA==6 but the second degree neighbors do not exist
      {
        double aalpha = 0.25;
        double aa = 2.0*(aalpha+2.0)/3.0;
        // lhs matrix
        ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx, 1.0, ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx_r, aalpha, ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx_l, aalpha, ADD_VALUES); CHKERRQ(ierr);
        //rhs matrix
        ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx_r, 0.5*aa/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx_l, -0.5*aa/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
      }
    }
    else { // wall
      PetscInt wall_normal = (!node_exists(node_idx-1)? + 1: -1);
      double aalpha = 3.0;
      double aa = -17.0/6.0*((double) wall_normal);
      double bb = 1.5*((double) wall_normal);
      double cc = 1.5*((double) wall_normal);
      double dd = -1./6.*((double) wall_normal);
      PetscInt glo_idx    = global_idx_of_local_node(node_idx);
      ierr = !node_exists(node_idx+wall_normal); CHKERRQ(ierr);
      ierr = !node_exists(node_idx+2*wall_normal); CHKERRQ(ierr);
      ierr = !node_exists(node_idx+3*wall_normal); CHKERRQ(ierr);
      PetscInt glo_idx_n    = global_idx_of_local_node(node_idx+wall_normal);
      PetscInt glo_idx_nn   = global_idx_of_local_node(node_idx+2*wall_normal);
      PetscInt glo_idx_nnn  = global_idx_of_local_node(node_idx+3*wall_normal);
      // lhs matrix
      ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx,      1.0, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(lhs_matrix, glo_idx, glo_idx_n,    aalpha, ADD_VALUES); CHKERRQ(ierr);
      // rhs matrix
      ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx,      aa/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx_n,    bb/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx_nn,   cc/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(rhs_matrix, glo_idx, glo_idx_nnn,  dd/get_delta_x(), ADD_VALUES); CHKERRQ(ierr);
    }
  }

  // Assemble the matrices
  ierr = MatAssemblyBegin(lhs_matrix, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(rhs_matrix, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (lhs_matrix, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (rhs_matrix, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return ierr;
}

void one_dimensional_uniform_grid::calculate_first_derivative_compact_fd(Vec node_sampled_function, Vec first_derivative_compact_finite_differences, const unsigned int &OOA) const
{
  Vec compact_fd_rhs = NULL;
  PetscErrorCode ierr = VecDuplicate(node_sampled_function, &compact_fd_rhs); CHKERRXX(ierr);
  Mat lhs_matrix  = NULL;
  Mat rhs_matrix  = NULL;
  create_compact_finite_differences_operators(lhs_matrix, rhs_matrix, OOA);
  ierr = MatMult(rhs_matrix, node_sampled_function, compact_fd_rhs);
  KSP krylov_solver = NULL;
  ierr = KSPCreate(mpicomm(), &krylov_solver); CHKERRXX(ierr);
  ierr = KSPSetOperators(krylov_solver, lhs_matrix, lhs_matrix, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  // Solve the system
  ierr = KSPSolve(krylov_solver, compact_fd_rhs, first_derivative_compact_finite_differences); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(first_derivative_compact_finite_differences, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (first_derivative_compact_finite_differences, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if (krylov_solver != NULL){
    ierr = KSPDestroy(krylov_solver); CHKERRXX(ierr); }
  if(lhs_matrix!=NULL){
    ierr = MatDestroy(lhs_matrix); CHKERRXX(ierr); }
  if(rhs_matrix!=NULL){
    ierr = MatDestroy(rhs_matrix); CHKERRXX(ierr); }
  if(compact_fd_rhs!=NULL){
    ierr = VecDestroy(compact_fd_rhs); CHKERRXX(ierr); }
}


void one_dimensional_uniform_grid::shuffle()
{
  srand (mpirank()*time(NULL));
  PetscInt nnodes_to_add = (rand()%n_owned)-n_owned/2;
  std::vector<PetscInt> n_to_add_on_proc(mpisize(), 0);
  // calculate how many nodes are globally added if no correction
  PetscInt total_added;
  MPI_Allreduce(&nnodes_to_add, &total_added, 1, MPIU_INT, MPI_SUM, mpi.comm());
  // correct the individual numbers to add to make sure that no global difference is made
  nnodes_to_add -= (total_added/mpisize() + ((total_added>0)?1:-1)*((mpirank() < abs(total_added%mpisize()))? 1:0));
#ifdef DEBUG
  MPI_Allreduce(&nnodes_to_add, &total_added, 1, MPIU_INT, MPI_SUM, mpi.comm());
  PetscErrorCode ierr = (total_added!=0); CHKERRXX(ierr);
  PetscInt n_global_before = n_global;
#endif
  // reset the grid partition
  set_partition_and_ghosts(n_owned+nnodes_to_add, ghost_layer_size);
#ifdef DEBUG
  ierr = (n_global!=n_global_before); CHKERRXX(ierr);
#endif
}

void one_dimensional_uniform_grid::remap(Vec vector_on_olg_grid, Vec vector_on_current_grid) const
{
  PetscErrorCode ierr = 0;
#ifdef DEBUG
  PetscInt size_on_old_grid, size_on_current_grid;
  ierr = VecGetSize(vector_on_olg_grid, &size_on_old_grid); CHKERRXX(ierr);
  ierr = VecGetSize(vector_on_current_grid, &size_on_current_grid); CHKERRXX(ierr);
  ierr = (size_on_old_grid != size_on_current_grid); CHKERRXX(ierr);
  ierr = (n_global != size_on_current_grid); CHKERRXX(ierr);
#endif

  // we will map values from the original vector (defined on an older grid) to values in the new vector defined on the new grid
  IS idx_set_origin, idx_set_destination;

  ISLocalToGlobalMapping l2g;
  ierr = VecGetLocalToGlobalMapping(vector_on_current_grid, &l2g); CHKERRXX(ierr); // l2g is a Petsc object that maps local petsc indices to global petsc indices

  const PetscInt *idx;
  PetscInt l2g_size;
  ierr = ISLocalToGlobalMappingGetIndices(l2g, &idx); CHKERRXX(ierr);
  ierr = ISLocalToGlobalMappingGetSize(l2g, &l2g_size); CHKERRXX(ierr);
  // after these operations, idx is an array pointing to the l2g_size global indices that are mapped to the local
  // indices of the petsc vector defined on the current grid. Given our definition of global indicies and since
  // the original and current grids are identical, the global indices of the origins and destinations are identical

  // one can choose between two rather equivalent approaches from there

//  // APPROACH 1: use the same global index sets for origins and destinations
//  ierr = ISCreateGeneral(mpicomm(), l2g_size, idx, PETSC_USE_POINTER, &idx_set_destination); CHKERRXX(ierr);
//  ierr = ISCreateGeneral(mpicomm(), l2g_size, idx, PETSC_USE_POINTER, &idx_set_origin); CHKERRXX(ierr);
//  VecScatter scatter_context;
//  ierr = VecScatterCreate(vector_on_olg_grid, idx_set_origin, vector_on_current_grid, idx_set_destination, &scatter_context); CHKERRXX(ierr);
//  ierr = ISLocalToGlobalMappingRestoreIndices(l2g, &idx); CHKERRXX(ierr);
//  // here again, one could perform some lcoal operations in between these two calls to hide communications
//  // however, this is not the purpose of this illustration
//  ierr = VecScatterBegin(scatter_context, vector_on_olg_grid, vector_on_current_grid, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecScatterEnd(scatter_context, vector_on_olg_grid, vector_on_current_grid, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // APPROACH 2: map globally indexed values from the origin vector directly to the local form of the destination vector
  // get the local form (includes ghosted values)
  Vec vector_on_current_grid_local;
  ierr = VecGhostGetLocalForm(vector_on_current_grid, &vector_on_current_grid_local); CHKERRXX(ierr);
  // the destination memory location are the array elements of index 0, 1, ..., n_owned+nghosts_1 of the local
  ierr = ISCreateStride(mpicomm(), l2g_size, 0, 1, &idx_set_destination); CHKERRXX(ierr);
  // the origin memory location are the global indices of the full, parallel origin vector
  ierr = ISCreateGeneral(mpicomm(), l2g_size, idx, PETSC_USE_POINTER, &idx_set_origin); CHKERRXX(ierr);
  VecScatter scatter_context;
  ierr = VecScatterCreate(vector_on_olg_grid, idx_set_origin, vector_on_current_grid_local, idx_set_destination, &scatter_context); CHKERRXX(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(l2g, &idx); CHKERRXX(ierr);
  // here again, one could perform some lcoal operations in between these two calls to hide communications
  // however, this is not the purpose of this illustration
  ierr = VecScatterBegin(scatter_context, vector_on_olg_grid, vector_on_current_grid_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecScatterEnd(scatter_context, vector_on_olg_grid, vector_on_current_grid_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vector_on_current_grid, &vector_on_current_grid_local); CHKERRXX(ierr);

  // APPROACH 2 may be better than APPROACH 1 if one wants the ghosted values to be synchronized immediately after the remapping
  // Indeed, approach one synchronizes all locally owned values on the relevant processes only, the ghosted values are not updated
  // on the relevant other processes after completion of APPROACH 1. To be fully equivalent, APPROACH ONE would require additional calls to
  //  ierr = VecGhostUpdateBegin(vector_on_current_grid, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //  ierr = VecGhostUpdateEnd(vector_on_current_grid, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // If you wan to check out the latter comment by yourself, uncomment the following 4 lines of code and run the two different approaches
//  const double *read_values;
//  ierr = VecGetArrayRead(vector_on_current_grid, &read_values); CHKERRXX(ierr);
//  std::cout << "The first ghosted value on proc " << mpirank() << " is " << read_values[array_idx_of_node(ghost_nodes[0].local_idx)] << std::endl;
//  ierr = VecRestoreArrayRead(vector_on_current_grid, &read_values); CHKERRXX(ierr);


  ierr = ISDestroy(idx_set_origin); CHKERRXX(ierr);
  ierr = ISDestroy(idx_set_destination); CHKERRXX(ierr);
  ierr = VecScatterDestroy(scatter_context); CHKERRXX(ierr);

}
