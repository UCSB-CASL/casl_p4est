/*
 * Test the cell based poisson solver with
 * - possibly very stretched grids
 * - neumann wall boundary conditions on the y-normal walls
 * - periodic boundary conditions along x and z
 * as likely to be found in SHS channel simulations
 */

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_shs_channel.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_shs_channel.h>
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

const double default_length         = 6.0;
const double default_height         = 2.0;
#ifdef P4_TO_P8
const double default_width          = 3.0;
#endif
const double default_pitch_to_tranverse_length = 0.125;
const double default_GF             = 0.5;
const unsigned int default_lmin     = 4;
const unsigned int default_lmax     = 6;
const unsigned int default_nsplits  = 4;

const int default_nx = 1;
const int default_ny = 1;
#ifdef P4_TO_P8
const int default_nz = 1;
#endif

const double default_rho          = 20.;

const int default_wall_layer      = 6;
const double default_lip          = 1.2;

const int default_kx              = 3;
#ifdef P4_TO_P8
const int default_kz              = 2;
const int default_kx_w            = 1;
const int default_kz_w            = -2;
#endif

const std::string default_export_dir = "/home/regan/workspace/projects/poisson_cells_shs";


struct p_exact : CF_DIM
{
  const my_p4est_shs_channel_t& channel;
  const int kx;
  const int kz;
  p_exact(const my_p4est_shs_channel_t& channel_, const int& kx_ = 1, const int& kz_ = 1) : channel(channel_), kx(kx_), kz(kz_) {};
  double my_f(DIM(double x, double y, double z)) const
  {
    return 1 + sin(kx*x*2.0*M_PI/channel.length() ONLY3D(+ kz*z*2.0*M_PI/channel.width())) - 0.3*cos(kx*x*2.0*M_PI/channel.length() - 1.3*y/channel.delta());
  }

  double my_df_dx(DIM(double x, double y, double z)) const
  {
    return (kx*2.0*M_PI/channel.length())*cos(kx*x*2.0*M_PI/channel.length() ONLY3D(+ kz*z*2.0*M_PI/channel.width())) + 0.3*(kx*2.0*M_PI/channel.length())*sin(kx*x*2.0*M_PI/channel.length() - 1.3*y/channel.delta());
  }

  double my_ddf_dx_dx(DIM(double x, double y, double z)) const
  {
    return -SQR(kx*2.0*M_PI/channel.length())*sin(kx*x*2.0*M_PI/channel.length() ONLY3D(+ kz*z*2.0*M_PI/channel.width())) + 0.3*SQR(kx*2.0*M_PI/channel.length())*cos(kx*x*2.0*M_PI/channel.length() - 1.3*y/channel.delta());
  }

  double my_df_dy(DIM(double x, double y, double)) const
  {
    return + 0.3*(-1.3/channel.delta())*sin(kx*x*2.0*M_PI/channel.length() - 1.3*y/channel.delta());
  }

  double my_ddf_dy_dy(DIM(double x, double y, double)) const
  {
    return + 0.3*SQR(-1.3/channel.delta())*cos(kx*x*2.0*M_PI/channel.length() - 1.3*y/channel.delta());
  }

#ifdef P4_TO_P8
  double my_df_dz(DIM(double x, double, double z)) const
  {
    return (kz*2.0*M_PI/channel.width())*cos(kx*x*2.0*M_PI/channel.length() + kz*z*2.0*M_PI/channel.width());
  }

  double my_ddf_dz_dz(DIM(double x, double, double z)) const
  {
    return -SQR(kz*2.0*M_PI/channel.width())*sin(kx*x*2.0*M_PI/channel.length() + kz*z*2.0*M_PI/channel.width());
  }
#endif

  double operator()(DIM(double x, double y, double z)) const
  {
    return atan(my_f(DIM(x, y, z)));
  }

  double laplace(const double* xyz) const
  {
    return SUMD(my_ddf_dx_dx(DIM(xyz[0], xyz[1], xyz[2])), my_ddf_dy_dy(DIM(xyz[0], xyz[1], xyz[2])), my_ddf_dz_dz(DIM(xyz[0], xyz[1], xyz[2])))/(1.0 + SQR(my_f(DIM(xyz[0], xyz[1], xyz[2])))) - 2.0*my_f(DIM(xyz[0], xyz[1], xyz[2]))*(SUMD(SQR(my_df_dx(DIM(xyz[0], xyz[1], xyz[2]))), SQR(my_df_dy(DIM(xyz[0], xyz[1], xyz[2]))), SQR(my_df_dz(DIM(xyz[0], xyz[1], xyz[2]))))/SQR(1.0 + SQR(my_f(DIM(xyz[0], xyz[1], xyz[2])))));
  }

  double derivative(DIM(const double x, const double y, const double z), const unsigned char& dir) const
  {
    switch (dir) {
    case dir::x:
      return my_df_dx(DIM(x, y, z))/(1.0 + SQR(my_f(DIM(x, y, z))));
      break;
    case dir::y:
      return my_df_dy(DIM(x, y, z))/(1.0 + SQR(my_f(DIM(x, y, z))));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return my_df_dz(DIM(x, y, z))/(1.0 + SQR(my_f(DIM(x, y, z))));
      break;
#endif
    default:
      throw std::runtime_error("p_exact::derivative::unkown direction");
      break;
    }
  }

  double derivative(const double* xyz, const unsigned char& dir) const
  {
    return derivative(DIM(xyz[0], xyz[1], xyz[2]), dir);
  }

};

struct n_dot_grad_p_exact : CF_DIM
{
  const p_exact& pex;
  n_dot_grad_p_exact (const p_exact& pex_) : pex(pex_) {};
  double operator()(DIM(double x, double y, double z)) const
  {
      return (y > 0.0 ? 1.0 : -1.0)*pex.derivative(DIM(x, y, z), dir::y);
  }
};

//void save_VTK(const std::string& out_dir, p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
//              Vec solution, Vec error, int compt)
//{
//  PetscErrorCode ierr;
//  ostringstream command;
//  command << "mkdir -p " << out_dir << "/vtu";
//  int sys_return = system(command.str().c_str()); (void) sys_return;
//  std::ostringstream oss;
//  oss << out_dir << "/vtu/cells_" << p4est->mpisize << "_"
//      << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] << ONLY3D("x" << brick->nxyztrees[2] <<) "." << compt;

//  const double *sol_cells_p, *err_cells_p;
//  ierr = VecGetArrayRead(solution, &sol_cells_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(error, &err_cells_p); CHKERRXX(ierr);

//  /* save the size of the leaves */
//  Vec leaf_level;
//  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
//  double *l_p;
//  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

//  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
//  {
//    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
//    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
//      l_p[tree->quadrants_offset + q] = p4est_quadrant_array_index(&tree->quadrants, q)->level;
//  }

//  for(size_t q = 0; q < ghost->ghosts.elem_count; ++q)
//    l_p[p4est->local_num_quadrants + q] = p4est_quadrant_array_index(&ghost->ghosts, q)->level;

//  my_p4est_vtk_write_all_general(p4est, nodes, ghost,
//                                 P4EST_TRUE, P4EST_TRUE,
//                                 0, 0, 0, 3, 0, 0, oss.str().c_str(),
//                                 VTK_CELL_SCALAR, "solution", sol_cells_p,
//                                 VTK_CELL_SCALAR, "error", err_cells_p,
//                                 VTK_CELL_SCALAR, "leaf_level", l_p);

//  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
//  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

//  ierr = VecRestoreArrayRead(error, &err_cells_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(solution, &sol_cells_p); CHKERRXX(ierr);

//  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
//}

double compute_grad_p(const n_dot_grad_p_exact& ngrad_p, Vec sol, my_p4est_faces_t* faces, my_p4est_cell_neighbors_t* ngbd_c, p4est_t* p4est, p4est_ghost_t* ghost, p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const unsigned char face_dir)
{
  PetscErrorCode ierr;

  p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);

  const double *sol_p;
  ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);

  if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
  {
    P4EST_ASSERT(face_dir/2 == dir::y);
    double xyz_quad[P4EST_DIM];
    quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);

    const double hy = faces->get_smallest_dxyz()[1]*((double)(1 << (((splitting_criteria_t*) p4est->user_pointer)->max_lvl - quad->level)));


    switch(face_dir)
    {
    case dir::f_0m0:
      return -ngrad_p(DIM(xyz_quad[0], xyz_quad[1] - hy*0.5, xyz_quad[2]));
    case dir::f_0p0:
      return ngrad_p(DIM(xyz_quad[0], xyz_quad[1] + hy*0.5, xyz_quad[2]));
    default:
      throw std::invalid_argument("[ERROR]: unknown wall boundary condition for evaluating gradient of solution.");
    }
  }
  else
  {
    set_of_neighboring_quadrants ngbd; ngbd.clear();
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, face_dir);

    /* multiple neighbor cells should never happen since this function is called for a given face,
     * and the faces are defined only for small cells.
     */
    if(ngbd.size() > 1)
      throw std::invalid_argument("[ERROR]: this cannot happen.");
    /* one neighbor cell of same size, check for interface */
    else if(ngbd.begin()->level == quad->level)
    {
      const double h = faces->get_smallest_dxyz()[face_dir/2]*((double)(1 << (((splitting_criteria_t*) p4est->user_pointer)->max_lvl - quad->level)));

      double grad_sol = sol_p[quad_idx] - sol_p[ngbd.begin()->p.piggy3.local_num];
      ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
      return (face_dir%2 == 0 ? grad_sol/h : -grad_sol/h);
    }
    /* one neighbor cell that is bigger, get common neighbors */
    else
    {
      p4est_quadrant_t quad_tmp = *ngbd.begin();
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, face_dir%2 == 0 ? face_dir + 1 : face_dir - 1);

      double dist = 0;
      double grad_sol = 0;
      double d0 = (double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN;

      for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
      {
        double dm = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;
        dist += pow(dm, P4EST_DIM - 1) * .5*(d0+dm);
        grad_sol += (sol_p[it->p.piggy3.local_num] - sol_p[quad_tmp.p.piggy3.local_num]) * pow(dm, P4EST_DIM - 1);
      }
      dist *= faces->get_tree_dimensions()[face_dir/2];

      ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
      return (face_dir%2 == 0 ? grad_sol/dist : -grad_sol/dist);
    }
  }
}


void evaluate_errors_on_gradients(double* err_grad_n, const p_exact& pex, const n_dot_grad_p_exact& n_grad_p, Vec sol, my_p4est_faces_t* faces, my_p4est_cell_neighbors_t* ngbd_c, p4est_t* p4est, p4est_ghost_t* ghost)
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    err_grad_n[dir] = 0.0;
    for (int f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx) {
      faces->f2q(f_idx, dir, quad_idx, tree_idx);
      unsigned char tmp = (faces->q2f(quad_idx, 2*dir) == f_idx ? 0 : 1);
      const double grad_component = compute_grad_p(n_grad_p, sol, faces, ngbd_c, p4est, ghost, quad_idx, tree_idx, 2*dir + tmp);
      double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dir, xyz_face);
      err_grad_n[dir] = MAX(err_grad_n[dir], fabs(grad_component - pex.derivative(xyz_face, dir)));
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, err_grad_n, P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}


int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("nx", "number of trees along x in the macromesh");
  cmd.add_option("ny", "number of trees along y in the macromesh");
  cmd.add_option("pitch", "pitch");
  cmd.add_option("GF", "gas fraction");
  cmd.add_option("rho", "rho (inverse of diffusion coefficient)");
  cmd.add_option("wall_layer", "number of finest cells desired to layer the channel walls, default is " + std::to_string(default_wall_layer));
  cmd.add_option("lip", "Lipschitz constant L for grid refinement. The levelset is defined as the negative distance to the top/bottom wall in case of spanwise grooves or to the closest no-slip region with streamwise grooves. \n\tWarning: this application uses a modified criterion comparin the levelset value to L\\Delta y (as opposed to L*diag(C)). \n\tDefault value is " + std::to_string(default_lip));
  cmd.add_option("length", "length of the domain");
  cmd.add_option("height", "height of the domain");
  cmd.add_option("kx", "kx for solution");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees along z in the macromesh");
  cmd.add_option("spanwise", "spanwise grooves if present");
  cmd.add_option("width",  "width of the domain");
  cmd.add_option("kz", "kz for solution");
#endif
  cmd.add_option("export_dir", "exportation direction");
  cmd.add_option("timing", "activates prints of the timing if present");

  if (cmd.parse(argc, argv))
    return 0;

  const unsigned int lmin     = cmd.get("lmin", default_lmin);
  const unsigned int lmax     = cmd.get("lmax", default_lmax);
  const unsigned int nsplits  = cmd.get("nb_splits", default_nsplits);
  const bool print_timing     = cmd.contains("timing");

  parStopWatch w;
  double total, grid_data, problem_setup, solve, measure_errors;
  total = grid_data = problem_setup = solve = measure_errors = 0.0;

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  const int n_xyz         [P4EST_DIM] = {DIM(cmd.get<int>("nx", default_nx), cmd.get<int>("ny", default_ny), cmd.get<int>("nz", default_nz))};
  const int periodicitity [P4EST_DIM] = {DIM(1, 0, 1)};
  const double domain_dim [P4EST_DIM] = {DIM(cmd.get<double>("length", default_length), cmd.get<double>("height", default_height), cmd.get<double>("width", default_width))};
  const double tree_dim [P4EST_DIM]   = {DIM(domain_dim[0]/n_xyz[0], domain_dim[1]/n_xyz[1], domain_dim[2]/n_xyz[2])};
  const double xyz_min    [P4EST_DIM] = {DIM(-0.5*domain_dim[0], -0.5*domain_dim[1], -0.5*domain_dim[2])};
  const double xyz_max    [P4EST_DIM] = {DIM( 0.5*domain_dim[0],  0.5*domain_dim[1],  0.5*domain_dim[2])};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodicitity);
  const double pitch                  = cmd.get<double>("pitch", default_pitch_to_tranverse_length*(ONLY3D(cmd.contains("spanwise") ?) domain_dim[0] ONLY3D( : domain_dim[2])));
  const double GF                     = cmd.get<double>("GF", default_GF);
  const double inv_rho                = 1.0/cmd.get<double>("rho", default_rho);

  const std::string export_dir        = cmd.get<std::string>("export_dir", default_export_dir);

  double err_grad_n[P4EST_DIM], err_grad_nm1[P4EST_DIM];

  my_p4est_shs_channel_t channel(mpi);
  p_exact p_comp(DIM(channel, cmd.get<int>("kx", default_kx), cmd.get<int>("kz", default_kz)));
  n_dot_grad_p_exact n_grad_p(p_comp);

  p4est_t       *p4est = NULL;
  p4est_nodes_t *nodes = NULL;
  p4est_ghost_t *ghost = NULL;
  splitting_criteria_cf_and_uniform_band_t* sp = NULL;

  for (unsigned int k = 0; k < nsplits; ++k) {
    ierr = PetscPrintf(mpi.comm(), "Grid levels %d / %d\n", lmin + k, lmax + k); CHKERRXX(ierr);

    // setup the grid
    w.start();
    if(k == 0)
    {
      channel.configure(&brick, DIM(pitch, GF, cmd.contains("spanwise")), lmax);
      channel.create_p4est_ghost_and_nodes(p4est, ghost, nodes, sp, connectivity, mpi, lmin,  cmd.get<unsigned int>("wall_layer", default_wall_layer), cmd.get<double>("lip", default_lip));
    }
    else
    {
      sp->min_lvl++;
      sp->max_lvl++;
      channel.configure(&brick, DIM(channel.get_pitch(), channel.GF(), channel.spanwise_grooves()), sp->max_lvl);
      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);
      p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
      p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
      ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_ghost_expand(p4est, ghost);
      if(third_degree_ghost_are_required(tree_dim))
        p4est_ghost_expand(p4est, ghost);
      nodes = my_p4est_nodes_new(p4est, ghost);
    }
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_cell_neighbors_t ngbd_c(&hierarchy);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy, nodes);
    my_p4est_faces_t faces(p4est, ghost, &brick, &ngbd_c);
    w.stop();
    grid_data = w.get_duration();
    total = grid_data;

    // setup the problam
    w.start();
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, channel, phi);
    channel.set_neumann_value_p(n_grad_p);

    BoundaryConditionsDIM* bc = channel.get_bc_on_pressure();

    Vec rhs;
    ierr = VecCreateNoGhostCells(p4est, &rhs);
    double* rhs_p;
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
      p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t qq = 0; qq < tree->quadrants.elem_count; ++qq) {
        p4est_locidx_t quad_idx = tree->quadrants_offset + qq;
        double xyz_quad[P4EST_DIM];
        quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
        rhs_p[quad_idx] = -inv_rho*p_comp.laplace(xyz_quad);
      }
    }
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

    my_p4est_poisson_cells_t solver(&ngbd_c, &ngbd_n);
    solver.set_phi(phi);
    solver.set_diagonal(0.0);
    solver.set_mu(inv_rho);
    solver.set_bc(*bc);
    solver.set_rhs(rhs);

    Vec sol; // , exact_solution;
    ierr = VecCreateGhostCells(p4est, ghost, &sol);
    w.stop();
    problem_setup = w.get_duration();
    total += problem_setup;

    // solve for the solution and shift to match averaged value
    w.start();
    solver.solve(sol);
    w.stop();
    solve = w.get_duration();
    total += solve;

    // measure error on components of gradient of solution on the faces
    w.start();

    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      err_grad_nm1[dir] = err_grad_n[dir];

    evaluate_errors_on_gradients(err_grad_n, p_comp, n_grad_p, sol, &faces, &ngbd_c, p4est, ghost);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = PetscPrintf(p4est->mpicomm, "Error on partial derivative along %s: %.5e, order = %g\n", (dir == dir::x ? "x" : ONLY3D(OPEN_PARENTHESIS dir == dir::y ?) "y" ONLY3D(: "z" CLOSE_PARENTHESIS)), err_grad_n[dir], log(err_grad_nm1[dir]/err_grad_n[dir])/log(2)); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);

    if(print_timing){
      ierr = PetscPrintf(mpi.comm(), "Check for grid %d/%d executed in %.1f s. \nConstruction of grid data:\t %2.1f\%; \nProblem setup: \t\t\t %2.1f\%; \nSolve step: \t\t\t %2.1f\%; \nMeasure of errors: \t\t %2.1f\%\n", lmin + k, lmax + k, total, 100.0*grid_data/total, 100.0*problem_setup/total, 100.0*solve/total, 100.0*measure_errors/total); CHKERRXX(ierr);
    }
  }

  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);

  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
