/*
 * Test the face based poisson solver with
 * - possibly very stretched grids
 * - mixed wall boundary conditions on the y-normal walls
 * - periodic boundary conditions along x and z
 * as likely to be found in SHS channel simulations
 */

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_shs_channel.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_shs_channel.h>
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

#ifdef P4_TO_P8
const double default_length         = 6.0;
#else
// In this example, I set 7 as default length in 2D, not 6. Why?
// In 2D, a length of 6 would not be enough to trigger original issues with y-normal, face-seeded Voronoi tesselation whereas 7 is.
// However, one can show that a length of 6 is enough to trigger those very issues in 3D (see comment in check_past_sharing_quad_is_required
// from my_p4est_faces.h for more details).
// Therefore, I did set 7 as default in this test file in 2D in order to originally analyze the issue with lower computational cost,
// investigate it easily and eventually fix if! (The implementation of the fix is made such that its generality carries over to 3D)
const double default_length         = 7.0;
#endif
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

const double default_mu           = 20.;
const double default_add_diagonal = 4.0;

const int default_wall_layer      = 6;
const double default_lip          = 1.2;

const int default_kx_u            = 3;
#ifdef P4_TO_P8
const int default_kz_u            = 2;
const int default_kx_w            = 1;
const int default_kz_w            = -2;
#endif

const std::string default_export_dir = "/home/regan/workspace/projects/poisson_faces_shs";

struct u_exact : CF_DIM
{
  const my_p4est_shs_channel_t& channel;
  const int kx;
#ifdef P4_TO_P8
  const int kz;
#endif
  u_exact(DIM(const my_p4est_shs_channel_t& channel_, const int& kx_ = 1, const int& kz_ = 1)) : DIM(channel(channel_), kx(kx_), kz(kz_)) {};
  double operator()(DIM(double x, double y, double z)) const
  {
    return MULTD(sin(kx*2.0*M_PI*x/channel.length()), (1.0 - SQR(y/channel.delta())), sin(kz*2.0*M_PI*z/channel.width()));
  }

  double laplace(const double* xyz)
  {
    return (-SQR(kx*2.0*M_PI/channel.length()))*MULTD(sin(kx*2.0*M_PI*xyz[0]/channel.length()), (1.0 - SQR(xyz[1]/channel.delta())), sin(kz*2.0*M_PI*xyz[2]/channel.width()))
        + MULTD(sin(kx*2.0*M_PI*xyz[0]/channel.length()), -2.0/SQR(channel.delta()), sin(kz*2.0*M_PI*xyz[2]/channel.width()))
        ONLY3D(+ (-SQR(kz*2.0*M_PI/channel.width()))*MULTD(sin(kx*2.0*M_PI*xyz[0]/channel.length()), (1.0 - SQR(xyz[1]/channel.delta())), sin(kz*2.0*M_PI*xyz[2]/channel.width())));
  }
};

struct n_dot_grad_u_exact : CF_DIM
{
  const u_exact& uex;
  n_dot_grad_u_exact (const u_exact& uex_) : uex(uex_) {};
  double operator()(DIM(double x, double y, double z)) const
  {
    return (y > 0.0 ? 1.0 : -1.0)*MULTD(sin(uex.kx*2.0*M_PI*x/uex.channel.length()), (-2.0*y/SQR(uex.channel.delta())), sin(uex.kz*2.0*M_PI*z/uex.channel.width()));
  }
};

struct v_exact : CF_DIM
{
  const my_p4est_shs_channel_t& channel;
  v_exact(const my_p4est_shs_channel_t& channel_) : channel(channel_) {};
  double operator()(DIM(double x, double y, double z)) const
  {
    return MULTD(exp(0.7*(cos(2.0*M_PI*(x - 0.2*channel.length())/channel.length()) + 1)), SQR(y/channel.delta()) - 0.3*pow(y/channel.delta(), 3.0), log(1 + 0.5*sin(2.0*M_PI*z/channel.width())));
  }
  double laplace(const double* xyz)
  {
    return MULTD(exp(0.7*(cos(2.0*M_PI*(xyz[0] - 0.2*channel.length())/channel.length()) + 1))*(SQR(0.7*sin(2.0*M_PI*(xyz[0] - 0.2*channel.length())/channel.length())*2.0*M_PI/channel.length()) - 0.7*SQR(2.0*M_PI/channel.length())*cos(2.0*M_PI*(xyz[0] - 0.2*channel.length())/channel.length())), SQR(xyz[1]/channel.delta()) - 0.3*pow(xyz[1]/channel.delta(), 3.0), log(1 + 0.5*sin(2.0*M_PI*xyz[2]/channel.width())))
        + MULTD(exp(0.7*(cos(2.0*M_PI*(xyz[0] - 0.2*channel.length())/channel.length()) + 1)), 2.0/SQR(channel.delta()) - 1.8*xyz[1]/pow(channel.delta(), 3.0), log(1 + 0.5*sin(2.0*M_PI*xyz[2]/channel.width())))
        ONLY3D(+ MULTD(exp(0.7*(cos(2.0*M_PI*(xyz[0] - 0.2*channel.length())/channel.length()) + 1)), SQR(xyz[1]/channel.delta()) - 0.3*pow(xyz[1]/channel.delta(), 3.0), (-0.5*SQR(2.0*M_PI/channel.width())*sin(2.0*M_PI*xyz[2]/channel.width()) - 0.25*SQR(2.0*M_PI/channel.width()))/SQR(1.0 + 0.5*sin(2.0*M_PI*xyz[2]/channel.width()))));
  }
};

#ifdef P4_TO_P8
struct w_exact : CF_DIM
{
  const my_p4est_shs_channel_t& channel;
  const int kx;
  const int kz;
  w_exact(const my_p4est_shs_channel_t& channel_, const int& kx_ = 1, const int& kz_ = 1) : channel(channel_), kx(kx_), kz(kz_) {};
  double my_f(double x, double y, double z) const
  {
    return 1 + sin(kx*x*2.0*M_PI/channel.length() + kz*z*2.0*M_PI/channel.width()) - 0.3*cos(kz*z*2.0*M_PI/channel.width() - 1.3*y/channel.delta());
  }

  double my_df_dx(double x, double, double z) const
  {
    return (kx*2.0*M_PI/channel.length())*cos(kx*x*2.0*M_PI/channel.length() + kz*z*2.0*M_PI/channel.width());
  }

  double my_ddf_dx_dx(double x, double, double z) const
  {
    return -SQR(kx*2.0*M_PI/channel.length())*sin(kx*x*2.0*M_PI/channel.length() + kz*z*2.0*M_PI/channel.width());
  }

  double my_df_dy(double, double y, double z) const
  {
    return + 0.3*(-1.3/channel.delta())*sin(kz*z*2.0*M_PI/channel.width() - 1.3*y/channel.delta());
  }

  double my_ddf_dy_dy(double, double y, double z) const
  {
    return + 0.3*SQR(-1.3/channel.delta())*cos(kz*z*2.0*M_PI/channel.width() - 1.3*y/channel.delta());
  }

  double my_df_dz(double x, double y, double z) const
  {
    return (kz*2.0*M_PI/channel.width())*cos(kx*x*2.0*M_PI/channel.length() + kz*z*2.0*M_PI/channel.width()) + 0.3*(kz*2.0*M_PI/channel.width())*sin(kz*z*2.0*M_PI/channel.width() - 1.3*y/channel.delta());
  }

  double my_ddf_dz_dz(double x, double y, double z) const
  {
    return -SQR(kz*2.0*M_PI/channel.width())*sin(kx*x*2.0*M_PI/channel.length() + kz*z*2.0*M_PI/channel.width()) + 0.3*SQR(kz*2.0*M_PI/channel.width())*cos(kz*z*2.0*M_PI/channel.width() - 1.3*y/channel.delta());
  }

  double operator()(double x, double y, double z) const
  {
    return atan(my_f(x, y, z));
  }

  double laplace(const double* xyz)
  {
    return ((my_ddf_dx_dx(xyz[0], xyz[1], xyz[2]) + my_ddf_dy_dy(xyz[0], xyz[1], xyz[2]) + my_ddf_dz_dz(xyz[0], xyz[1], xyz[2]))/(1.0 + SQR(my_f(xyz[0], xyz[1], xyz[2]))) - 2.0*my_f(xyz[0], xyz[1], xyz[2])*(SQR(my_df_dx(xyz[0], xyz[1], xyz[2])) + SQR(my_df_dy(xyz[0], xyz[1], xyz[2])) + SQR(my_df_dz(xyz[0], xyz[1], xyz[2])))/SQR(1.0 + SQR(my_f(xyz[0], xyz[1], xyz[2]))));
  }
};

struct n_dot_grad_w_exact : CF_DIM
{
  const w_exact& wex;
  n_dot_grad_w_exact (const w_exact& wex_) : wex(wex_) {};
  double operator()(double x, double y, double z) const
  {
      return (y > 0.0 ? 1.0 : -1.0)*(wex.my_df_dy(x, y, z)/(1.0 + SQR(wex.my_f(x, y, z))));
  }
};
#endif


void save_VTK(const std::string& out_dir, p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec error_cells, Vec *sol_nodes, Vec *err_nodes, int compt)
{
  PetscErrorCode ierr;
  ostringstream command;
  command << "mkdir -p " << out_dir << "/vtu";
  int sys_return = system(command.str().c_str()); (void) sys_return;
  std::ostringstream oss;
  oss << out_dir << "/vtu/faces_" << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] << ONLY3D("x" << brick->nxyztrees[2] <<) "." << compt;

  double *error_cells_p;
  ierr = VecGetArray(error_cells, &error_cells_p); CHKERRXX(ierr);

  double *sol_nodes_p[P4EST_DIM];
  double *err_nodes_p[P4EST_DIM];
  for(unsigned char d = 0; d < P4EST_DIM; ++d)
  {
    ierr = VecGetArray(sol_nodes[d], &sol_nodes_p[d]); CHKERRXX(ierr);
    ierr = VecGetArray(err_nodes[d], &err_nodes_p[d]); CHKERRXX(ierr);
  }

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for( size_t q = 0; q < tree->quadrants.elem_count; ++q)
      l_p[tree->quadrants_offset+q] = p4est_quadrant_array_index(&tree->quadrants, q)->level;
  }

  for(size_t q = 0; q < ghost->ghosts.elem_count; ++q)
    l_p[p4est->local_num_quadrants + q] = p4est_quadrant_array_index(&ghost->ghosts, q)->level;

  my_p4est_vtk_write_all_general(p4est, nodes, ghost,
                                 P4EST_TRUE, P4EST_TRUE,
                                 0, 2, 0, 1, 0, 1, oss.str().c_str(),
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "solution", DIM(sol_nodes_p[0], sol_nodes_p[1], sol_nodes_p[2]),
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "errors", DIM(err_nodes_p[0], err_nodes_p[1], err_nodes_p[2]),
                                 VTK_CELL_VECTOR_BLOCK, "error_faces", error_cells_p,
                                 VTK_CELL_SCALAR, "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(error_cells, &error_cells_p); CHKERRXX(ierr);

  for(unsigned char d = 0; d < P4EST_DIM; ++d)
  {
    ierr = VecRestoreArray(sol_nodes[d], &sol_nodes_p[d]); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_nodes[d], &err_nodes_p[d]); CHKERRXX(ierr);
  }

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
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
  cmd.add_option("save_voro", "save the Voronoi partition in vtk format");
  cmd.add_option("nx", "number of trees along x in the macromesh");
  cmd.add_option("ny", "number of trees along y in the macromesh");
  cmd.add_option("pitch", "pitch");
  cmd.add_option("GF", "gas fraction");
  cmd.add_option("mu", "mu (diffusion coefficient)");
  cmd.add_option("add_diag", "diagonal factor");
  cmd.add_option("wall_layer", "number of finest cells desired to layer the channel walls, default is " + std::to_string(default_wall_layer));
  cmd.add_option("lip", "Lipschitz constant L for grid refinement. The levelset is defined as the negative distance to the top/bottom wall in case of spanwise grooves or to the closest no-slip region with streamwise grooves. \n\tWarning: this application uses a modified criterion comparin the levelset value to L\\Delta y (as opposed to L*diag(C)). \n\tDefault value is " + std::to_string(default_lip));
  cmd.add_option("length", "length of the domain");
  cmd.add_option("height", "height of the domain");
  cmd.add_option("kx_u", "kx for u solution");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees along z in the macromesh");
  cmd.add_option("spanwise", "spanwise grooves if present");
  cmd.add_option("width",  "width of the domain");
  cmd.add_option("kz_u", "kz for u solution");
  cmd.add_option("kx_w", "kx for w solution");
  cmd.add_option("kz_w", "kz for w solution");
#endif
  cmd.add_option("export_dir", "exportation direction");
  cmd.add_option("timing", "activates prints of the timing if present");

  if (cmd.parse(argc, argv))
    return 0;

  const unsigned int lmin     = cmd.get("lmin", default_lmin);
  const unsigned int lmax     = cmd.get("lmax", default_lmax);
  const unsigned int nsplits  = cmd.get("nb_splits", default_nsplits);
  const bool save_vtk         = cmd.contains("save_vtk");
  const bool save_voro        = cmd.contains("save_voro");
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
  const double mu                     = cmd.get<double>("mu", default_mu);
  const double add_diag               = cmd.get<double>("add_diag", default_add_diagonal);

  const std::string export_dir        = cmd.get<std::string>("export_dir", default_export_dir);

  double err_n  [P4EST_DIM];
  double err_nm1[P4EST_DIM];
  double err_nodes_n  [P4EST_DIM];
  double err_nodes_nm1[P4EST_DIM];

  my_p4est_shs_channel_t channel(mpi);
  u_exact u_comp(DIM(channel, cmd.get<int>("kx_u", default_kx_u), cmd.get<int>("kz_u", default_kz_u)));
  n_dot_grad_u_exact n_grad_u(u_comp);
  v_exact v_comp(channel);
#ifdef P4_TO_P8
  w_exact w_comp(DIM(channel, cmd.get<int>("kx_w", default_kx_w), cmd.get<int>("kz_w", default_kz_w)));
  n_dot_grad_w_exact n_grad_w(w_comp);
#endif

  p4est_t       *p4est = NULL;
  p4est_nodes_t *nodes = NULL;
  p4est_ghost_t *ghost = NULL;
  splitting_criteria_cf_and_uniform_band_t* sp = NULL;

  for (unsigned int k = 0; k < nsplits; ++k) {
    ierr = PetscPrintf(mpi.comm(), "Grid levels %d / %d\n", lmin + k, lmax + k); CHKERRXX(ierr);

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

    w.start();
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, channel, phi);

    channel.set_dirichlet_value_u(u_comp); channel.set_neumann_value_u(n_grad_u);
    channel.set_dirichlet_value_v(v_comp);
#ifdef P4_TO_P8
    channel.set_dirichlet_value_w(w_comp); channel.set_neumann_value_w(n_grad_w);
#endif

    BoundaryConditionsDIM* bc = channel.get_bc_on_velocity();

    Vec rhs[P4EST_DIM];
    Vec dxyz_hodge[P4EST_DIM];
    Vec face_is_well_defined[P4EST_DIM];
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostFaces(p4est, &faces, &rhs[dir], dir); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est, &faces, &dxyz_hodge[dir], dir); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est, &faces, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
      double *rhs_p, *dxyz_hodge_p, *face_is_well_defined_p;
      ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
      ierr = VecGetArray(dxyz_hodge[dir], &dxyz_hodge_p); CHKERRXX(ierr);
      ierr = VecGetArray(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);
      for(size_t k = 0; k < faces.get_layer_size(dir); ++k)
      {
        p4est_locidx_t f_idx = faces.get_layer_face(dir, k);
        double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
        rhs_p[f_idx] = (dir == dir::x ? -mu*u_comp.laplace(xyz) + add_diag*u_comp(DIM(xyz[0], xyz[1], xyz[2])) : ONLY3D(OPEN_PARENTHESIS dir == dir::y ?) -mu*v_comp.laplace(xyz) + add_diag*v_comp(DIM(xyz[0], xyz[1], xyz[2])) ONLY3D(: -mu*w_comp.laplace(xyz) + add_diag*w_comp(DIM(xyz[0], xyz[1], xyz[2])) CLOSE_PARENTHESIS));
        dxyz_hodge_p[f_idx]           = 0.0;
        face_is_well_defined_p[f_idx] = true;
      }
      ierr = VecGhostUpdateBegin(rhs[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(face_is_well_defined[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      for(size_t k = 0; k < faces.get_local_size(dir); ++k)
      {
        p4est_locidx_t f_idx = faces.get_local_face(dir, k);
        double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
        rhs_p[f_idx] = (dir == dir::x ? -mu*u_comp.laplace(xyz) + add_diag*u_comp(DIM(xyz[0], xyz[1], xyz[2])) : ONLY3D(OPEN_PARENTHESIS dir == dir::y ?) -mu*v_comp.laplace(xyz) + add_diag*v_comp(DIM(xyz[0], xyz[1], xyz[2])) ONLY3D(: -mu*w_comp.laplace(xyz) + add_diag*w_comp(DIM(xyz[0], xyz[1], xyz[2])) CLOSE_PARENTHESIS));
        dxyz_hodge_p[f_idx]           = 0.0;
        face_is_well_defined_p[f_idx] = true;
      }
      ierr = VecGhostUpdateEnd(rhs[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(face_is_well_defined[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(dxyz_hodge[dir], &dxyz_hodge_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);
    }

    my_p4est_poisson_faces_t solver(&faces, &ngbd_n);
    solver.set_phi(phi);
    solver.set_diagonal(add_diag);
    solver.set_mu(mu);
    solver.set_bc(bc, dxyz_hodge, face_is_well_defined);
    solver.set_rhs(rhs);
    solver.set_compute_partition_on_the_fly(false);

    Vec sol[P4EST_DIM];
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
      ierr = VecCreateGhostFaces(p4est, &faces, &sol[dir], dir); CHKERRXX(ierr); }

    w.stop();
    problem_setup = w.get_duration();
    total += problem_setup;


    w.start();
    solver.solve(sol);
    w.stop();
    solve = w.get_duration();
    total += solve;

    if(save_voro)
    {
      ostringstream command;
      command << "mkdir -p " << export_dir << "/voro_grid";
      int sys_return = system(command.str().c_str()); (void) sys_return;

      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        char filename[PATH_MAX];
        sprintf(filename, "%s/voro_grid/voro_%s_face_%d.vtk", export_dir.c_str(), (dir == dir::x ? "x" : (dir == dir::y ? "y" : "z")), p4est->mpirank);
        solver.print_partition_VTK(filename, dir);
      }
    }

    w.start();
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      double *sol_p;
      ierr = VecGetArray(sol[dir], &sol_p); CHKERRXX(ierr);

      err_nm1[dir] = err_n[dir];
      err_n[dir] = 0;

      CF_DIM* exact_sol;
      switch (dir) {
      case dir::x:
        exact_sol = &u_comp;
        break;
      case dir::y:
        exact_sol = &v_comp;
        break;
#ifdef P4_TO_P8
      case dir::z:
        exact_sol = &w_comp;
        break;
#endif
      default:
        throw std::logic_error("What the hell happened here? looping through cartesian direction, but unknown direction was found...");
        break;
      }

      for(p4est_locidx_t f_idx = 0; f_idx < faces.num_local[dir]; ++f_idx)
      {
        double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
        err_n[dir] = MAX(err_n[dir], fabs(sol_p[f_idx] - (*exact_sol)(xyz)));
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n[dir], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      ierr = PetscPrintf(p4est->mpicomm, "Error for direction %d : %g, order = %g\n", dir, err_n[dir], log(err_nm1[dir]/err_n[dir])/log(2)); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol[dir], &sol_p); CHKERRXX(ierr);
    }

    /* interpolate the solution on the nodes */
    Vec sol_nodes[P4EST_DIM];
    Vec err_nodes[P4EST_DIM];

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateGhostNodes(p4est, nodes, &sol_nodes[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &err_nodes[dir]); CHKERRXX(ierr);
    }

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      double *sol_nodes_p, *err_p;
      ierr = VecGetArray(sol_nodes[dir], &sol_nodes_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_nodes[dir], &err_p); CHKERRXX(ierr);

      err_nodes_nm1[dir] = err_nodes_n[dir];
      err_nodes_n[dir] = 0;

      CF_DIM* exact_sol;
      switch (dir) {
      case dir::x:
        exact_sol = &u_comp;
        break;
      case dir::y:
        exact_sol = &v_comp;
        break;
#ifdef P4_TO_P8
      case dir::z:
        exact_sol = &w_comp;
        break;
#endif
      default:
        throw std::logic_error("What the hell happened here? looping through cartesian direction, but unknown direction was found...");
        break;
      }

      for(size_t k = 0; k < ngbd_n.get_layer_size(); ++k)
      {
        p4est_locidx_t node_idx = ngbd_n.get_layer_node(k);
        double xyz[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est, nodes, xyz);
        sol_nodes_p[node_idx] = interpolate_velocity_at_node_n(&faces, &ngbd_n, node_idx, sol[dir], dir, face_is_well_defined[dir], 2, bc);
        err_p[node_idx]       = fabs(sol_nodes_p[node_idx] - (*exact_sol)(xyz));
        err_nodes_n[dir] = max(err_nodes_n[dir], err_p[node_idx]);
      }
      ierr = VecGhostUpdateBegin(sol_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(err_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      for(size_t k = 0; k < ngbd_n.get_local_size(); ++k)
      {
        p4est_locidx_t node_idx = ngbd_n.get_local_node(k);
        double xyz[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est, nodes, xyz);
        sol_nodes_p[node_idx] = interpolate_velocity_at_node_n(&faces, &ngbd_n, node_idx, sol[dir], dir, face_is_well_defined[dir], 2, bc);
        err_p[node_idx]       = fabs(sol_nodes_p[node_idx] - (*exact_sol)(xyz));
        err_nodes_n[dir] = max(err_nodes_n[dir], err_p[node_idx]);
      }
      ierr = VecGhostUpdateEnd(sol_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(err_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol_nodes[dir], &sol_nodes_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_nodes[dir], &err_p); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_nodes_n[dir], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      ierr = PetscPrintf(p4est->mpicomm, "Error on nodes for direction %d : %g, order = %g\n", dir, err_nodes_n[dir], log(err_nodes_nm1[dir]/err_nodes_n[dir])/log(2)); CHKERRXX(ierr);
    }
    w.stop();
    measure_errors = w.get_duration();
    total += measure_errors;


    if(save_vtk)
    {
      Vec error_cells;
      double *error_cells_p;
      ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &error_cells); CHKERRXX(ierr);
      ierr = VecGetArray(error_cells, &error_cells_p); CHKERRXX(ierr);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        CF_DIM* exact_sol;
        switch (dir) {
        case dir::x:
          exact_sol = &u_comp;
          break;
        case dir::y:
          exact_sol = &v_comp;
          break;
  #ifdef P4_TO_P8
        case dir::z:
          exact_sol = &w_comp;
          break;
  #endif
        default:
          throw std::logic_error("What the hell happened here? looping through cartesian direction, but unknown direction was found...");
          break;
        }
        const double *sol_p;
        ierr = VecGetArrayRead(sol[dir], &sol_p); CHKERRXX(ierr);
        for (p4est_topidx_t tr_idx = p4est->first_local_tree; tr_idx <= p4est->last_local_tree ; ++tr_idx) {
          const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_idx);
          for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
            p4est_locidx_t quad_idx = q + tree->quadrants_offset;
            error_cells_p[P4EST_DIM*q + dir] = 0.0;
            // negative direction
            p4est_locidx_t f_idx = faces.q2f(quad_idx, 2*dir);
            if(f_idx != -1)
            {
              double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
              error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - (*exact_sol)(xyz)));
            }
            else
            {
              set_of_neighboring_quadrants ngbd; ngbd.clear();
              ngbd_c.find_neighbor_cells_of_cell(ngbd, quad_idx, tr_idx, DIM(dir == dir::x ? -1 : 0, dir == dir::y ? -1 : 0, dir == dir::z ? -1 : 0));
              for (set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end() ; ++it) {
                f_idx = faces.q2f(it->p.piggy3.local_num, 2*dir+1);
                double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
                error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - (*exact_sol)(xyz)));
              }
            }
            // positive direction
            f_idx = faces.q2f(quad_idx, 2*dir+1);
            if(f_idx!=-1)
            {
              double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
              error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - (*exact_sol)(xyz)));
            }
            else
            {
              set_of_neighboring_quadrants ngbd; ngbd.clear();
              ngbd_c.find_neighbor_cells_of_cell(ngbd, quad_idx, tr_idx, DIM(dir == dir::x ? +1 : 0, dir == dir::y ? +1 : 0, dir == dir::z ? +1 : 0));
              for (set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end() ; ++it) {
                f_idx = faces.q2f(it->p.piggy3.local_num, 2*dir);
                double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
                error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - (*exact_sol)(xyz)));
              }
            }
          }
        }
        ierr = VecRestoreArrayRead(sol[dir], &sol_p); CHKERRXX(ierr);
      }
      ierr = VecGhostUpdateBegin(error_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (error_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(error_cells, &error_cells_p); CHKERRXX(ierr);

      /* END OF TESTS */

      save_VTK(export_dir, p4est, ghost, nodes, &brick, error_cells, sol_nodes, err_nodes, k);

      ierr = VecDestroy(error_cells); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(rhs[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(sol[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_nodes[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(err_nodes[dir]); CHKERRXX(ierr);
    }

    double voro_global_volume[P4EST_DIM];
    solver.global_volume_of_voronoi_tesselation(voro_global_volume);
    // one should have EXACTLY the volume of the computational box for u and w components
    // and strictly less than the computational domain for v (because of face-wall alignment of Dirichlet boundary conditions --> the Voronoi cell is not even calculated there)
    const double expected_volume = MULTD(domain_dim[0], domain_dim[1], domain_dim[2]);
    if(mpi.rank() == 0 && fabs(voro_global_volume[0] - expected_volume) > 10.0*EPS*expected_volume)
      std::cerr << "The global volume of the Voronoi tesselation for faces of normal direction x is " << voro_global_volume[0] << " whereas it is expected to be " << expected_volume << " --> check the Voronoi tesselation (use option -save_voro)" << std::endl;
    if(mpi.rank() == 0 && voro_global_volume[1] >= expected_volume)
      std::cerr << "The global volume of the Voronoi tesselation for faces of normal direction y is " << voro_global_volume[1] << " which is greater than the volume of the computational box (" << expected_volume << "): this is NOT NORMAL --> check the Voronoi tesselation (use option -save_voro)" << std::endl;
#ifdef P4_TO_P8
    if(mpi.rank() == 0 && fabs(voro_global_volume[2] - expected_volume) > 10.0*EPS*expected_volume)
      std::cerr << "The global volume of the Voronoi tesselation for faces of normal direction z is " << voro_global_volume[2] << " whereas it is expected to be " << expected_volume << " --> check the Voronoi tesselation (use option -save_voro)" << std::endl;
#endif

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
