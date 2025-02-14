// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_jump_cells_xgfm.h>
#include <src/my_p8est_poisson_jump_cells_fv.h>
#else
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_jump_cells_xgfm.h>
#include <src/my_p4est_poisson_jump_cells_fv.h>
#endif

#include <src/Parser.h>

using namespace std;
#undef MIN
#undef MAX

const static string main_description =
    string("In this example, we test the two-phase divergence-free projection step. \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), July 2020 (I'll probably die in this lab...)\n");

const int default_lmin = 6;
const int default_lmax = 6;

const int default_ntree   = 2;

const interpolation_method default_interp_method_phi = linear;
const bool default_use_second_order_theta = false;
const bool default_subrefinement  = false;


const double xyz_mmm[P4EST_DIM] = {DIM(-2.0, -2.0, -2.0)};
const double xyz_ppp[P4EST_DIM] = {DIM(+2.0, +2.0, +2.0)};

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/two_phase_projection";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/two_phase_projection";
#else
const string default_work_folder = "/home/regan/workspace/projects/two_phase_projection";
#endif


enum solver_tag {
  GFM   = 0, // --> standard GFM solver ("A Boundary Condition Capturing Method for Poisson's Equation on Irregular Domains", JCP, 160(1):151-178, Liu, Fedkiw, Kand, 2000);
  xGFM  = 1, // --> xGFM solver ("xGFM: Recovering Convergence of Fluxes in the Ghost Fluid Method", JCP, Volume 409, 15 May 2020, 19351, R. Egan, F. Gibou);
  FV    = 2  // --> finite volume approach with duplicated unknowns in cut cells ("Solving Elliptic Interface Problems with Jump Conditions on Cartesian Grids", JCP, Volume 407, 15 April 2020, 109269, D. Bochkov, F. Gibou)
};

std::string convert_to_string(const solver_tag& tag)
{
  switch(tag){
  case GFM:
    return std::string("GFM");
    break;
  case xGFM:
    return std::string("xGFM");
    break;
  case FV:
    return std::string("FV");
    break;
  default:
    return std::string("unknown type of solver");
    break;
  }
}

std::istream& operator>> (std::istream& is, std::vector<solver_tag>& solvers_to_test)
{
  std::string str;
  is >> str;
  solvers_to_test.clear();

  std::vector<size_t> substr_found_at;
  case_insensitive_find_substr_in_str(str, "GFM", substr_found_at);
  for (size_t k = 0; k < substr_found_at.size(); ++k)
    if (substr_found_at[k] == 0 || !case_insenstive_char_compare(str[substr_found_at[k] - 1], 'x')) // make sure it's 'GFM' and not 'xGFM'
    {
      solvers_to_test.push_back(GFM);
      break;
    }

  case_insensitive_find_substr_in_str(str, "xGFM", substr_found_at);
  if(substr_found_at.size() > 0)
    solvers_to_test.push_back(xGFM);

  case_insensitive_find_substr_in_str(str, "FV", substr_found_at);
  if(substr_found_at.size() > 0)
    solvers_to_test.push_back(FV);
  return is;
}

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
public:
  BCWALLVAL() {}
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
};

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

void set_computational_grid_data(const mpi_environment_t &mpi, my_p4est_brick_t* brick, p4est_connectivity_t *connectivity, const splitting_criteria_cf_t* data,
                                 p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, Vec &phi,
                                 my_p4est_hierarchy_t* &hierarchy, my_p4est_node_neighbors_t* &ngbd_n, my_p4est_cell_neighbors_t* &ngbd_c, my_p4est_faces_t* &faces)
{
  if(p4est == NULL)
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*) data;

  for(int i = find_max_level(p4est); i < data->max_lvl; ++i) {
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
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
  sample_cf_on_nodes(p4est, nodes, *data->phi, phi);
  my_p4est_level_set_t ls_coarse(ngbd_n);
  ls_coarse.reinitialize_2nd_order(phi);

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  splitting_criteria_tag_t data_tag(data->min_lvl, data->max_lvl);
  p4est_t* new_p4est = p4est_copy(p4est, P4EST_FALSE);

  while(data_tag.refine_and_coarsen(new_p4est, nodes, phi_p))
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
  if(ngbd_c != NULL)
    delete ngbd_c;
  ngbd_c = new my_p4est_cell_neighbors_t(hierarchy);

  if(faces != NULL)
    delete faces;
  faces = new my_p4est_faces_t(p4est, ghost, brick, ngbd_c);
}

void build_interface_capturing_grid_data(p4est_t* p4est_comp, my_p4est_brick_t *brick, const splitting_criteria_cf_t* subrefined_data,
                                         p4est_t* &subrefined_p4est, p4est_ghost_t* &subrefined_ghost, p4est_nodes_t* &subrefined_nodes, Vec &subrefined_phi,
                                         my_p4est_hierarchy_t* &subrefined_hierarchy, my_p4est_node_neighbors_t* &subrefined_ngbd_n)
{
  if(subrefined_p4est != NULL)
    p4est_destroy(subrefined_p4est);
  subrefined_p4est = p4est_copy(p4est_comp, P4EST_FALSE);
  subrefined_p4est->user_pointer = (void*) subrefined_data;
  my_p4est_refine(subrefined_p4est, P4EST_FALSE, refine_levelset_cf, NULL);

  if(subrefined_ghost != NULL)
    p4est_ghost_destroy(subrefined_ghost);
  subrefined_ghost = my_p4est_ghost_new(subrefined_p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(subrefined_p4est, subrefined_ghost);

  if(subrefined_nodes != NULL)
    p4est_nodes_destroy(subrefined_nodes);
  subrefined_nodes = my_p4est_nodes_new(subrefined_p4est, subrefined_ghost);

  if(subrefined_hierarchy != NULL)
    subrefined_hierarchy->update(subrefined_p4est, subrefined_ghost);
  else
    subrefined_hierarchy = new my_p4est_hierarchy_t(subrefined_p4est, subrefined_ghost, brick);

  if(subrefined_ngbd_n != NULL)
    subrefined_ngbd_n->update(subrefined_hierarchy, subrefined_nodes);
  else
  {
    subrefined_ngbd_n = new my_p4est_node_neighbors_t(subrefined_hierarchy, subrefined_nodes);
    subrefined_ngbd_n->init_neighbors();
  }

  PetscErrorCode ierr;
  if (subrefined_phi != NULL){
    ierr = VecDestroy(subrefined_phi); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(subrefined_p4est, subrefined_nodes, &subrefined_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(subrefined_p4est, subrefined_nodes, *subrefined_data->phi, subrefined_phi);
  my_p4est_level_set_t ls(subrefined_ngbd_n);
  ls.reinitialize_2nd_order(subrefined_phi);

  const double *subrefined_phi_p;
  ierr = VecGetArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);
  splitting_criteria_tag_t data_tag(subrefined_data->min_lvl, subrefined_data->max_lvl);
  p4est_t *new_subrefined_p4est = p4est_copy(subrefined_p4est, P4EST_FALSE);

  while(data_tag.refine(new_subrefined_p4est, subrefined_nodes, subrefined_phi_p)) // not refine_and_coarsen, because we need the fine grid to be everywhere finer or as coarse as the coarse grid!
  {
    ierr = VecRestoreArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);
    my_p4est_interpolation_nodes_t interp_subrefined_nodes(subrefined_ngbd_n);
    interp_subrefined_nodes.set_input(subrefined_phi, linear);

    p4est_ghost_t *new_subrefined_ghost = my_p4est_ghost_new(new_subrefined_p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(new_subrefined_p4est, new_subrefined_ghost);
    p4est_nodes_t *new_subrefined_nodes  = my_p4est_nodes_new(new_subrefined_p4est, new_subrefined_ghost);
    Vec new_subrefined_phi;
    ierr = VecCreateGhostNodes(new_subrefined_p4est, new_subrefined_nodes, &new_subrefined_phi); CHKERRXX(ierr);
    for(size_t nn = 0; nn < new_subrefined_nodes->indep_nodes.elem_count; ++nn)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(nn, new_subrefined_p4est, new_subrefined_nodes, xyz);
      interp_subrefined_nodes.add_point(nn, xyz);
    }
    interp_subrefined_nodes.interpolate(new_subrefined_phi);


    p4est_destroy(subrefined_p4est); subrefined_p4est = new_subrefined_p4est; new_subrefined_p4est = p4est_copy(subrefined_p4est, P4EST_FALSE);
    p4est_ghost_destroy(subrefined_ghost); subrefined_ghost = new_subrefined_ghost;
    subrefined_hierarchy->update(subrefined_p4est, subrefined_ghost);
    p4est_nodes_destroy(subrefined_nodes); subrefined_nodes = new_subrefined_nodes;
    subrefined_ngbd_n->update(subrefined_hierarchy, subrefined_nodes);
    ls.update(subrefined_ngbd_n);

    ierr = VecDestroy(subrefined_phi); CHKERRXX(ierr); subrefined_phi = new_subrefined_phi;

    ierr = VecGetArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);

  p4est_destroy(new_subrefined_p4est);
}

class level_set_sphere : public CF_DIM
{
  const double radius;
public:
  level_set_sphere(const double &radius_)
    : radius(radius_) { }

  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(SUMD(SQR(x), SQR(y), SQR(z))) - radius;
  }

  double grad(const u_char& comp, const double *xyz) const
  {
    const double rr = sqrt(SUMD(SQR(xyz[0]), SQR(xyz[1]), SQR(xyz[2])));
    if(rr < EPS*radius)
      return 0.0;
    return xyz[comp]/rr;
  }

  double get_radius() const { return radius; }
};

class divergence_free_velocity_field_t
{
  level_set_sphere *ls;
public:
  divergence_free_velocity_field_t(level_set_sphere* ls_) : ls(ls_) { }

  double velocity_component(const char& sgn, const u_char& component, const double *xyz) const
  {
    P4EST_ASSERT(component < P4EST_DIM);
    switch (component) {
    case dir::x:
    {
      if(sgn < 0)
        return xyz[1]*(SQR(xyz[0]) + SQR(xyz[1]) - SQR(ls->get_radius())) + 1.0;
      else
        return xyz[1]*(ls->get_radius()/sqrt(SQR(xyz[0]) + SQR(xyz[1])) - 1.0) + 1.0;
    }
      break;
    case dir::y:
    {
      if(sgn < 0)
        return -xyz[0]*(SQR(xyz[0]) + SQR(xyz[1]) - SQR(ls->get_radius())) + 1.0;
      else
        return -xyz[0]*(ls->get_radius()/sqrt(SQR(xyz[0]) + SQR(xyz[1])) - 1.0) + 1.0;
    }
      break;
    default:
      throw std::invalid_argument("divergence_free_velocity_field::velocity_component() : unknown velocity component...");
      break;
    }
  }
};


class grad_scalar_part_t
{
  level_set_sphere *ls;
public:
  grad_scalar_part_t(level_set_sphere* ls_) : ls(ls_) { }

  double component(const u_char& component, const double xyz[P4EST_DIM]) const
  {
    // scalar field = SQR(x - xyz_mmm[0])*SQR(x - xyz_ppp[0])*SQR(y - xyz_mmm[1])*SQR(y - xyz_ppp[1])*SQR(SQR(x) + SQR(y) - SQR(radius))
    const double &x = xyz[0];
    const double &y = xyz[1];
    P4EST_ASSERT(component < P4EST_DIM);
    switch (component) {
    case dir::x:
    {
      return 2.0*(x - xyz_mmm[0])*SQR(x - xyz_ppp[0])*SQR(y - xyz_mmm[1])*SQR(y - xyz_ppp[1])*SQR(SQR(x) + SQR(y) - SQR(ls->get_radius()))/SQR(SQR(ls->get_radius()))
          + SQR(x - xyz_mmm[0])*2.0*(x - xyz_ppp[0])*SQR(y - xyz_mmm[1])*SQR(y - xyz_ppp[1])*SQR(SQR(x) + SQR(y) - SQR(ls->get_radius()))/SQR(SQR(ls->get_radius()))
          + SQR(x - xyz_mmm[0])*SQR(x - xyz_ppp[0])*SQR(y - xyz_mmm[1])*SQR(y - xyz_ppp[1])*2.0*(SQR(x) + SQR(y) - SQR(ls->get_radius()))*2.0*x/SQR(SQR(ls->get_radius()));
    }
      break;
    case dir::y:
    {
      return SQR(x - xyz_mmm[0])*SQR(x - xyz_ppp[0])*2.0*(y - xyz_mmm[1])*SQR(y - xyz_ppp[1])*SQR(SQR(x) + SQR(y) - SQR(ls->get_radius()))/SQR(SQR(ls->get_radius()))
          + SQR(x - xyz_mmm[0])*SQR(x - xyz_ppp[0])*SQR(y - xyz_mmm[1])*2.0*(y - xyz_ppp[1])*SQR(SQR(x) + SQR(y) - SQR(ls->get_radius()))/SQR(SQR(ls->get_radius()))
          + SQR(x - xyz_mmm[0])*SQR(x - xyz_ppp[0])*SQR(y - xyz_mmm[1])*SQR(y - xyz_ppp[1])*2.0*(SQR(x) + SQR(y) - SQR(ls->get_radius()))*2.0*y/SQR(SQR(ls->get_radius()));
    }
      break;
    default:
      throw std::invalid_argument("grad_scalar_part_t::component() : unknown velocity component...");
      break;
    }
  }
};

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin",            "min level of the tree in the computational grid, default is " + to_string(default_lmin));
  cmd.add_option("lmax",            "max level of the tree in the computational grid, default is " + to_string(default_lmax));
  cmd.add_option("save_vtk",        "saves the p4est's (computational and interface-capturing grids) in vtk format");
  cmd.add_option("work_dir",        "exportation directory, if not defined otherwise in the environment variable OUT_DIR. \n\
\tThis is required if saving vtk or summary files: work_dir/vtu for vtk files work_dir/summaries for summary files. Default is " + default_work_folder);
  cmd.add_option("second_order_ls", "activate second order interface localization if present. Default is " + string(default_use_second_order_theta ? "true" : "false"));
  cmd.add_option("ntree",           "number of trees in the macromesh along every dimension of the computational domain. Default value is " + to_string(default_ntree));
  cmd.add_option("subrefinement",   "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement");
  cmd.add_option("solver",          "solver(s) to be tested, possible choices are 'GFM', 'xGFM', 'FV' or any combination thereof (separated with comma(s), and no space characters) [default is all of them].");
  ostringstream oss;
  oss.str("");
  oss << default_interp_method_phi;
  cmd.add_option("phi_interp",      "interpolation method for the node-sampled levelset function (relevant only if not using subrefinement). Default is " + oss.str());
  if(cmd.parse(argc, argv, main_description))
    return 0;

  const int lmin                        = cmd.get<int>("lmin",  default_lmin);
  const int lmax                        = cmd.get<int>("lmax",  default_lmax);
  const int ntree                       = cmd.get<int>("ntree", default_ntree);
  const int n_xyz[P4EST_DIM]            = {DIM(ntree, ntree, ntree)};
  const int periodicity[P4EST_DIM]      = {DIM(0, 0, 0)};
  const bool use_second_order_theta     = default_use_second_order_theta || cmd.contains("second_order_ls");
  const bool save_vtk                   = cmd.contains("save_vtk");
  const bool use_subrefinement          = cmd.get<bool>("subrefinement", default_subrefinement);
  const interpolation_method phi_interp = cmd.get<interpolation_method>("phi_interp", default_interp_method_phi);

  std::vector<solver_tag> default_solvers_to_test; default_solvers_to_test.push_back(GFM); default_solvers_to_test.push_back(xGFM); default_solvers_to_test.push_back(FV);
  const std::vector<solver_tag> solvers_to_test = cmd.get<std::vector<solver_tag> >("solver", default_solvers_to_test);
  if(solvers_to_test.size() > 3)
    throw std::invalid_argument("main for testing my_p4est_poisson_jump_cells : do not duplicate the solvers to test, that is not allowed...");

  parStopWatch watch, watch_global;
  watch_global.start("Total run time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  BoundaryConditionsDIM bc;
  BCWALLTYPE bc_wall_type(NEUMANN); bc.setWallTypes(bc_wall_type);
  BCWALLVAL bc_wall_val;            bc.setWallValues(bc_wall_val);
  level_set_sphere ls(0.5);
  divergence_free_velocity_field_t divergence_free_velocity_field(&ls);
  grad_scalar_part_t grad_scalar_part(&ls);

  const string out_folder = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR"))) + "/";
  const string vtk_out = out_folder + "/vtu/" + (use_subrefinement ? "subresolved" : "standard");

  connectivity = my_p4est_brick_new(n_xyz, xyz_mmm, xyz_ppp, &brick, periodicity);
  splitting_criteria_cf_t       *data       = NULL, *subrefined_data        = NULL;
  p4est_t                       *p4est      = NULL, *subrefined_p4est       = NULL;
  p4est_nodes_t                 *nodes      = NULL, *subrefined_nodes       = NULL;
  p4est_ghost_t                 *ghost      = NULL, *subrefined_ghost       = NULL;
  my_p4est_hierarchy_t          *hierarchy  = NULL, *subrefined_hierarchy   = NULL;
  my_p4est_node_neighbors_t     *ngbd_n     = NULL, *subrefined_ngbd_n      = NULL;
  Vec                            phi        = NULL,  subrefined_phi         = NULL;
  my_p4est_cell_neighbors_t     *ngbd_c     = NULL;
  my_p4est_faces_t              *faces      = NULL;
  my_p4est_interface_manager_t  *interface_manager = NULL;

  ierr = PetscPrintf(mpi.comm(), "Building computational grid \n"); CHKERRXX(ierr);

  /* build/updates the computational grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods, its cell neighborhoods
   * the REINITIALIZED levelset on the computational grid
   */
  data = new splitting_criteria_cf_t(lmin, lmax, &ls);
  set_computational_grid_data(mpi, &brick, connectivity, data,
                              p4est, ghost, nodes, phi, hierarchy, ngbd_n, ngbd_c, faces);

  const my_p4est_node_neighbors_t* interface_capturing_ngbd_n;  // no creation here, just a renamed pointer to streamline the logic
  Vec interface_capturing_phi;                                  // no creation here, just a renamed pointer to streamline the logic
  if(use_subrefinement)
  {
    /* build/updates the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
     * the REINITIALIZED levelset on the interface-capturing grid
     */
    subrefined_data = new splitting_criteria_cf_t(data->min_lvl, data->max_lvl + 1, &ls);
    build_interface_capturing_grid_data(p4est, &brick, subrefined_data,
                                        subrefined_p4est, subrefined_ghost, subrefined_nodes, subrefined_phi, subrefined_hierarchy, subrefined_ngbd_n);
    interface_capturing_ngbd_n  = subrefined_ngbd_n;
    interface_capturing_phi     = subrefined_phi;
  }
  else
  {
    interface_capturing_ngbd_n  = ngbd_n;
    interface_capturing_phi     = phi;
  }

  Vec interface_capturing_phi_xxyyzz = NULL;
  interface_manager = new my_p4est_interface_manager_t(faces, nodes, interface_capturing_ngbd_n);
  if(use_second_order_theta || (!use_subrefinement && phi_interp != linear)){
    ierr = VecCreateGhostNodesBlock(interface_capturing_ngbd_n->get_p4est(), interface_capturing_ngbd_n->get_nodes(), P4EST_DIM, &interface_capturing_phi_xxyyzz); CHKERRXX(ierr);
    interface_capturing_ngbd_n->second_derivatives_central(interface_capturing_phi, interface_capturing_phi_xxyyzz);
  }
  interface_manager->evaluate_FD_theta_with_quadratics_if_second_derivatives_are_available(use_second_order_theta);
  interface_manager->set_levelset(interface_capturing_phi, (use_subrefinement ? linear : phi_interp), interface_capturing_phi_xxyyzz, true); // last argument set to true cause we'll need the gradient of phi
  if(use_subrefinement)
    interface_manager->set_under_resolved_levelset(phi);

  const double irho_minus  = 1.0;
  const double irho_plus   = 0.5;

  // Get the velocity fields to project
  Vec v_face_plus[P4EST_DIM], v_face_minus[P4EST_DIM];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecCreateGhostFaces(p4est, faces, &v_face_minus[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &v_face_plus[dim],  dim); CHKERRXX(ierr);
  }
  /* TEST THE JUMP SOLVER AND COMPARE TO ORIGINAL GFM */
  for(size_t k = 0; k < solvers_to_test.size(); ++k) {

    my_p4est_poisson_jump_cells_t* jump_solver = NULL;

    switch (solvers_to_test[k]) {
    case GFM:
    case xGFM:
    {
      my_p4est_poisson_jump_cells_xgfm_t *solver = new my_p4est_poisson_jump_cells_xgfm_t(ngbd_c, nodes);
      solver->activate_xGFM_corrections(solvers_to_test[k] == xGFM);
      interface_manager->clear_all_FD_interface_neighbors(); // for representative timing, if storing the maps
      jump_solver = solver;
    }
      break;
    case FV:
    {
      my_p4est_poisson_jump_cells_fv_t *solver = new my_p4est_poisson_jump_cells_fv_t(ngbd_c, nodes);
      jump_solver = solver;
    }
      break;
    default:
      throw std::runtime_error("main for testing two-phase projection : unkonwn solver type...");
      break;
    }

    jump_solver->set_interface(interface_manager);
    jump_solver->set_mus(irho_minus, irho_plus);
    jump_solver->set_jumps(NULL, NULL);
    jump_solver->set_diagonals(0.0, 0.0);
    jump_solver->set_bc(bc);

    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      double *vface_minus_p, *vface_plus_p;
      ierr = VecGetArray(v_face_minus[dim], &vface_minus_p); CHKERRXX(ierr);
      ierr = VecGetArray(v_face_plus[dim], &vface_plus_p); CHKERRXX(ierr);
      for (p4est_locidx_t k = 0; k < faces->num_local[dim] + faces->num_ghost[dim]; ++k) {
        double xyz_face[P4EST_DIM]; faces->xyz_fr_f(k, dim, xyz_face);
        vface_minus_p[k]  = divergence_free_velocity_field.velocity_component(-1, dim, xyz_face) + grad_scalar_part.component(dim, xyz_face)*irho_minus;
        vface_plus_p[k]   = divergence_free_velocity_field.velocity_component(+1, dim, xyz_face) + grad_scalar_part.component(dim, xyz_face)*irho_plus;
      }
      ierr = VecRestoreArray(v_face_minus[dim], &vface_minus_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(v_face_plus[dim], &vface_plus_p); CHKERRXX(ierr);
    }


    KSPType ksp_solver = (dynamic_cast<my_p4est_poisson_jump_cells_xgfm_t*>(jump_solver) != NULL ? KSPCG : KSPBCGS);
    watch.start("Total time:");
    for (int iter = 0; iter < 10; ++iter) {
      jump_solver->set_velocity_on_faces(v_face_minus, v_face_plus);
      jump_solver->solve_for_sharp_solution(ksp_solver, PCHYPRE);
      jump_solver->project_face_velocities(faces);

      double error_vface_minus[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
      double error_vface_plus[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};

      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        const double *vface_minus_p, *vface_plus_p;
        ierr = VecGetArrayRead(v_face_minus[dim], &vface_minus_p); CHKERRXX(ierr);
        ierr = VecGetArrayRead(v_face_plus[dim], &vface_plus_p); CHKERRXX(ierr);
        for (p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dim]; ++f_idx) {
          double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dim, xyz_face);
          if(fabs(vface_minus_p[f_idx]) < largest_dbl_smaller_than_dbl_max)
            error_vface_minus[dim]  = MAX(error_vface_minus[dim], fabs(vface_minus_p[f_idx] - divergence_free_velocity_field.velocity_component(-1, dim, xyz_face)));
          if(fabs(vface_plus_p[f_idx]) < largest_dbl_smaller_than_dbl_max)
            error_vface_plus[dim]   = MAX(error_vface_plus[dim],  fabs(vface_plus_p[f_idx] - divergence_free_velocity_field.velocity_component(+1, dim, xyz_face)));
        }
        ierr = VecRestoreArrayRead(v_face_minus[dim], &vface_minus_p); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(v_face_plus[dim], &vface_plus_p); CHKERRXX(ierr);
      }
      int mpiret;
      mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vface_minus, P4EST_DIM, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vface_plus,  P4EST_DIM, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
      if(mpi.rank() == 0)
      {
        std::cout << "error on vface minus for solver " << convert_to_string(solvers_to_test[k]) << " = " << error_vface_minus[0] << ", " << error_vface_minus[1] << std::endl;
        std::cout << "error on vface plus for solver " << convert_to_string(solvers_to_test[k]) << " = " << error_vface_plus[0] << ", " << error_vface_plus[1] << std::endl;
      }
    }

    watch.stop(); watch.read_duration();

//    {
//      if(create_directory(vtk_out, mpi.rank()))
//        throw std::runtime_error("impossible to create directory");
//      const double *solution_p, *rhs_copy_p;
//      ierr = VecGetArrayRead(jump_solver->get_solution(), &solution_p); CHKERRXX(ierr);
//      ierr = VecGetArrayRead(dynamic_cast<my_p4est_poisson_jump_cells_xgfm_t*>(jump_solver)->get_rhs_copy(), &rhs_copy_p); CHKERRXX(ierr);
//      my_p4est_vtk_write_all(p4est, nodes, ghost, P4EST_TRUE, P4EST_TRUE,
//                             0, 2, (vtk_out + "/solution").c_str(),
//                             VTK_CELL_SCALAR, "solution", solution_p,
//                             VTK_CELL_SCALAR, "rhs", rhs_copy_p);
//      ierr = VecGetArrayRead(dynamic_cast<my_p4est_poisson_jump_cells_xgfm_t*>(jump_solver)->get_rhs_copy(), &rhs_copy_p); CHKERRXX(ierr);
//      ierr = VecRestoreArrayRead(jump_solver->get_solution(), &solution_p); CHKERRXX(ierr);
//    }


    delete jump_solver;
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = delete_and_nullify_vector(v_face_minus[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(v_face_plus[dim]); CHKERRXX(ierr);
  }

  // destroy data created for this iteration
  ierr = VecDestroy(phi);               CHKERRXX(ierr); phi = NULL;
  if(subrefined_phi != NULL){
    ierr = VecDestroy(subrefined_phi);  CHKERRXX(ierr); subrefined_phi = NULL; }
  if(interface_capturing_phi_xxyyzz != NULL){
    ierr = VecDestroy(interface_capturing_phi_xxyyzz); CHKERRXX(ierr);  interface_capturing_phi_xxyyzz = NULL; }

  delete data;
  if(subrefined_data != NULL)
    delete subrefined_data;

  delete interface_manager;


  delete faces;
  delete ngbd_c;
  delete ngbd_n;              if(subrefined_ngbd_n != NULL)     delete subrefined_ngbd_n;
  delete hierarchy;           if(subrefined_hierarchy != NULL)  delete subrefined_hierarchy;
  p4est_nodes_destroy(nodes); if(subrefined_nodes != NULL)      p4est_nodes_destroy(subrefined_nodes);
  p4est_ghost_destroy(ghost); if(subrefined_ghost != NULL)      p4est_ghost_destroy(subrefined_ghost);
  p4est_destroy      (p4est); if(subrefined_p4est != NULL)      p4est_destroy(subrefined_p4est);

  my_p4est_brick_destroy(connectivity, &brick);

  watch_global.stop(); watch_global.read_duration();

  return 0;
}
