
/*
 * Test the cell based p4est.
 * 1 - solve a poisson equation on an irregular domain (circle)
 * 2 - interpolate from faces to nodes
 * 3 - extrapolate faces over interface
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/cube3.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/cube2.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

double xmin = -1;
double xmax =  1;
double ymin = -1;
double ymax =  1;
double zmin = -1;
double zmax =  1;

using namespace std;

int lmin = 4;
int lmax = 4;
int nb_splits = 6;

int nx = 1;
int ny = 1;
int nz = 1;

bool save_vtk = true;

double mu = 1.0;
double add_diagonal = 1.0;

/*
 * 0 - circle
 */
int interface_type = 0;

/*
 *  ********* 2D *********
 * 0 - sin(x)*cos(y)
 */
int test_number_m = 0;

/*
 *  ********* 2D *********
 * 0 - exp(x+y)
 */
int test_number_p = 0;

/*
 *  ********* 2D *********
 * 0 - 0
 * 1 - 1
 */
int diag_number = 0;


BoundaryConditionType bc_itype = NOINTERFACE;
BoundaryConditionType bc_wtype = DIRICHLET;

double r0 = 0.57;
double xC = 0.01;
double yC =-0.03;
double zC = 0.02;

#undef P4_TO_P8

#ifdef P4_TO_P8
#else

class NO_INTERFACE_CF : public CF_2
{
public:
  double operator()(double, double) const
  {
    return -1;
  }
} no_interface_cf;

/* level-set function and its derivatives */
class PHI_CF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
      return (r0 - sqrt(SQR(x - xC) + SQR(y - yC)));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_cf;

class PHI_X_CF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
        return -(x - xC)/sqrt(SQR(x - xC) + SQR(y - yC));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_x_cf;

class PHI_Y_CF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
        return -(y - yC)/sqrt(SQR(x - xC) + SQR(y - yC));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_y_cf;

/* test functions */
class UM_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_m)
    {
    case 0:
      return sin(x)*cos(y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} um_exact;

class UP_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_p)
    {
      case 0:
        return exp(x+y);
      case 1:
        return um_exact(x,y) + log(1.0+phi_cf(x,y))*exp(x+y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} up_exact;

/* first derivatives of test functions */
class UM_X_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_m)
    {
    case 0:
      return cos(x)*cos(y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} um_x_exact;

class UM_Y_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_m)
    {
    case 0:
      return -sin(x)*sin(y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} um_y_exact;

class UP_X_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_p)
    {
    case 0:
        return exp(x+y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} up_x_exact;

class UP_Y_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_p)
    {
    case 0:
      return exp(x+y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} up_y_exact;

/* laplacians of test functions */
class LAP_UM_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_m)
    {
    case 0:
      return -2.0*sin(x)*cos(y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} lap_um_exact;

class LAP_UP_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number_p)
    {
    case 0:
      return 2.0*exp(x+y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} lap_up_exact;

/* jump conditions */
class U_JUMP : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return um_exact(x,y) - up_exact(x,y);
  }
} u_jump;

class UN_JUMP : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;

    return (um_x_exact(x,y) - up_x_exact(x,y))*nx + (um_y_exact(x,y) - up_y_exact(x,y))*ny;
  }
} un_jump;

/* diagonal terms */
class DIAG_ADD : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(diag_number)
    {
      case 0:
        return 0.0;
      case 1:
        return 1.0;
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} diag_add;


/* force functions */
class FORCE_M : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return add_diagonal*um_exact(x,y) - mu*lap_um_exact(x,y);
  }
} force_m;

class FORCE_P : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return add_diagonal*up_exact(x,y) - mu*lap_up_exact(x,y);
  }
} force_p;

class BCWALLTYPE : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return bc_wtype;
  }
} bc_wall_type;

class U_EXACT : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (phi_cf(x,y) < 0.) return um_exact(x,y);
    else                  return up_exact(x,y);
  }
} u_exact;

#endif

void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err_nodes, Vec err_ex,
              int compt);

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
  cmd.add_option("save_vtk", "save the p4est in vtk format");

#ifdef P4_TO_P8
#else
  cmd.add_option("test_number_m", "test_number_m.\n\
                 0 - sin(x)*cos(y)");
  cmd.add_option("test_number_p", "test_number_p.\n\
                 0 - exp(x+y)");
  cmd.add_option("diag_number", "diag_number.\n\
                 0 - 0.0\n\
                 1 - 1.0");
#endif

  cmd.parse(argc, argv);

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);
  test_number_m = cmd.get("test_number_m", test_number_m);
  test_number_p = cmd.get("test_number_p", test_number_p);

  diag_number = cmd.get("diag_number", diag_number);

  bc_wtype = cmd.get("bc_wtype", bc_wtype);
  bc_itype = cmd.get("bc_itype", bc_itype);

  save_vtk = cmd.get("save_vtk", save_vtk);

  parStopWatch w;
  w.start("total time");

  if(0)
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const int n_xyz[] = {nx, ny, nz};
  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};
  const int periodic[] = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  double err_n;
  double err_nm1;

  double err_ex_n;
  double err_ex_nm1;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    srand(1);
//    splitting_criteria_random_t data(2, 7, 1000, 10000);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &phi_cf, 1.2);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, no_interface_cf, phi);

    Vec phi_interface;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_interface); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, phi_cf, phi_interface);

    my_p4est_level_set_t ls(&ngbd_n);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_interface);
//    ls.perturb_level_set_function(phi, EPS);

    /* find dx and dy smallest */
    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
    double dx = (xmax-xmin) / pow(2.,(double) data.max_lvl);
    double dy = (ymax-ymin) / pow(2.,(double) data.max_lvl);

    double dxyz[P4EST_DIM] = {dx, dy};

#ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
    double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
    double diag = sqrt(dx*dx + dy*dy + dz*dz);

    double dxyz[P4EST_DIM] = {dx, dy, dz};
#else
    double diag = sqrt(dx*dx + dy*dy);
#endif

    /* TEST THE NODES FUNCTIONS */
#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif

    bc.setWallTypes(bc_wall_type);
    bc.setWallValues(u_exact);
    bc.setInterfaceType(bc_itype);
    bc.setInterfaceValue(no_interface_cf);

    Vec rhs;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);


    Vec tmp;
    ierr = VecCreateGhostNodes(p4est, nodes, &tmp); CHKERRXX(ierr);

    Vec u_jump_vec;
    ierr = VecCreateGhostNodes(p4est, nodes, &u_jump_vec); CHKERRXX(ierr);

    Vec un_jump_vec;
    ierr = VecCreateGhostNodes(p4est, nodes, &un_jump_vec); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, u_jump, tmp);
    ls.extend_from_interface_to_whole_domain_TVD(phi_interface, tmp, u_jump_vec);

    sample_cf_on_nodes(p4est, nodes, un_jump, tmp);
    ls.extend_from_interface_to_whole_domain_TVD(phi_interface, tmp, un_jump_vec);

    ierr = VecDestroy(tmp); CHKERRXX(ierr);

    quad_neighbor_nodes_of_node_t qnnn;
    double p_000, p_m00, p_p00, p_0m0, p_0p0;
  #ifdef P4_TO_P8
    double p_00m, p_00p;
  #endif

    double *rhs_p, *phii_p, *u_jump_vec_p, *un_jump_vec_p;
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_interface, &phii_p); CHKERRXX(ierr);
    ierr = VecGetArray(u_jump_vec, &u_jump_vec_p); CHKERRXX(ierr);
    ierr = VecGetArray(un_jump_vec, &un_jump_vec_p); CHKERRXX(ierr);

    if (false)
    {
      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
#endif

        if (phi_cf(x,y) < 0) rhs_p[n] = force_m(x,y);
        else                 rhs_p[n] = force_p(x,y);

        double sign_of_phi = 1.0;

        if (phi_cf(x,y) < 0) sign_of_phi = -1.0;

        qnnn = ngbd_n.get_neighbors(n);
#ifdef P4_TO_P8
        qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
        if (p_000*p_m00<=0){
          p4est_locidx_t neigh;
          if     (qnnn.d_m00_m0==0 && qnnn.d_m00_0m==0) neigh = qnnn.node_m00_mm;
          else if(qnnn.d_m00_m0==0 && qnnn.d_m00_0p==0) neigh = qnnn.node_m00_mp;
          else if(qnnn.d_m00_p0==0 && qnnn.d_m00_0m==0) neigh = qnnn.node_m00_pm;
          else                                          neigh = qnnn.node_m00_pp;
          rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
        }
        if (p_000*p_p00<=0){
          p4est_locidx_t neigh;
          if     (qnnn.d_p00_m0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_mm;
          else if(qnnn.d_p00_m0==0 && qnnn.d_p00_0p==0) neigh = qnnn.node_p00_mp;
          else if(qnnn.d_p00_p0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_pm;
          else                                          neigh = qnnn.node_p00_pp;
          rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
        }
        if (p_000*p_0m0<=0){
          p4est_locidx_t neigh;
          if     (qnnn.d_0m0_m0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_mm;
          else if(qnnn.d_0m0_m0==0 && qnnn.d_0m0_0p==0) neigh = qnnn.node_0m0_mp;
          else if(qnnn.d_0m0_p0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_pm;
          else                                          neigh = qnnn.node_0m0_pp;
          rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
        }
        if (p_000*p_0p0<=0){
          p4est_locidx_t neigh;
          if     (qnnn.d_0p0_m0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_mm;
          else if(qnnn.d_0p0_m0==0 && qnnn.d_0p0_0p==0) neigh = qnnn.node_0p0_mp;
          else if(qnnn.d_0p0_p0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_pm;
          else                                          neigh = qnnn.node_0p0_pp;
          rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
        }
        if (p_000*p_00m<=0){
          p4est_locidx_t neigh;
          if     (qnnn.d_00m_m0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_mm;
          else if(qnnn.d_00m_m0==0 && qnnn.d_00m_0p==0) neigh = qnnn.node_00m_mp;
          else if(qnnn.d_00m_p0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_pm;
          else                                          neigh = qnnn.node_00m_pp;
          rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[2]) * scaling_jump;
        }
        if (p_000*p_00p<=0){
          p4est_locidx_t neigh;
          if     (qnnn.d_00p_m0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_mm;
          else if(qnnn.d_00p_m0==0 && qnnn.d_00p_0p==0) neigh = qnnn.node_00p_mp;
          else if(qnnn.d_00p_p0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_pm;
          else                                          neigh = qnnn.node_00p_pp;
          rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[2]) * scaling_jump;
        }
#else
        qnnn.ngbd_with_quadratic_interpolation(phii_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
        if (p_000*p_m00<=0){
          p4est_locidx_t neigh = (qnnn.d_m00_m0==0) ? qnnn.node_m00_mm : qnnn.node_m00_pm;
          rhs_p[n] += (fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + (p_m00 < 0 ? -1. : 1.)*u_jump_vec_p[neigh]) / SQR(dxyz[0]) * mu;
        }
        if (p_000*p_p00<=0){
          p4est_locidx_t neigh = (qnnn.d_p00_m0==0) ? qnnn.node_p00_mm : qnnn.node_p00_pm;
          rhs_p[n] += (fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + (p_p00 < 0 ? -1. : 1.)*u_jump_vec_p[neigh]) / SQR(dxyz[0]) * mu;
        }
        if (p_000*p_0m0<=0){
          p4est_locidx_t neigh = (qnnn.d_0m0_m0==0) ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
          rhs_p[n] += (fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + (p_0m0 < 0 ? -1. : 1.)*u_jump_vec_p[neigh]) / SQR(dxyz[1]) * mu;
        }
        if (p_000*p_0p0<=0){
          p4est_locidx_t neigh = (qnnn.d_0p0_m0==0) ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
          rhs_p[n] += (fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + (p_0p0 < 0 ? -1. : 1.)*u_jump_vec_p[neigh]) / SQR(dxyz[1]) * mu;
        }
#endif
      }

    } else {

      my_p4est_interpolation_nodes_t phi_interp(&ngbd_n);
      phi_interp.set_input(phi_interface, linear);

      my_p4est_interpolation_nodes_t un_jump_interp(&ngbd_n);
      un_jump_interp.set_input(un_jump_vec, linear);

      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
#endif

        if (fabs(phii_p[n]) > 2.*diag)
        {

          if (phii_p[n] < 0) rhs_p[n] = force_m(x,y);
          else               rhs_p[n] = force_p(x,y);

        } else {

          double xm = x - 0.5*dxyz[0];
          double xp = x + 0.5*dxyz[0];
          double ym = y - 0.5*dxyz[1];
          double yp = y + 0.5*dxyz[1];
  #ifdef P4_TO_P8
          double zm = z - 0.5*dxyz[2];
          double zp = z + 0.5*dxyz[2];
  #endif
          double P_000 = phii_p[n];
          double P_mm0 = phi_interp(xm, ym);
          double P_mp0 = phi_interp(xm, yp);
          double P_pm0 = phi_interp(xp, ym);
          double P_pp0 = phi_interp(xp, yp);
          double P_m00 = phi_interp(xm, y );
          double P_p00 = phi_interp(xp, y );
          double P_0m0 = phi_interp(x , ym);
          double P_0p0 = phi_interp(x , yp);

          bool is_one_positive = (P_000 > 0 ||
                                  P_m00 > 0 || P_p00 > 0 || P_0m0 > 0 || P_0p0 > 0 ||
                                  P_mm0 > 0 || P_pm0 > 0 || P_mp0 > 0 || P_pp0 > 0);
          bool is_one_negative = (P_000 < 0 ||
                                  P_m00 < 0 || P_p00 < 0 || P_0m0 < 0 || P_0p0 < 0 ||
                                  P_mm0 < 0 || P_pm0 < 0 || P_mp0 < 0 || P_pp0 < 0);

          bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;

          qnnn = ngbd_n.get_neighbors(n);
          qnnn.ngbd_with_quadratic_interpolation(phii_p, p_000, p_m00, p_p00, p_0m0, p_0p0);

          is_one_positive = (p_000 > 0 || p_m00 > 0 || p_p00 > 0 || p_0m0 > 0 || p_0p0 > 0);
          is_one_negative = (p_000 < 0 || p_m00 < 0 || p_p00 < 0 || p_0m0 < 0 || p_0p0 < 0);

          bool is_ngbd_crossed_dirichlet = is_one_negative && is_one_positive;

          if (is_ngbd_crossed_neumann || is_ngbd_crossed_dirichlet)
          {

#ifdef P4_TO_P8
            Cube3 cube;
#else
            Cube2 cube;
#endif


            double volume_m = 0.;
            double interface_area  = 0.;
            double integral_bc = 0;
            QuadValue phi_cube;
            QuadValue bc_value;
            for (short i = 0; i < P4EST_CHILDREN; ++i)
            {
              switch (i) {
                case 0: phi_cube.val00 = P_mm0; phi_cube.val01 = P_m00; phi_cube.val10 = P_0m0; phi_cube.val11 = P_000; break;
                case 1: phi_cube.val00 = P_mp0; phi_cube.val01 = P_0p0; phi_cube.val10 = P_m00; phi_cube.val11 = P_000; break;
                case 2: phi_cube.val00 = P_pm0; phi_cube.val01 = P_0m0; phi_cube.val10 = P_p00; phi_cube.val11 = P_000; break;
                case 3: phi_cube.val00 = P_pp0; phi_cube.val01 = P_p00; phi_cube.val10 = P_0p0; phi_cube.val11 = P_000; break;
              }

              switch (i) {
                case 0: case 2: cube.x0 = 0.; cube.y0 = 0.; cube.x1 = 0.5*dxyz[0]; cube.y1 = 0.5*dxyz[1]; break;
                case 1: case 3: cube.x0 = 0.; cube.y0 = 0.; cube.x1 = 0.5*dxyz[1]; cube.y1 = 0.5*dxyz[0]; break;
              }

//              switch (i) {
//                case 0: bc_value.val00 = un_jump(xm,ym); bc_value.val01 = un_jump(xm,y ); bc_value.val10 = un_jump(x ,ym); bc_value.val11 = un_jump(x,y); break;
//                case 1: bc_value.val00 = un_jump(xm,yp); bc_value.val01 = un_jump(x ,yp); bc_value.val10 = un_jump(xm,y ); bc_value.val11 = un_jump(x,y); break;
//                case 2: bc_value.val00 = un_jump(xp,ym); bc_value.val01 = un_jump(x ,ym); bc_value.val10 = un_jump(xp,y ); bc_value.val11 = un_jump(x,y); break;
//                case 3: bc_value.val00 = un_jump(xp,yp); bc_value.val01 = un_jump(xp,y ); bc_value.val10 = un_jump(x ,yp); bc_value.val11 = un_jump(x,y); break;
//              }

              switch (i) {
                case 0: bc_value.val00 = un_jump_interp(xm,ym); bc_value.val01 = un_jump_interp(xm,y ); bc_value.val10 = un_jump_interp(x ,ym); bc_value.val11 = un_jump_interp(x,y); break;
                case 1: bc_value.val00 = un_jump_interp(xm,yp); bc_value.val01 = un_jump_interp(x ,yp); bc_value.val10 = un_jump_interp(xm,y ); bc_value.val11 = un_jump_interp(x,y); break;
                case 2: bc_value.val00 = un_jump_interp(xp,ym); bc_value.val01 = un_jump_interp(x ,ym); bc_value.val10 = un_jump_interp(xp,y ); bc_value.val11 = un_jump_interp(x,y); break;
                case 3: bc_value.val00 = un_jump_interp(xp,yp); bc_value.val01 = un_jump_interp(xp,y ); bc_value.val10 = un_jump_interp(x ,yp); bc_value.val11 = un_jump_interp(x,y); break;
              }

              volume_m        += cube.area_In_Negative_Domain(phi_cube);
              interface_area  += cube.interface_Length_In_Cell(phi_cube);
              integral_bc     += cube.integrate_Over_Interface(bc_value, phi_cube);
            }

            double volume = dx*dy;
            double volume_p = volume - volume_m;

            double sm_m00 = 0.5 * dy * (fraction_Interval_Covered_By_Irregular_Domain(P_m00, P_mm0, dx, dy) +
                                        fraction_Interval_Covered_By_Irregular_Domain(P_m00, P_mp0, dx, dy));

            double sm_p00 = 0.5 * dy * (fraction_Interval_Covered_By_Irregular_Domain(P_p00, P_pm0, dx, dy) +
                                        fraction_Interval_Covered_By_Irregular_Domain(P_p00, P_pp0, dx, dy));

            double sm_0m0 = 0.5 * dx * (fraction_Interval_Covered_By_Irregular_Domain(P_0m0, P_mm0, dx, dy) +
                                        fraction_Interval_Covered_By_Irregular_Domain(P_0m0, P_pm0, dx, dy));

            double sm_0p0 = 0.5 * dx * (fraction_Interval_Covered_By_Irregular_Domain(P_0p0, P_mp0, dx, dy) +
                                        fraction_Interval_Covered_By_Irregular_Domain(P_0p0, P_pp0, dx, dy));

            double sp_m00 = dy - sm_m00;
            double sp_p00 = dy - sm_p00;
            double sp_0m0 = dx - sm_0m0;
            double sp_0p0 = dx - sm_0p0;

            double sign_of_phi = 1.0;

            if (phii_p[n] < 0) sign_of_phi = -1.0;

            rhs_p[n] = force_m(x,y)*volume_m + force_p(x,y)*volume_p + mu*integral_bc;

            if (phii_p[n] < 0)
              rhs_p[n] -= (fabs(phii_p[n]) * un_jump_vec_p[n] - u_jump_vec_p[n])
                *( add_diagonal*volume_p + mu*((sp_m00+sp_p00)/dxyz[0] + (sp_0m0+sp_0p0)/dxyz[1]));
            else
              rhs_p[n] -= (fabs(phii_p[n]) * un_jump_vec_p[n] + u_jump_vec_p[n])
                *( add_diagonal*volume_m + mu*((sm_m00+sm_p00)/dxyz[0] + (sm_0m0+sm_0p0)/dxyz[1]));

            qnnn = ngbd_n.get_neighbors(n);
    #ifdef P4_TO_P8
    #else
            p4est_locidx_t neigh;

            neigh = (qnnn.d_m00_m0==0) ? qnnn.node_m00_mm : qnnn.node_m00_pm;
            if (p_m00 < 0)  rhs_p[n] += mu*sp_m00/dxyz[0]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] - u_jump_vec_p[neigh]);
            else            rhs_p[n] += mu*sm_m00/dxyz[0]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + u_jump_vec_p[neigh]);

            neigh = (qnnn.d_p00_m0==0) ? qnnn.node_p00_mm : qnnn.node_p00_pm;
            if (p_p00 < 0)  rhs_p[n] += mu*sp_p00/dxyz[0]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] - u_jump_vec_p[neigh]);
            else            rhs_p[n] += mu*sm_p00/dxyz[0]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + u_jump_vec_p[neigh]);

            neigh = (qnnn.d_0m0_m0==0) ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
            if (p_0m0 < 0)  rhs_p[n] += mu*sp_0m0/dxyz[1]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] - u_jump_vec_p[neigh]);
            else            rhs_p[n] += mu*sm_0m0/dxyz[1]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + u_jump_vec_p[neigh]);

            neigh = (qnnn.d_0p0_m0==0) ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
            if (p_0p0 < 0)  rhs_p[n] += mu*sp_0p0/dxyz[1]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] - u_jump_vec_p[neigh]);
            else            rhs_p[n] += mu*sm_0p0/dxyz[1]*(fabs(phii_p[neigh]) * un_jump_vec_p[neigh] + u_jump_vec_p[neigh]);

            rhs_p[n] /= volume;
    #endif
          } else {
            if (phii_p[n] < 0) rhs_p[n] = force_m(x,y);
            else               rhs_p[n] = force_p(x,y);
          }
        }
      }
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_interface, &phii_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(u_jump_vec, &u_jump_vec_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(un_jump_vec, &un_jump_vec_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

    my_p4est_poisson_nodes_t solver(&ngbd_n);
    solver.set_phi(phi);
    solver.set_diagonal(add_diagonal);
    solver.set_mu(mu);
    solver.set_bc(bc);
    solver.set_rhs(rhs);

    Vec sol;
    ierr = VecDuplicate(rhs, &sol); CHKERRXX(ierr);

    solver.solve(sol);

    /* check the error */
    err_nm1 = err_n;
    err_n = 0;

    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    const double *sol_p;
    ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);

    Vec err_nodes;
    ierr = VecDuplicate(sol, &err_nodes); CHKERRXX(ierr);
    double *err_p;
    ierr = VecGetArray(err_nodes, &err_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(bc_itype==NOINTERFACE || phi_p[n]<0)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        err_p[n] = fabs(sol_p[n] - u_exact(x,y,z));
#else
        err_p[n] = fabs(sol_p[n] - u_exact(x,y));
#endif
        err_n = MAX(err_n, err_p[n]);
      }
      else
        err_p[n] = 0;
    }
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_nodes, &err_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(p4est->mpicomm, "Error on nodes : %g, order = %g\n", err_n, log(err_nm1/err_n)/log(2)); CHKERRXX(ierr);


    /* extrapolate the solution and check accuracy */
    double band = 4;

    if(bc_itype!=NOINTERFACE)
      ls.extend_Over_Interface_TVD(phi, sol, 100);

    ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    Vec err_ex;
    ierr = VecDuplicate(sol, &err_ex); CHKERRXX(ierr);
    double *err_ex_p;
    ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    err_ex_nm1 = err_ex_n;
    err_ex_n = 0;

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(phi_p[n]>0)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        if(phi_p[n]<band*diag)
          err_ex_p[n] = fabs(sol_p[n] - u_exact(x,y,z));
#else
        if(phi_p[n]<band*diag)
          err_ex_p[n] = fabs(sol_p[n] - u_exact(x,y));
#endif
        else
          err_ex_p[n] = 0;

        err_ex_n = MAX(err_ex_n, err_ex_p[n]);
      }
      else
        err_ex_p[n] = 0;
    }

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(p4est->mpicomm, "Error extrapolation : %g, order = %g\n", err_ex_n, log(err_ex_nm1/err_ex_n)/log(2)); CHKERRXX(ierr);


    if(save_vtk)
    {
      save_VTK(p4est, ghost, nodes, &brick, phi, sol, err_nodes, err_ex, iter);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);

    ierr = VecDestroy(err_nodes); CHKERRXX(ierr);
    ierr = VecDestroy(err_ex); CHKERRXX(ierr);

    ierr = VecDestroy(u_jump_vec); CHKERRXX(ierr);
    ierr = VecDestroy(un_jump_vec); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}


void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err_nodes, Vec err_ex,
              int compt)
{
  PetscErrorCode ierr;
  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) {
    out_dir = "out_dir/vtu";
    system((string("mkdir -p ")+out_dir).c_str());
  }

  std::ostringstream oss;

  oss << out_dir
      << "/Poisson_nodes_"
      << "proc="
      << p4est->mpisize << "_"
         << "brick="
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "_levels=(" <<lmin << "," << lmax << ")" <<
         ".split=" << compt;

  double *phi_p, *sol_p, *err_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(err_nodes, &err_p); CHKERRXX(ierr);

  double *err_ex_p;
  ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         4, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "err", err_p,
                         VTK_POINT_DATA, "err_ex", err_ex_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err_nodes, &err_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}
