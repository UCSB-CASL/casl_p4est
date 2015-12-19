// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

// p4est Library
#ifdef P4_TO_P8
// FIXME: implement this example in 3d
#error "Example not fully implemented in 3D"
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

/* Available options in 2d
 * 0 - frank sphere
 * 1 - a single seed
 * 2 - 20 seeds
 * 3 - a plane
 */
int test_number = 0;

double xmin, xmax;
double ymin, ymax;
#ifdef P4_TO_P8
double zmin, zmax;
#endif

// logging variables
PetscLogEvent log_compute_curvature;
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#endif
#ifndef CASL_LOG_FLOPS
#define PetscLogFlops(n) 0
#endif


double k_s = 1;
double k_l = 1;
double L = 1;
double G = 1;
double V = 0.1;
double t_interface = 0.;
double epsilon_c = -5e-6;
double epsilon_anisotropy = .5;
double N_anisotropy = 3;
double theta_0 = 0;

double tn;
double dt;

using namespace std;


/* error function */
double E1(double x)
{
  const double EULER=0.5772156649;
  const int    MAXIT=100;
  const double FPMIN=1.0e-20;

  int i,ii;
  double a,b,c,d,del,fact,h,psi,ans=0;

  int n   =1;
  int nm1 =0;

  if (x > 1.0)
  {        /* Lentz's algorithm */
    b=x+n;
    c=1.0/FPMIN;
    d=1.0/b;
    h=d;
    for (i=1;i<=MAXIT;i++)
    {
      a = -i*(nm1+i);
      b += 2.0;
      d=1.0/(a*d+b);    /* Denominators cannot be zero */
      c=b+a/c;
      del=c*d;
      h *= del;
      if (fabs(del-1.0) < EPS)
      {
        ans=h*exp(-x);
        return ans;
      }
    }
    printf("Continued fraction failed in expint\n");
  }
  else
  {
    ans = (nm1!=0 ? 1.0/nm1 : -log(x)-EULER);    /* Set first term */
    fact=1.0;
    for (i=1;i<=MAXIT;i++)
    {
      fact *= -x/i;
      if (i != nm1) del = -fact/(i-nm1);
      else
      {
        psi = -EULER;  /* Compute psi(n) */
        for (ii=1;ii<=nm1;ii++) psi += 1.0/ii;
        del=fact*(-log(x)+psi);
      }
      ans += del;
      if (fabs(del) < fabs(ans)*EPS) return ans;
    }
    printf("series failed in expint\n");
  }

  return ans;
}

double F (double s)
{
  return E1(.25*s*s);
}

double dF (double s)
{
  return -.5*s*exp(-s*s/4)/(s*s/4);
}

double t_frank_sphere(double x, double y, double t)
{
  double s = sqrt(x*x + y*y)/sqrt(t);
  double s0 = 0.5;
  double Tinf = .5*s0*F(s0)/dF(s0);
  if(s<=s0) return 0;
  else      return Tinf*(1-F(s)/F(s0));
}

//#ifdef P4_TO_P8
//#else
struct level_set_t : CF_2
{
  level_set_t() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return sqrt(x*x + y*y) - 0.5*sqrt(tn);
    case 1: return sqrt(x*x + y*y) - .01;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} level_set;

struct init_temperature_l_t:CF_2
{
  double operator()(double x, double y) const {
//    double s = sqrt(x*x + y*y)/sqrt(tn);
//    double s0 = 0.5;
//    double Tinf = .5*s0*F(s0)/dF(s0);
    switch(test_number)
    {
    case 0: return t_frank_sphere(x,y,tn);
//    case 0: return Tinf*(1-F(s)/F(s0));
    case 1: return -.25;
    case 2: return -.25;
    case 3: return G*(y - (ymin + .1*(ymax-ymin))) + t_interface;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} init_temperature_l;

struct init_temperature_s_t:CF_2
{
  double operator()(double x, double y) const {
    switch(test_number)
    {
    case 0: return t_frank_sphere(x,y,tn);
//    case 0: return 0;
    case 1: return 0;
    case 2: return 0;
    case 3: return (G+L/k_l*V)*(y - (ymin + .1*(ymax-ymin))) + t_interface;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} init_temperature_s;

struct bc_wall_type_t : WallBC2D
{
  BoundaryConditionType operator()( double, double ) const
  {
    switch(test_number)
    {
    case 0: return DIRICHLET;
    case 1: return NEUMANN;
    case 3: return NEUMANN;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} bc_wall_type;

struct bc_wall_value_t : CF_2
{
  double operator()( double x, double y) const
  {
    switch(test_number)
    {
    case 0: return t_frank_sphere(x,y,tn+dt);
    case 1: return 0;
    case 3:
      if     (ABS(y-ymax)<EPS) return G;
      else if(ABS(y-ymin)<EPS) return -G - L/k_s * V;
      else                     return 0;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} bc_wall_value;

struct BCInterfaceValue : CF_2 {
private:
  my_p4est_interpolation_nodes_t interp;
  my_p4est_interpolation_nodes_t interp_phi_x;
  my_p4est_interpolation_nodes_t interp_phi_y;
public:
  BCInterfaceValue( my_p4est_node_neighbors_t *ngbd_,
                    Vec *d_phi, Vec kappa_)
    : interp(ngbd_),
      interp_phi_x(ngbd_),
      interp_phi_y(ngbd_)
  {
    interp.set_input(kappa_, linear);
    interp_phi_x.set_input(d_phi[0], linear);
    interp_phi_y.set_input(d_phi[1], linear);
  }

  double operator() (double x, double y) const
  {
    /* frank sphere: no surface tension */
    if(test_number==0) return 0;

    double theta = atan2( interp_phi_y(x,y) , interp_phi_x(x,y) );
    return t_interface + epsilon_c * (1. - epsilon_anisotropy * cos(N_anisotropy*(theta + theta_0))) * interp(x,y);
    /* T = -eps_c kappa - eps_v V */
  }
};
//#endif

void save_VTK(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_brick_t *brick, Vec phi, Vec T_l, Vec T_s,
              Vec *v, Vec kappa, int compt)
{
  PetscErrorCode ierr;
  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir)
    out_dir = "out_dir";

  std::ostringstream oss, command;
  oss << out_dir << "/vtu";

  command << "mkdir -p " << oss.str();
  system(command.str().c_str());

  struct stat st;
  if(stat(oss.str().data(),&st)!=0 || !S_ISDIR(st.st_mode))
  {
    ierr = PetscPrintf(p4est->mpicomm, "Trying to save files in ... %s\n", oss.str().data());
    throw std::invalid_argument("[ERROR]: the directory specified to export vtu images does not exist.");
  }

  oss << "/stefan_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  const double *phi_p, *t_l_p, *t_s_p, *kappa_p;
  const double *v_p[P4EST_DIM];

  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(T_s, &t_s_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(T_l, &t_l_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(kappa , &kappa_p ); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
  }

  /* compute the temperature in the domain */
  Vec t;
  ierr = VecDuplicate(phi, &t); CHKERRXX(ierr);
  double *t_p;
  ierr = VecGetArray(t, &t_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    t_p[n] = phi_p[n]<0 ? t_s_p[n] : t_l_p[n];

  my_p4est_vtk_write_all(  p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                         #ifdef P4_TO_P8
                           6,
                         #else
                           5,
                         #endif
                           0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "temperature", t_p,
//                           VTK_POINT_DATA, "temp_liquid", t_l_p,
//                           VTK_POINT_DATA, "temp_solid" , t_s_p,
                           VTK_POINT_DATA, "vx", v_p[0],
                           VTK_POINT_DATA, "vy", v_p[1],
                         #ifdef P4_TO_P8
                           VTK_POINT_DATA, "vz", v_p[2],
                         #endif
                           VTK_POINT_DATA, "kappa", kappa_p);

  ierr = VecRestoreArray(t, &t_p); CHKERRXX(ierr);
  ierr = VecDestroy(t); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(T_l, &t_l_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(T_s, &t_s_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa , &kappa_p ); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
  }

  ierr = PetscPrintf(p4est->mpicomm, "Saved in ... %s\n", oss.str().data()); CHKERRXX(ierr);
}



void update_p4est(my_p4est_brick_t *brick, p4est_t *&p4est, p4est_ghost_t *&ghost, p4est_nodes_t *&nodes,
                  my_p4est_hierarchy_t *&hierarchy, my_p4est_node_neighbors_t *&ngbd,
                  Vec &phi, Vec *d_phi, Vec *v, Vec &t_l, Vec &t_s,
                  double dt)
{
  PetscErrorCode ierr;

  p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);
  sl.update_p4est(v, dt, phi);

  /* interpolate the quanities on the new mesh */
  Vec tnp1_l, tnp1_s;
  ierr = VecDuplicate(phi, &tnp1_l); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &tnp1_s); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t interp(ngbd);

  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  interp.set_input(t_l, quadratic);
  interp.interpolate(tnp1_l);

  interp.set_input(t_s, quadratic);
  interp.interpolate(tnp1_s);

  ierr = VecDestroy(t_l); CHKERRXX(ierr);
  t_l = tnp1_l;

  ierr = VecDestroy(t_s); CHKERRXX(ierr);
  t_s = tnp1_s;

  p4est_destroy(p4est); p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  delete hierarchy; hierarchy = new my_p4est_hierarchy_t(p4est, ghost, brick);
  delete ngbd; ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
  ngbd->init_neighbors();

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(d_phi[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &d_phi[dir]); CHKERRXX(ierr);

    ierr = VecDestroy(v[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(d_phi[dir], &v[dir]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi, 20);
  ls.perturb_level_set_function(phi, EPS);
}


void compute_normal_and_curvature(my_p4est_node_neighbors_t *ngbd, Vec phi, Vec *d_phi, Vec kappa)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_compute_curvature, phi, kappa, 0, 0); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;

  /* compute normal */
  double *phi_p, *d_phi_p[P4EST_DIM];
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArray(d_phi[dir], &d_phi_p[dir]); CHKERRXX(ierr);
  }
  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    ngbd->get_neighbors(n, qnnn);
    d_phi_p[0][n] = qnnn.dx_central(phi_p);
    d_phi_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    d_phi_p[2][n] = qnnn.dz_central(phi_p);
#endif

#ifdef P4_TO_P8
    double norm = sqrt(SQR(d_phi_p[0][n]) + SQR(d_phi_p[1][n]) + SQR(d_phi_p[2][n]));
#else
    double norm = sqrt(SQR(d_phi_p[0][n]) + SQR(d_phi_p[1][n]));
#endif

    d_phi_p[0][n] = norm>EPS ? d_phi_p[0][n]/norm : 0;
    d_phi_p[1][n] = norm>EPS ? d_phi_p[1][n]/norm : 0;
#ifdef P4_TO_P8
    d_phi_p[2][n] = norm>EPS ? d_phi_p[2][n]/norm : 0;
#endif
  }
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(d_phi[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    ngbd->get_neighbors(n, qnnn);
    d_phi_p[0][n] = qnnn.dx_central(phi_p);
    d_phi_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    d_phi_p[2][n] = qnnn.dz_central(phi_p);
#endif

#ifdef P4_TO_P8
    double norm = sqrt(SQR(d_phi_p[0][n]) + SQR(d_phi_p[1][n]) + SQR(d_phi_p[2][n]));
#else
    double norm = sqrt(SQR(d_phi_p[0][n]) + SQR(d_phi_p[1][n]));
#endif

    d_phi_p[0][n] = norm>EPS ? d_phi_p[0][n]/norm : 0;
    d_phi_p[1][n] = norm>EPS ? d_phi_p[1][n]/norm : 0;
#ifdef P4_TO_P8
    d_phi_p[2][n] = norm>EPS ? d_phi_p[2][n]/norm : 0;
#endif
  }
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(d_phi[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* compute curvature */
  Vec kappa_tmp;
  ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);
  double *kappa_p;
  ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    ngbd->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    kappa_p[n] = qnnn.dx_central(d_phi_p[0]) + qnnn.dy_central(d_phi_p[1]) + qnnn.dz_central(d_phi_p[2]);
#else
    kappa_p[n] = qnnn.dx_central(d_phi_p[0]) + qnnn.dy_central(d_phi_p[1]);
#endif
  }

  ierr = VecGhostUpdateBegin(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    ngbd->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    kappa_p[n] = qnnn.dx_central(d_phi_p[0]) + qnnn.dy_central(d_phi_p[1]) + qnnn.dz_central(d_phi_p[2]);
#else
    kappa_p[n] = qnnn.dx_central(d_phi_p[0]) + qnnn.dy_central(d_phi_p[1]);
#endif
  }

  ierr = VecGhostUpdateEnd(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi  , &phi_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(d_phi[dir], &d_phi_p[dir]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);

  ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_compute_curvature, phi, kappa, 0, 0); CHKERRXX(ierr);
}


void solve_temperature(my_p4est_node_neighbors_t *ngbd, Vec phi_s, Vec phi_l, Vec *d_phi, Vec kappa, double dt, Vec t_l, Vec t_s)
{
#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  BCInterfaceValue bc_interface_value(ngbd, d_phi, kappa);

  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_value);
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  /* solve for the liquid phase */
  my_p4est_poisson_nodes_t solver_l(ngbd);
  solver_l.set_phi(phi_l);
  solver_l.set_mu(k_l*dt);
  solver_l.set_diagonal(1);
  solver_l.set_bc(bc);
  solver_l.set_rhs(t_l);

  solver_l.solve(t_l);

  /* solve for the solid phase */
  my_p4est_poisson_nodes_t solver_s(ngbd);
  solver_s.set_phi(phi_s);
  solver_s.set_mu(k_s*dt);
  solver_s.set_diagonal(1);
  solver_s.set_bc(bc);
  solver_s.set_rhs(t_s);

  solver_s.solve(t_s);
}


void extend_temperatures_over_interface(my_p4est_node_neighbors_t *ngbd, Vec phi_s, Vec phi_l, Vec t_l, Vec t_s)
{
  my_p4est_level_set_t ls(ngbd);
  ls.extend_Over_Interface_TVD(phi_l, t_l);
  ls.extend_Over_Interface_TVD(phi_s, t_s);
}


void compute_velocity(my_p4est_node_neighbors_t *ngbd, Vec phi, Vec t_l, Vec t_s, Vec *v)
{
  PetscErrorCode ierr;

  double *t_l_p, *t_s_p;
  ierr = VecGetArray(t_l, &t_l_p ); CHKERRXX(ierr);
  ierr = VecGetArray(t_s, &t_s_p ); CHKERRXX(ierr);

  Vec jump[P4EST_DIM];
  double *jump_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(v[dir], &jump[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(jump[dir], &jump_p[dir]); CHKERRXX(ierr);
  }

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    ngbd->get_neighbors(n, qnnn);
    jump_p[0][n] = (k_s*qnnn.dx_central(t_s_p) - k_l*qnnn.dx_central(t_l_p)) / L;
    jump_p[1][n] = (k_s*qnnn.dy_central(t_s_p) - k_l*qnnn.dy_central(t_l_p)) / L;
#ifdef P4_TO_P8
    jump_p[2][n] = (k_s*qnnn.dz_central(t_s_p) - k_l*qnnn.dz_central(t_l_p)) / L;
#endif
  }
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(jump[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    ngbd->get_neighbors(n, qnnn);
    jump_p[0][n] = (k_s*qnnn.dx_central(t_s_p) - k_l*qnnn.dx_central(t_l_p)) / L;
    jump_p[1][n] = (k_s*qnnn.dy_central(t_s_p) - k_l*qnnn.dy_central(t_l_p)) / L;
#ifdef P4_TO_P8
    jump_p[2][n] = (k_s*qnnn.dz_central(t_s_p) - k_l*qnnn.dz_central(t_l_p)) / L;
#endif
  }
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(jump[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(t_l, &t_l_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_s, &t_s_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(jump[dir], &jump_p[dir] ); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(phi, jump[dir], v[dir], 20);
    ierr = VecDestroy(jump[dir]); CHKERRXX(ierr);
  }
}


void check_error_frank_sphere(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec t_l)
{
  PetscErrorCode ierr;

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dxyzmin = MAX(xmax-xmin, ymax-ymin) / (1<<data->max_lvl);

  const double *t_l_p, *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(t_l, &t_l_p); CHKERRXX(ierr);

  double err[] = {0, 0};
  double r = .5*sqrt(tn);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);

    if(fabs(phi_p[n])<dxyzmin)
    {
      double phi_exact = sqrt(x*x+y*y) - r;
      err[0] = max(err[0], fabs(phi_p[n]-phi_exact));
    }

    if(phi_p[n]>0)
      err[1] = max(err[1], fabs(t_frank_sphere(x,y,tn)-t_l_p[n]));
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(t_l, &t_l_p); CHKERRXX(ierr);

  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = PetscPrintf(p4est->mpicomm, "Error on phi: %g\nError on t_l: %g\n", err[0], err[1]); CHKERRXX(ierr);
}


int main (int argc, char* argv[])
{
  mpi_enviroment_t mpi;
  mpi.init(argc, argv);

  PetscErrorCode ierr;

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nx", "number of trees in the x direction");
  cmd.add_option("ny", "number of trees in the y direction");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z direction");
#endif
  cmd.add_option("tf", "final time");
  cmd.add_option("save_vtk", "1 to export vtu images, 0 otherwise");
  cmd.add_option("save_every_n", "export images every n iterations");
  cmd.add_option("max_iter", "maximum number of iterations");
  cmd.add_option("n_times_dt", "CFL restriction, dt = n_times_dt * dx/umax");
  cmd.add_option("test", "the test to run. Available options are\
                 \t 0 - frank sphere\n\
                 \t 1 - a single seed\n\
                 \t 2 - 20 seeds\n\
                 \t 3 - a plane\n");
  cmd.parse(argc, argv);

  int lmin = cmd.get("lmin", 0);
  int lmax = cmd.get("lmax", 6);
  test_number = cmd.get("test", test_number);
  bool save_vtk = cmd.get("save_vtk", 1);
  int save_every_n = cmd.get("save_every_n", 1);
  int max_iter = cmd.get("max_iter", INT_MAX);
  double n_times_dt = cmd.get("n_times_dt", 1);

  splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.2);

  int nx, ny;
#ifdef P4_TO_P8
  int nz;
#endif
  double tf;

  switch(test_number)
  {
  case 0: nx=2; ny=2; xmin=-1; xmax= 1; ymin=-1; ymax= 1; k_s=1; k_l=1; tn=1; tf=2.89; break;
  case 1: nx=2; ny=2; xmin=-1; xmax= 1; ymin=-1; ymax= 1; tn=0; tf=DBL_MAX; break;
  default: throw std::invalid_argument("[ERROR]: choose a valid test.");
  }

  nx = cmd.get("nx", nx);
  ny = cmd.get("ny", ny);
#ifdef P4_TO_P8
  nz = cmd.get("nz", nz);
#endif
  tf = cmd.get("tf", tf);

  // Create the connectivity object
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  int n_xyz [] = {nx, ny, nz};
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
#else
  int n_xyz [] = {nx, ny};
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
#endif
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick);

  double dxyz_min[P4EST_DIM];
  dxyz_min[0] = (xmax-xmin)/nx/(1<<lmax);
  dxyz_min[1] = (ymax-ymin)/ny/(1<<lmax);
#ifdef P4_TO_P8
  dxyz_min[2] = (zmax-zmin)/nz/(1<<lmax);
#endif

  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);
  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
  ngbd->init_neighbors();

  /* Initialize the level-set function */
  Vec phi_s;
  Vec phi_l;
  Vec kappa;
  Vec d_phi[P4EST_DIM];
  Vec t_l, t_s;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_s); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_s, &phi_l); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_s, &t_l); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_s, &t_s); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_s, &kappa); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, level_set, phi_s);
  sample_cf_on_nodes(p4est, nodes, init_temperature_l, t_l);
  sample_cf_on_nodes(p4est, nodes, init_temperature_s, t_s);

  double *phi_l_p;
  const double *phi_s_p;
  ierr = VecGetArray(phi_l, &phi_l_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_s, &phi_s_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count ; ++n)
    phi_l_p[n] = -phi_s_p[n];
  ierr = VecRestoreArray(phi_l, &phi_l_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_s, &phi_s_p); CHKERRXX(ierr);

  /* Initialize the velocity field */
  Vec v[P4EST_DIM];
  double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &d_phi[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(d_phi[dir], &v[dir]); CHKERRXX(ierr);
  }

  compute_normal_and_curvature(ngbd, phi_s, d_phi, kappa);
  extend_temperatures_over_interface(ngbd, phi_s, phi_l, t_l, t_s);
  compute_velocity(ngbd, phi_s, t_l, t_s, v);

  /* save the initial state */
  if(save_vtk)
    save_VTK(p4est, nodes, &brick, phi_s, t_l, t_s, v, kappa, 0);

  // loop over time
  int iter = 1;
  dt = 0;

  while(tn<tf && iter<max_iter)
  {
//    ierr = PetscPrintf(p4est->mpicomm, "Iteration #%d, tn=%g\n", iter, tn);

    /* compute the time step dt */
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGetArray(v[dir], &v_p[dir]); CHKERRXX(ierr);
    }

    double max_norm_u = 0;
    ierr = VecGetArrayRead(phi_s, &phi_s_p); CHKERRXX(ierr);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
#ifdef P4_TO_P8
      if(fabs(phi_s_p[n])<3*MIN(dxyz_min[0],dxyz_min[1],dxyz_min[2]))
        max_norm_u = max(max_norm_u, sqrt( SQR(v_p[0][n]) + SQR(v_p[1][n]) + SQR(v_p[2][n]) ) );
#else
      if(fabs(phi_s_p[n])<3*MIN(dxyz_min[0],dxyz_min[1]))
        max_norm_u = max(max_norm_u, sqrt( SQR(v_p[0][n]) + SQR(v_p[1][n]) ) );
#endif
    }
    ierr = VecRestoreArrayRead(phi_s, &phi_s_p); CHKERRXX(ierr);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecRestoreArray(v[dir], &v_p[dir]); CHKERRXX(ierr);
    }

    MPI_Allreduce(MPI_IN_PLACE, &max_norm_u, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);

#ifdef P4_TO_P8
    dt = min(1.,1/max_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
    dt = min(1.,1/max_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1]);
#endif

    if(tn+dt>tf) dt = tf-tn;

    /* contruct the mesh at time tn+dt */
    update_p4est(&brick, p4est, ghost, nodes, hierarchy, ngbd, phi_s, d_phi, v, t_l, t_s, dt);

    ierr = VecDestroy(phi_l);
    ierr = VecDuplicate(phi_s, &phi_l); CHKERRXX(ierr);
    ierr = VecGetArray(phi_l, &phi_l_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi_s, &phi_s_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count ; ++n)
      phi_l_p[n] = -phi_s_p[n];
    ierr = VecRestoreArray(phi_l, &phi_l_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi_s, &phi_s_p); CHKERRXX(ierr);

    /* compute the curvature for boundary conditions */
    ierr = VecDestroy(kappa); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_s, &kappa); CHKERRXX(ierr);
    compute_normal_and_curvature(ngbd, phi_s, d_phi, kappa);

    /* solve for the temperatures */
    solve_temperature(ngbd, phi_s, phi_l, d_phi, kappa, dt, t_l, t_s);

    /* extend the temperature over the interface */
    extend_temperatures_over_interface(ngbd, phi_s, phi_l, t_l, t_s);

    /* compute the velocity of the interface */
    compute_velocity(ngbd, phi_s, t_l, t_s, v);

    tn += dt;

    if(save_vtk && iter % save_every_n == 0)
      save_VTK(p4est, nodes, &brick, phi_s, t_l, t_s, v, kappa, iter/save_every_n);

    iter++;
//    break;
  }

  ierr = PetscPrintf(mpi.comm(), "Final time of the simulation: tf=%g\n", tf); CHKERRXX(ierr);

  if(test_number==0)
    check_error_frank_sphere(p4est, nodes, phi_s, t_l);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(d_phi[dir]); CHKERRXX(ierr);
    ierr = VecDestroy(v[dir]);     CHKERRXX(ierr);
  }
  ierr = VecDestroy(kappa); CHKERRXX(ierr);
  ierr = VecDestroy(phi_s);   CHKERRXX(ierr);
  ierr = VecDestroy(phi_l);   CHKERRXX(ierr);
  ierr = VecDestroy(t_s);  CHKERRXX(ierr);
  ierr = VecDestroy(t_l);  CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
