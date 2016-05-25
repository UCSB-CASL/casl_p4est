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
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

int lmin = 4;
int lmax = 6;
int nb_splits = 4;

int nx = 2;
int ny = 2;
int nz = 2;

double xmin = -1;
double xmax = 3;
double ymin = -1;
double ymax = 4;
double zmin = 0;
double zmax = 3;

bool save_vtk = false;
bool save_voro = false;
bool save_stats = false;
bool check_partition = false;

/*
 * 0 - circle
 */
int level_set_type = 0;

/*
 *  ********* 2D *********
 * 0 - u_m=1+log(r/r0), u_p=1, mu_m=mu_p=1, diag_add=0
 * 1 - u_m=u_p=cos(x)*sin(y), mu_m=mu_p, BC dirichlet
 * 2 - u_m=u_p=sin(x)*sin(y), mu_m=mu_p, BC neumann
 * 3 - u_m=exp(x), u_p=cos(x)*sin(y), mu_m=y*y*ln(x+2)+4, mu_p=exp(-y)   article example 4.4
 *
 *  ********* 3D *********
 * 0 - u_m=exp(z), u_p=cos(x)*sin(y), mu_m=mu_p
 * 1 - u_m=exp(z), u_p=cos(x)*sin(y), mu_m=y*y*ln(x+2)+4, mu_p=exp(-z)   article example 4.6
 * 2 - u_m=u_p=cos(x)*sin(y)*exp(z), mu_m=mu_p, BC dirichlet
 * 3 - u_m=u_p=cos(x)*sin(y)*exp(z), mu_m=mu_p=exp(x)*ln(y+z+2), BC dirichlet
 * 4 - u_m=y*z*sin(x), u_p=x*y*y+z*z*z, mu_m=y*y+5, mu_p=exp(x+z)        article example 4.7
 * 5 - u_m=u_p=cos(x)*sin(y)*exp(z), mu_m=y*y+5, mu_p=exp(x+z)
 */
int test_number = 1;

double diag_add = 0;

#ifdef P4_TO_P8
double r0 = (double) MIN(xmax-xmin,ymax-ymin,zmax-zmin) / 4;
#else
double r0 = (double) MIN(xmax-xmin,ymax-ymin) / 4;
#endif



#ifdef P4_TO_P8

class LEVEL_SET: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(level_set_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - (xmin+xmax)/2) + SQR(y - (ymin+ymax)/2) + SQR(z - (zmin+zmax)/2));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class ONE: public CF_3
{
public:
  double operator()(double , double, double ) const
  {
    return -1;
  }
} one;

double phi_x(double x, double y, double z)
{
  switch(level_set_type)
  {
  case 0:
    return -(x-(xmin+xmax)/2)/sqrt(SQR(x-(xmin+xmax)/2)+SQR(y-(ymin+ymax)/2)+SQR(z-(zmin+zmax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

double phi_y(double x, double y, double z)
{
  switch(level_set_type)
  {
  case 0:
    return -(y-(ymin+ymax)/2)/sqrt(SQR(x-(xmin+xmax)/2)+SQR(y-(ymin+ymax)/2)+SQR(z-(zmin+zmax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

double phi_z(double x, double y, double z)
{
  switch(level_set_type)
  {
  case 0:
    return -(z-(zmin+zmax)/2)/sqrt(SQR(x-(xmin+xmax)/2)+SQR(y-(ymin+ymax)/2)+SQR(z-(zmin+zmax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

class MU_M: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
      return SQR(y)*log(x+2)+4;
    case 2:
      return 1.45;
    case 3:
      return exp(x)*log(y+z+2);
    case 4:
      return y*y+5;
    case 5:
      return y*y+5;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_m;

class MU_P: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
      return exp(-z);
    case 2:
      return 1.45;
    case 3:
      return exp(x)*log(y+z+2);
    case 4:
      return exp(x+z);
    case 5:
      return exp(x+z);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_p;

double u_m(double x, double y, double z)
{
  switch(test_number)
  {
  case 0:
    return exp(z);
  case 1:
    return exp(z);
  case 2:
    return cos(x)*sin(y)*exp(z);
  case 3:
    return cos(x)*sin(y)*exp(z);
  case 4:
    return y*z*sin(x);
  case 5:
    return cos(x)*sin(y)*exp(z);
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double u_p(double x, double y, double z)
{
  switch(test_number)
  {
  case 0:
    return cos(x)*sin(y);
  case 1:
    return cos(x)*sin(y);
  case 2:
    return cos(x)*sin(y)*exp(z);
  case 3:
    return cos(x)*sin(y)*exp(z);
  case 4:
    return x*y*y + z*z*z;
  case 5:
    return cos(x)*sin(y)*exp(z);
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

class U_JUMP: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return u_p(x,y,z) - u_m(x,y,z);
  }
} u_jump;

double u_exact(double x, double y, double z)
{
  if(level_set(x,y,z)>0) return u_p(x,y,z);
  else                   return u_m(x,y,z);
}

double grad_u_m(double x, double y, double z)
{
  double ux, uy, uz;
  switch(test_number)
  {
  case 0:
    ux = 0;
    uy = 0;
    uz = exp(z);
    break;
  case 1:
    ux = 0;
    uy = 0;
    uz = exp(z);
    break;
  case 2:
    ux = -sin(x)*sin(y)*exp(z);
    uy =  cos(x)*cos(y)*exp(z);
    uz =  cos(x)*sin(y)*exp(z);
    break;
  case 3:
    ux = -sin(x)*sin(y)*exp(z);
    uy =  cos(x)*cos(y)*exp(z);
    uz =  cos(x)*sin(y)*exp(z);
    break;
  case 4:
    ux = y*z*cos(x);
    uy = z*sin(x);
    uz = y*sin(x);
    break;
  case 5:
    ux = -sin(x)*sin(y)*exp(z);
    uy =  cos(x)*cos(y)*exp(z);
    uz =  cos(x)*sin(y)*exp(z);
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y,z);
  double phiy = phi_y(x,y,z);
  double phiz = phi_z(x,y,z);

  return ux*phix + uy*phiy + uz*phiz;
}

double grad_u_p(double x, double y, double z)
{
  double ux, uy, uz;
  switch(test_number)
  {
  case 0:
    ux = -sin(x)*sin(y);
    uy =  cos(x)*cos(y);
    uz = 0;
    break;
  case 1:
    ux = -sin(x)*sin(y);
    uy =  cos(x)*cos(y);
    uz = 0;
    break;
  case 2:
    ux = -sin(x)*sin(y)*exp(z);
    uy =  cos(x)*cos(y)*exp(z);
    uz =  cos(x)*sin(y)*exp(z);
    break;
  case 3:
    ux = -sin(x)*sin(y)*exp(z);
    uy =  cos(x)*cos(y)*exp(z);
    uz =  cos(x)*sin(y)*exp(z);
    break;
  case 4:
    ux = y*y;
    uy = 2*x*y;
    uz = 3*z*z;
    break;
  case 5:
    ux = -sin(x)*sin(y)*exp(z);
    uy =  cos(x)*cos(y)*exp(z);
    uz =  cos(x)*sin(y)*exp(z);
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y,z);
  double phiy = phi_y(x,y,z);
  double phiz = phi_z(x,y,z);

  return ux*phix + uy*phiy + uz*phiz;
}


class MU_GRAD_U_JUMP: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return mu_p(x,y,z)*grad_u_p(x,y,z) - mu_m(x,y,z)*grad_u_m(x,y,z);
  }
} mu_grad_u_jump;

class BC_WALL_TYPE : public WallBC3D
{
public:
  BoundaryConditionType operator() (double , double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type;


class BC_WALL_VALUE : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return level_set(x,y,z)<0 ? u_m(x,y,z) : u_p(x,y,z);
  }
} bc_wall_value;

#else

class LEVEL_SET: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(level_set_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - (xmin+xmax)/2) + SQR(y - (ymin+ymax)/2));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class ONE: public CF_2
{
public:
  double operator()(double , double ) const
  {
    return -1;
  }
} one;

double phi_x(double x, double y)
{
  switch(level_set_type)
  {
  case 0:
    return -(x-(xmin+xmax)/2)/sqrt(SQR(x-(xmin+xmax)/2)+SQR(y-(ymin+ymax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

double phi_y(double x, double y)
{
  switch(level_set_type)
  {
  case 0:
    return -(y-(ymin+ymax)/2)/sqrt(SQR(x-(xmin+xmax)/2)+SQR(y-(ymin+ymax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

class MU_M: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
    case 2:
      return 4.3;
    case 3:
      return SQR(y)*log(x+2)+4;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_m;

class MU_P: public CF_2
{
public:
  double operator()(double , double y) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
    case 2:
      return 4.3;
    case 3:
      return exp(-y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_p;

struct U_M : CF_2
{
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-(xmin+xmax)/2) + SQR(y-(ymin+ymax)/2));
    switch(test_number)
    {
    case 0:
      return 1+log(r/.5);
    case 1:
      return cos(x)*sin(y);
    case 2:
      return sin(x)*sin(y);
    case 3:
      return exp(x);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} u_m;

struct U_P : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
      return cos(x)*sin(y);
    case 2:
      return sin(x)*sin(y);
    case 3:
      return cos(x)*sin(y);
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} u_p;

class U_JUMP: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return u_p(x,y) - u_m(x,y);
  }
} u_jump;

double u_exact(double x, double y)
{
  if(level_set(x,y)>0) return u_p(x,y);
  else                 return u_m(x,y);
}

double grad_u_m(double x, double y)
{
  double ux, uy;
  switch(test_number)
  {
  case 0:
    ux = (x-(xmin+xmax)/2)/(SQR(x-(xmin+xmax)/2)+SQR(y-(ymin+ymax)/2));
    uy = (y-(ymin+ymax)/2)/(SQR(x-(xmin+xmax)/2)+SQR(y-(ymin+ymax)/2));
    break;
  case 1:
    ux = -sin(x)*sin(y);
    uy =  cos(x)*cos(y);
    break;
  case 2:
    ux =  cos(x)*sin(y);
    uy =  sin(x)*cos(y);
    break;
  case 3:
    ux = exp(x);
    uy = 0;
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y);
  double phiy = phi_y(x,y);

  return ux*phix + uy*phiy;
}

double grad_u_p(double x, double y)
{
  double ux, uy;
  switch(test_number)
  {
  case 0:
    ux = 0;
    uy = 0;
    break;
  case 1:
    ux = -sin(x)*sin(y);
    uy =  cos(x)*cos(y);
    break;
  case 2:
    ux =  cos(x)*sin(y);
    uy =  sin(x)*cos(y);
    break;
  case 3:
    ux = -sin(x)*sin(y);
    uy =  cos(x)*cos(y);
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y);
  double phiy = phi_y(x,y);

  return ux*phix + uy*phiy;
}

class MU_GRAD_U_JUMP: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return mu_p(x,y)*grad_u_p(x,y) - mu_m(x,y)*grad_u_m(x,y);
  }
} mu_grad_u_jump;

class BC_WALL_TYPE : public WallBC2D
{
public:
  BoundaryConditionType operator() (double , double ) const
  {
    if(test_number==2) return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type;

class BC_WALL_VALUE : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if(bc_wall_type(x,y)==DIRICHLET)
      return level_set(x,y)<0 ? u_m(x,y) : u_p(x,y);
    else
    {
      if(ABS(x-   0)<EPS) return -cos(x)*sin(y);
      if(ABS(x-xmax)<EPS) return  cos(x)*sin(y);
      if(ABS(y-   0)<EPS) return -sin(x)*cos(y);
      return  sin(x)*cos(y);
    }
  }
} bc_wall_value;

#endif



void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err,
              int compt)
{
  PetscErrorCode ierr;
  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save visuals\n"); CHKERRXX(ierr);
    return;
  }

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/jump_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p, *sol_p, *err_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);

  Vec mu;
  ierr = VecDuplicate(phi, &mu); CHKERRXX(ierr);
  double *mu_p_;
  ierr = VecGetArray(mu, &mu_p_); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
    mu_p_[n] = phi_p[n]<0 ? mu_m(x,y,z) : mu_p(x,y,z);
#else
    mu_p_[n] = phi_p[n]<0 ? mu_m(x,y) : mu_p(x,y);
#endif
  }

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
                         VTK_POINT_DATA, "mu", mu_p_,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "err", err_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(mu, &mu_p_); CHKERRXX(ierr);
  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(mu); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



void shift_Neumann_Solution(p4est_t *p4est, p4est_nodes_t *nodes, Vec sol)
{
  PetscErrorCode ierr;

  ierr = PetscPrintf(p4est->mpicomm, "Shifting all neumann solution\n");

  double ex_int = 0;
  switch(test_number)
  {
  case 2:
    ex_int = (1-cos((double) nx))*(1-cos((double) ny));
    break;
  default:
    ex_int = 0;
  }

  Vec ones;
  ierr = VecDuplicate(sol, &ones); CHKERRXX(ierr);
  ierr = VecSet(ones, -1); CHKERRXX(ierr);
  double sol_int = integrate_over_negative_domain(p4est, nodes, ones, sol);

  double *sol_p;
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    sol_p[n] += (ex_int - sol_int)/((xmax-xmin)*(ymax-ymin));

  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
}



void solve_Poisson_Jump( p4est_t *p4est, p4est_nodes_t *nodes,
                         my_p4est_node_neighbors_t *ngbd_n, my_p4est_cell_neighbors_t *ngbd_c,
                         Vec phi, Vec sol)
{
  PetscErrorCode ierr;

  Vec rhs_m, rhs_p;
  Vec mu_m_, mu_p_;
  Vec u_jump_;
  Vec mu_grad_u_jump_;

  ierr = VecDuplicate(phi, &rhs_m); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs_p); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &mu_m_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &mu_p_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &u_jump_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &mu_grad_u_jump_); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, mu_m, mu_m_);
  sample_cf_on_nodes(p4est, nodes, mu_p, mu_p_);
  sample_cf_on_nodes(p4est, nodes, u_jump, u_jump_);
  sample_cf_on_nodes(p4est, nodes, mu_grad_u_jump, mu_grad_u_jump_);

  double *rhs_m_p, *rhs_p_p;
  ierr = VecGetArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
    switch(test_number)
    {
    case 0:
      rhs_m_p[n] = -exp(z);
      rhs_p_p[n] = 2*cos(x)*sin(y);
      break;
    case 1:
      rhs_m_p[n] = -exp(z)*(y*y*log(x+2)+4);
      rhs_p_p[n] = exp(-z)*2*cos(x)*sin(y);
      break;
    case 2:
      rhs_m_p[n] = (mu_m(x,y,z) + diag_add)*cos(x)*sin(y)*exp(z);
      rhs_p_p[n] = (mu_p(x,y,z) + diag_add)*cos(x)*sin(y)*exp(z);
      break;
    case 3:
      rhs_m_p[n] = exp(x+z)*(log(y+z+2)*sin(y)*(cos(x)+sin(x)) - cos(x)*(cos(y)+sin(y))/(y+z+2));
      rhs_p_p[n] = exp(x+z)*(log(y+z+2)*sin(y)*(cos(x)+sin(x)) - cos(x)*(cos(y)+sin(y))/(y+z+2));
      break;
    case 4:
      rhs_m_p[n] = y*z*sin(x)*(y*y+3);
      rhs_p_p[n] = -exp(x+z)*(y*y+2*x+3*z*(z+2));
      break;
    case 5:
      rhs_m_p[n] = exp(z)*cos(x)*(sin(y)*(y*y+5) - 2*y*cos(y));
      rhs_p_p[n] = exp(x+2*z)*sin(x)*sin(y);
      break;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
#else
    switch(test_number)
    {
    case 0:
      rhs_m_p[n] = 0;
      rhs_p_p[n] = 0;
      break;
    case 1:
      rhs_m_p[n] = (2*mu_m(x,y) + diag_add) * cos(x)*sin(y);
      rhs_p_p[n] = (2*mu_p(x,y) + diag_add) * cos(x)*sin(y);
      break;
    case 2:
      rhs_m_p[n] = (2*mu_m(x,y) + diag_add) * sin(x)*sin(y);
      rhs_p_p[n] = (2*mu_p(x,y) + diag_add) * sin(x)*sin(y);
      break;
    case 3:
      rhs_m_p[n] = -exp(x)*(y*y*log(x+2)+y*y/(x+2)+4);
      rhs_p_p[n] = exp(-y)*cos(x)*(cos(y)+2*sin(y));
      break;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
#endif
  }

  ierr = VecRestoreArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  my_p4est_poisson_jump_nodes_voronoi_t solver(ngbd_n, ngbd_c);
  solver.set_phi(phi);
  solver.set_bc(bc);
  solver.set_mu(mu_m_, mu_p_);
  solver.set_u_jump(u_jump_);
  solver.set_mu_grad_u_jump(mu_grad_u_jump_);
  solver.set_rhs(rhs_m, rhs_p);

  solver.solve(sol);
  //  solver.compute_voronoi_points();
  if(check_partition)
    solver.check_voronoi_partition();
  //  if(p4est->mpirank==0)
  //  solver.compute_voronoi_mesh();
  //  solver.setup_negative_laplace_matrix();
  //  solver.setup_negative_laplace_rhsvec();
//  sample_cf_on_nodes(p4est, nodes, u_m, sol);

  char out_path[1000];
  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save stats\n"); CHKERRXX(ierr);
  }
  else
  {
    if(save_stats)
    {
      sprintf(out_path, "%s/stats.dat", out_dir);
      solver.write_stats(out_path);
    }

    if(save_voro)
    {
      snprintf(out_path,1000, "%s/vtu/voronoi", out_dir);
      solver.print_voronoi_VTK(out_path);
    }
  }

  if(solver.get_matrix_has_nullspace())
    shift_Neumann_Solution(p4est, nodes, sol);

  ierr = VecDestroy(rhs_m); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_p); CHKERRXX(ierr);
  ierr = VecDestroy(mu_m_); CHKERRXX(ierr);
  ierr = VecDestroy(mu_p_); CHKERRXX(ierr);
  ierr = VecDestroy(u_jump_); CHKERRXX(ierr);
  ierr = VecDestroy(mu_grad_u_jump_); CHKERRXX(ierr);
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
  cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
  cmd.add_option("save_voro", "1 to save voronoi partition, 0 otherwise");
  cmd.add_option("save_stats", "1 to save statistics about the voronoi partition, 0 otherwise");
  cmd.add_option("check_partition", "1 to check if the voronoi partition is symmetric, 0 otherwise");
#ifdef P4_TO_P8
  cmd.add_option("test", "choose a test.\n\
                 0 - u_m=1+log(r/r0), u_p=1, mu=1\n\
                 1 - u_m=exp(z), u_p=cos(x)*sin(y), mu_m=y*y*ln(x+2)+4, mu_p=exp(-z)   article example 4.6");
#else
  cmd.add_option("test", "choose a test.\n\
                 0 - u_m=1+log(r/r0), u_p=1, mu=1\n\
                 1 - u_m=u_p=cos(x)*sin(y), mu_m=mu_p, BC dirichlet\n\
                 2 - u_m=u_p=sin(x)*sin(y), mu_m=mu_p, BC neumann\n\
                 3 - u_m=exp(x), u_p=cos(x)*sin(y), mu_m=y*y*ln(x+2)+4, mu_p=exp(-y)   article example 4.4");
#endif
  cmd.parse(argc, argv);

  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);
  test_number = cmd.get("test", test_number);
  save_vtk = cmd.get("save_vtk", save_vtk);
  save_voro= cmd.get("save_voro", save_voro);
  save_stats = cmd.get("save_stats", save_stats);
  check_partition = cmd.get("check_partition", check_partition);

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

  const int n_xyz []       = {nx, ny, nz};
  const double xyz_min []  = {xmin, ymin, zmin};
  const double xyz_max []  = {xmax, ymax, zmax};
  const int periodic []    = {0, 0, 0};

  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  double err_n   = 0;
  double err_nm1 = 0;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    //    srand(1);
    //    splitting_criteria_random_t data(4, 6, 1000, 10000);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set, 1.2);
    p4est->user_pointer = (void*)(&data);

    for(int i=0; i<lmax+iter; ++i)
    {
      //    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
      my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    if(p4est->mpirank==0)
    {
      p4est_gloidx_t nb_nodes = 0;
      for(int r=0; r<p4est->mpisize; ++r)
        nb_nodes += nodes->global_owned_indeps[r];
      ierr = PetscPrintf(p4est->mpicomm, "number of nodes : %d\n", nb_nodes); CHKERRXX(ierr);
    }

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);

    my_p4est_cell_neighbors_t ngbd_c(&hierarchy);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set, phi);
    // bousouf
//    sample_cf_on_nodes(p4est, nodes, one, phi);

    my_p4est_level_set_t ls(&ngbd_n);
    ls.perturb_level_set_function(phi, EPS);

    Vec sol;
    ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

    solve_Poisson_Jump(p4est, nodes, &ngbd_n, &ngbd_c, phi, sol);

    /* compute the error on the tree*/
    Vec err;
    ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
    double *err_p, *sol_p;
    ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
    err_nm1 = err_n;
    err_n = 0;
    double x_err=-1, y_err=-1, z_err=-1;
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
      err_p[n] = fabs(u_exact(x,y,z) - sol_p[n]);
#else
      err_p[n] = fabs(u_exact(x,y) - sol_p[n]);
#endif
      if(err_p[n]>err_n)
      {
        x_err = x;
        y_err = y;
#ifdef P4_TO_P8
        z_err = z;
#endif
      }
      err_n = max(err_n, err_p[n]);
    }

    MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
    PetscPrintf(p4est->mpicomm, "Iter %d : %g, \t order : %g\n", iter, err_n, log(err_nm1/err_n)/log(2));
    PetscPrintf(p4est->mpicomm, "error at %g, %g, %g, qh = %g, dist_interface = %g\n", x_err, y_err, z_err, (double) 2/pow(2,lmax+iter+1), fabs(sqrt(SQR(x_err-1) + SQR(y_err-1) + SQR(z_err-1))-r0));

    ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);

    if(save_vtk)
      save_VTK(p4est, ghost, nodes, &brick, phi, sol, err, iter);

    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);
    ierr = VecDestroy(err); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
