#include "my_p4est_epitaxy.h"

#include <random>
#include <stack>

#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_interpolation_nodes.h>

my_p4est_epitaxy_t::my_p4est_epitaxy_t(my_p4est_node_neighbors_t *ngbd)
  : brick(ngbd->myb), p4est(ngbd->p4est), connectivity(p4est->connectivity),
    ghost(ngbd->ghost), nodes(ngbd->nodes), hierarchy(ngbd->hierarchy),
    ngbd(ngbd)
{
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_g); CHKERRXX(ierr);
  rho.resize(1);
  ierr = VecDuplicate(rho_g, &rho[0]); CHKERRXX(ierr);

  rho_np1.resize(1);
  ierr = VecDuplicate(rho_g, &rho_np1[0]); CHKERRXX(ierr);

  Vec loc;
  ierr = VecGhostGetLocalForm(rho_g, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho_g, &loc); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(rho[0], &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho[0], &loc); CHKERRXX(ierr);

  Nuc = 0;
  new_island = 0;
  alpha = 1.05;
  dx_dy_dz(p4est, dxyz);
//  dt_n = MIN(dxyz[0],dxyz[1]);
  dt_n = 1e-3;

  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<3; i++)
    xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
  for (short i=0; i<3; i++)
    xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];

  L = xyz_max[0]-xyz_min[0];

  island_nucleation_scaling = 1;
//  island_nucleation_scaling = L*L;
}


my_p4est_epitaxy_t::~my_p4est_epitaxy_t()
{
  if(rho_g !=NULL) { ierr = VecDestroy(rho_g); CHKERRXX(ierr); }

  for(unsigned int i=0; i<phi.size(); ++i)
  {
    if(phi[i]  !=NULL) { ierr = VecDestroy(phi[i]) ; CHKERRXX(ierr); }
    if(v[0][i] !=NULL) { ierr = VecDestroy(v[0][i]); CHKERRXX(ierr); }
    if(v[1][i] !=NULL) { ierr = VecDestroy(v[1][i]); CHKERRXX(ierr); }
  }

  for(unsigned int i=0; i<rho.size(); ++i)
  {
    if(rho[i] !=NULL) { ierr = VecDestroy(rho[i]); CHKERRXX(ierr); }
  }

  /* destroy the p4est and its connectivity structure */
  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, brick);
}



void my_p4est_epitaxy_t::set_parameters(double D, double F, double alpha)
{
  this->D = D;
  this->F = F;
  this->alpha = alpha;
}



void my_p4est_epitaxy_t::compute_velocity()
{
  if(v[0].size()>0)
  {
    Vec vtmp[2];
    ierr = VecDuplicate(v[0][0], &vtmp[0]); CHKERRXX(ierr);
    ierr = VecDuplicate(v[1][0], &vtmp[1]); CHKERRXX(ierr);

    double *v_p[2];
    quad_neighbor_nodes_of_node_t qnnn;

    const double *rho_0, *rho_1;

    my_p4est_level_set_t ls(ngbd);

    for(unsigned int level=0; level<phi.size(); ++level)
    {
      ierr = VecGetArray(vtmp[0], &v_p[0]); CHKERRXX(ierr);
      ierr = VecGetArray(vtmp[1], &v_p[1]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(rho[level  ], &rho_0); CHKERRXX(ierr);
      ierr = VecGetArrayRead(rho[level+1], &rho_1); CHKERRXX(ierr);

      for(size_t i=0; i<ngbd->get_layer_size(); ++i)
      {
        p4est_locidx_t n = ngbd->get_layer_node(i);
        ngbd->get_neighbors(n, qnnn);

        v_p[0][n] = -dxyz[0]*dxyz[1]*D*(qnnn.dx_central(rho_1) - qnnn.dx_central(rho_0));
        v_p[1][n] = -dxyz[0]*dxyz[1]*D*(qnnn.dy_central(rho_1) - qnnn.dy_central(rho_0));
//        v_p[0][n] = -D*(qnnn.dx_central(rho_1) - qnnn.dx_central(rho_0));
//        v_p[1][n] = -D*(qnnn.dy_central(rho_1) - qnnn.dy_central(rho_0));
      }

      ierr = VecGhostUpdateBegin(vtmp[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vtmp[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      for(size_t i=0; i<ngbd->get_local_size(); ++i)
      {
        p4est_locidx_t n = ngbd->get_local_node(i);
        ngbd->get_neighbors(n, qnnn);

        v_p[0][n] = -dxyz[0]*dxyz[1]*D*(qnnn.dx_central(rho_1) - qnnn.dx_central(rho_0));
        v_p[1][n] = -dxyz[0]*dxyz[1]*D*(qnnn.dy_central(rho_1) - qnnn.dy_central(rho_0));
//        v_p[0][n] = -D*(qnnn.dx_central(rho_1) - qnnn.dx_central(rho_0));
//        v_p[1][n] = -D*(qnnn.dy_central(rho_1) - qnnn.dy_central(rho_0));
      }

      ierr = VecGhostUpdateEnd(vtmp[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vtmp[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(vtmp[0], &v_p[0]); CHKERRXX(ierr);
      ierr = VecRestoreArray(vtmp[1], &v_p[1]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(rho[level  ], &rho_0); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(rho[level+1], &rho_1); CHKERRXX(ierr);

      ls.extend_from_interface_to_whole_domain_TVD(phi[level], vtmp[0], v[0][level]);
      ls.extend_from_interface_to_whole_domain_TVD(phi[level], vtmp[1], v[1][level]);
    }

    ierr = VecDestroy(vtmp[0]); CHKERRXX(ierr);
    ierr = VecDestroy(vtmp[1]); CHKERRXX(ierr);
  }
}


void my_p4est_epitaxy_t::fill_island(const double *phi_p, std::vector<int> &color, int col, size_t n)
{
  std::stack<size_t> st;
  st.push(n);
  quad_neighbor_nodes_of_node_t qnnn;
  while(!st.empty())
  {
    size_t k = st.top();
    st.pop();
    color[k] = col;
    ngbd->get_neighbors(k, qnnn);
    if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && color[qnnn.node_m00_mm]==0) st.push(qnnn.node_m00_mm);
    if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && color[qnnn.node_m00_pm]==0) st.push(qnnn.node_m00_pm);
    if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && color[qnnn.node_p00_mm]==0) st.push(qnnn.node_p00_mm);
    if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && color[qnnn.node_p00_pm]==0) st.push(qnnn.node_p00_pm);

    if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && color[qnnn.node_0m0_mm]==0) st.push(qnnn.node_0m0_mm);
    if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && color[qnnn.node_0m0_pm]==0) st.push(qnnn.node_0m0_pm);
    if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && color[qnnn.node_0p0_mm]==0) st.push(qnnn.node_0p0_mm);
    if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && color[qnnn.node_0p0_pm]==0) st.push(qnnn.node_0p0_pm);
  }
}


void my_p4est_epitaxy_t::compute_average_islands_velocity()
{
  if(phi.size()==0)
    return;

  Vec phi_tmp;
  ierr = VecDuplicate(rho_g, &phi_tmp); CHKERRXX(ierr);

  Vec vn;
  ierr = VecDuplicate(rho_g, &vn); CHKERRXX(ierr);

  std::vector<int> color(nodes->num_owned_indeps);
  my_p4est_level_set_t ls(ngbd);

  for(unsigned int level=0; level<phi.size(); ++level)
  {
    int nb_col = 1;
    std::fill(color.begin(), color.end(), 0);
    const double *phi_p;
    ierr = VecGetArrayRead(phi[level], &phi_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(phi_p[n]>0 && color[n]==0)
      {
        fill_island(phi_p, color, nb_col, n);
        nb_col++;
      }
    }
    ierr = VecRestoreArrayRead(phi[level], &phi_p); CHKERRXX(ierr);

    /* first compute the normal velocity for this level */
    double *vn_p;
    ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);
    double *v_p[2];
    ierr = VecGetArray(v[0][level], &v_p[0]); CHKERRXX(ierr);
    ierr = VecGetArray(v[1][level], &v_p[1]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi[level], &phi_p); CHKERRXX(ierr);
    quad_neighbor_nodes_of_node_t qnnn;
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      ngbd->get_neighbors(n, qnnn);
      double nx = -qnnn.dx_central(phi_p);
      double ny = -qnnn.dy_central(phi_p);
      double norm = sqrt(nx*nx+ny*ny);
      if(norm>EPS) { nx /= norm; ny /= norm; }
      else         { nx = 0; ny = 0; }
      vn_p[n] = v_p[0][n]*nx + v_p[1][n]*ny;
    }
    ierr = VecGhostUpdateBegin(vn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      ngbd->get_neighbors(n, qnnn);
      double nx = -qnnn.dx_central(phi_p);
      double ny = -qnnn.dy_central(phi_p);
      double norm = sqrt(nx*nx+ny*ny);
      if(norm>EPS) { nx /= norm; ny /= norm; }
      else         { nx = 0; ny = 0; }
      vn_p[n] = v_p[0][n]*nx + v_p[1][n]*ny;
    }
    ierr = VecGhostUpdateEnd(vn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vn, &vn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(v[0][level], &v_p[0]); CHKERRXX(ierr);
    ierr = VecRestoreArray(v[1][level], &v_p[1]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi[level], &phi_p); CHKERRXX(ierr);

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &nb_col, 1, MPI_INT, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    /* for each color/island, compute average velocity */
    for(int col=1; col<nb_col; ++col)
    {
      /* first build level-set function */
      double *phi_tmp_p;
      ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        phi_tmp_p[n] = color[n]==col ? -1 : 1;
      }
      ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

      ls.reinitialize_1st_order_time_2nd_order_space(phi_tmp);

      /* compute the average normal velocity for this island */
      double vn_avg = integrate_over_interface(p4est, nodes, phi_tmp, vn) / interface_length(p4est, nodes, phi_tmp);

      /* set the velocity inside the corresponding island */
      ierr = VecGetArrayRead(phi[level], &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(v[0][level], &v_p[0]); CHKERRXX(ierr);
      ierr = VecGetArray(v[1][level], &v_p[1]); CHKERRXX(ierr);

      for(size_t i=0; i<ngbd->get_layer_size(); ++i)
      {
        p4est_locidx_t n = ngbd->get_layer_node(i);
        if(color[n]==col)
        {
          ngbd->get_neighbors(n, qnnn);
          double nx = -qnnn.dx_central(phi_p);
          double ny = -qnnn.dy_central(phi_p);
          double norm = sqrt(nx*nx+ny*ny);
          if(norm>EPS) { nx /= norm; ny /= norm; }
          else         { nx = 0; ny = 0; }
          v_p[0][n] = vn_avg*nx;
          v_p[1][n] = vn_avg*ny;
        }
      }
      ierr = VecGhostUpdateBegin(v[0][level], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(v[1][level], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      for(size_t i=0; i<ngbd->get_local_size(); ++i)
      {
        p4est_locidx_t n = ngbd->get_local_node(i);
        if(color[n]==col)
        {
          ngbd->get_neighbors(n, qnnn);
          double nx = -qnnn.dx_central(phi_p);
          double ny = -qnnn.dy_central(phi_p);
          double norm = sqrt(nx*nx+ny*ny);
          if(norm>EPS) { nx /= norm; ny /= norm; }
          else         { nx = 0; ny = 0; }
          v_p[0][n] = vn_avg*nx;
          v_p[1][n] = vn_avg*ny;
        }
      }
      ierr = VecGhostUpdateEnd(v[0][level], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(v[1][level], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArrayRead(phi[level], &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(v[0][level], &v_p[0]); CHKERRXX(ierr);
      ierr = VecRestoreArray(v[1][level], &v_p[1]); CHKERRXX(ierr);
    }

    /* Extend the velocity accross the interface for continuous velocity field */
    double *phi_neg_p;
    ierr = VecGetArray(phi[level], &phi_neg_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      phi_neg_p[n] *= -1;
    ierr = VecRestoreArray(phi[level], &phi_neg_p); CHKERRXX(ierr);

    ls.extend_Over_Interface_TVD(phi[level], v[0][level], 20, 0);
    ls.extend_Over_Interface_TVD(phi[level], v[1][level], 20, 0);

    ierr = VecGetArray(phi[level], &phi_neg_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      phi_neg_p[n] *= -1;
    ierr = VecRestoreArray(phi[level], &phi_neg_p); CHKERRXX(ierr);
  }

  ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
}



void my_p4est_epitaxy_t::compute_dt()
{
  dt_n *= 4;

  dt_n = MIN(dt_n, 0.01/F);

  if(phi.size()!=0)
  {
    double vmax = 0;
    double m, M;
    for(unsigned int i=0; i<phi.size(); ++i)
    {
      ierr = VecMax(v[0][i], NULL, &M); CHKERRXX(ierr);
      ierr = VecMin(v[0][i], NULL, &m); CHKERRXX(ierr);
      vmax = MAX(vmax, fabs(M), fabs(m));

      ierr = VecMax(v[1][i], NULL, &M); CHKERRXX(ierr);
      ierr = VecMin(v[1][i], NULL, &m); CHKERRXX(ierr);
      vmax = MAX(vmax, fabs(M), fabs(m));
    }
    ierr = PetscPrintf(p4est->mpicomm, "Maximum velocity = %e\n", vmax); CHKERRXX(ierr);

    if(vmax>EPS)
    {
      dt_n = MIN(dt_n, MIN(dxyz[0],dxyz[1])/vmax);
    }
  }

  ierr = PetscPrintf(p4est->mpicomm, "time step dt_n = %e\n", dt_n); CHKERRXX(ierr);
}


void my_p4est_epitaxy_t::update_grid()
{
  p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  if(phi.size()>0)
  {
    my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, brick, ngbd);

    Vec velo[2];
    velo[0] = v[0][0];
    velo[1] = v[1][0];
    sl.update_p4est(velo, dt_n, phi[0]);
  }

  /* interpolate the quantities on the new grid */
  my_p4est_interpolation_nodes_t interp(ngbd);

  double xyz[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  Vec rho_g_tmp;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &rho_g_tmp); CHKERRXX(ierr);
  interp.set_input(rho_g, quadratic_non_oscillatory);
  interp.interpolate(rho_g_tmp);
  ierr = VecDestroy(rho_g); CHKERRXX(ierr);
  rho_g = rho_g_tmp;

  for(unsigned int i=0; i<rho.size(); ++i)
  {
    Vec tmp;
    ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
    interp.set_input(rho[i], quadratic_non_oscillatory);
    interp.interpolate(tmp);
    ierr = VecDestroy(rho[i]); CHKERRXX(ierr);
    rho[i] = tmp;
  }

  for(unsigned int i=0; i<phi.size(); ++i)
  {
    ierr = VecDestroy(v[0][i]); CHKERRXX(ierr);
    ierr = VecDestroy(v[1][i]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v[0][i]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v[1][i]); CHKERRXX(ierr);
  }

  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  hierarchy->update(p4est, ghost);
  ngbd->update(hierarchy, nodes);

  /* reinitialize and perturb phi */
  my_p4est_level_set_t ls(ngbd);
  for(unsigned int level=0; level<phi.size(); ++level)
  {
    ls.reinitialize_1st_order_time_2nd_order_space(phi[level]);
    ls.perturb_level_set_function(phi[level], EPS);
  }
}


/*
 * solve the density heat equation
 */
void my_p4est_epitaxy_t::solve_rho()
{
  /* if there is no island yet, just compute the deposition directly ... there is no diffusion */
  if(phi.size()==0)
  {
    if(new_island!=0)
    {
      throw std::runtime_error("There cannot be an island generated with no level-set ...");
    }
    double *rho_p, *rho_np1_p, *rho_g_p;
    ierr = VecGetArray(rho[0]    , &rho_p    ); CHKERRXX(ierr);
    ierr = VecGetArray(rho_np1[0], &rho_np1_p); CHKERRXX(ierr);
    ierr = VecGetArray(rho_g     , &rho_g_p  ); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      rho_np1_p[n] = rho_p[n] + dt_n*F;
      rho_g_p[n] = rho_np1_p[n];
    }
    ierr = VecRestoreArray(rho[0]    , &rho_p    ); CHKERRXX(ierr);
    ierr = VecRestoreArray(rho_np1[0], &rho_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rho_g     , &rho_g_p  ); CHKERRXX(ierr);
  }
  else
  {
    Vec rhs;
    ierr = VecDuplicate(rho_g, &rhs); CHKERRXX(ierr);

    Vec phi_i;
    ierr = VecDuplicate(rho_g, &phi_i); CHKERRXX(ierr);

    std::vector<const double *> phi_p(phi.size());
    for(unsigned int i=0; i<phi.size(); ++i)
    {
      ierr = VecGetArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
    }

    my_p4est_level_set_t ls(ngbd);

    for(unsigned int level=0; level<rho.size(); ++level)
    {
      const double *rho_p;
      ierr = VecGetArrayRead(rho[level], &rho_p); CHKERRXX(ierr);

      double *rhs_p, *phi_i_p;
      ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
      ierr = VecGetArray(phi_i, &phi_i_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
//        rhs_p[n] = rho_p[n] + dt_n*(F - new_island*2*D*sigma1*rho_sqr_avg);
        rhs_p[n] = rho_p[n] + dt_n*(F - .1*new_island*2*D*sigma1*SQR(rho_p[n]));

        phi_i_p[n] = -4*L;
        if(level<phi.size()) phi_i_p[n] = phi_p[level][n];
        if(level>0)          phi_i_p[n] = MAX(phi_i_p[n], -phi_p[level-1][n]);
      }

      ierr = VecRestoreArrayRead(rho[level], &rho_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_i, &phi_i_p); CHKERRXX(ierr);

      BoundaryConditions2D bc;
      bc.setInterfaceType(DIRICHLET);
      bc.setInterfaceValue(zero);

      my_p4est_poisson_nodes_t solver(ngbd);
      solver.set_phi(phi_i);
      solver.set_bc(bc);
      solver.set_diagonal(1);
      solver.set_mu(dt_n*D);
      solver.set_rhs(rhs);
      solver.solve(rho_np1[level]);

      ls.extend_Over_Interface_TVD(phi_i, rho_np1[level]);

      double *rho_g_p;
      const double *rho_np1_p;
      ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);
      ierr = VecGetArray(phi_i, &phi_i_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(rho_np1[level], &rho_np1_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        if(phi_i_p[n]<0)
          rho_g_p[n] = rho_np1_p[n];
      }
      ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_i, &phi_i_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(rho_np1[level], &rho_np1_p); CHKERRXX(ierr);
    }

    for(unsigned int i=0; i<phi.size(); ++i)
    {
      ierr = VecRestoreArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi_i); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);
  }


  double rho_max;
  VecMax(rho_np1[0], NULL, &rho_max);
  ierr = PetscPrintf(p4est->mpicomm, "Maximum density = %e\n", rho_max); CHKERRXX(ierr);
}



void my_p4est_epitaxy_t::update_nucleation()
{
  Vec ones;
  ierr = VecDuplicate(rho[0], &ones); CHKERRXX(ierr);
  Vec loc;
  ierr = VecGhostGetLocalForm(ones, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, -1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(ones, &loc); CHKERRXX(ierr);

  rho_avg_np1 = integrate_over_negative_domain(p4est, nodes, ones, rho_g)/(L*L);
  sigma1_np1 = 4*PI/log((1/alpha)*rho_avg_np1*D/F);

  Vec rho_sqr;
  ierr = VecDuplicate(rho[0], &rho_sqr); CHKERRXX(ierr);
  double *rho_sqr_p;
  const double *rho_g_p;
  ierr = VecGetArray(rho_sqr, &rho_sqr_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(rho_g, &rho_g_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    rho_sqr_p[n] = SQR(rho_g_p[n]);
  }
  ierr = VecRestoreArray(rho_sqr, &rho_sqr_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(rho_g, &rho_g_p); CHKERRXX(ierr);

  rho_sqr_avg_np1 = integrate_over_negative_domain(p4est, nodes, ones, rho_sqr)/(L*L);

  Nuc_np1 = Nuc + dt_n * D*sigma1_np1*rho_sqr_avg_np1;

  ierr = VecDestroy(rho_sqr); CHKERRXX(ierr);
  ierr = VecDestroy(ones); CHKERRXX(ierr);
}


void my_p4est_epitaxy_t::nucleate_new_island()
{
  /* check for new island */
  if(floor(Nuc*island_nucleation_scaling) != floor(Nuc_np1*island_nucleation_scaling))
  {
    ierr = PetscPrintf(p4est->mpicomm, "Nucleating new island !\n"); CHKERRXX(ierr);
    double xc, yc;
    double r = 4*dxyz[0];

    /* first island created */
    if(phi.size()==0)
    {
      /* select nucleation point */
      xc = ((double)rand()/RAND_MAX)*L;
      yc = ((double)rand()/RAND_MAX)*L;
    }
    /* islands already exist */
    else
    {
      my_p4est_interpolation_nodes_t interp(ngbd);
      interp.set_input(phi[0], quadratic_non_oscillatory);

      /* find the max value of rho */
      const double *phi_p, *rho_g_p;
      ierr = VecGetArrayRead(phi[0], &phi_p  ); CHKERRXX(ierr);
      ierr = VecGetArrayRead(rho_g , &rho_g_p); CHKERRXX(ierr);

      std::default_random_engine generator;
      /* create a gaussian noise generator, constructor(mean, standard deviation) */
      std::normal_distribution<double> distribution(1,.1);
      double phi_c;

      /* find the nucleation point, maximum of (rho * gaussian_perturbation) */
      std::vector<double> comm(4*p4est->mpisize);
      do{
        comm[4*p4est->mpirank] = 0;

        for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
        {
          if(phi_p[n]<0)
          {
            double rho_perturb = SQR(rho_g_p[n])*distribution(generator);
            if(rho_perturb > comm[4*p4est->mpirank])
            {
              comm[4*p4est->mpirank + 0] = rho_perturb;
              comm[4*p4est->mpirank + 1] = node_x_fr_n(n, p4est, nodes);
              comm[4*p4est->mpirank + 2] = node_y_fr_n(n, p4est, nodes);
              comm[4*p4est->mpirank + 3] = phi_p[n];
            }
          }
        }

        int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &comm[0], 4, MPI_DOUBLE, p4est->mpicomm); SC_CHECK_MPI(mpiret);

        int rank = 0;
        for(int p=1; p<p4est->mpisize; ++p)
        {
          if(comm[4*p]>comm[4*rank])
            rank = p;
        }

        xc    = comm[4*rank+1];
        yc    = comm[4*rank+2];
        phi_c = comm[4*rank+3];
      } while(phi_c>r);

      ierr = VecRestoreArrayRead(phi[0], &phi_p  ); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(rho_g , &rho_g_p); CHKERRXX(ierr);
    }
    circle_t circle(xc, yc, r, this);

    /* update the forest with the new island */
    p4est_t *p4est_new = p4est_copy(p4est, P4EST_FALSE);
    splitting_criteria_cf_t *sp_old = (splitting_criteria_cf_t*)p4est->user_pointer;
    splitting_criteria_cf_t sp(sp_old->min_lvl, sp_old->max_lvl, &circle, sp_old->lip);
    p4est_new->user_pointer = (void*)(&sp);
    my_p4est_refine(p4est_new, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est_new, P4EST_FALSE, NULL);
    p4est_new->user_pointer = (void*)sp_old;

    p4est_ghost_t *ghost_new = my_p4est_ghost_new(p4est_new, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_new = my_p4est_nodes_new(p4est_new, ghost_new);

    my_p4est_interpolation_nodes_t interp(ngbd);
    double xyz[P4EST_DIM];
    for(size_t n=0; n<nodes_new->indep_nodes.elem_count; ++n)
    {
      node_xyz_fr_n(n, p4est_new, nodes_new, xyz);
      interp.add_point(n, xyz);
    }

    Vec rho_g_new;
    ierr = VecCreateGhostNodes(p4est_new, nodes_new, &rho_g_new); CHKERRXX(ierr);
    interp.set_input(rho_g, quadratic_non_oscillatory);
    interp.interpolate(rho_g_new);
    ierr = VecDestroy(rho_g ); CHKERRXX(ierr);
    rho_g  = rho_g_new;

    for(unsigned int i=0; i<rho.size(); ++i)
    {
      Vec rho_new;
      ierr = VecDuplicate(rho_g_new, &rho_new); CHKERRXX(ierr);
      interp.set_input(rho[i], quadratic_non_oscillatory);
      interp.interpolate(rho_new);
      ierr = VecDestroy(rho[i]); CHKERRXX(ierr);
      rho[i] = rho_new;
    }

    /* first island created */
    if(phi.size()==0)
    {
      phi.resize(1);
      ierr = VecDuplicate(rho_g, &phi[0]); CHKERRXX(ierr);
      double *phi_p, *rho_p, *rho_g_p;
      ierr = VecGetArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho[0], &rho_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes_new->indep_nodes.elem_count; ++n)
      {
        double x = node_x_fr_n(n, p4est_new, nodes_new);
        double y = node_y_fr_n(n, p4est_new, nodes_new);
        double tmp = circle(x,y);
        phi_p[n] = tmp;
//        if(tmp>0)
//        {
//          rho_p[n] = 0;
//          rho_g_p[n] = 0;
//        }
      }
      ierr = VecRestoreArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho[0], &rho_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);

      Vec tmp, loc[2];
      ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
      rho.push_back(tmp);

      ierr = VecGhostGetLocalForm(rho[0], &loc[0]); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(rho[1], &loc[1]); CHKERRXX(ierr);
      ierr = VecCopy(rho[0], rho[1]); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(rho[0], &loc[0]); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(rho[1], &loc[1]); CHKERRXX(ierr);

      rho_np1.resize(2);

      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &tmp); CHKERRXX(ierr);
      v[0].push_back(tmp);
      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &tmp); CHKERRXX(ierr);
      v[1].push_back(tmp);
    }
    /* islands already exist at that level */
    else
    {
      Vec phi_0_new;
      ierr = VecDuplicate(rho_g, &phi_0_new); CHKERRXX(ierr);
      interp.set_input(phi[0], quadratic_non_oscillatory); CHKERRXX(ierr);
      interp.interpolate(phi_0_new);
      ierr = VecDestroy(phi[0]); CHKERRXX(ierr);
      phi[0] = phi_0_new;

      double *phi_p, *rho_p[2], *rho_g_p;
      ierr = VecGetArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho[0], &rho_p[0]); CHKERRXX(ierr);
      ierr = VecGetArray(rho[1], &rho_p[1]); CHKERRXX(ierr);
      ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes_new->indep_nodes.elem_count; ++n)
      {
        double x = node_x_fr_n(n, p4est_new, nodes_new);
        double y = node_y_fr_n(n, p4est_new, nodes_new);
        double tmp = circle(x,y);
        phi_p[n] = MAX(phi_p[n],tmp);
        if(tmp>-2*MAX(dxyz[0],dxyz[1]))
        {
          rho_p[1][n] = rho_p[0][n];
//          rho_p[0][n] = 0;
//          rho_p[1][n] = 0;
//          rho_g_p[n]  = 0;
        }
      }
      ierr = VecRestoreArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho[0], &rho_p[0]); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho[1], &rho_p[1]); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);

      ierr = VecDestroy(v[0][0]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &v[0][0]); CHKERRXX(ierr);
      ierr = VecDestroy(v[1][0]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &v[1][0]); CHKERRXX(ierr);
    }

    p4est_destroy(p4est);       p4est = p4est_new;
    p4est_ghost_destroy(ghost); ghost = ghost_new;
    p4est_nodes_destroy(nodes); nodes = nodes_new;
    hierarchy->update(p4est, ghost);
    ngbd->update(hierarchy, nodes);

    my_p4est_level_set_t ls(ngbd);
    ls.perturb_level_set_function(phi[0], EPS);

    new_island = 1;
  }
  else
  {
    new_island = 0;
  }

  for(unsigned int i=0; i<rho.size(); ++i)
  {
    ierr = VecDuplicate(rho_g, &rho_np1[i]); CHKERRXX(ierr);
  }

  Nuc = Nuc_np1;
  sigma1 = sigma1_np1;
  rho_sqr_avg = rho_sqr_avg_np1;
  rho_avg = rho_avg_np1;
}


bool my_p4est_epitaxy_t::check_time_step()
{
  /* if rho is negative or if more than one island is being nucleated, then the timestep must be reduced */
  double rho_min;
  ierr = VecMin(rho_g, NULL, &rho_min); CHKERRXX(ierr);

  if(floor(Nuc_np1*island_nucleation_scaling) < floor(Nuc*island_nucleation_scaling) || floor(Nuc_np1*island_nucleation_scaling) > floor(Nuc*island_nucleation_scaling)+1 || rho_min<0)
  {
    dt_n *= .25;
    if(dt_n<1e-15)
      throw std::runtime_error("The time step is smaller than 1e-15 !!");
    ierr = PetscPrintf(p4est->mpicomm, "Reducing time step, Old Nuc = %e, New Nuc = %e, rho_min = %e\n", Nuc*island_nucleation_scaling, Nuc_np1*island_nucleation_scaling, rho_min); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "time step dt_n = %e\n", dt_n); CHKERRXX(ierr);
    return false;
  }

  /* if the time step passed the test, then update the quantities */
  for(unsigned int i=0; i<rho.size(); ++i)
  {
    ierr = VecDestroy(rho[i]); CHKERRXX(ierr);
    rho[i] = rho_np1[i];
  }

  ierr = PetscPrintf(p4est->mpicomm, "Time step is fine, Old Nuc = %e, New Nuc = %e, rho_min = %e\n", Nuc*island_nucleation_scaling, Nuc_np1*island_nucleation_scaling, rho_min); CHKERRXX(ierr);
  return true;
}


void my_p4est_epitaxy_t::save_vtk(int iter)
{
  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to the desired path to save visuals.\n");
    return;
  }

  char name[1000];
  snprintf(name, 1000, "%s/vtu/epitaxy_%04d", out_dir, iter);

  /* first build a level-set combining all levels */
  Vec phi_g;
  ierr = VecDuplicate(rho_g, &phi_g); CHKERRXX(ierr);
  std::vector<const double *> phi_p(phi.size());
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    ierr = VecGetArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
  }
  double *phi_g_p;
  ierr = VecGetArray(phi_g, &phi_g_p); CHKERRXX(ierr);
  for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    phi_g_p[n] = -2*L;
    for(unsigned int i=0; i<phi.size(); ++i)
    {
      phi_g_p[n] = (i%2==0 ? MAX(phi_g_p[n],phi_p[i][n]) : MIN(phi_g_p[n],phi_p[i][n]));
    }
  }
  ierr = VecRestoreArray(phi_g, &phi_g_p); CHKERRXX(ierr);
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    ierr = VecRestoreArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
  }


  /* since forest is periodic, need to build temporary non-periodic forest for visualization */
  my_p4est_brick_t brick_vis;
  p4est_connectivity_t *connectivity_vis = my_p4est_brick_new(brick->nxyztrees[0], brick->nxyztrees[1], xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], &brick_vis, 0, 0);
  p4est_t *p4est_vis = my_p4est_new(p4est->mpicomm, connectivity_vis, 0, NULL, NULL);
  p4est_ghost_t *ghost_vis = my_p4est_ghost_new(p4est_vis, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_vis = my_p4est_nodes_new(p4est_vis, ghost_vis);
  Vec phi_vis;
  double *phi_vis_p;
  ierr = VecCreateGhostNodes(p4est_vis, nodes_vis, &phi_vis); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t interp(ngbd);
  for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_vis, nodes_vis, xyz);
    interp.add_point(n, xyz);
  }
  interp.set_input(phi_g, linear);
  interp.interpolate(phi_vis);

  splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;
  bool is_grid_changing = true;

  while(is_grid_changing)
  {
    ierr = VecGetArray(phi_vis, &phi_vis_p); CHKERRXX(ierr);
    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est_vis, nodes_vis, phi_vis_p);
    ierr = VecRestoreArray(phi_vis, &phi_vis_p); CHKERRXX(ierr);

    if(is_grid_changing)
    {
      my_p4est_partition(p4est_vis, P4EST_TRUE, NULL);
      p4est_ghost_destroy(ghost_vis); ghost_vis = my_p4est_ghost_new(p4est_vis, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes_vis); nodes_vis = my_p4est_nodes_new(p4est_vis, ghost_vis);
      ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_vis, nodes_vis, &phi_vis); CHKERRXX(ierr);

      interp.clear();
      for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est_vis, nodes_vis, xyz);
        interp.add_point(n, xyz);
      }
      interp.set_input(phi_g, linear);
      interp.interpolate(phi_vis);
    }
  }
  ierr = VecDestroy(phi_g); CHKERRXX(ierr);


  Vec rho_g_vis;
  ierr = VecDuplicate(phi_vis, &rho_g_vis); CHKERRXX(ierr);
  interp.set_input(rho_g, linear);
  interp.interpolate(rho_g_vis);

  const double *phi_v_p, *rho_g_v_p;
  ierr = VecGetArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
  ierr = VecGetArrayRead(rho_g_vis, &rho_g_v_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(p4est_vis, nodes_vis, ghost_vis, P4EST_TRUE, P4EST_TRUE, 2, 0, name,
                         VTK_POINT_DATA, "phi", phi_v_p,
                         VTK_POINT_DATA, "rho", rho_g_v_p);

  ierr = VecRestoreArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(rho_g_vis, &rho_g_v_p); CHKERRXX(ierr);

  ierr = VecDestroy(rho_g_vis); CHKERRXX(ierr);
  ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
  p4est_nodes_destroy(nodes_vis);
  p4est_ghost_destroy(ghost_vis);
  p4est_destroy(p4est_vis);
  my_p4est_brick_destroy(connectivity_vis, &brick_vis);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", name);
}
