#include "my_p4est_epitaxy.h"

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

  Vec loc;
  ierr = VecGhostGetLocalForm(rho_g, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho_g, &loc); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(rho[0], &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho[0], &loc); CHKERRXX(ierr);

  new_island = 0;
  alpha = 1;
  dx_dy_dz(p4est, dxyz);
  dt_n = MIN(dxyz[0],dxyz[1]);

  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<3; i++)
    xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
  for (short i=0; i<3; i++)
    xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];

  L = xyz_max[0]-xyz_min[0];
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

        v_p[0][n] = D*(qnnn.dx_central(rho_0) - qnnn.dx_central(rho_1));
        v_p[1][n] = D*(qnnn.dy_central(rho_0) - qnnn.dy_central(rho_1));
      }

      ierr = VecGhostUpdateBegin(vtmp[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vtmp[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      for(size_t i=0; i<ngbd->get_local_size(); ++i)
      {
        p4est_locidx_t n = ngbd->get_local_node(i);
        ngbd->get_neighbors(n, qnnn);

        v_p[0][n] = D*(qnnn.dx_central(rho_0) - qnnn.dx_central(rho_1));
        v_p[1][n] = D*(qnnn.dy_central(rho_0) - qnnn.dy_central(rho_1));
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


/*
 * solve the density heat equation
 */
void my_p4est_epitaxy_t::solve_rho()
{
  /* if there is no island yet, just compute the deposition directly ... there is no diffusion */
  if(phi.size()==0)
  {
    double *rho_p, *rho_g_p;
    ierr = VecGetArray(rho[0], &rho_p); CHKERRXX(ierr);
    ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      rho_p[n] = rho_p[n] + dt_n*(F - new_island*2*D*sigma1*rho_sqr_avg);
      rho_g_p[n] = rho_p[n];
    }
    ierr = VecRestoreArray(rho[0], &rho_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);
  }
  else
  {
    Vec rhs;
    ierr = VecDuplicate(rho_g, &rhs); CHKERRXX(ierr);

    Vec phi_i;
    ierr = VecDuplicate(rho_g, &phi_i); CHKERRXX(ierr);

    const double *phi_p[phi.size()];
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
        rhs_p[n] = rho_p[n] + dt_n*(F - new_island*2*D*sigma1*rho_sqr_avg);

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
      solver.solve(rho[level]);

      ls.extend_Over_Interface_TVD(phi_i, rho[level]);

      double *rho_g_p;
      ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);
      ierr = VecGetArray(phi_i, &phi_i_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(rho[level], &rho_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        if(phi_i_p[n]<0)
          rho_g_p[n] = rho_p[n];
      }
      ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_i, &phi_i_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(rho[level], &rho_p); CHKERRXX(ierr);
    }

    for(unsigned int i=0; i<phi.size(); ++i)
    {
      ierr = VecRestoreArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi_i); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);
  }


  double rho_max;
  VecMax(rho[0], NULL, &rho_max);
  ierr = PetscPrintf(p4est->mpicomm, "Maximum density = %e\n", rho_max); CHKERRXX(ierr);
}



void my_p4est_epitaxy_t::compute_dt()
{
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

    dt_n = vmax<EPS ? dxyz[0] : MIN(dxyz[0],dxyz[1])/vmax;
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
    ierr = VecCreateGhostNodes(p4est, nodes, &v[0][i]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &v[1][i]); CHKERRXX(ierr);
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


void my_p4est_epitaxy_t::update_nucleation()
{
  Vec ones;
  ierr = VecDuplicate(rho[0], &ones); CHKERRXX(ierr);
  Vec loc;
  ierr = VecGhostGetLocalForm(ones, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, -1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(ones, &loc); CHKERRXX(ierr);

  rho_avg = integrate_over_negative_domain(p4est, nodes, ones, rho_g)/(L*L);
  sigma1 = 4*PI/log((1/alpha)*rho_avg*D/F);

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

  rho_sqr_avg = integrate_over_negative_domain(p4est, nodes, ones, rho_sqr)/(L*L);
  std::cout << rho_sqr_avg << ", " << sigma1 << std::endl;

  double new_Nuc = Nuc + dt_n * D*sigma1*rho_sqr_avg;
  ierr = PetscPrintf(p4est->mpicomm, "Nucleation is now : %e\n", new_Nuc); CHKERRXX(ierr);

  ierr = VecDestroy(rho_sqr); CHKERRXX(ierr);
  ierr = VecDestroy(ones); CHKERRXX(ierr);

  /* check for new island */
  if(floor(Nuc*L*L) != floor(new_Nuc*L*L))
  {
    ierr = PetscPrintf(p4est->mpicomm, "Nucleating new island !\n"); CHKERRXX(ierr);
    if(phi.size()==0)
    {
      Vec tmp;
      ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
      Vec loc;
      ierr = VecGhostGetLocalForm(tmp, &loc); CHKERRXX(ierr);
      ierr = VecSet(loc, 0); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(tmp, &loc); CHKERRXX(ierr);
      rho.push_back(tmp);

      /* select nucleation point */
      double xc = ((double)rand()/RAND_MAX)*L;
      double yc = ((double)rand()/RAND_MAX)*L;
      double r = 4*dxyz[0];
      phi.resize(1);
      ierr = VecDuplicate(rho_g, &phi[0]); CHKERRXX(ierr);
      double *phi_p, *rho_p, *rho_g_p;
      ierr = VecGetArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho[0], &rho_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
        phi_p[n] = -4*L;
        for(int i=-1; i<2; ++i)
          for(int j=-1; j<2; ++j)
          {
            phi_p[n] = MAX(phi_p[n], r-sqrt( SQR(x+i*(xyz_max[0]-xyz_min[0])-xc) + SQR(y+j*(xyz_max[1]-xyz_min[1])-yc) ) );
          }
        if(phi_p[n]>0)
        {
          rho_p[n] = 0;
          rho_g_p[n] = 0;
        }
      }
      ierr = VecRestoreArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho[0], &rho_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);

      ierr = VecCreateGhostNodes(p4est, nodes, &tmp); CHKERRXX(ierr);
      v[0].push_back(tmp);
      ierr = VecCreateGhostNodes(p4est, nodes, &tmp); CHKERRXX(ierr);
      v[1].push_back(tmp);
    }
    else
    {
      /* find nucleation point */
      my_p4est_interpolation_nodes_t interp(ngbd);
      interp.set_input(phi[0], quadratic_non_oscillatory);
      double r = 4*dxyz[0];
      double xc, yc;
      do{
        xc = ((double)rand()/RAND_MAX)*L;
        yc = ((double)rand()/RAND_MAX)*L;
      } while(interp(xc,yc)>r);

      double *phi_p, *rho_p, *rho_g_p;
      ierr = VecGetArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho[0], &rho_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
        for(int i=-1; i<2; ++i)
          for(int j=-1; j<2; ++j)
          {
            phi_p[n] = MAX(phi_p[n], r-sqrt( SQR(x+i*(xyz_max[0]-xyz_min[0])-xc) + SQR(y+j*(xyz_max[1]-xyz_min[1])-yc) ) );
            if(r-sqrt( SQR(x+i*(xyz_max[0]-xyz_min[0])-xc) + SQR(y+j*(xyz_max[1]-xyz_min[1])-yc) ) > 0)
            {
              rho_p[n] = 0;
              rho_g_p[n] = 0;
            }
          }
      }
      ierr = VecRestoreArray(phi[0], &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho[0], &rho_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);
    }
//    new_island = 1;
  }
  else
  {
    new_island = 0;
  }

  Nuc = new_Nuc;
}

void my_p4est_epitaxy_t::one_step()
{
  solve_rho();
  compute_velocity();
  update_nucleation();

  compute_dt();

  update_grid();
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
  const double *phi_p[phi.size()];
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
