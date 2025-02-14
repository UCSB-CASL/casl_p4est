#include "my_p4est_epitaxy.h"

#include <time.h>
#include <stack>
/* c++11 support for the "random" procedures is broken for some reason with icpc / gcc 4.4.7 or gcc 4.8.0 */
//#if defined(COMET) || defined(STAMPEDE)

// the following is needed on stampede when using intel compiler
#ifdef __INTEL_COMPILER
namespace std {
      typedef decltype(nullptr) nullptr_t;
}
#endif
#include <boost/random.hpp>
//#else
//#include <random>
//#endif

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
  ngbd->init_neighbors();

  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<P4EST_DIM; i++)
    xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
  for (short i=0; i<P4EST_DIM; i++)
    xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];

  L = xyz_max[0]-xyz_min[0];

  ierr = VecCreateGhostNodes(p4est, nodes, &rho_g); CHKERRXX(ierr);
  rho.resize(1);
  ierr = VecDuplicate(rho_g, &rho[0]); CHKERRXX(ierr);
  ierr = VecDuplicate(rho_g, &phi_g); CHKERRXX(ierr);

  rho_np1.resize(1);
  ierr = VecDuplicate(rho_g, &rho_np1[0]); CHKERRXX(ierr);

  Vec loc;
  ierr = VecGhostGetLocalForm(rho_g, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho_g, &loc); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(phi_g, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, -2*L); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi_g, &loc); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(rho[0], &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho[0], &loc); CHKERRXX(ierr);

  if(bc_type==ROBIN)
  {
    ierr = VecDuplicate(rho_g, &robin_coef); CHKERRXX(ierr);
  }
  else
    robin_coef = NULL;

  Nuc = 0;
  new_island = 0;
  alpha = 1.05;
  dxyz_min(p4est, dxyz);
//  dt_n = MIN(dxyz[0],dxyz[1]);
  dt_n = 1e-3;

  nb_levels_deleted = 0;

//  island_nucleation_scaling = 1;
  island_nucleation_scaling = L*L;

//  srand(0);
  srand(time(NULL));
}


my_p4est_epitaxy_t::~my_p4est_epitaxy_t()
{
  if(rho_g !=NULL) { ierr = VecDestroy(rho_g); CHKERRXX(ierr); }
  if(phi_g !=NULL) { ierr = VecDestroy(phi_g); CHKERRXX(ierr); }
  if(robin_coef != NULL) { ierr = VecDestroy(robin_coef); CHKERRXX(ierr); }

  for(unsigned int i=0; i<phi.size(); ++i)
  {
    if(island_number[i] != NULL) { ierr = VecDestroy(island_number[i]); CHKERRXX(ierr); }
    if(phi[i]           != NULL) { ierr = VecDestroy(phi[i]) ; CHKERRXX(ierr); }
    if(v[0][i]          != NULL) { ierr = VecDestroy(v[0][i]); CHKERRXX(ierr); }
    if(v[1][i]          != NULL) { ierr = VecDestroy(v[1][i]); CHKERRXX(ierr); }
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



void my_p4est_epitaxy_t::set_parameters(double D, double F, double alpha, double lattice_spacing, BoundaryConditionType bc_type, double barrier)
{
  this->D = D;
  this->F = F;
  this->alpha = alpha;
  this->lattice_spacing = lattice_spacing;
  this->bc_type = bc_type;
  this->barrier = barrier;

  Dp = barrier*D;
  Dm = 0.95*D;
  Dcurl = 10;
  DE = D;

  double c0 = DE/(2*DE + (2*Dp/Dm + 1)*Dcurl);
  double c1 = (2*DE + (3*Dp/Dm +1)*Dcurl) / (2*DE + (4*Dp/Dm +2)*Dcurl);
  double Pe = F*L/DE;
  rho_eq = pow(.5,2./3.) * pow(Pe,2./3.) * Dcurl/Dm * pow(c0,2./3.) * pow(c1,1./3.);
}


void my_p4est_epitaxy_t::compute_phi_g()
{
  ierr = VecDestroy(phi_g); CHKERRXX(ierr);
  ierr = VecDuplicate(rho_g, &phi_g); CHKERRXX(ierr);

  std::vector<const double*> phi_p(phi.size());
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    ierr = VecGetArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
  }
  double *phi_g_p;
  ierr = VecGetArray(phi_g, &phi_g_p); CHKERRXX(ierr);
  for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    phi_g_p[n] = -2*L;
    for(unsigned int level=0; level<phi.size(); ++level)
    {
      phi_g_p[n] = (level%2==0 ? MAX(phi_g_p[n],phi_p[level][n]) : MIN(phi_g_p[n],-phi_p[level][n]));
    }
  }
  ierr = VecRestoreArray(phi_g, &phi_g_p); CHKERRXX(ierr);
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    ierr = VecRestoreArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
  }
}


void my_p4est_epitaxy_t::compute_velocity()
{
  if(phi.size()>0)
  {
    Vec vtmp[2];
    ierr = VecDuplicate(v[0][0], &vtmp[0]); CHKERRXX(ierr);
    ierr = VecDuplicate(v[1][0], &vtmp[1]); CHKERRXX(ierr);

    double *v_p[2];

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
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];

        v_p[0][n] = -SQR(lattice_spacing)*D*(qnnn.dx_central(rho_1) - qnnn.dx_central(rho_0));
        v_p[1][n] = -SQR(lattice_spacing)*D*(qnnn.dy_central(rho_1) - qnnn.dy_central(rho_0));
      }

      ierr = VecGhostUpdateBegin(vtmp[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vtmp[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      for(size_t i=0; i<ngbd->get_local_size(); ++i)
      {
        p4est_locidx_t n = ngbd->get_local_node(i);
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];

        v_p[0][n] = -SQR(lattice_spacing)*D*(qnnn.dx_central(rho_1) - qnnn.dx_central(rho_0));
        v_p[1][n] = -SQR(lattice_spacing)*D*(qnnn.dy_central(rho_1) - qnnn.dy_central(rho_0));
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



void my_p4est_epitaxy_t::fill_island(const double *phi_p, double *island_number_p, int number, p4est_locidx_t n)
{
  std::stack<size_t> st;
  st.push(n);
  while(!st.empty())
  {
    size_t k = st.top();
    st.pop();
    island_number_p[k] = number;
    const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[k];
    if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && island_number_p[qnnn.node_m00_mm]<0) st.push(qnnn.node_m00_mm);
    if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && island_number_p[qnnn.node_m00_pm]<0) st.push(qnnn.node_m00_pm);
    if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && island_number_p[qnnn.node_p00_mm]<0) st.push(qnnn.node_p00_mm);
    if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && island_number_p[qnnn.node_p00_pm]<0) st.push(qnnn.node_p00_pm);

    if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && island_number_p[qnnn.node_0m0_mm]<0) st.push(qnnn.node_0m0_mm);
    if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && island_number_p[qnnn.node_0m0_pm]<0) st.push(qnnn.node_0m0_pm);
    if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && island_number_p[qnnn.node_0p0_mm]<0) st.push(qnnn.node_0p0_mm);
    if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && island_number_p[qnnn.node_0p0_pm]<0) st.push(qnnn.node_0p0_pm);
  }
}


void my_p4est_epitaxy_t::find_connected_ghost_islands(const double *phi_p, double *island_number_p, p4est_locidx_t n, std::vector<double> &connected, std::vector<bool> &visited)
{
  std::stack<size_t> st;
  st.push(n);
  while(!st.empty())
  {
    size_t k = st.top();
    st.pop();
    visited[k] = true;
    const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[k];
    if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && !visited[qnnn.node_m00_mm]) st.push(qnnn.node_m00_mm);
    if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && !visited[qnnn.node_m00_pm]) st.push(qnnn.node_m00_pm);
    if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && !visited[qnnn.node_p00_mm]) st.push(qnnn.node_p00_mm);
    if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && !visited[qnnn.node_p00_pm]) st.push(qnnn.node_p00_pm);

    if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && !visited[qnnn.node_0m0_mm]) st.push(qnnn.node_0m0_mm);
    if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && !visited[qnnn.node_0m0_pm]) st.push(qnnn.node_0m0_pm);
    if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && !visited[qnnn.node_0p0_mm]) st.push(qnnn.node_0p0_mm);
    if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && !visited[qnnn.node_0p0_pm]) st.push(qnnn.node_0p0_pm);

    /* check connected ghost island and add to list if new */
    if(qnnn.node_m00_mm>=nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && !contains(connected, island_number_p[qnnn.node_m00_mm])) connected.push_back(island_number_p[qnnn.node_m00_mm]);
    if(qnnn.node_m00_pm>=nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && !contains(connected, island_number_p[qnnn.node_m00_pm])) connected.push_back(island_number_p[qnnn.node_m00_pm]);
    if(qnnn.node_p00_mm>=nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && !contains(connected, island_number_p[qnnn.node_p00_mm])) connected.push_back(island_number_p[qnnn.node_p00_mm]);
    if(qnnn.node_p00_pm>=nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && !contains(connected, island_number_p[qnnn.node_p00_pm])) connected.push_back(island_number_p[qnnn.node_p00_pm]);

    if(qnnn.node_0m0_mm>=nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && !contains(connected, island_number_p[qnnn.node_0m0_mm])) connected.push_back(island_number_p[qnnn.node_0m0_mm]);
    if(qnnn.node_0m0_pm>=nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && !contains(connected, island_number_p[qnnn.node_0m0_pm])) connected.push_back(island_number_p[qnnn.node_0m0_pm]);
    if(qnnn.node_0p0_mm>=nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && !contains(connected, island_number_p[qnnn.node_0p0_mm])) connected.push_back(island_number_p[qnnn.node_0p0_mm]);
    if(qnnn.node_0p0_pm>=nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && !contains(connected, island_number_p[qnnn.node_0p0_pm])) connected.push_back(island_number_p[qnnn.node_0p0_pm]);
  }
}


void my_p4est_epitaxy_t::compute_islands_numbers()
{
  int nb_islands_total = 0;
  int proc_padding = 1e6;
  for(unsigned int level=0; level<phi.size(); ++level)
  {
    Vec loc;
    ierr = VecGhostGetLocalForm(island_number[level], &loc); CHKERRXX(ierr);
    ierr = VecSet(loc, -1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(island_number[level], &loc); CHKERRXX(ierr);

    /* first everyone compute the local numbers */
    std::vector<int> nb_islands(p4est->mpisize);
    nb_islands[p4est->mpirank] = p4est->mpirank*proc_padding;

    const double *phi_p;
    ierr = VecGetArrayRead(phi[level], &phi_p); CHKERRXX(ierr);

    double *island_number_p;
    ierr = VecGetArray(island_number[level], &island_number_p); CHKERRXX(ierr);

    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      if(phi_p[n]>0 && island_number_p[n]<0)
      {
        fill_island(phi_p, island_number_p, nb_islands[p4est->mpirank], n);
        nb_islands[p4est->mpirank]++;
      }
    }
    ierr = VecGhostUpdateBegin(island_number[level], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      if(phi_p[n]>0 && island_number_p[n]<0)
      {
        fill_island(phi_p, island_number_p, nb_islands[p4est->mpirank], n);
        nb_islands[p4est->mpirank]++;
      }
    }
    ierr = VecGhostUpdateEnd(island_number[level], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* get remote number of islands to prepare graph communication structure */
    int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &nb_islands[0], 1, MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    /* compute offset for each process */
    std::vector<int> proc_offset(p4est->mpisize+1);
    proc_offset[0] = 0;
    for(int p=0; p<p4est->mpisize; ++p)
      proc_offset[p+1] = proc_offset[p] + (nb_islands[p]%proc_padding);

    /* build a local graph with
     *   - vertices = island number
     *   - edges = connected islands
     * in order to simplify the communications, the graph is stored as a full matrix. Given the sparsity, this can be optimized ...
     */
    int nb_islands_g = proc_offset[p4est->mpisize];
    std::vector<int> graph(nb_islands_g*nb_islands_g, 0);
    /* note that the only reason this is double and not int is that Petsc works with doubles, can't do Vec of int ... */
    std::vector<double> connected;
    std::vector<bool> visited(nodes->num_owned_indeps, false);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(island_number_p[n]>=0 && !visited[n])
      {
        /* find the connected islands and add the connection information to the graph */
        find_connected_ghost_islands(phi_p, island_number_p, n, connected, visited);
        for(unsigned int i=0; i<connected.size(); ++i)
        {
          int local_id = proc_offset[p4est->mpirank]+static_cast<int>(island_number_p[n])%proc_padding;
          int remote_id = proc_offset[static_cast<int>(connected[i])/proc_padding] + (static_cast<int>(connected[i])%proc_padding);
          graph[nb_islands_g*local_id + remote_id] = 1;
        }

        connected.clear();
      }
    }

    std::vector<int> rcvcounts(p4est->mpisize);
    std::vector<int> displs(p4est->mpisize);
    for(int p=0; p<p4est->mpisize; ++p)
    {
      rcvcounts[p] = (nb_islands[p]%proc_padding) * nb_islands_g;
      displs[p] = proc_offset[p]*nb_islands_g;
    }

    mpiret = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &graph[0], &rcvcounts[0], &displs[0], MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    /* now we can color the graph connecting the islands, and thus obtain a unique numbering for all the islands */
    std::vector<int> graph_numbering(nb_islands_g,-1);
    std::stack<int> st;
    nb_islands_per_level[level] = 0;
    for(int i=0; i<nb_islands_g; ++i)
    {
      if(graph_numbering[i]==-1)
      {
        st.push(i);
        while(!st.empty())
        {
          int k = st.top();
          st.pop();
          graph_numbering[k] = nb_islands_total;
          for(int j=0; j<nb_islands_g; ++j)
          {
            int nj = k*nb_islands_g+j;
            if(graph[nj] && graph_numbering[j]==-1)
              st.push(j);
          }
        }
        nb_islands_total++;
        nb_islands_per_level[level]++;
      }
    }

    /* and finally assign the correct number to the islands of this level */
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      if(island_number_p[n]>=0)
      {
        int index = proc_offset[static_cast<int>(island_number_p[n])/proc_padding] + (static_cast<int>(island_number_p[n])%proc_padding);
        island_number_p[n] = graph_numbering[index];
      }
    }

    ierr = VecRestoreArrayRead(phi[level], &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(island_number[level], &island_number_p); CHKERRXX(ierr);
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

  compute_islands_numbers();

  my_p4est_level_set_t ls(ngbd);

  for(unsigned int level=0; level<phi.size(); ++level)
  {
    /* first compute the normal velocity for this level */
    double *vn_p;
    ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);
    double *v_p[2];
    ierr = VecGetArray(v[0][level], &v_p[0]); CHKERRXX(ierr);
    ierr = VecGetArray(v[1][level], &v_p[1]); CHKERRXX(ierr);
    const double *phi_p;
    ierr = VecGetArrayRead(phi[level], &phi_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
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
      const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
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

    /* for each color/island, compute average velocity */
    const double *island_number_p;
    ierr = VecGetArrayRead(island_number[level], &island_number_p); CHKERRXX(ierr);

    for(int island=0; island<nb_islands_per_level[level]; ++island)
    {
      /* first build level-set function */
      double *phi_tmp_p;
      ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        phi_tmp_p[n] = island_number_p[n]==island ? -1 : 1;
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
        if(island_number_p[n]==island)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
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
        if(island_number_p[n]==island)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
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
    ierr = VecRestoreArrayRead(island_number[level], &island_number_p); CHKERRXX(ierr);

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
  ierr = VecDestroy(vn); CHKERRXX(ierr);
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
  splitting_criteria_t* sp = (splitting_criteria_t*)p4est->user_pointer;
  p4est_t *p4est_np1 = my_p4est_new(p4est->mpicomm, p4est->connectivity, 0, NULL, (void*)sp);
  for(int lvl=0; lvl<sp->min_lvl; ++lvl)
  {
    my_p4est_refine(p4est_np1, P4EST_FALSE, refine_every_cell, NULL);
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
  }
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_interpolation_nodes_t interp(ngbd);
  double xyz[P4EST_DIM];

  if(phi.size()>0)
  {
    my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

    sl.update_p4est(v, dt_n, phi);

    /* check if the lowest level disappeared, and remove the corresponding data if needed */
    double phi_min;
    ierr = VecMin(phi[0], PETSC_NULL, &phi_min); CHKERRXX(ierr);
    if(phi_min>0)
    {
      ierr = VecDestroy(phi[0]); CHKERRXX(ierr);
      phi.erase(phi.begin());

      ierr = VecDestroy(rho[0]); CHKERRXX(ierr);
      rho.erase(rho.begin());

      ierr = VecDestroy(v[0][0]); CHKERRXX(ierr);
      v[0].erase(v[0].begin());
      ierr = VecDestroy(v[1][0]); CHKERRXX(ierr);
      v[1].erase(v[1].begin());

      ierr = VecDestroy(island_number[0]); CHKERRXX(ierr);
      island_number.erase(island_number.begin());

      nb_islands_per_level.resize(phi.size());

      nb_levels_deleted++;
    }
  }


  /* interpolate the quantities on the new grid */
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
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v[0][i]); CHKERRXX(ierr);
    ierr = VecDestroy(v[1][i]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v[1][i]); CHKERRXX(ierr);
  }

  for(unsigned int i=0; i<island_number.size(); ++i)
  {
    ierr = VecDestroy(island_number[i]); CHKERRXX(ierr);
    ierr = VecDuplicate(rho_g, &island_number[i]); CHKERRXX(ierr);
  }

  if(bc_type==ROBIN)
  {
    ierr = VecDestroy(robin_coef); CHKERRXX(ierr);
    ierr = VecDuplicate(rho_g, &robin_coef); CHKERRXX(ierr);
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
  compute_phi_g();
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
        rhs_p[n] = rho_p[n] + dt_n*(F - 1*new_island*2*D*sigma1*rho_sqr_avg);
        //        rhs_p[n] = rho_p[n] + dt_n*(F - .1*new_island*2*D*sigma1*SQR(rho_p[n]));

        phi_i_p[n] = -4*L;
        if(level<phi.size()) phi_i_p[n] = phi_p[level][n];
        if(level>0)          phi_i_p[n] = MAX(phi_i_p[n], -phi_p[level-1][n]);
      }

      ierr = VecRestoreArrayRead(rho[level], &rho_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_i, &phi_i_p); CHKERRXX(ierr);

      my_p4est_poisson_nodes_t solver(ngbd);
      solver.set_phi(phi_i);
      solver.set_diagonal(1);
      solver.set_mu(dt_n*D);
      solver.set_rhs(rhs);

      BoundaryConditions2D bc;
      bc.setInterfaceValue(zero);
      if(bc_type==DIRICHLET)
      {
        bc.setInterfaceType(DIRICHLET);
      }
      else if(bc_type==ROBIN)
      {
        double *robin_coef_p;
        ierr = VecGetArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
        double Dn;
        for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        {

          if     (level==0)          Dn = Dm;
          else if(level==phi.size()) Dn = Dp;
          else                       Dn = fabs(phi_p[level-1][n])<fabs(phi_p[level  ][n]) ? Dp : Dm;

          robin_coef_p[n] = Dn/(D-Dn);
        }
        ierr = VecRestoreArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);

        bc.setInterfaceType(ROBIN);
        solver.set_robin_coef(robin_coef);
      }
      else
        throw std::invalid_argument("invalid boundary condition type for rho. Choose either dirichlet or robin.");
      solver.set_bc(bc);

      solver.solve(rho_np1[level]);

      /* shift the solution because b.c. is drho/dn + D'/(D-D') (rho - rho_eq) = 0 */
      if(bc_type==ROBIN)
      {
        double *rho_np1_p;
        ierr = VecGetArray(rho_np1[level], &rho_np1_p); CHKERRXX(ierr);
        for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        {
          rho_np1_p[n] += rho_eq;
        }
        ierr = VecRestoreArray(rho_np1[level], &rho_np1_p); CHKERRXX(ierr);
      }

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
    /* NOTE: the new islands are disks, in the articles they are squares ... */
    double r = MAX(2*MIN(dxyz[0],dxyz[1]), sqrt(2/PI)*lattice_spacing);
    /* the level of the new island */
    unsigned int level;

    /* first island created */
    if(phi.size()==0)
    {
      /* select nucleation point */
      xc = ((double)rand()/RAND_MAX)*L;
      yc = ((double)rand()/RAND_MAX)*L;
      level = 0;
    }
    /* islands already exist */
    else
    {
      /* find the max value of rho */
      const double *phi_g_p, *rho_g_p;
      ierr = VecGetArrayRead(phi_g, &phi_g_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(rho_g, &rho_g_p); CHKERRXX(ierr);

      std::vector<const double*> phi_p(phi.size());
      for(unsigned int i=0; i<phi.size(); ++i)
      {
        ierr = VecGetArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
      }

//#if defined(COMET) || defined(STAMPEDE)
      boost::mt19937 rng;
      rng.seed(time(NULL));
//      rng.seed(0);
      boost::normal_distribution<> distribution(1,.02);
      boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(rng, distribution);
//#else
//      std::default_random_engine generator(time(NULL));
//      /* create a gaussian noise generator, constructor(mean, standard deviation) */
//      std::normal_distribution<double> distribution(1,.1);
//#endif
      double phi_c;

      /* find the nucleation point, maximum of (rho * gaussian_perturbation) */
      std::vector<double> comm(5*p4est->mpisize);
      do{
        comm[5*p4est->mpirank] = 0;

        for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
        {
          if(!one_level_only || phi_g_p[n]<0)
          {
//#if defined(COMET) || defined(STAMPEDE)
						double rho_perturb = SQR(rho_g_p[n])*dist();
//#else
//            double rho_perturb = SQR(rho_g_p[n])*distribution(generator);
//#endif

            if(rho_perturb > comm[5*p4est->mpirank])
            {
              comm[5*p4est->mpirank + 0] = rho_perturb;
              comm[5*p4est->mpirank + 1] = node_x_fr_n(n, p4est, nodes);
              comm[5*p4est->mpirank + 2] = node_y_fr_n(n, p4est, nodes);
              comm[5*p4est->mpirank + 3] = phi_g_p[n];

              /* find the level of the newly nucleated island */
              unsigned int l = 0;
              while(l<phi.size() && phi_p[l][n]>0) ++l;
              comm[5*p4est->mpirank + 4] = l;
            }
          }
        }

        int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &comm[0], 5, MPI_DOUBLE, p4est->mpicomm); SC_CHECK_MPI(mpiret);

        int rank = 0;
        for(int p=1; p<p4est->mpisize; ++p)
        {
          if(comm[5*p]>comm[5*rank])
            rank = p;
        }

        xc    = comm[5*rank+1];
        yc    = comm[5*rank+2];
        phi_c = comm[5*rank+3];
        level = comm[5*rank+4];
      } while(fabs(phi_c)<1.2*r);

      ierr = VecRestoreArrayRead(phi_g, &phi_g_p  ); CHKERRXX(ierr);
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

    Vec tmp;
    ierr = VecCreateGhostNodes(p4est_new, nodes_new, &tmp); CHKERRXX(ierr);
    interp.set_input(rho_g, quadratic_non_oscillatory);
    interp.interpolate(tmp);
    ierr = VecDestroy(rho_g ); CHKERRXX(ierr);
    rho_g  = tmp;

    for(unsigned int i=0; i<rho.size(); ++i)
    {
      ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
      interp.set_input(rho[i], quadratic_non_oscillatory);
      interp.interpolate(tmp);
      ierr = VecDestroy(rho[i]); CHKERRXX(ierr);
      rho[i] = tmp;
    }

    for(unsigned int i=0; i<phi.size(); ++i)
    {
      ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
      interp.set_input(phi[i], quadratic_non_oscillatory);
      interp.interpolate(tmp);
      ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
      phi[i] = tmp;
    }

    for(unsigned int l=0; l<phi.size(); ++l)
    {
      ierr = VecDestroy(v[0][l]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &v[0][l]); CHKERRXX(ierr);
      ierr = VecDestroy(v[1][l]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &v[1][l]); CHKERRXX(ierr);

      ierr = VecDestroy(island_number[l]); CHKERRXX(ierr);
      ierr = VecDuplicate(rho_g, &island_number[l]); CHKERRXX(ierr);
    }

    /* the island spawns a new level */
    if(level>=phi.size())
    {
      Vec tmp;
      ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
      phi.push_back(tmp);

      double *phi_p;
      ierr = VecGetArray(phi[level], &phi_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes_new->indep_nodes.elem_count; ++n)
      {
        double x = node_x_fr_n(n, p4est_new, nodes_new);
        double y = node_y_fr_n(n, p4est_new, nodes_new);
        double tmp = circle(x,y);
        phi_p[n] = tmp;
      }
      ierr = VecRestoreArray(phi[level], &phi_p); CHKERRXX(ierr);

      Vec loc[2];
      ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
      rho.push_back(tmp);

      ierr = VecGhostGetLocalForm(rho[level  ], &loc[0]); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(rho[level+1], &loc[1]); CHKERRXX(ierr);
      ierr = VecCopy(loc[0], loc[1]); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(rho[level  ], &loc[0]); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(rho[level+1], &loc[1]); CHKERRXX(ierr);

      rho_np1.resize(rho.size());

      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &tmp); CHKERRXX(ierr);
      v[0].push_back(tmp);
      ierr = VecCreateGhostNodes(p4est_new, nodes_new, &tmp); CHKERRXX(ierr);
      v[1].push_back(tmp);

      ierr = VecDuplicate(rho_g, &tmp); CHKERRXX(ierr);
      island_number.push_back(tmp);
      nb_islands_per_level.resize(phi.size());
    }
    /* islands already exist at that level */
    else
    {
      double *phi_p, *rho_p[2];
      ierr = VecGetArray(phi[level  ], &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(rho[level  ], &rho_p[0]); CHKERRXX(ierr);
      ierr = VecGetArray(rho[level+1], &rho_p[1]); CHKERRXX(ierr);

      for(size_t n=0; n<nodes_new->indep_nodes.elem_count; ++n)
      {
        double x = node_x_fr_n(n, p4est_new, nodes_new);
        double y = node_y_fr_n(n, p4est_new, nodes_new);
        double tmp = circle(x,y);
        phi_p[n] = MAX(phi_p[n],tmp);
        if(tmp>0)
        {
          rho_p[1][n] = rho_p[0][n];
        }
      }
      ierr = VecRestoreArray(phi[level  ], &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho[level  ], &rho_p[0]); CHKERRXX(ierr);
      ierr = VecRestoreArray(rho[level+1], &rho_p[1]); CHKERRXX(ierr);
    }

    if(bc_type==ROBIN)
    {
      ierr = VecDestroy(robin_coef); CHKERRXX(ierr);
      ierr = VecDuplicate(rho_g, &robin_coef); CHKERRXX(ierr);
    }

    p4est_destroy(p4est);       p4est = p4est_new;
    p4est_ghost_destroy(ghost); ghost = ghost_new;
    p4est_nodes_destroy(nodes); nodes = nodes_new;
    hierarchy->update(p4est, ghost);
    ngbd->update(hierarchy, nodes);

    my_p4est_level_set_t ls(ngbd);
    for(unsigned int l=0; l<phi.size(); ++l)
      ls.perturb_level_set_function(phi[l], EPS);

    compute_phi_g();

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


double my_p4est_epitaxy_t::compute_coverage()
{
  if(phi.size()==0)
    return 0;

  double area = area_in_negative_domain(p4est, nodes, phi[0]);
  return (L*L-area)/(L*L);
}


void my_p4est_epitaxy_t::compute_statistics()
{
  if(phi.size()==0)
    return;

  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to the desired path before saving the statistics for the simulation.\n"); CHKERRXX(ierr);
    return;
  }

  char name[1000];
  snprintf(name, 1000, "%s/stats2.dat", out_dir);
  FILE *fp = fopen(name, "a");
  if(fp==NULL)
    throw std::invalid_argument("Could not open file for statistics ...");

  compute_islands_numbers();

  double theta = compute_coverage();

//  my_p4est_level_set_t ls(ngbd);

  Vec phi_tmp;
  ierr = VecDuplicate(rho_g, &phi_tmp); CHKERRXX(ierr);

  for(unsigned int level=0; level<phi.size(); ++level)
  {
    /* compute the area of each island */
    const double *island_number_p;
    ierr = VecGetArrayRead(island_number[level], &island_number_p); CHKERRXX(ierr);
    for(int island=0; island<nb_islands_per_level[level]; ++island)
    {
      PetscPrintf(p4est->mpicomm, "checking stats for island %d\n", island);
      /* first build level-set function */
      double *phi_tmp_p;
      ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        phi_tmp_p[n] = island_number_p[n]==island ? -1 : 1;
      }
      ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

//      ls.reinitialize_1st_order_time_2nd_order_space(phi_tmp);

      /* then compute the area of the island */
      double area = area_in_negative_domain(p4est, nodes, phi_tmp);
      if(p4est->mpirank==0)
        fprintf(fp, "%e %e\n", theta, area);
    }
    ierr = VecRestoreArrayRead(island_number[level], &island_number_p); CHKERRXX(ierr);
  }

  fclose(fp);
  ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
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

  /* first build a level-set and island number combining all levels */
  compute_islands_numbers();
  Vec island_number_g;
  ierr = VecDuplicate(rho_g, &island_number_g); CHKERRXX(ierr);
  std::vector<const double*> island_number_p(island_number.size());
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    ierr = VecGetArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);
  }
  double *island_number_g_p;
  ierr = VecGetArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
  for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    island_number_g_p[n] = -1;
    for(unsigned int level=0; level<phi.size(); ++level)
    {
      island_number_g_p[n] = MAX(island_number_g_p[n], island_number_p[level][n]);
    }
  }
  ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    ierr = VecRestoreArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);
  }


  /* since forest is periodic, need to build temporary non-periodic forest for visualization */
  my_p4est_brick_t brick_vis;
  int non_periodic[] = {0, 0};
  p4est_connectivity_t *connectivity_vis = my_p4est_brick_new(brick->nxyztrees, xyz_min, xyz_max, &brick_vis, non_periodic);
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

  Vec island_number_g_vis;
  ierr = VecDuplicate(phi_vis, &island_number_g_vis); CHKERRXX(ierr);
  interp.set_input(island_number_g, linear);
  interp.interpolate(island_number_g_vis);
  ierr = VecDestroy(island_number_g); CHKERRXX(ierr);

  Vec rho_g_vis;
  ierr = VecDuplicate(phi_vis, &rho_g_vis); CHKERRXX(ierr);
  interp.set_input(rho_g, linear);
  interp.interpolate(rho_g_vis);

  /* also export the level of each island */
  std::vector<int> island_offset_per_level(phi.size()+1);
  for(unsigned int i=0; i<phi.size(); ++i)
    island_offset_per_level[i+1] = island_offset_per_level[i]+nb_islands_per_level[i];

  Vec island_level_vis;
  ierr = VecDuplicate(phi_vis, &island_level_vis); CHKERRXX(ierr);
  double *island_level_vis_p;
  ierr = VecGetArray(island_level_vis, &island_level_vis_p); CHKERRXX(ierr);
  const double *island_number_g_v_p;
  ierr = VecGetArrayRead(island_number_g_vis, &island_number_g_v_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
  {
    unsigned int level = 0;
    while(level<phi.size() && island_offset_per_level[level]<=island_number_g_v_p[n])
      level++;
    island_level_vis_p[n] = level+nb_levels_deleted;
  }

  const double *phi_v_p, *rho_g_v_p;
  ierr = VecGetArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
  ierr = VecGetArrayRead(rho_g_vis, &rho_g_v_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(p4est_vis, nodes_vis, ghost_vis, P4EST_TRUE, P4EST_TRUE, 4, 0, name,
                         VTK_POINT_DATA, "phi", phi_v_p,
                         VTK_POINT_DATA, "rho", rho_g_v_p,
                         VTK_POINT_DATA, "island_number", island_number_g_v_p,
                         VTK_POINT_DATA, "level", island_level_vis_p);

  ierr = VecRestoreArray(island_level_vis, &island_level_vis_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(rho_g_vis, &rho_g_v_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(island_number_g_vis, &island_number_g_v_p); CHKERRXX(ierr);

  ierr = VecDestroy(island_level_vis); CHKERRXX(ierr);
  ierr = VecDestroy(island_number_g_vis); CHKERRXX(ierr);
  ierr = VecDestroy(rho_g_vis); CHKERRXX(ierr);
  ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
  p4est_nodes_destroy(nodes_vis);
  p4est_ghost_destroy(ghost_vis);
  p4est_destroy(p4est_vis);
  my_p4est_brick_destroy(connectivity_vis, &brick_vis);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", name);
}
