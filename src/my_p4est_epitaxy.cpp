#include "my_p4est_epitaxy.h"

#include <time.h>
#include <stack>
/* c++11 support for the "random" procedures is broken for some reason with icpc / gcc 4.4.7 or gcc 4.8.0 */
//#if defined(COMET) || defined(STAMPEDE)
//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>
//#else
//#include <random>
//#endif
#include <algorithm>
#include <src/my_p4est_utils.h>
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
    ierr = VecDuplicate(rho_g, &capture_zone); CHKERRXX(ierr);
  //  ierr = VecDuplicate(rho_g, &mask); CHKERRXX(ierr);
    
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
    
    ierr = VecGhostGetLocalForm(capture_zone, &loc); CHKERRXX(ierr);
    ierr = VecSet(loc, -1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(capture_zone, &loc); CHKERRXX(ierr);
    
//    ierr = VecGhostGetLocalForm(mask, &loc); CHKERRXX(ierr);
//    ierr = VecSet(loc, -1); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(mask, &loc); CHKERRXX(ierr);
    
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
    dt_n = MIN(dxyz[0],dxyz[1]);
    
    nb_levels_deleted = 0;
    
    island_nucleation_scaling = L*L;
    
    deltaNuc = 0.0;
    
    //srand(1);
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


double my_p4est_epitaxy_t::getRhoAverage() {
    return rho_avg;
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
            phi_g_p[n] = ( level%2==0 ? MAX(phi_g_p[n],phi_p[level][n]) : MIN(phi_g_p[n],-phi_p[level][n]) );
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
         *   - edges    = connected islands
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


void my_p4est_epitaxy_t::set_dt(double dt_user)
{
    dt_n = dt_user;
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
    // Frederic: got rid of the following statement to fix a bug:
    //    dt_n *= 4;                // This is to set back to the dt chosen by the user. Then the suite of algorithms will reduce it if needed. There is a bug here!
    
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
        
        if(vmax>EPS)
        {
            dt_n = MIN(dt_n, MIN(dxyz[0],dxyz[1])/vmax);
        }
    }
    
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
 * solve the equation for the adatom density for one time step dt_n
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
                rhs_p[n] = rho_p[n] + dt_n*(F - 2*D*sigma1*rho_sqr_avg);
                
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
    
    Vec ones;
    ierr = VecDuplicate(rho[0], &ones); CHKERRXX(ierr);
    Vec loc;
    ierr = VecGhostGetLocalForm(ones, &loc); CHKERRXX(ierr);
    ierr = VecSet(loc, -1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(ones, &loc); CHKERRXX(ierr);
    
    rho_avg = integrate_over_negative_domain(p4est, nodes, ones, rho_g)/(L*L);
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
    if (sigma1_np1 < 1) sigma1_np1 = 1;   // Sigma1 is set to 1 at the minimum.
    
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
    double sigma1_np1 = 4*PI/log((1/alpha)*rho_avg_np1*D/F);
    if (sigma1_np1 < 1) sigma1_np1 = 1;   // Sigma1 is set to 1 at the minimum.
    
    Vec ones;
    ierr = VecDuplicate(rho[0], &ones); CHKERRXX(ierr);
    Vec loc;
    ierr = VecGhostGetLocalForm(ones, &loc); CHKERRXX(ierr);
    ierr = VecSet(loc, -1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(ones, &loc); CHKERRXX(ierr);
    rho_avg_np1 = integrate_over_negative_domain(p4est, nodes, ones, rho_g)/(L*L);
    
    /* check for new island */
    if(floor(Nuc*island_nucleation_scaling) != floor(Nuc_np1*island_nucleation_scaling))
    {
        ierr = PetscPrintf(p4est->mpicomm, "Nucleating new island!\n"); CHKERRXX(ierr);
        double xc, yc;
        
        /* NOTE: the new islands are disks, in the articles they are squares ... */
        double r = MAX(2*MIN(dxyz[0],dxyz[1]), sqrt(2/PI)*lattice_spacing);
        
        /* the level of the new island */
        unsigned int level;
        
        /* first island created, i.e. we seed an island on the substrate */
        if(phi.size()==0)
        {
            /* select nucleation point */
            xc = ((double)rand()/RAND_MAX)*L;
            yc = ((double)rand()/RAND_MAX)*L;
            level = 0; ierr = PetscPrintf(p4est->mpicomm, "First island created! xc=%g and yc=%g\n", xc, yc); CHKERRXX(ierr);
        }
        /* islands already exist */
        else
        {
            //PAM: compute area*rho^2 for each node
            Vec rho_sqr1;
            ierr = VecDuplicate(rho[0], &rho_sqr1); CHKERRXX(ierr);
            double *rho_sqr_p;
            const double  *rho_g_p1;
            ierr = VecGetArray(rho_sqr1, &rho_sqr_p); CHKERRXX(ierr);
            ierr = VecGetArrayRead(rho_g, &rho_g_p1); CHKERRXX(ierr);
            for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
            {
                rho_sqr_p[n] = SQR(rho_g_p1[n]);
            }
            ierr = VecRestoreArrayRead(rho_g, &rho_g_p1); CHKERRXX(ierr);
            ierr = VecRestoreArray(rho_sqr1, &rho_sqr_p); CHKERRXX(ierr);
            Vec rho_sqr_area;
            ierr = VecDuplicate(rho[0], &rho_sqr_area); CHKERRXX(ierr);
            multiply_values_by_area(p4est, nodes, rho_sqr1, rho_sqr_area);
            
            
            ierr = VecDestroy(rho_sqr1); CHKERRXX(ierr);
            // compute partial sum in parallel for rho_sqr_area vector
            Vec rho_sqr_cum;
            ierr = VecDuplicate(rho_sqr_area, &rho_sqr_cum); CHKERRXX(ierr);
            
            double *rho_sqr_area_p, *rho_sqr_cum_p;
            ierr = VecGetArray(rho_sqr_area, &rho_sqr_area_p); CHKERRXX(ierr);
            ierr = VecGetArray(rho_sqr_cum, &rho_sqr_cum_p); CHKERRXX(ierr);
            
            for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
            {
                if(n==0)
                {
                    rho_sqr_cum_p[0] = rho_sqr_area_p[0];
                }
                else
                {
                    rho_sqr_cum_p[n] = rho_sqr_cum_p[n-1] + rho_sqr_area_p[n];
                    
                }
            }
            
            ierr = VecRestoreArray(rho_sqr_area, &rho_sqr_area_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(rho_sqr_cum, &rho_sqr_cum_p); CHKERRXX(ierr);
            
            ierr = VecGetArray(rho_sqr_cum, &rho_sqr_cum_p); CHKERRXX(ierr);
            double last;
            std::vector<double> last_store(p4est->mpisize);
            
            
            last_store[p4est->mpirank] = 0;
            last = rho_sqr_cum_p[nodes->indep_nodes.elem_count-1];
            last_store[p4est->mpirank] = last;
            int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &last_store[0], 1, MPI_DOUBLE, p4est->mpicomm); SC_CHECK_MPI(mpiret);
            for(int p=1; p<p4est->mpisize; ++p)
            {
                last_store[p] += last_store[p-1];
                
            }
            
            
            
            ierr = VecRestoreArray(rho_sqr_cum, &rho_sqr_cum_p); CHKERRXX(ierr);
            PetscInt begin, end;
            
            
            if(p4est->mpirank==0)
                last = 0;
            else
                last = last_store[p4est->mpirank-1];
            
            
            
            VecGetArray(rho_sqr_cum, &rho_sqr_cum_p);
            for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
            {
                rho_sqr_cum_p[n] += last;
            }
            VecRestoreArray(rho_sqr_cum, &rho_sqr_cum_p);
            
            
            
            
            // scale the cumulative sum by its maximum
            PetscReal rho_max;
            PetscInt imax;
            ierr = VecMax(rho_sqr_cum,&imax,&rho_max); CHKERRXX(ierr);
            
            Vec l;
            ierr = VecGhostGetLocalForm(rho_sqr_cum, &l); CHKERRXX(ierr);
            ierr = VecScale(l, 1./rho_max); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(rho_sqr_cum, &l); CHKERRXX(ierr);
            //ierr = VecDestroy(l); CHKERRXX(ierr);
            // check the output
            //VecView(rho_sqr_cum,PETSC_VIEWER_STDOUT_WORLD);
            
            
            
            
            /* find the max value of rho */
            const double *phi_g_p, *rho_p2;
            ierr = VecGetArrayRead(phi_g, &phi_g_p); CHKERRXX(ierr);
            
            std::vector<const double*> phi_p(phi.size());
            for(unsigned int i=0; i<phi.size(); ++i)
            {
                ierr = VecGetArrayRead(phi[i], &phi_p[i]); CHKERRXX(ierr);
            }
            
            
            double phi_c;
            
            /* find the nucleation point, maximum of (rho * gaussian_perturbation) */
            std::vector<double> comm(5*p4est->mpisize);
            
            double random_nb;
            do{
                Vec rho_sqr_cum_tmp;
                ierr = VecDuplicate(rho_sqr_cum, &rho_sqr_cum_tmp);
                Vec l_tmp;
                ierr = VecGhostGetLocalForm(rho_sqr_cum, &l); CHKERRXX(ierr);
                ierr = VecGhostGetLocalForm(rho_sqr_cum_tmp, &l_tmp); CHKERRXX(ierr);
                ierr = VecCopy(l, l_tmp); CHKERRXX(ierr);
                ierr = VecGhostRestoreLocalForm(rho_sqr_cum, &l); CHKERRXX(ierr);
                ierr = VecGhostRestoreLocalForm(rho_sqr_cum_tmp, &l_tmp); CHKERRXX(ierr);
                //VecView(rho_sqr_cum_tmp,PETSC_VIEWER_STDOUT_WORLD);
                
                // choose the nucleation point: the nearest point to random number
                random_nb = -((double)rand()/RAND_MAX);
                MPI_Barrier(p4est->mpicomm);
                MPI_Bcast(&random_nb, 1, MPI_DOUBLE, 0, p4est->mpicomm);
                MPI_Barrier(p4est->mpicomm);
                
                
                
                // produce the vector to find its maximum
                double *rho_sqr_cum_tmp_p;
                ierr = VecGetArray(rho_sqr_cum_tmp, &rho_sqr_cum_tmp_p); CHKERRXX(ierr);
                for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
                {
                    rho_sqr_cum_tmp_p[n] = 1./(1+fabs(rho_sqr_cum_tmp_p[n]+random_nb));
                }
                ierr = VecRestoreArray(rho_sqr_cum_tmp, &rho_sqr_cum_tmp_p); CHKERRXX(ierr);
                
                //PetscPrintf(p4est->mpicomm, "random_nb: %g\n", random_nb);
                //VecView(rho_sqr_cum_tmp,PETSC_VIEWER_STDOUT_WORLD);
                ierr = VecGetArrayRead(rho_sqr_cum_tmp, &rho_p2); CHKERRXX(ierr);
                
                
                comm[5*p4est->mpirank] = 0;
                for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
                {
                    if(!one_level_only || phi_g_p[n]<0)
                    {
                        double rho_perturb = rho_p2[n];
                        
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
                
                
                mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &comm[0], 5, MPI_DOUBLE, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                
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
               
                //PetscPrintf(p4est->mpicomm, "final value (must be close to 1.0) is: %g\n", comm[5*rank]);
                ierr = VecRestoreArrayRead(rho_sqr_cum_tmp , &rho_p2); CHKERRXX(ierr);
                ierr = VecDestroy(rho_sqr_cum_tmp); CHKERRXX(ierr);
                
            } while(fabs(phi_c)<1.2*r);
            
            ierr = VecRestoreArrayRead(phi_g, &phi_g_p  ); CHKERRXX(ierr);
            ierr = VecDestroy(rho_sqr_cum); CHKERRXX(ierr);
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
        /* islands already exist at that level, thus we need to seed the new island on top of it */
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
            ierr = VecRestoreArray(phi[level  ], &phi_p   ); CHKERRXX(ierr);
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
        return false;
    }
    
    /* if the time step passed the test, then update the quantities */
    for(unsigned int i=0; i<rho.size(); ++i)
    {
        ierr = VecDestroy(rho[i]); CHKERRXX(ierr);
        rho[i] = rho_np1[i];
    }
    
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
    snprintf(name, 1000, "%s/statistics_CSD.dat", out_dir);
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
    
    //PAM
    Vec capture_zone_vis;
    ierr = VecDuplicate(phi_vis, &capture_zone_vis); CHKERRXX(ierr);
    interp.set_input(capture_zone, linear);
    interp.interpolate(capture_zone_vis);
    
 /*   Vec mask_vis;
    ierr = VecDuplicate(phi_vis, &mask_vis); CHKERRXX(ierr);
    interp.set_input(mask, linear);
    interp.interpolate(mask_vis); */
    //
    
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
    
    const double *phi_v_p, *rho_g_v_p, *capture_zone_v_p, *mask_v_p;
    ierr = VecGetArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
    ierr = VecGetArrayRead(rho_g_vis, &rho_g_v_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(capture_zone_vis  , &capture_zone_v_p  ); CHKERRXX(ierr);
  //  ierr = VecGetArrayRead(mask_vis  , &mask_v_p  ); CHKERRXX(ierr);
    
    my_p4est_vtk_write_all(p4est_vis, nodes_vis, ghost_vis, P4EST_TRUE, P4EST_TRUE, 5, 0, name,
                           VTK_POINT_DATA, "phi", phi_v_p,
                           VTK_POINT_DATA, "rho", rho_g_v_p,
                           VTK_POINT_DATA, "island_number", island_number_g_v_p,
                           VTK_POINT_DATA, "level", island_level_vis_p,
                           VTK_POINT_DATA, "capture_zone", capture_zone_v_p);
     //                      VTK_POINT_DATA, "mask", mask_v_p);
    
    ierr = VecRestoreArray(island_level_vis, &island_level_vis_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(rho_g_vis, &rho_g_v_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(island_number_g_vis, &island_number_g_v_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(capture_zone_vis  , &capture_zone_v_p  ); CHKERRXX(ierr);
 //   ierr = VecRestoreArrayRead(mask_vis  , &mask_v_p  ); CHKERRXX(ierr);
    
    ierr = VecDestroy(island_level_vis); CHKERRXX(ierr);
    ierr = VecDestroy(island_number_g_vis); CHKERRXX(ierr);
    ierr = VecDestroy(rho_g_vis); CHKERRXX(ierr);
    ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
    ierr = VecDestroy(capture_zone_vis); CHKERRXX(ierr);
   // ierr = VecDestroy(mask_vis); CHKERRXX(ierr);
    
    p4est_nodes_destroy(nodes_vis);
    p4est_ghost_destroy(ghost_vis);
    p4est_destroy(p4est_vis);
    my_p4est_brick_destroy(connectivity_vis, &brick_vis);
    
    PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", name);
}


void my_p4est_epitaxy_t::multiply_values_by_area(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec input, Vec output)
{
    Vec output_loc;
    ierr = VecGhostGetLocalForm(output, &output_loc); CHKERRXX(ierr);
    ierr = VecSet(output_loc, 0.0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(output, &output_loc); CHKERRXX(ierr);
    
    double *input_ptr, *output_ptr;
    ierr = VecGetArray(input, &input_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(output, &output_ptr); CHKERRXX(ierr);
    
    const p4est_locidx_t *q2n = nodes->local_nodes;
    
    // loop through local quadrants
    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        
        p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
        p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];
        
        double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
        double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
        double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
        double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
        double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
        double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif
        
        for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
        {
            const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
            
            p4est_locidx_t quad_idx_forest = quad_idx + tree->quadrants_offset;
            
            // get volume of a quadrant
            double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
            double dV = (tree_xmax-tree_xmin)*dmin*(tree_ymax-tree_ymin)*dmin/4.0;
#ifdef P4_TO_P8
            dV *= (tree_zmax-tree_zmin)*dmin/2.0;
#endif
            
            // loop through nodes of a quadrant and put weights on those nodes, which are local
            p4est_locidx_t offset = quad_idx_forest*P4EST_CHILDREN;
            for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
            {
                p4est_locidx_t node_idx = q2n[offset + child_idx];
                if (node_idx < nodes->num_owned_indeps)
                    output_ptr[node_idx] += input_ptr[node_idx]*dV;
            }
        }
    }
    // loop through ghosts
    for (p4est_locidx_t ghost_idx = 0; ghost_idx < ghost->ghosts.elem_count; ++ghost_idx)
    {
        // get a ghost quadrant
        p4est_quadrant_t* quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
        
        // get a tree to which the ghost quadrant belongs
        p4est_topidx_t tree_idx = quad->p.piggy3.which_tree;
        
        // get coordinates of the tree
        p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
        p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];
        
        double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
        double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
        double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
        double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
        double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
        double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif
        
        // calculate volume per each node of the ghost quadrant
        double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
        double dV = (tree_xmax-tree_xmin)*dmin*(tree_ymax-tree_ymin)*dmin/4.0;
#ifdef P4_TO_P8
        dV *= (tree_zmax-tree_zmin)*dmin/2.0;
#endif
        
        // loop through nodes of a quadrant and put weights on those nodes, which are local
        p4est_locidx_t offset = (p4est->local_num_quadrants + ghost_idx)*P4EST_CHILDREN;
        for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
        {
            p4est_locidx_t node_idx = q2n[offset + child_idx];
            if (node_idx < nodes->num_owned_indeps)
                output_ptr[node_idx] += input_ptr[node_idx]*dV;
        }
    }
    
    ierr = VecRestoreArray(input, &input_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(output, &output_ptr); CHKERRXX(ierr);
}


void my_p4est_epitaxy_t::compute_capture_zone()
{
if(phi.size()>0)
    {
   // compute island numbers which we need to find their corresponding capture zones 
  	    Vec island_number_g;
            ierr = VecDuplicate(phi_g, &island_number_g); CHKERRXX(ierr);
            compute_islands_numbers();
            
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


    Vec mask, top_mask;
    ierr = VecDuplicate(phi_g, &mask); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_g, &top_mask); CHKERRXX(ierr);
    compute_topmost_layer_mask(mask, top_mask);


    double *mask_p, *rho_g_p, *top_mask_p;
/*
	double *cz_p;
	VecDestroy(capture_zone);
        VecDuplicate(rho_g, &capture_zone);
	 ierr = VecGetArray(top_mask, &top_mask_p); CHKERRXX(ierr);
	ierr = VecGetArray(capture_zone, &cz_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)

    {
	cz_p[n] = top_mask_p[n];
    }
	ierr = VecRestoreArray(top_mask, &top_mask_p); CHKERRXX(ierr);
	ierr = VecRestoreArray(capture_zone, &cz_p); CHKERRXX(ierr);
*/
/*    ierr = VecGetArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);
    ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
    ierr = VecGetArray(top_mask, &top_mask_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
	if (top_mask_p[n]<0 || mask_p[n]==0)
	{
		island_number_g_p[n] = -1;
		rho_g_p[n] *= mask_p[n];
	}
    }

    ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rho_g, &rho_g_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(top_mask, &top_mask_p); CHKERRXX(ierr);
*/
// at this point, the troubles are resolved! We have the masked island numbers (only the top ones!) as well as the penultimate density field! Below, we do capture zone identification on this density field, when we hit the central island we adopt its number to the CZ, if the gradients took us to -1
    
        ierr = PetscPrintf(p4est->mpicomm, "compute capture zone!\n"); CHKERRXX(ierr);
          
        Vec loc1;
        VecDestroy(capture_zone);
        VecDuplicate(rho_g, &capture_zone);
        ierr = VecGhostGetLocalForm(capture_zone, &loc1); CHKERRXX(ierr);
        ierr = VecSet(loc1, -1); CHKERRXX(ierr);
        ierr = VecGhostRestoreLocalForm(capture_zone, &loc1); CHKERRXX(ierr);
	

    
        Vec grad_phi_x, grad_phi_y;
        ierr = VecDuplicate(phi_g, &grad_phi_x); CHKERRXX(ierr);
        ierr = VecDuplicate(phi_g, &grad_phi_y); CHKERRXX(ierr);
   
       
            Vec loc;
            ierr = VecGhostGetLocalForm(grad_phi_x, &loc); CHKERRXX(ierr);
            ierr = VecSet(loc, 0); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(grad_phi_x, &loc); CHKERRXX(ierr);
            
            ierr = VecGhostGetLocalForm(grad_phi_y, &loc); CHKERRXX(ierr);
            ierr = VecSet(loc, 0); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(grad_phi_y, &loc); CHKERRXX(ierr);
            
            double *phi_p, *grad_phi_p_x, *grad_phi_p_y;
            
            
            ierr = VecGetArray(rho_g, &rho_g_p); CHKERRXX(ierr);	//PAM: naming issue for phi_p is actually rho! :)
            
            ierr = VecGetArray(grad_phi_x, &grad_phi_p_x); CHKERRXX(ierr);
            ierr = VecGetArray(grad_phi_y, &grad_phi_p_y); CHKERRXX(ierr);
            for(size_t i=0; i<ngbd->get_layer_size(); ++i)
            {
                p4est_locidx_t n = ngbd->get_layer_node(i);
                const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
                double nx = qnnn.dx_central(rho_g_p);
                double ny = qnnn.dy_central(rho_g_p);
                double norm = sqrt(SQR(nx) + SQR(ny));
                nx = norm<EPS ? 0 : nx/norm;
                ny = norm<EPS ? 0 : ny/norm;
                grad_phi_p_x[n] = -nx;
                grad_phi_p_y[n] = -ny;
            }
            ierr = VecGhostUpdateBegin(grad_phi_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            for(size_t i=0; i<ngbd->get_local_size(); ++i)
            {
                p4est_locidx_t n = ngbd->get_local_node(i);
                const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
                double nx = qnnn.dx_central(rho_g_p);
                double ny = qnnn.dy_central(rho_g_p);
                double norm = sqrt(SQR(nx) + SQR(ny));
                nx = norm<EPS ? 0 : nx/norm;
                grad_phi_p_x[n] = -nx;
            }
            ierr = VecGhostUpdateEnd(grad_phi_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecGhostUpdateBegin(grad_phi_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            for(size_t i=0; i<ngbd->get_local_size(); ++i)
            {
                p4est_locidx_t n = ngbd->get_local_node(i);
                const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
                double nx = qnnn.dx_central(rho_g_p);
                double ny = qnnn.dy_central(rho_g_p);
                double norm = sqrt(SQR(nx) + SQR(ny));
                ny = norm<EPS ? 0 : ny/norm;
                grad_phi_p_y[n] = -ny;
            }
            ierr = VecGhostUpdateEnd(grad_phi_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            
            VecRestoreArray(rho_g, &rho_g_p);		//PAM
            VecRestoreArray(grad_phi_x, &grad_phi_p_x);
            VecRestoreArray(grad_phi_y, &grad_phi_p_y);

            
            ierr = VecGetArray(grad_phi_x, &grad_phi_p_x); CHKERRXX(ierr);
            ierr = VecGetArray(grad_phi_y, &grad_phi_p_y); CHKERRXX(ierr);
            
            // compute the capture zones. first start from the local nodes and label the chains of nodes 
            int proc_padding = 1e6;   // one could use for example = (1000*number of nodes)/(number of processors) instead
            std::vector<int> nb_chains(p4est->mpisize);
            nb_chains[p4est->mpirank] = (p4est->mpirank+1)*proc_padding;
            
            double *cz_p;
            ierr = VecGetArray(capture_zone, &cz_p); CHKERRXX(ierr);
            ierr = VecGetArray(phi_g, &phi_p); CHKERRXX(ierr);
            ierr = VecGetArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);		//PAM
	    ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
    	    ierr = VecGetArray(top_mask, &top_mask_p); CHKERRXX(ierr);
            bool dangling_chain = false;
            for(size_t i=0; i<ngbd->get_layer_size(); ++i)
            {
                p4est_locidx_t n = ngbd->get_layer_node(i);
                if(cz_p[n]>=0)
                    continue;
                double grad_x = grad_phi_p_x[n];
                double grad_y = grad_phi_p_y[n];
                
                dangling_chain = fill_capture_zone(phi_p, island_number_g_p, cz_p, nb_chains[p4est->mpirank], n, grad_x, grad_y, grad_phi_p_x, grad_phi_p_y, top_mask_p, mask_p, dangling_chain);
                if(dangling_chain)
                    nb_chains[p4est->mpirank]++;
            }
            dangling_chain = false;
            ierr = VecGhostUpdateBegin(capture_zone, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            for(size_t i=0; i<ngbd->get_local_size(); ++i)
            {
                p4est_locidx_t n = ngbd->get_local_node(i);
                if(cz_p[n]>=0)
                    continue;
                double grad_x = grad_phi_p_x[n];
                double grad_y = grad_phi_p_y[n];
                
                dangling_chain = fill_capture_zone(phi_p, island_number_g_p, cz_p, nb_chains[p4est->mpirank], n, grad_x, grad_y, grad_phi_p_x, grad_phi_p_y, top_mask_p, mask_p, dangling_chain);
                if(dangling_chain)
                    nb_chains[p4est->mpirank]++;
            }


            ierr = VecGhostUpdateEnd(capture_zone, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecRestoreArray(capture_zone, &cz_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);	//PAM
            ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(top_mask, &top_mask_p); CHKERRXX(ierr);
            // get remote number of chains to prepare graph communication structure
            int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &nb_chains[0], 1, MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);
            
            // compute offset for each process
            std::vector<int> proc_offset(p4est->mpisize+1);
            proc_offset[0] = 0;
            for(int p=0; p<p4est->mpisize; ++p)
                proc_offset[p+1] = proc_offset[p] + (nb_chains[p]%proc_padding);
            
            // build a local graph with
            // - vertices = chain number
            // - edges    = connected chains
            // in order to simplify the communications, the graph is stored as a full matrix. Given the sparsity, this can be optimized ...
            
            ierr = VecGetArray(capture_zone, &cz_p); CHKERRXX(ierr);
            int nb_chains_g = proc_offset[p4est->mpisize];
		
            std::vector<int> graph(nb_chains_g*nb_chains_g, -1);
            // note that the only reason this is double and not int is that Petsc works with doubles, can't do Vec of int ...
            std::vector<double> connected;
            std::vector<bool> visited(nodes->num_owned_indeps, false);
            std::vector<bool> visited_chains(nb_chains[p4est->mpirank]%proc_padding, false);
            
            
            for(size_t i=0; i<ngbd->get_layer_size(); ++i)
            {
                p4est_locidx_t n = ngbd->get_layer_node(i);
                if((cz_p[n]>=proc_padding ) && !visited[n])
                {
                    //PetscPrintf(p4est->mpicomm, "Wrong! %d\n", n);
                    visited_chains[(int) cz_p[n]%proc_padding] =true;
                    double grad_x = grad_phi_p_x[n];
                    double grad_y = grad_phi_p_y[n];
                    // find the connected chains and add the connection information to the graph
                    find_connected_ghost_chains(phi_p, cz_p, n, connected, visited, grad_x, grad_y, grad_phi_p_x, grad_phi_p_y);
                    if(!connected.size()) throw std::runtime_error("graph not fully connected!");
                    for(unsigned int i=0; i<connected.size(); ++i)
                    {
                        //ierr = PetscPrintf(p4est->mpicomm, "debug %g cz_p %g\n", connected[i], cz_p[n]); CHKERRXX(ierr);
                        int local_id = proc_offset[p4est->mpirank]+static_cast<int>(cz_p[n])%proc_padding;
                        int remote_id;
                        if(connected[i]>=proc_padding)
                            remote_id = proc_offset[static_cast<int>(connected[i]-proc_padding)/proc_padding] + (static_cast<int>(connected[i])%proc_padding);
                        else
                            remote_id = local_id; //proc_offset[static_cast<int>(connected[i])/proc_padding] + (static_cast<int>(connected[i])%proc_padding);
                        //	PetscPrintf(p4est->mpicomm, "debug nb_chains_g %d id %d \n", nb_chains_g, remote_id);
                        graph[nb_chains_g*local_id + remote_id] = (int) connected[i];
                        graph[nb_chains_g*remote_id + local_id] = (int) connected[i];
                    }
                    
                    connected.clear();
                }
            }
            
            
            ierr = VecRestoreArray(capture_zone, &cz_p); CHKERRXX(ierr);

            ierr = VecRestoreArray(phi_g, &phi_p); CHKERRXX(ierr);
            VecRestoreArray(grad_phi_x, &grad_phi_p_x);
            VecRestoreArray(grad_phi_y, &grad_phi_p_y);
            
            
            std::vector<int> rcvcounts(p4est->mpisize);
            std::vector<int> displs(p4est->mpisize);
            for(int p=0; p<p4est->mpisize; ++p)
            {
                rcvcounts[p] = (nb_chains[p]%proc_padding) * nb_chains_g;
                displs[p] = proc_offset[p]*nb_chains_g;
            }
            
            mpiret = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &graph[0], &rcvcounts[0], &displs[0], MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);
            
            
            // now we can color the graph connecting the chains, and thus obtain a unique numbering for all the chains
            if(nb_chains_g){
                std::vector<int> graph_numbering(nb_chains_g,-2);
                std::vector<p4est_locidx_t> st;
                for(int i=0; i<nb_chains_g; ++i)
                {
                    if(graph_numbering[i]==-2)
                    {
                        st.push_back(i);
                        while(st.size())
                        {
                            int k = st.back();
                            for(int j=0; j<nb_chains_g; ++j)
                            {
                                int nj = k*nb_chains_g+j;
                                
                                if(graph[nj]>=proc_padding)
                                {
                                    if(std::find(st.begin(), st.end(), j)!=st.end())
                                    {
                                        while(st.size())
                                            st.pop_back();
                                        break;
                                    }
                                    st.push_back(j);
                                    break;
                                } else if(graph[nj]>=0)
                                {
                                    graph_numbering[k] = graph[nj];
                                    graph_numbering[i] = graph[nj];
                                    while(st.size())
                                    {
                                        int g = st.back();
                                        graph_numbering[g] = graph[nj];
                                        st.pop_back();
                                    }
                                    break;
                                } else if(j==nb_chains_g-1){
                                    while(st.size())
                                    {
                                        //	for (int q = 0; q <nb_chains_g; ++q)
                                        //	{
                                        //		int jj = st.top()*nb_chains_g+q;
                                        //		PetscPrintf(p4est->mpicomm, "%d \t ", graph[jj]);
                                        //	}
                                        //	PetscPrintf(p4est->mpicomm, " : %d \t nb_chains %d\n", st.top(), nb_chains_g);
                                        st.pop_back();
                                    }
                                    //throw std::runtime_error("Wrong graph numbering!");
                                }
                                
                            }
                        }
                        st.clear();
                    }
                }
                
                // and finally assign the correct number to the chains

                ierr = VecGetArray(capture_zone, &cz_p); CHKERRXX(ierr);
                VecGetArray(grad_phi_x, &grad_phi_p_x);
                VecGetArray(grad_phi_y, &grad_phi_p_y);
		
                for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
                {
		    
                    if(cz_p[n]>=proc_padding)
                    {
                        int index = proc_offset[static_cast<int>(cz_p[n]-proc_padding)/proc_padding] + (static_cast<int>(cz_p[n])%proc_padding);
                        if(graph_numbering[index]>=proc_padding || graph_numbering[index]==-2)
                            rename_dangling_chains(cz_p, -1, n, grad_phi_p_x, grad_phi_p_y);
                        else
                            rename_dangling_chains(cz_p, graph_numbering[index], n, grad_phi_p_x, grad_phi_p_y);
                    } 
		   
                    
                }

                ierr = VecRestoreArray(capture_zone, &cz_p); CHKERRXX(ierr);
                VecRestoreArray(grad_phi_x, &grad_phi_p_x);
                VecRestoreArray(grad_phi_y, &grad_phi_p_y); 
            } // end of coloring capture zone!
            
	
	VecDestroy(island_number_g);
        VecDestroy(grad_phi_x);
        VecDestroy(grad_phi_y);
	VecDestroy(mask);
  	VecDestroy(top_mask);

	//remove_holes();
  } 
}


p4est_locidx_t my_p4est_epitaxy_t::choose_next_node(const my_p4est_node_neighbors_t *ngbd, p4est_locidx_t n, double grad_x, double grad_y)
{
    /*
     * This function finds the next node from the list of neighbors which is closest to the direction of the gradient.
     */
    if(fabs(grad_x)<EPS && fabs(grad_y)<EPS)
        return n;
    
    
    const quad_neighbor_nodes_of_node_t qnnn = ngbd->get_neighbors(n);
    
    double d_m00 = qnnn.d_m00; double d_m00_m0 = qnnn.d_m00_m0; double d_m00_p0 = qnnn.d_m00_p0;
    double d_p00 = qnnn.d_p00; double d_p00_m0 = qnnn.d_p00_m0; double d_p00_p0 = qnnn.d_p00_p0;
    double d_0m0 = qnnn.d_0m0; double d_0m0_m0 = qnnn.d_0m0_m0; double d_0m0_p0 = qnnn.d_0m0_p0;
    double d_0p0 = qnnn.d_0p0; double d_0p0_m0 = qnnn.d_0p0_m0; double d_0p0_p0 = qnnn.d_0p0_p0;
    
    p4est_locidx_t node_m00_mm = qnnn.node_m00_mm; p4est_locidx_t node_m00_pm = qnnn.node_m00_pm;
    p4est_locidx_t node_p00_mm = qnnn.node_p00_mm; p4est_locidx_t node_p00_pm = qnnn.node_p00_pm;
    p4est_locidx_t node_0m0_mm = qnnn.node_0m0_mm; p4est_locidx_t node_0m0_pm = qnnn.node_0m0_pm;
    p4est_locidx_t node_0p0_mm = qnnn.node_0p0_mm; p4est_locidx_t node_0p0_pm = qnnn.node_0p0_pm;
    
    double theta_grad = atan2(grad_y, grad_x);
    if (theta_grad < 0)
        theta_grad += 2*PI;
    
    
    double theta_p00_pm = 20*PI;
    double theta_p00_mm = 20*PI;
    double theta_0p0_pm = 20*PI;
    double theta_0p0_mm = 20*PI;
    double theta_m00_pm = 20*PI;
    double theta_m00_mm = 20*PI;
    double theta_0m0_mm = 20*PI;
    double theta_0m0_pm = 20*PI;
    
    if (d_p00>0)
    {
        theta_p00_pm = atan (d_p00_p0/d_p00);
        theta_p00_mm = 2*PI - atan(d_p00_m0/d_p00);
    }
    if (d_0p0>0)
    {
        theta_0p0_pm = PI/2 - atan(d_0p0_p0/d_0p0);
        theta_0p0_mm = PI - atan(d_0p0_m0/d_0p0);
    }
    if (d_m00>0)
    {
        theta_m00_pm = PI - atan(d_m00_p0/d_m00);
        theta_m00_mm = PI + atan(d_m00_m0/d_m00);
    }
    if (d_0m0>0)
    {
        theta_0m0_mm = 3*PI/2 - atan(d_0m0_m0/d_0m0);
        theta_0m0_pm = 3*PI/2 + atan(d_0m0_p0/d_0m0);
    }
    
    std::vector<double> theta (8);
    theta[0] = fabs(theta_grad - theta_p00_pm); theta[1] = fabs(theta_grad - theta_p00_mm); theta[2] = fabs(theta_grad - theta_0p0_pm); theta[3] = fabs(theta_grad - theta_0p0_mm);
    theta[4] = fabs(theta_grad - theta_m00_pm); theta[5] = fabs(theta_grad - theta_m00_mm); theta[6] = fabs(theta_grad - theta_0m0_mm); theta[7] = fabs(theta_grad - theta_0m0_pm);
    
    int arg = 0 ;
    double result = theta[0] ;
    for (int i = 1; i < theta.size(); ++i)
    {
        if (theta[i] < result)
        {
            result = theta[i] ;
            arg = i ;
        }
    }
    
    
    if(arg == 0)
        return node_p00_pm;
    if(arg == 1)
        return node_p00_mm;
    if(arg == 2)
        return node_0p0_pm;
    if(arg == 3)
        return node_0p0_mm;
    if(arg == 4)
        return node_m00_pm;
    if(arg == 5)
        return node_m00_mm;
    if(arg == 6)
        return node_0m0_mm;
    if(arg == 7)
        return node_0m0_pm;
}

bool my_p4est_epitaxy_t::fill_capture_zone(double *phi_p, double *island_number_g_p, double *cz_p, int number, p4est_locidx_t n, double grad_x, double grad_y, double *grad_phi_p_x, double *grad_phi_p_y, double *top_mask_p, double *mask_p, bool dangling_chain=false)
{
    /*
     * this function follows a node to make chains, if it hits an island it assigns the island number to the chain, otherwise it is a loose chain
     * and it will assign a number to that dangling chain, to be later determined.
     */

	// initially test if we are in the masked zones.
    if ((mask_p[n] == 0))
    {
	cz_p[n] = -1;
	return false;
    }

   // double phi_val = phi_p[n];
    std::vector<p4est_locidx_t> chain;
    chain.push_back(n);
    p4est_locidx_t next_node;
    next_node = n;
    double cz_node = cz_p[n];
    double island_nb_val = island_number_g_p[n];
    bool go=false;
    if(island_nb_val>=0)
        go=false;
    
    bool local;
    dangling_chain=false;
   
    while(mask_p[next_node]==1) //(island_nb_val<0) && (phi_val<0))
    {
        next_node = choose_next_node(ngbd, next_node, grad_x, grad_y);
        
        local = next_node<nodes->num_owned_indeps;
        /* if not in the local processor, it is a loose chain, assign the dangling number */
        if(!local){
            chain.push_back(next_node);
            island_nb_val = number;
            dangling_chain = true;
            break;
        }
        
       // phi_val = phi_p[next_node];
	if(mask_p[next_node]==0 && top_mask_p[next_node]<0)
	{
		island_nb_val = -1;
		break;
	} else if (mask_p[next_node]==0 && top_mask_p[next_node]>0)
	{
		island_nb_val = island_number_g_p[next_node];
		break;
	}
        grad_x = grad_phi_p_x[next_node];
        grad_y = grad_phi_p_y[next_node];
        island_nb_val = island_number_g_p[next_node];
        if (std::find(chain.begin(), chain.end(), next_node)!=chain.end()){
            island_nb_val = number;
            dangling_chain=true;	
            break;
        }
        chain.push_back(next_node);
        cz_node = cz_p[next_node];
        if(cz_node>=0){
            island_nb_val = cz_node;
            break;
        }
        
    }
    if(!go){
        for (int i=0; i<chain.size();i++)
        {
            p4est_locidx_t node = chain[i];
            cz_p[node] = island_nb_val;
        }
    }
    chain.clear();
    return dangling_chain;
}

void my_p4est_epitaxy_t::rename_dangling_chains(double *cz_p, int label, p4est_locidx_t n, double *grad_phi_p_x, double *grad_phi_p_y)
{
    p4est_locidx_t next_node;
    next_node = n;
    double grad_x = grad_phi_p_x[next_node];
    double grad_y = grad_phi_p_y[next_node];
    while(cz_p[next_node]>=1e6)
    {
        cz_p[next_node] = label;
        next_node = choose_next_node(ngbd, next_node, grad_x, grad_y);
        grad_x = grad_phi_p_x[next_node];
        grad_y = grad_phi_p_y[next_node];
    }
}





void my_p4est_epitaxy_t::find_connected_ghost_chains(double *phi_p, double *cz_p, p4est_locidx_t n, std::vector<double> &connected, std::vector<bool> &visited, double grad_x, double grad_y, double *grad_phi_p_x, double *grad_phi_p_y)
{
    std::vector<p4est_locidx_t> st;
    //std::stack<size_t> st;
    st.push_back(n);
    
    p4est_locidx_t next_node;
    next_node = n;
    visited[n] = true;
    if(p4est->mpisize==1){
        //size_t k = st.top();
        size_t k = st.back();
        st.pop_back();
        visited[k] = true;
        connected.push_back(-1);
    }
    
    while(true)
    {
        
        size_t k = st.back();
        
        
        grad_x = grad_phi_p_x[k];
        grad_y = grad_phi_p_y[k];
        next_node = choose_next_node(ngbd, k, grad_x, grad_y);
        
        int counter = 0;
        while(std::find(st.begin(), st.end(), next_node)!=st.end() && counter<10){
            counter += 1;
            grad_x = ((double)rand()/RAND_MAX)-0.5;
            grad_y = ((double)rand()/RAND_MAX)-0.5;
            next_node = choose_next_node(ngbd, k, grad_x, grad_y);
        }
        if(std::find(st.begin(), st.end(), next_node)!=st.end())
        {
            connected.push_back(-1);
            break;
        }
        
        if(next_node<nodes->num_owned_indeps){
            st.push_back(next_node);
            visited[k] = true;
        }
        
        if(next_node>=nodes->num_owned_indeps)
        {
            connected.push_back(cz_p[next_node]);
            st.clear();
            break;
        }   
    }
    st.clear();
    
}


void my_p4est_epitaxy_t::stochastic_reversibility()
{
    // Compute island_number_g. Can we avoid repeating this section of code in compute_capture_zone()?
    compute_islands_numbers();
    
    Vec island_number_g, levels_nb_g;
    ierr = VecDuplicate(rho_g, &island_number_g); CHKERRXX(ierr);
    ierr = VecDuplicate(rho_g, &levels_nb_g); CHKERRXX(ierr);
    std::vector<const double*> island_number_p(island_number.size());
    for(unsigned int i=0; i<phi.size(); ++i)
    {
        ierr = VecGetArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);
    }
    
    double *island_number_g_p, *levels_nb_g_p;
    ierr = VecGetArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    ierr = VecGetArray(levels_nb_g, &levels_nb_g_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        island_number_g_p[n] = -1;
        levels_nb_g_p[n] = -1;
        for(unsigned int level=0; level<phi.size(); ++level)
        {
            island_number_g_p[n] = MAX(island_number_g_p[n], island_number_p[level][n]);
            if(island_number_p[level][n]>=0)
                levels_nb_g_p[n] += 1;
        }
    }
    ierr = VecRestoreArray(levels_nb_g, &levels_nb_g_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    for(unsigned int i=0; i<phi.size(); ++i)
    {
        ierr = VecRestoreArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);
    }
    
    // Compute total number of islands
    double nb_islands_total = 0;
    VecMax(island_number_g, NULL, &nb_islands_total);
    nb_islands_total = (int) nb_islands_total + 1;
    
    /*  for(unsigned int level=0; level<nb_islands_per_level.size(); ++level)
     nb_islands_total += nb_islands_per_level[level];
     */
    
    // Compute signed distance from islands (using phi_g)
    //compute_phi_g(); // Is this redundant?
    my_p4est_level_set_t ls(ngbd);
    //  ls.reinitialize_reinitialize_1st_order_time_2nd_order_space(phi_g);
    
    // Compute capture_zone, which should be -k inside CZs (k is associated island number)
    compute_capture_zone();
    
    // Begin to set up Poisson solver
    Vec p_escape; // Solution of laplace p = 0 in CZ, p=0 on island, p=1 on bdry CZ
    ierr = VecDuplicate(rho_g, &p_escape); CHKERRXX(ierr);
    Vec rhs; // RHS is zero for the Laplace equation
    ierr = VecDuplicate(rho_g, &rhs); CHKERRXX(ierr);
    Vec loc;
    ierr = VecGhostGetLocalForm(rhs, &loc); CHKERRXX(ierr);
    ierr = VecSet(loc, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(rhs, &loc); CHKERRXX(ierr);
    
    Vec phi_cz; // Level set function ultimately used in solver.set_phi() (CZ minus island)
    ierr = VecDuplicate(rho_g, &phi_cz); CHKERRXX(ierr);
    
    //  double *phi_cz_p;
    //  ierr = VecGetArray(phi_cz, &phi_cz_p); CHKERRXX(ierr);
    
    double *cz_p;
    ierr = VecGetArray(capture_zone, &cz_p); CHKERRXX(ierr);
    
    /* MAIN LOOP. For each island:
     *  Compute island area
     *  if(area < A_cut)
     *    Set BCs: Need signed distance from islands and CZs
     *    Solve for p_escape
     *    Integrate p_escape over interface distance a from island
     *    Decide number to detach (sample Poisson distribution)
     *    Detach/Renucleate
     *    Distribute mass along CZ
     */
    
    std::stack<struct RenucData> datastore;
    for(unsigned int island=0; island<nb_islands_total; island++) // Is this the appropriate range?
    {
        // Get level set function which is -1 inside the island, and 1 elsewhere
        Vec phi_island;
        double *phi_island_p;
        VecDuplicate(phi_g, &phi_island);
        VecGetArray(island_number_g, &island_number_g_p);
        VecGetArray(phi_island, &phi_island_p);
        VecGetArray(levels_nb_g, &levels_nb_g_p);
        int level = 0;
        for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
        {
            phi_island_p[n] = (island_number_g_p[n] == (double)island) ? -1 : 1;
            if(island_number_g_p[n] == (double)island && level == 0) level = (int) levels_nb_g_p[n];
        }
        VecRestoreArray(levels_nb_g, &levels_nb_g_p);
        VecRestoreArray(island_number_g, &island_number_g_p);
        VecRestoreArray(phi_island, &phi_island_p);
        
        double area = area_in_negative_domain(p4est, nodes, phi_island);
        
        double A_cut = 180*180;
        //PetscPrintf(p4est->mpicomm, "A is %g and A_cut is %g, lattice spacing is %g\n", area, A_cut, lattice_spacing);
        if(area<A_cut)
        {
            // Set up BCs: Get signed distance from island and CZ
            ls.reinitialize_1st_order_time_2nd_order_space(phi_island); // Signed distance from island
            
            // Get signed distance from CZ (including the island)
            double *phi_cz_p;
            ierr = VecGetArray(phi_cz, &phi_cz_p); CHKERRXX(ierr);
            for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
                phi_cz_p[n] = (cz_p[n] == (double)island) ? -1 : 1;
            VecRestoreArray(phi_cz, &phi_cz_p);
            ls.reinitialize_1st_order_time_2nd_order_space(phi_cz);
            /*
             for(size_t i=0; i<ngbd->get_layer_size(); ++i)
             
             {
             
             p4est_locidx_t n = ngbd->get_layer_node(i);
             
             if( cz_p[n] == -1*island )
             
             {
             
             phi_cz_p[n] = -1;
             
             }
             
             else
             
             phi_cz_p[n] = 1;
             
             }
             */
            
            
            // Set BCs using phi_island and phi_cz
            my_p4est_interpolation_nodes_t phi0(ngbd), phi1(ngbd);
            phi0.set_input(phi_island, linear);
            phi1.set_input(phi_cz, linear);
            bc_escape.set_bc(phi0,phi1);
            
            // Convert phi_cz to level set for CZ (without the island)
            ierr = VecGetArray(phi_cz, &phi_cz_p); CHKERRXX(ierr);
            ierr = VecGetArray(phi_island, &phi_island_p); CHKERRXX(ierr);
            for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
                if(phi_island_p[n]<0) phi_cz_p[n] = 1;
            ierr = VecRestoreArray(phi_cz, &phi_cz_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(phi_island, &phi_island_p); CHKERRXX(ierr);
            
            // Compute p_esc
            my_p4est_poisson_nodes_t solver(ngbd);
            solver.set_phi(phi_cz);
            //      solver.set_diagonal(0); // JS: Not sure what this is. It is set to 1 in solve_rho(), which I think solves U_{k+1} + mu*laplacian U_{k+1} = RHS, where index for U is the timestep, mu = dt*Diffusivity, and RHS = U_k + dt*Forcing.
            solver.set_mu(dt_n*D);
            solver.set_rhs(rhs);
            
            BoundaryConditions2D bc;
            bc.setInterfaceValue(bc_escape); // JS: How to distinguish boundaries?
            bc.setInterfaceType(DIRICHLET);
            solver.set_bc(bc);
            solver.solve(p_escape);
            
            
            
            // Compute number of detachment events (reinitialize, integrate_over_interface, boost's RNG)
            // Find interface that is one lattice constant away from island and put it in phi_tmp
            // JS: How much refinement is there near this interface?
            Vec phi_tmp;
            ierr = VecDuplicate(phi_island, &phi_tmp); CHKERRXX(ierr);
            double *phi_tmp_p;
            ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
            ierr = VecGetArray(phi_island, &phi_island_p); CHKERRXX(ierr);
            for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
                phi_tmp_p[n] = (phi_island_p[n]<lattice_spacing) ? -1 : 1;
            ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(phi_island, &phi_island_p); CHKERRXX(ierr);
            
            double n_edge_adatoms = 2; // JS: Could be a function of island size/shape
            double D_det = 1e-5;
            double D_eff = D_det*n_edge_adatoms*integrate_over_interface(p4est, nodes, phi_tmp, p_escape);
            VecDestroy(p_escape);
            double r = ((double)rand()/RAND_MAX);
            double prob = exp(-1*D_eff*dt_n);
            //PetscPrintf(p4est->mpicomm, "r %g, prob %g, D_eff %g, dt %g\n", r, prob, D_eff, dt_n);
            int n_detach = 0;
            while(r>prob)
            {
                n_detach++;
                prob += pow(D_eff*dt_n, n_detach)*exp(-1*D_eff*dt_n)/Factorial(n_detach);
            }
            double new_area = area - n_detach*lattice_spacing*lattice_spacing;
            //PetscPrintf(p4est->mpicomm, "area %g, new_area %g\n", area, new_area);
            if(new_area<2*lattice_spacing*lattice_spacing) new_area = 0; // Island dissociates
            
            
            if(new_area<area)
            {
                double xc, yc;  // x- and y-coordinates of center of mass
                compute_island_com(island, &xc, &yc);
                
                struct RenucData data;
                data.xc = xc; data.yc = yc;
                data.new_area = new_area;
                data.level = level;
                datastore.push(data);
                dissolve_island(island, level);
                if(nb_islands_total>1)
                    redistribute_mass_on_boundary(island, area - new_area, level);
                else
                    redistribute_mass_uniformly(island, area - new_area, level);
            }
        } // end area<A_cut
    } // end loop over islands
    ierr = VecRestoreArray(capture_zone, &cz_p); CHKERRXX(ierr);
    
    nucleate_new_island(&datastore);
    
    //    if(nb_islands_total>1)
    //          redistribute_mass_on_boundary(island, area - new_area, level);
    //    else
    //          redistribute_mass_uniformly(island, area - new_area, level);
    
    
    
    
    // Destroy vecs
    ierr = VecDestroy(phi_cz); CHKERRXX(ierr);
    //  ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    
} // end STOCHASTIC_REVERSIBILITY

void my_p4est_epitaxy_t::compute_island_com(int island, double *xc_p, double *yc_p)
{
    PetscPrintf(p4est->mpicomm, "center of mass for island begins... \n");
    
    Vec island_number_g;
    ierr = VecDuplicate(rho_g, &island_number_g); CHKERRXX(ierr);
    std::vector<const double*> island_number_p(island_number.size());
    for(unsigned int i=0; i<phi.size(); ++i)
    {
        ierr = VecGetArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);
    }
    //        PetscPrintf(p4est->mpicomm, "maybe_100 \n");
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
    //	PetscPrintf(p4est->mpicomm, "maybe_101 \n");
    ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    for(unsigned int i=0; i<phi.size(); ++i)
    {
        ierr = VecRestoreArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);
    }
    
    
    // Compute area of island
    Vec phi_tmp, weight;
    ierr = VecDuplicate(island_number_g, &phi_tmp); CHKERRXX(ierr);
    ierr = VecDuplicate(island_number_g, &weight); CHKERRXX(ierr);
    double *phi_tmp_p;
    //   PetscPrintf(p4est->mpicomm, "maybe_0 \n");
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
    ierr = VecGetArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
        phi_tmp_p[n] = (island_number_g_p[n] == (double)island) ? -1 : 1;
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    
    double area = area_in_negative_domain(p4est, nodes, phi_tmp);
    
    // Compute moment wrt x-coordinate
    double *weight_p;
    //    PetscPrintf(p4est->mpicomm, "maybe_1 \n");
    ierr = VecGetArray(weight, &weight_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
        weight_p[n] = node_x_fr_n(n, p4est, nodes);
    ierr = VecRestoreArray(weight, &weight_p); CHKERRXX(ierr);
    //	PetscPrintf(p4est->mpicomm, "maybe_2 \n");
    *xc_p = integrate_over_negative_domain(p4est, nodes, phi_tmp, weight)/area;
    //	PetscPrintf(p4est->mpicomm, "maybe_3 \n");
    // Compute moment wrt y-coordinate
    ierr = VecGetArray(weight, &weight_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
        weight_p[n] = node_y_fr_n(n, p4est, nodes);
    ierr = VecRestoreArray(weight, &weight_p); CHKERRXX(ierr);
    *yc_p = integrate_over_negative_domain(p4est, nodes, phi_tmp, weight)/area;
} // end COMPUTE_ISLAND_COM



void my_p4est_epitaxy_t::dissolve_island(int island, int level)
{
    PetscPrintf(p4est->mpicomm, "dissolve island begins at level %d... \n", level);
    double *island_number_p, *phi_p;
    
    VecGetArray(island_number[level], &island_number_p);
    VecGetArray(phi[level], &phi_p);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        if(island_number_p[n] == (double) island)
        {
            phi_p[n] = -1;
        }
    }
    VecRestoreArray(island_number[level], &island_number_p);
    VecRestoreArray(phi[level], &phi_p);
}


void my_p4est_epitaxy_t::nucleate_new_island(std::stack <struct RenucData> *datastore)
{
    /* update the forest with the new island */
    //    p4est_t *p4est_new = p4est_copy(p4est, P4EST_FALSE);
    //    splitting_criteria_cf_t *sp_old = (splitting_criteria_cf_t*)p4est->user_pointer;
    while(!datastore->empty())
    {
        PetscPrintf(p4est->mpicomm, "renucleate island begins... \n");
        struct RenucData data;
        data = datastore->top();
        datastore->pop();
        double xc = data.xc;
        double yc = data.yc;
        double area = data.new_area;
        double level = data.level;
        
        if(area < 0)
        {
            //	PetscPrintf(p4est->mpicomm, "Complain! the area to be nucleated is negative! \n");
            throw std::runtime_error("The area to be nucleated is negative!\n");
            
        }
        
        double r = sqrt(area/PI);  // No factor of lattice_spacing since area is dimensional
        circle_t circle(xc, yc, r, this);
        
        /* update the forest with the new island */
        /*    p4est_t *p4est_new = p4est_copy(p4est, P4EST_FALSE);
         splitting_criteria_cf_t *sp_old = (splitting_criteria_cf_t*)p4est->user_pointer;
         splitting_criteria_cf_t sp(sp_old->min_lvl, sp_old->max_lvl, &circle, sp_old->lip);
         p4est_new->user_pointer = (void*)(&sp);
         my_p4est_refine(p4est_new, P4EST_TRUE, refine_levelset_cf, NULL);
         my_p4est_partition(p4est_new, P4EST_FALSE, NULL);
         p4est_new->user_pointer = (void*)sp_old;
         
         p4est_ghost_t *ghost_new = my_p4est_ghost_new(p4est_new, P4EST_CONNECT_FULL);
         p4est_nodes_t *nodes_new = my_p4est_nodes_new(p4est_new, ghost_new);
         */
        
        
        
        /*
         my_p4est_interpolation_nodes_t interp(ngbd);
         double xyz[P4EST_DIM];
         for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
         {
         node_xyz_fr_n(n, p4est, nodes, xyz);
         interp.add_point(n, xyz);
         }
         
         
         Vec tmp;
         ierr = VecCreateGhostNodes(p4est, nodes, &tmp); CHKERRXX(ierr);
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
         
         
         */
        double *phi_p; //, *rho_p[2];
        ierr = VecGetArray(phi[level  ], &phi_p); CHKERRXX(ierr);
        //      ierr = VecGetArray(rho[level  ], &rho_p[0]); CHKERRXX(ierr);
        //      ierr = VecGetArray(rho[level+1], &rho_p[1]); CHKERRXX(ierr);
        
        for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        {
            double x = node_x_fr_n(n, p4est, nodes);
            double y = node_y_fr_n(n, p4est, nodes);
            double tmp = circle(x,y);
            phi_p[n] = MAX(phi_p[n],tmp);
            //        if(tmp>0)
            //        {
            //          rho_p[1][n] = rho_p[0][n];
            //        }
        }
        ierr = VecRestoreArray(phi[level  ], &phi_p   ); CHKERRXX(ierr);
        //      ierr = VecRestoreArray(rho[level  ], &rho_p[0]); CHKERRXX(ierr);
        //      ierr = VecRestoreArray(rho[level+1], &rho_p[1]); CHKERRXX(ierr);
        
        
        
        /*    if(bc_type==ROBIN)
         {
         ierr = VecDestroy(robin_coef); CHKERRXX(ierr);
         ierr = VecDuplicate(rho_g, &robin_coef); CHKERRXX(ierr);
         }
         */
    }
    /*    p4est_destroy(p4est);       p4est = p4est_new;
     p4est_ghost_destroy(ghost); ghost = ghost_new;
     p4est_nodes_destroy(nodes); nodes = nodes_new;
     hierarchy->update(p4est, ghost);
     ngbd->update(hierarchy, nodes);
     */
    my_p4est_level_set_t ls(ngbd);
    for(unsigned int l=0; l<phi.size(); ++l)
        ls.perturb_level_set_function(phi[l], EPS);
    
    compute_phi_g();
    
    
} // end NUCLEATE_NEW_ISLAND with 3 arguements


void my_p4est_epitaxy_t::redistribute_mass_on_boundary(int island, double area, int level)
{
    PetscPrintf(p4est->mpicomm, "redistribute mass begins...\n");
    // find the boundary nodes on stack and find total node area
    double total_node_area;
    std::stack<size_t> st;
    Vec node_areas, boundary;
    ierr = VecDuplicate(rho_g, &node_areas); CHKERRXX(ierr);
    ierr = VecDuplicate(rho_g, &boundary); CHKERRXX(ierr);
    
    Vec loc;
    ierr = VecGhostGetLocalForm(boundary, &loc); CHKERRXX(ierr);
    ierr = VecSet(loc, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(boundary, &loc); CHKERRXX(ierr);
    double *boundary_p, *cz_p;
    VecGetArray(boundary, &boundary_p);
    VecGetArray(capture_zone, &cz_p);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        if (cz_p[n] == (double) island && is_boundary(n, island))
        {
            st.push(n);
            boundary_p[n] = 1;
        }
    }
    VecRestoreArray(boundary, &boundary_p);
    VecRestoreArray(capture_zone, &cz_p);
    multiply_values_by_area(p4est, nodes, boundary, node_areas);
    VecSum(node_areas, &total_node_area);
    // update rho add rho += (node_area/total_node_area)*area/lattice_spacing/lattice_spacing/node_area = (area/total_node_area)/lattice_spacing/lattice_spacing
    double *rho_p;
    VecGetArray(rho[level], &rho_p);
    while(!st.empty())
    {
        p4est_locidx_t bn = st.top();
        st.pop();
        rho_p[bn] += (area/total_node_area)/lattice_spacing/lattice_spacing;
    }
    VecRestoreArray(rho[level], &rho_p);
}

bool my_p4est_epitaxy_t::is_boundary(p4est_locidx_t node, int island)
{
    double *cz_p;
    VecGetArray(capture_zone, &cz_p);
    const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[node];
    if (cz_p[qnnn.node_p00_pm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    if (cz_p[qnnn.node_p00_mm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    if (cz_p[qnnn.node_0m0_pm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    if (cz_p[qnnn.node_0m0_mm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    if (cz_p[qnnn.node_m00_mm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    if (cz_p[qnnn.node_m00_pm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    if (cz_p[qnnn.node_0p0_mm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    if (cz_p[qnnn.node_0p0_pm] != island)
    {
        VecRestoreArray(capture_zone, &cz_p);
        return true;
    }
    VecRestoreArray(capture_zone, &cz_p);
    return false;
}

void my_p4est_epitaxy_t::redistribute_mass_uniformly(int island, double area, int level)
{
    PetscPrintf(p4est->mpicomm, "redistribute mass unifomrly begins...\n");
    // find the boundary nodes on stack and find total node area
    double total_node_area;
    std::stack<size_t> st;
    Vec node_areas, boundary;
    ierr = VecDuplicate(rho_g, &node_areas); CHKERRXX(ierr);
    ierr = VecDuplicate(rho_g, &boundary); CHKERRXX(ierr);
    
    Vec loc;
    ierr = VecGhostGetLocalForm(boundary, &loc); CHKERRXX(ierr);
    ierr = VecSet(loc, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(boundary, &loc); CHKERRXX(ierr);
    double *boundary_p, *island_nb_p;
    VecGetArray(boundary, &boundary_p);
    VecGetArray(island_number[level], &island_nb_p);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        if (island_nb_p[n] != (double) island)
        {
            st.push(n);
            boundary_p[n] = 1;
        }
    }
    VecRestoreArray(boundary, &boundary_p);
    VecRestoreArray(island_number[level], &island_nb_p);
    multiply_values_by_area(p4est, nodes, boundary, node_areas);
    VecSum(node_areas, &total_node_area);
    // update rho add rho += (node_area/total_node_area)*area/lattice_spacing/lattice_spacing/node_area = (area/total_node_area)/lattice_spacing/lattice_spacing
    double *rho_p;
    VecGetArray(rho[level], &rho_p);
    while(!st.empty())
    {
        p4est_locidx_t bn = st.top();
        st.pop();
        rho_p[bn] += (area/total_node_area)/lattice_spacing/lattice_spacing;
    }
    VecRestoreArray(rho[level], &rho_p);
}


 void my_p4est_epitaxy_t::compute_topmost_layer_mask(Vec mask, Vec top_mask)
 {
   Vec bottom_loc, top_loc;
   VecDuplicate(phi_g, &bottom_loc);
   VecDuplicate(phi_g, &top_loc);
   
// first build a level-set and island number combining all levels 
    compute_islands_numbers();
    Vec island_number_g;
    ierr = VecDuplicate(phi_g, &island_number_g); CHKERRXX(ierr);
    std::vector<const double*> island_number_p(island_number.size());
    for(unsigned int i=0; i<phi.size(); ++i)
    	ierr = VecGetArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);
    
    double *island_number_g_p, *bottom_loc_p, *top_loc_p;
    ierr = VecGetArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    ierr = VecGetArray(bottom_loc, &bottom_loc_p); CHKERRXX(ierr);
    ierr = VecGetArray(top_loc, &top_loc_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        island_number_g_p[n] = -1;
	top_loc_p[n] = -1;
	bottom_loc_p[n] = -2;
	// first build the slice of topmost and botultimate island numbers.
	int level=0;
	while(level<phi.size())	
	{
		bottom_loc_p[n] = top_loc_p[n];
		top_loc_p[n] = MAX(island_number_g_p[n], island_number_p[level][n]);
            	island_number_g_p[n] = MAX(island_number_g_p[n], island_number_p[level][n]);  
		level+=1;	
	}   
    }
    ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(bottom_loc, &bottom_loc_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(top_loc, &top_loc_p); CHKERRXX(ierr);
    
    for(unsigned int i=0; i<phi.size(); ++i)
        ierr = VecRestoreArrayRead(island_number[i], &island_number_p[i]); CHKERRXX(ierr);

    // construct local top-bottom lists on each processor.
    std::vector<double> top_list;
    std::vector<double> bot_list;
    ierr = VecGetArray(top_loc, &top_loc_p); CHKERRXX(ierr);
    ierr = VecGetArray(bottom_loc, &bottom_loc_p); CHKERRXX(ierr);
    top_list.push_back(top_loc_p[0]);
    bot_list.push_back(bottom_loc_p[0]);

    for(unsigned int n=1; n<nodes->indep_nodes.elem_count; ++n)
    {
	double item_top = top_loc_p[n];
	double item_bot = bottom_loc_p[n];
	bool not_found = true;
	for (int i=0;i<top_list.size();++i)
	{
		if ((top_list[i] == item_top) && (bot_list[i] == item_bot))
		{
			not_found = false;
			break;
		}
	}
	if (not_found)
	{
		top_list.push_back(item_top);
    		bot_list.push_back(item_bot);
	}
	//if (std::find(bot_list.begin(), bot_list.end(), item_bot) != bot_list.end() && std::find(top_list.begin(), top_list.end(), item_top) != top_list.end())
	//	continue;
	//else{
	//	top_list.push_back(item_top);
    	//	bot_list.push_back(item_bot);	
	//}
	
    }
    ierr = VecRestoreArray(top_loc, &top_loc_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(bottom_loc, &bottom_loc_p); CHKERRXX(ierr);

    const int root = 0;
    //std::vector<int> recvcounts(p4est->mpisize);
    int gsize = p4est->mpisize;
    int *recvcounts = (int *)malloc(gsize*sizeof(int));
  
    // Only root has the received data 
    int mylen = top_list.size();
    double top_list_arr[mylen], bot_list_arr[mylen];
    for(int i=0;i<mylen;++i)
    {
	top_list_arr[i] = top_list[i];
	bot_list_arr[i] = bot_list[i];
    }
    MPI_Gather(&mylen, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, p4est->mpicomm);

    
    //  Figure out the total length of vectors, 
     // and displacements for each rank 
     

    int totlen = 0;
    int *displs = (int *)malloc(gsize*sizeof(int));
    if (p4est->mpirank == 0) { 
        displs[0] = 0;
        totlen += recvcounts[0];

        for (int i=1; i<gsize; i++) {
           totlen += recvcounts[i];   
           displs[i] = displs[i-1] + recvcounts[i-1];
        }
     }
           
   MPI_Bcast(&totlen, 1, MPI_INT, 0, p4est->mpicomm);
   MPI_Bcast(displs, p4est->mpisize, MPI_INT, 0, p4est->mpicomm);
      
     // Now we have the receive buffer, counts, and displacements, and 
     // can gather the vectors: total_top and total_bot on the root process
     
    double *top_buff = (double *)malloc(gsize*mylen*sizeof(double));
    double *bot_buff = (double *)malloc(gsize*mylen*sizeof(double));

    MPI_Gatherv(top_list_arr, mylen, MPI_DOUBLE, top_buff, recvcounts, displs, MPI_DOUBLE, 0, p4est->mpicomm);
    MPI_Gatherv(bot_list_arr, mylen, MPI_DOUBLE, bot_buff, recvcounts, displs, MPI_DOUBLE, 0, p4est->mpicomm);

// clean the total lists to remove duplicates!
    double *top_buff_clean = (double *)malloc(gsize*totlen*sizeof(double));
    double *bot_buff_clean = (double *)malloc(gsize*totlen*sizeof(double));
    if (p4est->mpirank == 0) {
    	std::vector<double> total_top(totlen);
    	std::vector<double> total_bot(totlen);
    	for(int i=0;i<totlen;++i)
    	{
		total_top[i] = top_buff[i];
		total_bot[i] = bot_buff[i];
    	}
	// remove global duplicates.
	for(int n=0;n<totlen;++n)
	{
		double item_top = total_top[n];
		double item_bot = total_bot[n];
		int m=n+1;
		while(m<totlen)
		{
			if(item_top==total_top[m] && item_bot==total_bot[m])
			{
				total_top.erase(total_top.begin()+m);
				total_bot.erase(total_bot.begin()+m);
				m-=1;	
				totlen--;
			}
			m++;
		}
    	}
	// a top can not be a bottom at other nodes. 
	for(int n=0;n<totlen;++n)
	{
		double item_top = total_top[n];
		int m=n+1;
		while(m<totlen)
		{
			if(item_top==total_bot[m])
			{
				total_top.erase(total_top.begin()+n);
				total_bot.erase(total_bot.begin()+n);	
				totlen--;
				break;
			}
			m++;
		}
    	}
    
    	for(int i=0;i<totlen;++i)
    	{
		top_buff_clean[i] = total_top[i];
		bot_buff_clean[i] = total_bot[i];
    	}
    }
    free(top_buff);
    free(bot_buff);
    free(recvcounts);
    free(displs);
// total_top and total_bot contain the island numbers of the two topmost layers, it is on processor 0.
// now construct the masks.
    MPI_Bcast(&totlen, 1, MPI_INT, 0, p4est->mpicomm);
    MPI_Bcast(top_buff_clean, totlen, MPI_DOUBLE, 0, p4est->mpicomm);
    MPI_Bcast(bot_buff_clean, totlen, MPI_DOUBLE, 0, p4est->mpicomm);
    

    std::vector<double> total_top_vec(totlen);
    std::vector<double> total_bot_vec(totlen);
    for(int i=0;i<totlen;++i)
    {
	total_top_vec[i] = top_buff_clean[i];
	total_bot_vec[i] = bot_buff_clean[i];
        //std::cout<<"Process "<<p4est->mpirank<<" total bottom is "<<total_bot_vec[i]<< " total top is "<<total_top_vec[i]<<std::endl;
    }

    ierr = VecGetArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
    int island;
    double *mask_p, *is_top_p;
    VecGetArray(mask, &mask_p);
    VecGetArray(top_mask, &is_top_p);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {	
	mask_p[n] = 0;
	island = island_number_g_p[n];
	if (std::find(total_top_vec.begin(), total_top_vec.end(), island) != total_top_vec.end() && island>=0)
	{	
		is_top_p[n]=1;
        }
	else if (std::find(total_bot_vec.begin(), total_bot_vec.end(), island) != total_bot_vec.end())
		mask_p[n]=1;
	
    }
    VecRestoreArray(mask, &mask_p);
    VecRestoreArray(top_mask, &is_top_p);
    ierr = VecRestoreArray(island_number_g, &island_number_g_p); CHKERRXX(ierr);
}
 
void my_p4est_epitaxy_t::compute_masked_vector(Vec mask,Vec input, Vec output, double default_value)
{
    //	Vec loc;
    //        ierr = VecGhostGetLocalForm(output, &loc); CHKERRXX(ierr);
    //        ierr = VecSet(loc, default_value); CHKERRXX(ierr);
    //        ierr = VecGhostRestoreLocalForm(output, &loc); CHKERRXX(ierr);
    
    double *mask_p, *input_p, *output_p;
    VecGetArray(mask, &mask_p);
    VecGetArray(input, &input_p);
    VecGetArray(output, &output_p);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        output_p[n] = input_p[n]*mask_p[n];
    }
    VecRestoreArray(mask, &mask_p);
    VecRestoreArray(input, &input_p);
    VecRestoreArray(output, &output_p);
}


void my_p4est_epitaxy_t::remove_holes()
{
    PetscPrintf(p4est->mpicomm, "Remove holes in the capture zone field.\n");
    double *cz_p;
    VecGetArray(capture_zone, &cz_p);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
    	const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
	int capture_at_n = cz_p[n];

	if (capture_at_n>-1)
		continue;

	std::vector<int> neighbs;
	if(qnnn.node_p00_pm>-1)
		neighbs.push_back(cz_p[qnnn.node_p00_pm]);
	if(qnnn.node_p00_mm>-1)
		neighbs.push_back(cz_p[qnnn.node_p00_mm]);
	if(qnnn.node_0m0_pm>-1)
		neighbs.push_back(cz_p[qnnn.node_0m0_pm]);
	if(qnnn.node_0m0_mm>-1)
		neighbs.push_back(cz_p[qnnn.node_0m0_mm]);
	if(qnnn.node_m00_mm>-1)
		neighbs.push_back(cz_p[qnnn.node_m00_mm]);
	if(qnnn.node_m00_pm>-1)
		neighbs.push_back(cz_p[qnnn.node_m00_pm]);
	if(qnnn.node_0p0_mm>-1)
		neighbs.push_back(cz_p[qnnn.node_0p0_mm]);
	if(qnnn.node_0p0_pm>-1)
		neighbs.push_back(cz_p[qnnn.node_0p0_pm]);


	if( std::equal(neighbs.begin() + 1, neighbs.end(), neighbs.begin()) )	
   		cz_p[n] = neighbs[0];
	neighbs.clear();
    }
    VecRestoreArray(capture_zone, &cz_p);
}

