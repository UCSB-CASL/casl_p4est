#include "my_p4est_epidemics.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_poisson_nodes.h>
#include <random>
#include <time.h>
#include <stack>
#include <fstream>
#include <ANN/ANN.h>



my_p4est_epidemics_t::my_p4est_epidemics_t(my_p4est_node_neighbors_t *ngbd)
    : brick(ngbd->myb), p4est(ngbd->p4est), connectivity(p4est->connectivity),
      ghost(ngbd->ghost), nodes(ngbd->nodes), hierarchy(ngbd->hierarchy),
      ngbd(ngbd), interp_density(this)
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
    dt_n = 1e-3;
    dxyz_min(p4est, dxyz);
    srand(time(NULL));

    ierr = VecCreateGhostNodes(p4est, nodes, &phi_g); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &U_n); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &V_n); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &W_n); CHKERRXX(ierr);
    Vec phi_A, phi_B, phi_AB;
    VecDuplicate(U_n, &phi_A);
    VecDuplicate(V_n, &phi_B);
    VecDuplicate(W_n, &phi_AB);
    phi.push_back(phi_A);
    phi.push_back(phi_B);
    phi.push_back(phi_AB);


}




my_p4est_epidemics_t::~my_p4est_epidemics_t()
{

    if(phi_g != NULL) { ierr = VecDestroy(phi_g); CHKERRXX(ierr); }
    for(unsigned int i=0; i<phi.size(); ++i)
    {
        if(phi[i]           != NULL) { ierr = VecDestroy(phi[i]) ; CHKERRXX(ierr); }
        if(v[0][i]          != NULL) { ierr = VecDestroy(v[0][i]); CHKERRXX(ierr); }
        if(v[1][i]          != NULL) { ierr = VecDestroy(v[1][i]); CHKERRXX(ierr); }
    }


    /* destroy the p4est and its connectivity structure */
    delete ngbd;
    delete hierarchy;
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, brick);
}

double my_p4est_epidemics_t::get_density(double x, double y)
{
    return interp_density(x,y);
}

//void my_p4est_epidemics_t::read(const std::string &census) {

//    // only read on rank 0 and then broadcast the result to others
//    if (p4est->mpirank == 0) {
//        std::ifstream infile(census);

//#ifdef CASL_THROWS
//        if (!infile.fail())
//            throw std::invalid_argument("could not open the census file");
//#endif

//        // parse line by line
//        while(!infile.eof()) {
//            Tract tract;
//            infile >> tract.id;
//            infile >> tract.x;
//            infile >> tract.y;
//            infile >> tract.density;
//            infile >> tract.pop;
//            infile >> tract.area;
//            tracts.push_back(tract);
//        }
//        infile.close();
//    }

//    size_t msg_size = tracts.size()*sizeof(Tract);
//    MPI_Bcast(&msg_size, 1, MPI_UNSIGNED_LONG, 0, p4est->mpicomm);
//    if (p4est->mpirank != 0)
//        tracts.resize(msg_size/sizeof(Tract));
//    MPI_Bcast(&tracts[0], msg_size, MPI_BYTE, 0, p4est->mpicomm);

//    // compute the center of mass
//    xc_ = 0;
//    yc_ = 0;
//    for (size_t i = 0; i<tracts.size(); i++){
//        xc_ += tracts[i].x;
//        yc_ += tracts[i].y;
//        densities.push_back(tracts[i].density);
//    }
//    xc_ /= tracts.size();
//    yc_ /= tracts.size();


//    // compute the size of the bounding box
//    Lx_max= 0;
//    Lx_min=0;
//    Ly_max = 0;
//    Ly_min = 0;
//    for (size_t i = 0; i<tracts.size(); i++){
//        Lx_max = MAX(Lx_max, tracts[i].x);
//        Ly_max = MAX(Ly_max, tracts[i].y);

//        Lx_min = MIN(Lx_min, tracts[i].x);
//        Ly_min = MIN(Ly_min, tracts[i].y);
//    }
//    // make room from boundaries
//    Lx_min *= 1.1;
//    Lx_max *= 1.1;
//    Ly_min *= 1.1;
//    Ly_max *= 1.1;
//    // scale and recenter the tracts to middle
//    translate(Lx_min, Ly_min);   // shift coordinate system to the bottom left corner
//    unit_scaling();
//    translate(0, -0.2);

//    int maxPts = tracts.size();
//    dataPts = annAllocPts(maxPts, P4EST_DIM); // allocate data points


//    for(int n=0; n<tracts.size(); ++n)
//    {
//        dataPts[n][0] = tracts[n].x;
//        dataPts[n][1] = tracts[n].y;
//    }

//    if(p4est->mpirank==0)
//    {
//        std::ofstream fout;
//        fout.open("locs.txt");
//        for(int n=0; n<tracts.size(); ++n)
//            fout << tracts[n].x << "\t" << tracts[n].y << "\n";
//        fout.close();
//    }
//    set_density();
//}

//void my_p4est_epidemics_t::translate(double x_shift, double y_shift) {
//    // move the tracts to the new location
//    for (size_t i = 0; i<tracts.size(); i++){
//        tracts[i].x -= x_shift;
//        tracts[i].y -= y_shift;
//    }
//    xc_ -= x_shift;
//    yc_ -= y_shift;
//}

//void my_p4est_epidemics_t::unit_scaling() {
//    // scale coordinate to be unit times unit in length
//    double scale = MAX((Lx_max - Lx_min),(Ly_max - Ly_min));
//    for (size_t i = 0; i<tracts.size(); i++){
//        tracts[i].x /= scale;
//        tracts[i].y /= scale;
//    }
//    xc_ /= scale;
//    yc_ /= scale;
//}


//void my_p4est_epidemics_t::set_density()
//{
//    int nPts = tracts.size();
//    kdTree = new ANNkd_tree(dataPts,  nPts, 2, 1);  // build search structure
//}


//double my_p4est_epidemics_t::interp_density(double x, double y)
//{
//    ANNpoint queryPt;           // query point
//    queryPt = annAllocPt(2);  // allocate query point

//    queryPt[0] = x;
//    queryPt[1] = y;

//    ANNidxArray nnIdx;          // near neighbor indices
//    ANNdistArray dists;         // near neighbor distances
//    nnIdx = new ANNidx[k_neighs];      // allocate near neigh indices
//    dists = new ANNdist[k_neighs];     // allocate near neighbor dists
//    kdTree->annkSearch( queryPt, k_neighs, nnIdx, dists, 0);

//    double interpolated_density = 0;
//    double denom = 0;
//    double max_neigh_dens = 0;
//    double min_neigh_dens = DBL_MAX;
//    for (int i = 0; i < k_neighs; i++)
//    {
//        int nid = nnIdx[i];
//        double dist = sqrt(dists[i]);
//        if(dists[i]<=R_eff)
//        {
//            double neigh_dens = tracts[nid].density;
//            double weight = 1/(0.1*R_eff + dist);               // softening length is 10% of effective radius
//            interpolated_density += weight*neigh_dens;
//            denom += weight;
//            max_neigh_dens = MAX(max_neigh_dens, neigh_dens);
//            min_neigh_dens = MIN(min_neigh_dens, neigh_dens);
//        }
//    }

//    if(denom>EPS)
//        interpolated_density /= denom;
//    else
//        interpolated_density = 0;

//    if(interpolated_density>max_neigh_dens)
//        interpolated_density = max_neigh_dens;

//    if(interpolated_density<min_neigh_dens)
//        interpolated_density = min_neigh_dens;

//    if(min_neigh_dens>1e20)
//        interpolated_density = 0;

//    delete [] nnIdx;
//    delete [] dists;

//    return interpolated_density;
//}



void my_p4est_epidemics_t::compute_phi_g()
{
    double *phi_g_p, *phi_A_p, *phi_B_p, *phi_AB_p;
    ierr = VecGetArray(phi_g, &phi_g_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi[0], &phi_A_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi[1], &phi_B_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi[2], &phi_AB_p); CHKERRXX(ierr);
    for(unsigned int n=0; n<nodes->indep_nodes.elem_count; ++n)
        phi_g_p[n] = MAX(phi_A_p[n], phi_B_p[n], phi_AB_p[n]);

    ierr = VecRestoreArray(phi_g, &phi_g_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi[0], &phi_A_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi[1], &phi_B_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi[2], &phi_AB_p); CHKERRXX(ierr);


}


void my_p4est_epidemics_t::set_parameters(double R_A,
                                          double R_B,
                                          double Xi_A,
                                          double Xi_B)
{
    this->R_A     = R_A;
    this->R_B     = R_B;
    this->Xi_A    = Xi_A;
    this->Xi_B    = Xi_B;
}



void my_p4est_epidemics_t::set_D(double D_A, double D_B, double D_AB)
{
    this->D_A = D_A;
    this->D_B = D_B;
    this->D_AB = D_AB;
}





void my_p4est_epidemics_t::compute_velocity()
{
    // v[][0,1,2]>> 0: A, 1:B, 2:AB
    const double *U_n_p, *V_n_p, *W_n_p;
    my_p4est_level_set_t ls(ngbd);
    ierr = VecGetArrayRead(U_n, &U_n_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(V_n, &V_n_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(W_n, &W_n_p); CHKERRXX(ierr);

    /* velocity of region A */
    Vec vtmp[2];
    double *v_A_p[2];
    ierr = VecDuplicate(v[0][0], &vtmp[0]); CHKERRXX(ierr);
    ierr = VecDuplicate(v[1][0], &vtmp[1]); CHKERRXX(ierr);
    double *v_p[2];
    ierr = VecGetArray(vtmp[0], &v_A_p[0]); CHKERRXX(ierr);
    ierr = VecGetArray(vtmp[1], &v_A_p[1]); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_layer_node(i);
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        v_A_p[0][n] = -(D_AB*qnnn.dx_central(W_n_p) - D_A*qnnn.dx_central(U_n_p));
        v_A_p[1][n] = -(D_AB*qnnn.dy_central(W_n_p) - D_A*qnnn.dy_central(U_n_p));
    }
    ierr = VecGhostUpdateBegin(vtmp[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vtmp[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_local_node(i);
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        v_A_p[0][n] =  (D_AB*qnnn.dx_central(W_n_p) - D_A*qnnn.dx_central(U_n_p));
        v_A_p[1][n] = (D_AB*qnnn.dy_central(W_n_p) - D_A*qnnn.dy_central(U_n_p));
    }
    ierr = VecGhostUpdateEnd(vtmp[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vtmp[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(vtmp[0], &v_A_p[0]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vtmp[1], &v_A_p[1]); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(phi[0], vtmp[0], v[0][0]);
    ls.extend_from_interface_to_whole_domain_TVD(phi[0], vtmp[1], v[1][0]);



    /* velocity of region B */
    Vec vtmp_new[2];
    ierr = VecDuplicate(v[0][1], &vtmp_new[0]); CHKERRXX(ierr);
    ierr = VecDuplicate(v[1][1], &vtmp_new[1]); CHKERRXX(ierr);
    double *v_B_p[2];
    ierr = VecGetArray(vtmp_new[0], &v_B_p[0]); CHKERRXX(ierr);
    ierr = VecGetArray(vtmp_new[1], &v_B_p[1]); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_layer_node(i);
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        v_B_p[0][n] = (D_AB*qnnn.dx_central(W_n_p) - D_B*qnnn.dx_central(V_n_p));
        v_B_p[1][n] = (D_AB*qnnn.dy_central(W_n_p) - D_B*qnnn.dy_central(V_n_p));
    }
    ierr = VecGhostUpdateBegin(vtmp_new[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vtmp_new[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_local_node(i);
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        v_B_p[0][n] = (D_AB*qnnn.dx_central(W_n_p) - D_B*qnnn.dx_central(V_n_p));
        v_B_p[1][n] = (D_AB*qnnn.dy_central(W_n_p) - D_B*qnnn.dy_central(V_n_p));
    }
    ierr = VecGhostUpdateEnd(vtmp_new[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vtmp_new[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(vtmp_new[0], &v_B_p[0]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vtmp_new[1], &v_B_p[1]); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(phi[1], vtmp_new[0], v[0][1]);
    ls.extend_from_interface_to_whole_domain_TVD(phi[1], vtmp_new[1], v[1][1]);

    /* velocity of region AB */
    Vec vtmp_AB[2];
    ierr = VecDuplicate(v[0][2], &vtmp_AB[0]); CHKERRXX(ierr);
    ierr = VecDuplicate(v[1][2], &vtmp_AB[1]); CHKERRXX(ierr);
    double *v_t_p[2];
    ierr = VecGetArray(vtmp_AB[0], &v_t_p[0]); CHKERRXX(ierr);
    ierr = VecGetArray(vtmp_AB[1], &v_t_p[1]); CHKERRXX(ierr);
    double *v_AB_p[2];
    VecGetArray(v[0][2], &v_AB_p[0]);
    VecGetArray(v[1][2], &v_AB_p[1]);
    VecGetArray(vtmp[0], &v_A_p[0]);
    VecGetArray(vtmp[1], &v_A_p[1]);
    VecGetArray(vtmp_new[0], &v_B_p[0]);
    VecGetArray(vtmp_new[1], &v_B_p[1]);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_local_node(i);
        v_t_p[0][n] = v_A_p[0][n] - v_B_p[0][n];
        v_t_p[1][n] = v_A_p[1][n] - v_B_p[1][n];
    }
    ierr = VecRestoreArray(vtmp_AB[0], &v_t_p[0]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vtmp_AB[1], &v_t_p[1]); CHKERRXX(ierr);
    VecRestoreArray(v[0][2], &v_AB_p[0]);
    VecRestoreArray(v[1][2], &v_AB_p[1]);
    VecRestoreArray(vtmp[0], &v_A_p[0]);
    VecRestoreArray(vtmp[1], &v_A_p[1]);
    VecRestoreArray(vtmp_new[0], &v_B_p[0]);
    VecRestoreArray(vtmp_new[1], &v_B_p[1]);


    ls.extend_from_interface_to_whole_domain_TVD(phi[2], vtmp_AB[0], v[0][2]);
    ls.extend_from_interface_to_whole_domain_TVD(phi[2], vtmp_AB[1], v[1][2]);

    ierr = VecRestoreArrayRead(U_n, &U_n_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(V_n, &V_n_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(W_n, &W_n_p); CHKERRXX(ierr);
    ierr = VecDestroy(vtmp[0]); CHKERRXX(ierr);
    ierr = VecDestroy(vtmp[1]); CHKERRXX(ierr);
    ierr = VecDestroy(vtmp_new[0]); CHKERRXX(ierr);
    ierr = VecDestroy(vtmp_new[1]); CHKERRXX(ierr);
    ierr = VecDestroy(vtmp_AB[0]); CHKERRXX(ierr);
    ierr = VecDestroy(vtmp_AB[1]); CHKERRXX(ierr);
}

void my_p4est_epidemics_t::compute_dt()
{
    dt_n *= 1;
    double vmax = 0;
    double m, M;
    for(unsigned int i=0; i<2; ++i)
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
    ierr = PetscPrintf(p4est->mpicomm, "time step dt_n = %e\n", dt_n); CHKERRXX(ierr);
}


void my_p4est_epidemics_t::update_grid()
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


    my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

    sl.update_p4est(v, dt_n, phi);


    /* interpolate the quantities on the new grid */
    for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
    {
        node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
        interp.add_point(n, xyz);
    }


    Vec U_tmp;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &U_tmp); CHKERRXX(ierr);
    interp.set_input(U_n, quadratic_non_oscillatory);
    interp.interpolate(U_tmp);
    ierr = VecDestroy(U_n); CHKERRXX(ierr);
    U_n = U_tmp;


    Vec V_tmp;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &V_tmp); CHKERRXX(ierr);
    interp.set_input(V_n, quadratic_non_oscillatory);
    interp.interpolate(V_tmp);
    ierr = VecDestroy(V_n); CHKERRXX(ierr);
    V_n = V_tmp;

    Vec W_tmp;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &W_tmp); CHKERRXX(ierr);
    interp.set_input(W_n, quadratic_non_oscillatory);
    interp.interpolate(W_tmp);
    ierr = VecDestroy(W_n); CHKERRXX(ierr);
    W_n = W_tmp;


    for(unsigned int i=0; i<phi.size(); ++i)
    {
        ierr = VecDestroy(v[0][i]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v[0][i]); CHKERRXX(ierr);
        ierr = VecDestroy(v[1][i]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v[1][i]); CHKERRXX(ierr);
    }


    p4est_destroy(p4est);       p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    hierarchy->update(p4est, ghost);
    ngbd->update(hierarchy, nodes);

    /* reinitialize and perturb phi */
    my_p4est_level_set_t ls(ngbd);
    for(unsigned int agent=0; agent<phi.size(); ++agent)
    {
        ls.reinitialize_1st_order_time_2nd_order_space(phi[agent]);
        ls.perturb_level_set_function(phi[agent], EPS);
    }
    VecDestroy(phi_g);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_g); CHKERRXX(ierr);
    compute_phi_g();
}


void my_p4est_epidemics_t::solve(int iter)
{



    double *U_n_p, *V_n_p, *W_n_p, *phi_A_p, *phi_B_p, *phi_AB_p, *land_p;
    if(iter==0)
    {
        VecGetArray(U_n, &U_n_p);
        VecGetArray(V_n, &V_n_p);
        VecGetArray(W_n, &W_n_p);
        VecGetArray(phi[0], &phi_A_p);
        VecGetArray(phi[1], &phi_B_p);
        VecGetArray(phi[2], &phi_AB_p);
        VecDuplicate(U_n, &land);
        VecGetArray(land, &land_p);
        for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        {
            double x = node_x_fr_n(n, p4est, nodes);
            double y = node_y_fr_n(n, p4est, nodes);
            double local_density = get_density(x,y);
            if(land_p[n]>0)
                land_p[n] = 1;
            else
                land_p[n] = -1;

            if(phi_A_p[n]>0)
                U_n_p[n] = local_density;

            if(phi_B_p[n]>0)
                V_n_p[n] = local_density;
            if(phi_AB_p[n]>0)
                W_n_p[n] = 0;
            if(phi_A_p[n]>=0 && phi_B_p[n]>=0)
            {
                U_n_p[n] = local_density/2;
                V_n_p[n] = local_density/2;
            }
        }
        VecRestoreArray(land, &land_p);
        VecRestoreArray(U_n, &U_n_p);
        VecRestoreArray(V_n, &V_n_p);
        VecRestoreArray(W_n, &W_n_p);
        VecRestoreArray(phi[0], &phi_A_p);
        VecRestoreArray(phi[1], &phi_B_p);
        VecRestoreArray(phi[2], &phi_AB_p);
    } else {

        BoundaryConditions2D bc;
        bc.setInterfaceValue(interp_density);
        bc.setInterfaceType(DIRICHLET);

        my_p4est_poisson_nodes_t solver_u(ngbd);
        //solver_u.set_phi(phi[0]);
        solver_u.set_phi(land);
        solver_u.set_mu(dt_n*D_A);
        solver_u.set_diagonal(1);
        solver_u.set_bc(bc);

        my_p4est_poisson_nodes_t solver_v(ngbd);
        //solver_v.set_phi(phi[1]);
        solver_v.set_phi(land);
        solver_v.set_mu(dt_n*D_B);
        solver_v.set_diagonal(1);
        solver_v.set_bc(bc);

        my_p4est_poisson_nodes_t solver_w(ngbd);
        //solver_w.set_phi(phi[2]);
        solver_w.set_phi(land);
        solver_w.set_mu(dt_n*D_AB);
        solver_w.set_diagonal(1);
        solver_w.set_bc(bc);


        Vec rhs_u, rhs_v, rhs_w;

        ierr = VecDuplicate(U_n, &rhs_u); CHKERRXX(ierr);
        ierr = VecDuplicate(V_n, &rhs_v); CHKERRXX(ierr);
        ierr = VecDuplicate(W_n, &rhs_w); CHKERRXX(ierr);
        ierr = VecDuplicate(U_n, &U_np1); CHKERRXX(ierr);
        ierr = VecDuplicate(V_n, &V_np1); CHKERRXX(ierr);
        ierr = VecDuplicate(W_n, &W_np1); CHKERRXX(ierr);

         //my_p4est_level_set_t ls(ngbd);
        int counter = 0;
        while(counter <1)
        {
            double *rhs_u_p, *rhs_v_p, *rhs_w_p;
            ierr = VecGetArray(rhs_u, &rhs_u_p); CHKERRXX(ierr);
            ierr = VecGetArray(rhs_v, &rhs_v_p); CHKERRXX(ierr);
            ierr = VecGetArray(rhs_w, &rhs_w_p); CHKERRXX(ierr);
            ierr = VecGetArray(U_n, &U_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(V_n, &V_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(W_n, &W_n_p); CHKERRXX(ierr);

            for(size_t n=0; n<nodes->indep_nodes.elem_count ; ++n)
            {
                double x = node_x_fr_n(n, p4est, nodes);
                double y = node_y_fr_n(n, p4est, nodes);
                double up = U_n_p[n];
                double vp = V_n_p[n];
                double wp = W_n_p[n];
                double frac = get_density(x,y);
                double s = frac - up - vp + wp;

                if(up>frac)
                    up = frac;
                if(up<0)
                    up = 0;
                if(vp>frac)
                    vp = frac;
                if(vp<0)
                    vp=0;
                if(wp>frac)
                    wp=frac;
                if(wp<0)
                    wp=0;

                rhs_u_p[n] = up + dt_n*(R_A*s*up + Xi_B*R_A*(vp - wp)*up - up);
                rhs_v_p[n] = vp + dt_n*(R_B*s*vp + Xi_A*R_B*(up - wp)*vp - vp);
                rhs_w_p[n] = wp + dt_n*(Xi_A*R_B*(up - wp)*vp + Xi_B*R_A*(vp - wp)*up - 2*wp);
            }
            ierr = VecRestoreArray(rhs_u, &rhs_u_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(rhs_v, &rhs_v_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(rhs_w, &rhs_w_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(U_n, &U_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(V_n, &V_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(W_n, &W_n_p); CHKERRXX(ierr);

            solver_u.set_rhs(rhs_u);
            solver_u.solve(U_np1);

            solver_v.set_rhs(rhs_v);
            solver_v.solve(V_np1);

            solver_w.set_rhs(rhs_w);
            solver_w.solve(W_np1);

//            ls.extend_Over_Interface_TVD(phi[0], U_np1);
//            ls.extend_Over_Interface_TVD(phi[1], V_np1);
//            ls.extend_Over_Interface_TVD(phi[2], W_np1);

            double *U_np1_p, *V_np1_p, *W_np1_p;
            ierr = VecGetArray(U_n, &U_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(V_n, &V_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(W_n, &W_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(U_np1, &U_np1_p); CHKERRXX(ierr);
            ierr = VecGetArray(V_np1, &V_np1_p); CHKERRXX(ierr);
            ierr = VecGetArray(W_np1, &W_np1_p); CHKERRXX(ierr);
            for(size_t n=0; n<nodes->indep_nodes.elem_count ; ++n)
            {
                U_n_p[n] = U_np1_p[n];
                V_n_p[n] = V_np1_p[n];
                W_n_p[n] = W_np1_p[n];
            }
            ierr = VecRestoreArray(U_n, &U_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(V_n, &V_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(W_n, &W_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(U_np1, &U_np1_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(V_np1, &V_np1_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(W_np1, &W_np1_p); CHKERRXX(ierr);
            counter++;
        }

        ierr = VecDestroy(rhs_u); CHKERRXX(ierr);
        ierr = VecDestroy(rhs_v); CHKERRXX(ierr);
        ierr = VecDestroy(rhs_w); CHKERRXX(ierr);
    }
}

void my_p4est_epidemics_t::initialize_infections()
{
    double xc, yc;
    double r = MAX(2*MIN(dxyz[0],dxyz[1]), 0.2);

    Vec tmpp;
    VecDuplicate(phi_g, &tmpp);
    for(unsigned int i=0; i<phi.size(); ++i)
    {
        v[0].push_back(tmpp);
        v[1].push_back(tmpp);
    }

    for(int infection=0;infection<3;++infection)
    {
        if (infection==2)
        {
            r = 0;
            xc = 0.5;
            yc = 0.5;
        }
        /* select nucleation point */
        //xc = ((double)rand()/RAND_MAX)*L;
        //yc = ((double)rand()/RAND_MAX)*L;
        xc = 0.6 ;
        yc = 0.4 + infection*0.2;
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
        interp.set_input(U_n, quadratic_non_oscillatory);
        interp.interpolate(tmp);
        ierr = VecDestroy(U_n); CHKERRXX(ierr);
        U_n  = tmp;



        Vec tmp2;
        ierr = VecCreateGhostNodes(p4est_new, nodes_new, &tmp2); CHKERRXX(ierr);
        interp.set_input(V_n, quadratic_non_oscillatory);
        interp.interpolate(tmp2);
        ierr = VecDestroy(V_n); CHKERRXX(ierr);
        V_n  = tmp2;

        Vec tmp3;
        ierr = VecCreateGhostNodes(p4est_new, nodes_new, &tmp3); CHKERRXX(ierr);
        interp.set_input(W_n, quadratic_non_oscillatory);
        interp.interpolate(tmp3);
        ierr = VecDestroy(W_n); CHKERRXX(ierr);
        W_n  = tmp3;
        for(unsigned int i=0; i<phi.size(); ++i)
        {
            ierr = VecDuplicate(U_n, &tmp); CHKERRXX(ierr);
            interp.set_input(phi[i], quadratic_non_oscillatory);
            interp.interpolate(tmp);
            ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
            phi[i] = tmp;
        }
        interp.clear();

        for(unsigned int l=0; l<phi.size(); ++l)
        {
            ierr = VecCreateGhostNodes(p4est_new, nodes_new, &v[0][l]); CHKERRXX(ierr);\
            ierr = VecCreateGhostNodes(p4est_new, nodes_new, &v[1][l]); CHKERRXX(ierr);
        }


        double *phi_p;
        ierr = VecGetArray(phi[infection  ], &phi_p); CHKERRXX(ierr);

        for(size_t n=0; n<nodes_new->indep_nodes.elem_count; ++n)
        {
            double x = node_x_fr_n(n, p4est_new, nodes_new);
            double y = node_y_fr_n(n, p4est_new, nodes_new);
            double tmp = circle(x,y);
            phi_p[n] = tmp;
        }
        ierr = VecRestoreArray(phi[infection  ], &phi_p); CHKERRXX(ierr);

        p4est_destroy(p4est);       p4est = p4est_new;
        p4est_ghost_destroy(ghost); ghost = ghost_new;
        p4est_nodes_destroy(nodes); nodes = nodes_new;
        hierarchy->update(p4est, ghost);
        ngbd->update(hierarchy, nodes);

        my_p4est_level_set_t ls(ngbd);
        for(unsigned int l=0; l<phi.size(); ++l)
            ls.perturb_level_set_function(phi[l], EPS);
    }

    double *phi_p1, *phi_p2, *phi_p3;
    VecGetArray(phi[0], &phi_p1);
    VecGetArray(phi[1], &phi_p2);
    VecGetArray(phi[2], &phi_p3);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        if (phi_p1[n]>=0 && phi_p2[n]>=0)
            phi_p3[n] = 1;
        else
            phi_p3[n] = -1;
    }
    VecRestoreArray(phi[0], &phi_p1);
    VecRestoreArray(phi[1], &phi_p2);
    VecRestoreArray(phi[2], &phi_p3);

    //    VecDestroy(phi[2]);
    //    ierr = VecCreateGhostNodes(p4est, nodes, &phi[2]); CHKERRXX(ierr);

    VecDestroy(phi_g);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_g); CHKERRXX(ierr);
    compute_phi_g();

}




void my_p4est_epidemics_t::save_vtk(int iter)
{
    char *out_dir = NULL;
    out_dir = getenv("OUT_DIR");
    if(out_dir==NULL)
    {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to the desired path to save visuals.\n");
        return;
    }

    char name[1000];
    snprintf(name, 1000, "%s/vtu/epidemics_%04d", out_dir, iter);

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


    Vec U_vis;
    ierr = VecDuplicate(phi_vis, &U_vis); CHKERRXX(ierr);
    interp.set_input(U_n, linear);
    interp.interpolate(U_vis);

    Vec V_vis;
    ierr = VecDuplicate(phi_vis, &V_vis); CHKERRXX(ierr);
    interp.set_input(V_n, linear);
    interp.interpolate(V_vis);

    Vec W_vis;
    ierr = VecDuplicate(phi_vis, &W_vis); CHKERRXX(ierr);
    interp.set_input(W_n, linear);
    interp.interpolate(W_vis);

    Vec phi_vis_A;
    ierr = VecDuplicate(phi_vis, &phi_vis_A); CHKERRXX(ierr);
    interp.set_input(phi[0], linear);
    interp.interpolate(phi_vis_A);
    Vec phi_vis_B;
    ierr = VecDuplicate(phi_vis, &phi_vis_B); CHKERRXX(ierr);
    interp.set_input(phi[1], linear);
    interp.interpolate(phi_vis_B);
    Vec phi_vis_AB;
    ierr = VecDuplicate(phi_vis, &phi_vis_AB); CHKERRXX(ierr);
    interp.set_input(phi[2], linear);
    interp.interpolate(phi_vis_AB);

    Vec density_vis;
    ierr = VecDuplicate(phi_vis, &density_vis); CHKERRXX(ierr);


    double *den_vis_p;
    VecGetArray(density_vis, &den_vis_p);
    for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
    {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est_vis, nodes_vis, xyz);
        den_vis_p[n] = get_density(xyz[0],xyz[1]);

    }
    VecRestoreArray(density_vis, &den_vis_p);

    const double *phi_v_p, *U_v_p, *V_v_p, *W_v_p, *phi_vA_p, *phi_vB_p, *phi_vAB_p;
    ierr = VecGetArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
    ierr = VecGetArrayRead(U_vis, &U_v_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(V_vis, &V_v_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(W_vis, &W_v_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi_vis_A, &phi_vA_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi_vis_B, &phi_vB_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi_vis_AB, &phi_vAB_p); CHKERRXX(ierr);
    ierr = VecGetArray(density_vis, &den_vis_p); CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est_vis, nodes_vis, ghost_vis, P4EST_TRUE, P4EST_TRUE, 8, 0, name,
                           VTK_POINT_DATA, "phi", phi_v_p,
                           VTK_POINT_DATA, "A", U_v_p,
                           VTK_POINT_DATA, "B", V_v_p,
                           VTK_POINT_DATA, "AB", W_v_p,
                           VTK_POINT_DATA, "phiA", phi_vA_p,
                           VTK_POINT_DATA, "phiAB", phi_vAB_p,
                           VTK_POINT_DATA, "phiB", phi_vB_p,
                           VTK_POINT_DATA, "population density", den_vis_p);


    ierr = VecRestoreArrayRead(phi_vis  , &phi_v_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(U_vis, &U_v_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(V_vis, &V_v_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(W_vis, &W_v_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi_vis_A, &phi_vA_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi_vis_B, &phi_vB_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi_vis_AB, &phi_vAB_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(density_vis, &den_vis_p); CHKERRXX(ierr);

    ierr = VecDestroy(U_vis); CHKERRXX(ierr);
    ierr = VecDestroy(V_vis); CHKERRXX(ierr);
    ierr = VecDestroy(W_vis); CHKERRXX(ierr);
    ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
    ierr = VecDestroy(density_vis); CHKERRXX(ierr);
    p4est_nodes_destroy(nodes_vis);
    p4est_ghost_destroy(ghost_vis);
    p4est_destroy(p4est_vis);
    my_p4est_brick_destroy(connectivity_vis, &brick_vis);

    PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", name);
}




