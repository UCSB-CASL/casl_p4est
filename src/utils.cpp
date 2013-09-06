#include "utils.h"
#include "my_p4est_tools.h"
#include "my_p4est_node_neighbors.h"
#include <vector>

void c2p_coordinate_transform(p4est_t *p4est, p4est_topidx_t tree_id, double *x, double *y, double *z){
    // We first need to determine the refference point (i.e lower left corner of the current tree)
    p4est_topidx_t *v_ref = p4est->connectivity->tree_to_vertex + tree_id*P4EST_CHILDREN;

    // Now get the xyz coordinates of the four corners of the tree
    double ref_xyz[P4EST_CHILDREN][3];
    for (int i=0; i<P4EST_CHILDREN; i++){
        for (int j=0; j<3; j++){
            ref_xyz[i][j] = p4est->connectivity->vertices[3*v_ref[i]+j];
        }
    }

#ifdef CASL_THROWS
    if(x == NULL)
        throw std::invalid_argument("[CASL_ERROR]: In computational domain, x-ccordinate cannot be NULL");
    if(y == NULL)
        throw std::invalid_argument("[CASL_ERROR]: In computational domain, y-ccordinate cannot be NULL");
#endif
    double eta_x = *x, eta_y = *y;

#ifdef CASL_THROWS
    if (eta_x<0 || eta_x>1.0)
        throw std::invalid_argument("[CASL_ERROR]: In computational domain, x-coordinates should run in [0,1]. ");
    if (eta_y<0 || eta_y>1.0)
        throw std::invalid_argument("[CASL_ERROR]: In computational domain, y-coordinates should run in [0,1]. ");
#endif


    *x = (1.0-eta_x)*(1.0-eta_y)*ref_xyz[0][0] +
            (1.0-eta_x)*(    eta_y)*ref_xyz[2][0] +
            (    eta_x)*(1.0-eta_y)*ref_xyz[1][0] +
            (    eta_x)*(    eta_y)*ref_xyz[3][0];

    *y = (1.0-eta_x)*(1.0-eta_y)*ref_xyz[0][1] +
            (1.0-eta_x)*(    eta_y)*ref_xyz[2][1] +
            (    eta_x)*(1.0-eta_y)*ref_xyz[1][1] +
            (    eta_x)*(    eta_y)*ref_xyz[3][1];

    if (NULL != z)
        *z = (1.0-eta_x)*(1.0-eta_y)*ref_xyz[0][2] +
            (1.0-eta_x)*(    eta_y)*ref_xyz[2][2] +
            (    eta_x)*(1.0-eta_y)*ref_xyz[1][2] +
            (    eta_x)*(    eta_y)*ref_xyz[3][2];

}

void dx_dy_dz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *dx, double *dy, double *dz){
    double *v = p4est->connectivity->vertices;
    p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
    p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);

    if (dx != NULL){
        double dx_tree = v[3*t2v[tree_id*P4EST_CHILDREN+1]] - v[3*t2v[tree_id*P4EST_CHILDREN]];
        *dx = dx_tree * (double)(qh)/(double)(P4EST_ROOT_LEN);
    }

    if (dy != NULL){
        double dy_tree = v[3*t2v[tree_id*P4EST_CHILDREN+2] + 1] - v[3*t2v[tree_id*P4EST_CHILDREN] + 1];
        *dy = dy_tree * (double)(qh)/(double)(P4EST_ROOT_LEN);
    }

    if (dz != NULL){
        double dz_tree = v[3*t2v[tree_id*P4EST_CHILDREN+4] + 2] - v[3*t2v[tree_id*P4EST_CHILDREN] + 2];
        *dz = dz_tree * static_cast<double>(qh)/static_cast<double>(P4EST_ROOT_LEN);
    }
}

void xyz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *x, double *y, double *z){
    p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);

    if (x != NULL){
        *x = (double)(quad->x + 0.5 * qh)/(double)(P4EST_ROOT_LEN);
    }

    if (y != NULL){
        *y = (double)(quad->y + 0.5 * qh)/(double)(P4EST_ROOT_LEN);
    }

    c2p_coordinate_transform(p4est, tree_id, x, y, z);

}

double bilinear_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, double *F, const double *xy_global)
{
    p4est_topidx_t lower_left_vertex  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
    p4est_topidx_t upper_right_vertex = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 3];

    double tree_xmin = p4est->connectivity->vertices[3*lower_left_vertex + 0];
    double tree_ymin = p4est->connectivity->vertices[3*lower_left_vertex + 1];

    double tree_xmax = p4est->connectivity->vertices[3*upper_right_vertex + 0];
    double tree_ymax = p4est->connectivity->vertices[3*upper_right_vertex + 1];

    double x = (xy_global[0] - tree_xmin)/(tree_xmax - tree_xmin);
    double y = (xy_global[1] - tree_ymin)/(tree_ymax - tree_ymin);

#ifdef CASL_THROWS
    if (x<0 || x>1 || y<0 || y>1)
    {
        std::ostringstream oss;
        oss << "[CASL_ERROR]: Point (" << xy_global[0] << ", " << xy_global[1] << ") is not located inside given tree (= " << tree_id << ")" << std::endl;
        throw std::invalid_argument(oss.str());
    }
#endif

    double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
    double xmin = (double)quad.x / (double)(P4EST_ROOT_LEN);
    double ymin = (double)quad.y / (double)(P4EST_ROOT_LEN);

    double d_m0 = x - xmin;
    double d_p0 = qh - d_m0;
    double d_0m = y - ymin;
    double d_0p = qh - d_0m;

    return ( (F[0]*(d_0p*d_p0) + F[1]*(d_m0*d_0p) + F[2]*(d_p0*d_0m) + F[3]*(d_m0*d_0m))/(qh*qh) );
}

double quadratic_interpolation_nonoscilatory(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fxx, const double *Fyy, const double *xy_global)
{
    p4est_topidx_t lower_left_vertex  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
    p4est_topidx_t upper_right_vertex = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 3];

    double tree_xmin = p4est->connectivity->vertices[3*lower_left_vertex + 0];
    double tree_ymin = p4est->connectivity->vertices[3*lower_left_vertex + 1];

    double tree_xmax = p4est->connectivity->vertices[3*upper_right_vertex + 0];
    double tree_ymax = p4est->connectivity->vertices[3*upper_right_vertex + 1];

    double x = (xy_global[0] - tree_xmin)/(tree_xmax - tree_xmin);
    double y = (xy_global[1] - tree_ymin)/(tree_ymax - tree_ymin);

#ifdef CASL_THROWS
    if (x<0 || x>1 || y<0 || y>1)
    {
        std::ostringstream oss;
        oss << "[CASL_ERROR]: Point (" << xy_global[0] << ", " << xy_global[1] << ") is not located inside given tree (= " << tree_id << ")" << std::endl;
        throw std::invalid_argument(oss.str());
    }
#endif

    double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
    double xmin = (double)quad.x / (double)(P4EST_ROOT_LEN);
    double ymin = (double)quad.y / (double)(P4EST_ROOT_LEN);

    double d_m0 = x - xmin;
    double d_p0 = qh - d_m0;
    double d_0m = y - ymin;
    double d_0p = qh - d_0m;

    double fxx_minmod = Fxx[0];
    double fyy_minmod = Fyy[0];

    for (short i=1; i<P4EST_CHILDREN; i++)
    {
      fxx_minmod = MINMOD(fxx_minmod, Fxx[i]);
      fyy_minmod = MINMOD(fyy_minmod, Fyy[i]);
    }

    return ( (F[0]*(d_0p*d_p0) + F[1]*(d_m0*d_0p) + F[2]*(d_p0*d_0m) + F[3]*(d_m0*d_0m))/(qh*qh) - 0.5*d_m0*d_p0*fxx_minmod - 0.5*d_0m*d_0p*fyy_minmod);
}

double quadratic_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fxx, const double *Fyy, const double *xy_global)
{
    p4est_topidx_t lower_left_vertex  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
    p4est_topidx_t upper_right_vertex = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 3];

    double tree_xmin = p4est->connectivity->vertices[3*lower_left_vertex + 0];
    double tree_ymin = p4est->connectivity->vertices[3*lower_left_vertex + 1];

    double tree_xmax = p4est->connectivity->vertices[3*upper_right_vertex + 0];
    double tree_ymax = p4est->connectivity->vertices[3*upper_right_vertex + 1];

    double x = (xy_global[0] - tree_xmin)/(tree_xmax - tree_xmin);
    double y = (xy_global[1] - tree_ymin)/(tree_ymax - tree_ymin);

#ifdef CASL_THROWS
    if (x<0 || x>1 || y<0 || y>1)
    {
        std::ostringstream oss;
        oss << "[CASL_ERROR]: Point (" << xy_global[0] << ", " << xy_global[1] << ") is not located inside given tree (= " << tree_id << ")" << std::endl;
        throw std::invalid_argument(oss.str());
    }
#endif

    double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
    double xmin = (double)quad.x / (double)(P4EST_ROOT_LEN);
    double ymin = (double)quad.y / (double)(P4EST_ROOT_LEN);

    double d_m0 = x - xmin;
    double d_p0 = qh - d_m0;
    double d_0m = y - ymin;
    double d_0p = qh - d_0m;

    double fxx = (Fxx[0]*(d_0p*d_p0) + Fxx[1]*(d_m0*d_0p) + Fxx[2]*(d_p0*d_0m) + Fxx[3]*(d_m0*d_0m))/(qh*qh);
    double fyy = (Fyy[0]*(d_0p*d_p0) + Fyy[1]*(d_m0*d_0p) + Fyy[2]*(d_p0*d_0m) + Fyy[3]*(d_m0*d_0m))/(qh*qh);

    return ( (F[0]*(d_0p*d_p0) + F[1]*(d_m0*d_0p) + F[2]*(d_p0*d_0m) + F[3]*(d_m0*d_0m))/(qh*qh) - 0.5*d_m0*d_p0*fxx - 0.5*d_0m*d_0p*fyy);
}

PetscErrorCode VecCreateGhost(p4est_t *p4est, p4est_nodes_t *nodes, Vec* v)
{
    PetscErrorCode ierr = 0;
    p4est_locidx_t num_local = nodes->num_owned_indeps;

    std::vector<PetscInt> ghost_nodes(nodes->indep_nodes.elem_count - num_local, 0);
    std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

    // Calculate the global number of points
    for (int r = 0; r<p4est->mpisize; ++r)
        global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)nodes->global_owned_indeps[r];

    PetscInt num_global = global_offset_sum[p4est->mpisize];

    for (p4est_locidx_t i = 0; i<nodes->offset_owned_indeps; ++i)
    {
        p4est_indep_t *ni  = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
        ghost_nodes[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i]];
    }
    for (size_t i = nodes->offset_owned_indeps+num_local; i<nodes->indep_nodes.elem_count; ++i)
    {
        p4est_indep_t* ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
        ghost_nodes[i-num_local] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i-num_local]];
    }

    ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global, ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
    ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

    return ierr;
}

p4est_locidx_t p4est2petsc_local_numbering(p4est_nodes_t *nodes, p4est_locidx_t p4est_node_locidx)
{
#ifdef CASL_THROWS
    if (p4est_node_locidx < 0 || p4est_node_locidx >= (p4est_locidx_t)nodes->indep_nodes.elem_count)
    {
        std::stringstream oss; oss << "[CASL_ERROR]: node index " << p4est_node_locidx << " is out of bound" << std::endl;
        throw std::invalid_argument(oss.str());
    }
#endif
    p4est_locidx_t petsc_node_locidx;

    if (p4est_node_locidx < nodes->offset_owned_indeps)
        petsc_node_locidx = p4est_node_locidx + nodes->num_owned_indeps;
    else if (p4est_node_locidx >= nodes->offset_owned_indeps && p4est_node_locidx < nodes->offset_owned_indeps + nodes->num_owned_indeps)
        petsc_node_locidx = p4est_node_locidx - nodes->offset_owned_indeps;
    else
        petsc_node_locidx = p4est_node_locidx;

    return petsc_node_locidx;
}









