#include "utils.h"

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

    if (NULL != z){
        *z = (1.0-eta_x)*(1.0-eta_y)*ref_xyz[0][2] +
                (1.0-eta_x)*(    eta_y)*ref_xyz[2][2] +
                (    eta_x)*(1.0-eta_y)*ref_xyz[1][2] +
                (    eta_x)*(    eta_y)*ref_xyz[3][2];
    }

}

void dx_dy_dz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *dx, double *dy, double *dz){
    double *v = p4est->connectivity->vertices;
    p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
    p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);

    if (dx != NULL){
        double dx_tree = v[3*t2v[tree_id*P4EST_CHILDREN+1]] - v[3*t2v[tree_id*P4EST_CHILDREN]];
        *dx = dx_tree * static_cast<double>(qh)/static_cast<double>(P4EST_ROOT_LEN);
    }

    if (dy != NULL){
        double dy_tree = v[3*t2v[tree_id*P4EST_CHILDREN+2] + 1] - v[3*t2v[tree_id*P4EST_CHILDREN] + 1];
        *dy = dy_tree * static_cast<double>(qh)/static_cast<double>(P4EST_ROOT_LEN);
    }

    if (dz != NULL){
        double dz_tree = v[3*t2v[tree_id*P4EST_CHILDREN+4] + 2] - v[3*t2v[tree_id*P4EST_CHILDREN] + 2];
        *dz = dz_tree * static_cast<double>(qh)/static_cast<double>(P4EST_ROOT_LEN);
    }
}

void xyz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *x, double *y, double *z){
    p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);

    if (x != NULL){
        *x = static_cast<double>(quad->x + 0.5 * qh)/static_cast<double>(P4EST_ROOT_LEN);
    }

    if (y != NULL){
        *y = static_cast<double>(quad->y + 0.5 * qh)/static_cast<double>(P4EST_ROOT_LEN);
    }

    c2p_coordinate_transform(p4est, tree_id, x, y, z);

}

double bilinear_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, p4est_quadrant_t *quad, double *F, double x_global, double y_global)
{
  p4est_topidx_t lower_left_vertex  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
  p4est_topidx_t upper_right_vertex = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 3];

  double tree_xmin = p4est->connectivity->vertices[3*lower_left_vertex + 0];
  double tree_ymin = p4est->connectivity->vertices[3*lower_left_vertex + 1];

  double tree_xmax = p4est->connectivity->vertices[3*upper_right_vertex + 0];
  double tree_ymax = p4est->connectivity->vertices[3*upper_right_vertex + 1];

  double x = (x_global - tree_xmin)/(tree_xmax - tree_xmin);
  double y = (y_global - tree_ymin)/(tree_ymax - tree_ymin);

#ifdef CASL_THROWS
  if (x<0 || x>1 || y<0 || y>1)
  {
    std::ostringstream oss;
    oss << "[CASL_ERROR]: Point (" << x_global << ", " << y_global << ") is not located inside given tree (= " << tree_id << ")" << std::endl;
    throw std::invalid_argument(oss.str());
  }
#endif

  double qh   = (double)P4EST_QUADRANT_LEN(quad->level) / (double)(P4EST_ROOT_LEN);
  double xmin = (double)quad->x / (double)(P4EST_ROOT_LEN);
  double ymin = (double)quad->y / (double)(P4EST_ROOT_LEN);

  double d_m0 = x - xmin;
  double d_p0 = qh - d_m0;
  double d_0m = y - ymin;
  double d_0p = qh - d_0m;

  return ( (F[0]*(d_0p*d_p0) + F[1]*(d_m0*d_0p) + F[2]*(d_p0*d_0m) + F[3]*(d_m0*d_0m))/(qh*qh) );
}


