#include "my_p4est_trajectory_of_point.h"

#ifdef P4_TO_P8
#include "my_p8est_interpolation_nodes.h"
#else
#include "my_p4est_interpolation_nodes.h"
#endif


#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_trajectory_from_np1_to_n;
extern PetscLogEvent log_trajectory_from_np1_to_nm1;
extern PetscLogEvent log_trajectory_from_np1_to_n_faces;
extern PetscLogEvent log_trajectory_from_np1_to_nm1_faces;
extern PetscLogEvent log_trajectory_from_np1_bunch_of_points;
#endif

void trajectory_from_np1_to_n( p4est_t *p4est, p4est_nodes_t *nodes,
                               my_p4est_node_neighbors_t *ngbd_n,
                               double dt,
                               Vec v[P4EST_DIM],
                               std::vector<double> xyz_d[P4EST_DIM] )
{
  trajectory_from_np1_to_n(p4est, nodes, ngbd_n, dt, v, NULL, xyz_d);
}

void trajectory_from_np1_to_n( p4est_t *p4est, p4est_nodes_t *nodes,
                               my_p4est_node_neighbors_t *ngbd_n,
                               double dt,
                               Vec v[P4EST_DIM], Vec second_derivatives_v[P4EST_DIM][P4EST_DIM],
                               std::vector<double> xyz_d[P4EST_DIM] )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_trajectory_from_np1_to_n, 0, 0, 0, 0); CHKERRXX(ierr);

  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];

  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (int dir=0; dir<P4EST_DIM; dir++)
  {
    xyz_min[dir] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
    xyz_max[dir] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];
  }

  my_p4est_interpolation_nodes_t interp(ngbd_n);
  bool use_second_derivatives_v = (second_derivatives_v!=NULL);
  for (short dim = 0; dim < P4EST_DIM; ++dim)
    for (short dd = 0; dd < P4EST_DIM; ++dd)
      use_second_derivatives_v = use_second_derivatives_v && (second_derivatives_v[dd][dim] != NULL);

  const double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr); }

  for (p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);

    /* find the intermediate point */
    double xyz_star[] =
    {
      xyz[0] - .5*dt*v_p[0][n],
      xyz[1] - .5*dt*v_p[1][n]
  #ifdef P4_TO_P8
      ,xyz[2] - .5*dt*v_p[2][n]
  #endif
    };

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      if      (is_periodic(p4est,dir) && xyz_star[dir]<xyz_min[dir]) xyz_star[dir] += xyz_max[dir]-xyz_min[dir];
      else if (is_periodic(p4est,dir) && xyz_star[dir]>xyz_max[dir]) xyz_star[dir] -= xyz_max[dir]-xyz_min[dir];
      else                                                           xyz_star[dir] = MAX(xyz_min[dir], MIN(xyz_max[dir], xyz_star[dir]));
    }

    interp.add_point(n, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vstar[P4EST_DIM];
  double *data_star[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
    vstar[dir].resize(nodes->num_owned_indeps);
    data_star[dir] = vstar[dir].data();
  }
  if(use_second_derivatives_v)
#ifdef P4_TO_P8
    interp.set_input(v, second_derivatives_v[0], second_derivatives_v[1], second_derivatives_v[2], quadratic, P4EST_DIM);
#else
    interp.set_input(v, second_derivatives_v[0], second_derivatives_v[1], quadratic, P4EST_DIM);
#endif
  else
    interp.set_input(v, quadratic, P4EST_DIM);
  interp.interpolate(data_star, P4EST_DIM);

  /* find the departure points */
  for (short dir = 0; dir < P4EST_DIM; ++dir)
    xyz_d[dir].resize(nodes->num_owned_indeps);
  for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);

    xyz_d[0][n] = xyz[0] - dt*vstar[0][n];
    xyz_d[1][n] = xyz[1] - dt*vstar[1][n];
#ifdef P4_TO_P8
    xyz_d[2][n] = xyz[2] - dt*vstar[2][n];
#endif

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      if      (is_periodic(p4est,dir) && xyz_d[dir][n]<xyz_min[dir]) xyz_d[dir][n] += xyz_max[dir]-xyz_min[dir];
      else if (is_periodic(p4est,dir) && xyz_d[dir][n]>xyz_max[dir]) xyz_d[dir][n] -= xyz_max[dir]-xyz_min[dir];
      else                                                           xyz_d[dir][n] = MAX(xyz_min[dir], MIN(xyz_max[dir], xyz_d[dir][n]));
    }
  }

  ierr = PetscLogEventEnd(log_trajectory_from_np1_to_n, 0, 0, 0, 0); CHKERRXX(ierr);
}


void trajectory_from_np1_to_nm1( p4est_t *p4est_n, p4est_nodes_t *nodes_n,
                                 my_p4est_node_neighbors_t *ngbd_nm1,
                                 my_p4est_node_neighbors_t *ngbd_n,
                                 Vec vnm1[P4EST_DIM],
                                 Vec vn[P4EST_DIM],
                                 double dt_nm1, double dt_n,
                                 std::vector<double> xyz_nm1[P4EST_DIM],
                                 std::vector<double> xyz_n[P4EST_DIM] )
{
  trajectory_from_np1_to_nm1(p4est_n, nodes_n, ngbd_nm1, ngbd_n, vnm1, NULL, vn, NULL, dt_nm1, dt_n, xyz_nm1, xyz_n);
}
void trajectory_from_np1_to_nm1( p4est_t *p4est_n, p4est_nodes_t *nodes_n,
                                 my_p4est_node_neighbors_t *ngbd_nm1,
                                 my_p4est_node_neighbors_t *ngbd_n,
                                 Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                 Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                 double dt_nm1, double dt_n,
                                 std::vector<double> xyz_nm1[P4EST_DIM],
                                 std::vector<double> xyz_n[P4EST_DIM] )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_trajectory_from_np1_to_nm1, 0, 0, 0, 0); CHKERRXX(ierr);

  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];

  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (int dir=0; dir<P4EST_DIM; dir++)
  {
    xyz_min[dir] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
    xyz_max[dir] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];
  }

  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
  bool use_second_derivatives_vnm1 = (second_derivatives_vnm1!=NULL);
  bool use_second_derivatives_vn = (second_derivatives_vn!=NULL);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      use_second_derivatives_vnm1 = use_second_derivatives_vnm1 && (second_derivatives_vnm1[dd][dim] != NULL);
      use_second_derivatives_vn   = use_second_derivatives_vn   && (second_derivatives_vn[dd][dim] != NULL);
    }
  }

  const double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir){ ierr = VecGetArrayRead(vn[dir], &v_p[dir]); CHKERRXX(ierr); }

  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);

    /* find the intermediate point */
    double xyz_star[] =
    {
      xyz[0] - .5*dt_n*v_p[0][n],
      xyz[1] - .5*dt_n*v_p[1][n]
  #ifdef P4_TO_P8
      ,xyz[2] - .5*dt_n*v_p[2][n]
  #endif
    };

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      if      (is_periodic(p4est_n,dir) && xyz_star[dir]<xyz_min[dir]) xyz_star[dir] += xyz_max[dir]-xyz_min[dir];
      else if (is_periodic(p4est_n,dir) && xyz_star[dir]>xyz_max[dir]) xyz_star[dir] -= xyz_max[dir]-xyz_min[dir];
      else                                                             xyz_star[dir] = MAX(xyz_min[dir], MIN(xyz_max[dir], xyz_star[dir]));
    }

    interp_n  .add_point(n, xyz_star);
    interp_nm1.add_point(n, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vnm1_star[P4EST_DIM];
  std::vector<double> vn_star  [P4EST_DIM];
  double *data_star_nm1[P4EST_DIM];
  double *data_star_n[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vnm1_star[dir].resize(nodes_n->num_owned_indeps);
    data_star_nm1[dir] = vnm1_star[dir].data();
    vn_star[dir].resize(nodes_n->num_owned_indeps);
    data_star_n[dir] = vn_star[dir].data();
  }
  if(use_second_derivatives_vnm1)
#ifdef P4_TO_P8
    interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2], quadratic, P4EST_DIM);
#else
    interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], quadratic, P4EST_DIM);
#endif
  else
    interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
  interp_nm1.interpolate(data_star_nm1, P4EST_DIM);
  if(use_second_derivatives_vn)
#ifdef P4_TO_P8
    interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2], quadratic, P4EST_DIM);
#else
    interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], quadratic, P4EST_DIM);
#endif
  else
    interp_n.set_input(vn, quadratic, P4EST_DIM);
  interp_n.interpolate(data_star_nm1, P4EST_DIM);
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (short dim = 0; dim < P4EST_DIM; ++dim)
    xyz_n[dim].resize(nodes_n->num_owned_indeps);
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);

    xyz_n[0][n] = xyz[0] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[0][n] - .5*dt_n/dt_nm1*vnm1_star[0][n] );
    xyz_n[1][n] = xyz[1] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[1][n] - .5*dt_n/dt_nm1*vnm1_star[1][n] );
#ifdef P4_TO_P8
    xyz_n[2][n] = xyz[2] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[2][n] - .5*dt_n/dt_nm1*vnm1_star[2][n] );
#endif

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      if      (is_periodic(p4est_n,dir) && xyz_n[dir][n]<xyz_min[dir]) xyz_n[dir][n] += xyz_max[dir]-xyz_min[dir];
      else if (is_periodic(p4est_n,dir) && xyz_n[dir][n]>xyz_max[dir]) xyz_n[dir][n] -= xyz_max[dir]-xyz_min[dir];
      else                                                             xyz_n[dir][n] = MAX(xyz_min[dir], MIN(xyz_max[dir], xyz_n[dir][n]));
    }
  }

  /* proceed similarly for the departure point at time nm1 */
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);

    /* find the intermediate point */
    double xyz_star[] =
    {
      xyz[0] - .5*(dt_n+dt_nm1)*v_p[0][n],
      xyz[1] - .5*(dt_n+dt_nm1)*v_p[1][n]
  #ifdef P4_TO_P8
      ,xyz[2] - .5*(dt_n+dt_nm1)*v_p[2][n]
  #endif
    };

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      if      (is_periodic(p4est_n,dir) && xyz_star[dir]<xyz_min[dir]) xyz_star[dir] += xyz_max[dir]-xyz_min[dir];
      else if (is_periodic(p4est_n,dir) && xyz_star[dir]>xyz_max[dir]) xyz_star[dir] -= xyz_max[dir]-xyz_min[dir];
      else                                                             xyz_star[dir] = MAX(xyz_min[dir], MIN(xyz_max[dir], xyz_star[dir]));
    }

    interp_n  .add_point(n, xyz_star);
    interp_nm1.add_point(n, xyz_star);
  }
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vn[dir], &v_p[dir]); CHKERRXX(ierr);
  }

  /* compute the velocities at the intermediate point */
  if(use_second_derivatives_vnm1)
#ifdef P4_TO_P8
    interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2], quadratic, P4EST_DIM);
#else
    interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], quadratic, P4EST_DIM);
#endif
  else
    interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
  interp_nm1.interpolate(data_star_nm1, P4EST_DIM);
  if(use_second_derivatives_vn)
#ifdef P4_TO_P8
    interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2], quadratic, P4EST_DIM);
#else
    interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], quadratic, P4EST_DIM);
#endif
  else
    interp_n.set_input(vn, quadratic, P4EST_DIM);
  interp_n.interpolate(data_star_n, P4EST_DIM);
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time nm1 */
  for (short dim = 0; dim < P4EST_DIM; ++dim)
    xyz_nm1[dim].resize(nodes_n->num_owned_indeps);
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);

    xyz_nm1[0][n] = xyz[0] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[0][n] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[0][n] );
    xyz_nm1[1][n] = xyz[1] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[1][n] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[1][n] );
#ifdef P4_TO_P8
    xyz_nm1[2][n] = xyz[2] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[2][n] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[2][n] );
#endif

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      if      (is_periodic(p4est_n,dir) && xyz_nm1[dir][n]<xyz_min[dir]) xyz_nm1[dir][n] += xyz_max[dir]-xyz_min[dir];
      else if (is_periodic(p4est_n,dir) && xyz_nm1[dir][n]>xyz_max[dir]) xyz_nm1[dir][n] -= xyz_max[dir]-xyz_min[dir];
      else                                                               xyz_nm1[dir][n] = MAX(xyz_min[dir], MIN(xyz_max[dir], xyz_nm1[dir][n]));
    }
  }
}


void trajectory_from_np1_to_n( p4est_t *p4est_n, my_p4est_faces_t *faces_n,
                               my_p4est_node_neighbors_t *ngbd_nm1,
                               my_p4est_node_neighbors_t *ngbd_n,
                               Vec vnm1[P4EST_DIM],
                               Vec vn[P4EST_DIM],
                               double dt_nm1, double dt_n,
                               std::vector<double> xyz_n[P4EST_DIM],
                               int dir )
{
  trajectory_from_np1_to_nm1(p4est_n, faces_n, ngbd_nm1, ngbd_n, vnm1, vn, dt_nm1, dt_n, NULL, xyz_n, dir);
}


void trajectory_from_np1_to_nm1( p4est_t *p4est_n, my_p4est_faces_t *faces_n,
                                 my_p4est_node_neighbors_t *ngbd_nm1,
                                 my_p4est_node_neighbors_t *ngbd_n,
                                 Vec vnm1[P4EST_DIM],
                                 Vec vn[P4EST_DIM],
                                 double dt_nm1, double dt_n,
                                 std::vector<double> xyz_nm1[P4EST_DIM],
                                 std::vector<double> xyz_n[P4EST_DIM],
                                 int dir )
{
  std::vector<double> xyz_np1[P4EST_DIM];
  std::vector<double> *pointers_to_xyz_nm1[1][P4EST_DIM];
  std::vector<double> *pointers_to_xyz_n[1][P4EST_DIM];
  for (short comp = 0; comp < P4EST_DIM; ++comp)
  {
    xyz_np1[comp].resize(faces_n->num_local[dir], 0);
    pointers_to_xyz_n[0][comp]    = &xyz_n[comp];
    pointers_to_xyz_nm1[0][comp]  = ((xyz_nm1 != NULL)? &xyz_nm1[comp]: NULL);
  }
  for (unsigned int f_idx = 0; f_idx < (unsigned int)faces_n->num_local[dir]; ++f_idx) {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);
    xyz_np1[0][f_idx] = xyz[0];
    xyz_np1[1][f_idx] = xyz[1];
#ifdef P4_TO_P8
    xyz_np1[2][f_idx] = xyz[2];
#endif
  }
  trajectory_from_np1_bunch_of_points(p4est_n, ngbd_nm1, ngbd_n,
                                      vnm1, NULL, vn, NULL,
                                      dt_nm1, dt_n,
                                      &xyz_np1,
                                      pointers_to_xyz_n, ((xyz_nm1 != NULL)? pointers_to_xyz_nm1: NULL),
                                      1);
}

void trajectory_from_np1_all_faces( p4est_t *p4est_n, my_p4est_faces_t *faces_n,
                                    my_p4est_node_neighbors_t *ngbd_nm1,
                                    my_p4est_node_neighbors_t *ngbd_n,
                                    Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                    Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                    double dt_nm1, double dt_n,
                                    std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM])
{
  trajectory_from_np1_all_faces(p4est_n, faces_n, ngbd_nm1, ngbd_n, vnm1, second_derivatives_vnm1, vn, second_derivatives_vn, dt_nm1, dt_n, xyz_n, NULL);
}

void trajectory_from_np1_all_faces( p4est_t *p4est_n, my_p4est_faces_t *faces_n,
                                    my_p4est_node_neighbors_t *ngbd_nm1,
                                    my_p4est_node_neighbors_t *ngbd_n,
                                    Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                    Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                    double dt_nm1, double dt_n,
                                    std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM],
                                    std::vector<double> xyz_nm1[P4EST_DIM][P4EST_DIM])
{

  std::vector<double> xyz_np1[P4EST_DIM][P4EST_DIM];
  std::vector<double> *pointers_to_xyz_n[P4EST_DIM][P4EST_DIM];
  std::vector<double> *pointers_to_xyz_nm1[P4EST_DIM][P4EST_DIM];
  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    for (short comp = 0; comp < P4EST_DIM; ++comp) {
      xyz_np1[dir][comp].resize(faces_n->num_local[dir], 0);
      pointers_to_xyz_n[dir][comp] = &xyz_n[dir][comp];
      pointers_to_xyz_nm1[dir][comp] = ((xyz_nm1 != NULL)? &xyz_nm1[dir][comp] : NULL);
    }
    for (unsigned int f_idx = 0; f_idx < (unsigned int)faces_n->num_local[dir]; ++f_idx) {
      double xyz[P4EST_DIM];
      faces_n->xyz_fr_f(f_idx, dir, xyz);
      xyz_np1[dir][0][f_idx] = xyz[0];
      xyz_np1[dir][1][f_idx] = xyz[1];
#ifdef P4_TO_P8
      xyz_np1[dir][2][f_idx] = xyz[2];
#endif
    }
  }
  trajectory_from_np1_bunch_of_points(p4est_n, ngbd_nm1, ngbd_n,
                                      vnm1, second_derivatives_vnm1, vn, second_derivatives_vn,
                                      dt_nm1, dt_n,
                                      xyz_np1,
                                      pointers_to_xyz_n,
                                      ((xyz_nm1 != NULL)? pointers_to_xyz_nm1: NULL),
                                      P4EST_DIM);
}

void trajectory_from_np1_bunch_of_points( p4est_t *p4est_n,
                                          my_p4est_node_neighbors_t *ngbd_nm1,
                                          my_p4est_node_neighbors_t *ngbd_n,
                                          Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                          Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                          double dt_nm1, double dt_n,
                                          const std::vector<double> xyz_np1[][P4EST_DIM],
                                          std::vector<double>* xyz_n[][P4EST_DIM],
                                          std::vector<double>* xyz_nm1[][P4EST_DIM],
                                          unsigned int n_lists)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_trajectory_from_np1_bunch_of_points, 0, 0, 0, 0); CHKERRXX(ierr);

  unsigned int ndata = 0;
  vector<unsigned int> ndata_in_list(n_lists, 0);
  for (unsigned int k = 0; k < n_lists; ++k) {
    ndata_in_list[k] = xyz_np1[k][0].size();
#ifdef P4_TO_P8
    P4EST_ASSERT((ndata_in_list[k]==xyz_np1[k][1].size()) && (ndata_in_list[k]==xyz_np1[k][2].size()));
#else
    P4EST_ASSERT((ndata_in_list[k]==xyz_np1[k][1].size()));
#endif
    ndata += ndata_in_list[k];
  }
  P4EST_ASSERT(ndata > 0);

  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];

  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (int dd=0; dd<P4EST_DIM; dd++)
  {
    xyz_min[dd] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dd];
    xyz_max[dd] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dd];
  }

  /* first find the velocity at the np1 points */
  my_p4est_interpolation_nodes_t interp_np1(ngbd_n);
  bool use_second_derivatives_vnm1 = (second_derivatives_vnm1!=NULL);
  bool use_second_derivatives_vn = (second_derivatives_vn!=NULL);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      use_second_derivatives_vnm1 = use_second_derivatives_vnm1 && (second_derivatives_vnm1[dd][dim] != NULL);
      use_second_derivatives_vn   = use_second_derivatives_vn   && (second_derivatives_vn[dd][dim] != NULL);
    }
  }
  unsigned int idx = 0;
  for (unsigned int k = 0; k < n_lists; ++k) {
    for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
      double xyz[P4EST_DIM] = {xyz_np1[k][0][lidx], xyz_np1[k][1][lidx]
                         #ifdef P4_TO_P8
                               , xyz_np1[k][2][lidx]
                         #endif
                              };
      interp_np1.add_point(idx++, xyz);
    }
  }
  P4EST_ASSERT(idx == ndata); idx = 0;
  std::vector<double> vnp1[P4EST_DIM];
  double *data[P4EST_DIM];
  for(short dir=0; dir<P4EST_DIM; ++dir)
  {
    vnp1[dir].resize(ndata);
    data[dir] = vnp1[dir].data();
  }
  if (use_second_derivatives_vn)
#ifdef P4_TO_P8
    interp_np1.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2], quadratic, P4EST_DIM);
#else
    interp_np1.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], quadratic, P4EST_DIM);
#endif
  else
    interp_np1.set_input(vn, quadratic, P4EST_DIM);
  interp_np1.interpolate(data, P4EST_DIM);

  /* find xyz_star */
  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );

  for (unsigned int k = 0; k < n_lists; ++k) {
    for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
      /* find the intermediate point */
      double xyz_star[] =
      {
        xyz_np1[k][0][lidx] - .5*dt_n*vnp1[0][idx],
        xyz_np1[k][1][lidx] - .5*dt_n*vnp1[1][idx]
    #ifdef P4_TO_P8
        ,xyz_np1[k][2][lidx] - .5*dt_n*vnp1[2][idx]
    #endif
      };

      for(int dd=0; dd<P4EST_DIM; ++dd)
      {
        if      (is_periodic(p4est_n,dd) && xyz_star[dd]<xyz_min[dd]) xyz_star[dd] += xyz_max[dd]-xyz_min[dd];
        else if (is_periodic(p4est_n,dd) && xyz_star[dd]>xyz_max[dd]) xyz_star[dd] -= xyz_max[dd]-xyz_min[dd];
        else                                                          xyz_star[dd] = MAX(xyz_min[dd], MIN(xyz_max[dd], xyz_star[dd]));
      }

      interp_nm1.add_point(idx, xyz_star);
      interp_n  .add_point(idx++, xyz_star);
    }
  }
  P4EST_ASSERT(idx == ndata); idx = 0;

  /* compute the velocities at the intermediate point */
  std::vector<double> vn_star  [P4EST_DIM];
  std::vector<double> vnm1_star[P4EST_DIM];
  double *data_star_nm1[P4EST_DIM];
  double *data_star_n[P4EST_DIM];
  for(short dir=0; dir<P4EST_DIM; ++dir)
  {
    vnm1_star[dir].resize(ndata);
    data_star_nm1[dir] = vnm1_star[dir].data();
    vn_star[dir].resize(ndata);
    data_star_n[dir] = vn_star[dir].data();
  }
  if(use_second_derivatives_vnm1)
#ifdef P4_TO_P8
    interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2], quadratic, P4EST_DIM);
#else
    interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], quadratic, P4EST_DIM);
#endif
  else
    interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
  interp_nm1.interpolate(data_star_nm1, P4EST_DIM);
  if(use_second_derivatives_vn)
#ifdef P4_TO_P8
    interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2], quadratic, P4EST_DIM);
#else
    interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], quadratic, P4EST_DIM);
#endif
  else
    interp_n.set_input(vn, quadratic, P4EST_DIM);
  interp_n.interpolate(data_star_n, P4EST_DIM);
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (unsigned int k = 0; k < n_lists; ++k) {
    (*xyz_n[k][0]).resize(ndata_in_list[k]);
    (*xyz_n[k][1]).resize(ndata_in_list[k]);
#ifdef P4_TO_P8
    (*xyz_n[k][2]).resize(ndata_in_list[k]);
#endif
    for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
      (*xyz_n[k][0])[lidx] = xyz_np1[k][0][lidx] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[0][idx] - .5*dt_n/dt_nm1*vnm1_star[0][idx] );
      (*xyz_n[k][1])[lidx] = xyz_np1[k][1][lidx] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[1][idx] - .5*dt_n/dt_nm1*vnm1_star[1][idx] );
    #ifdef P4_TO_P8
      (*xyz_n[k][2])[lidx] = xyz_np1[k][2][lidx] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[2][idx] - .5*dt_n/dt_nm1*vnm1_star[2][idx] );
    #endif

      for(short dd=0; dd<P4EST_DIM; ++dd)
      {
        if      (is_periodic(p4est_n,dd) && (*xyz_n[k][dd])[lidx]<xyz_min[dd]) (*xyz_n[k][dd])[lidx] += xyz_max[dd]-xyz_min[dd];
        else if (is_periodic(p4est_n,dd) && (*xyz_n[k][dd])[lidx]>xyz_max[dd]) (*xyz_n[k][dd])[lidx] -= xyz_max[dd]-xyz_min[dd];
        else                                                                (*xyz_n[k][dd])[lidx] = MAX(xyz_min[dd], MIN(xyz_max[dd], (*xyz_n[k][dd])[lidx]));
      }
      idx++;
    }
  }
  P4EST_ASSERT(idx == ndata); idx = 0;

  // EXTRA STUFF FOR FINDING xyz_nm1 ONLY (for second-order bdf advection terms, for instance)
  if(xyz_nm1 != NULL)
  {
    /* proceed similarly for the departure point at time nm1 */
    for (unsigned int k = 0; k < n_lists; ++k) {
      for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
        /* find the intermediate point */
        double xyz_star[] =
        {
          xyz_np1[k][0][lidx] - .5*(dt_n+dt_nm1)*vnp1[0][idx],
          xyz_np1[k][1][lidx] - .5*(dt_n+dt_nm1)*vnp1[1][idx]
    #ifdef P4_TO_P8
          , xyz_np1[k][2][lidx] - .5*(dt_n+dt_nm1)*vnp1[2][idx]
    #endif
        };

        for(int dd=0; dd<P4EST_DIM; ++dd)
        {
          if      (is_periodic(p4est_n,dd) && xyz_star[dd]<xyz_min[dd]) xyz_star[dd] += xyz_max[dd]-xyz_min[dd];
          else if (is_periodic(p4est_n,dd) && xyz_star[dd]>xyz_max[dd]) xyz_star[dd] -= xyz_max[dd]-xyz_min[dd];
          else                                                          xyz_star[dd] = MAX(xyz_min[dd], MIN(xyz_max[dd], xyz_star[dd]));
        }

        interp_n  .add_point(idx, xyz_star);
        interp_nm1.add_point(idx++, xyz_star);
      }
    }
    P4EST_ASSERT(idx == ndata); idx = 0;

    /* compute the velocities at the intermediate point */
    if(use_second_derivatives_vnm1)
#ifdef P4_TO_P8
      interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2], quadratic, P4EST_DIM);
#else
      interp_nm1.set_input(vnm1, second_derivatives_vnm1[0], second_derivatives_vnm1[1], quadratic, P4EST_DIM);
#endif
    else
      interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
    interp_nm1.interpolate(data_star_nm1, P4EST_DIM);
    if(use_second_derivatives_vn)
#ifdef P4_TO_P8
      interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2], quadratic, P4EST_DIM);
#else
      interp_n.set_input(vn, second_derivatives_vn[0], second_derivatives_vn[1], quadratic, P4EST_DIM);
#endif
    else
      interp_n.set_input(vn, quadratic, P4EST_DIM);
    interp_n.interpolate(data_star_n, P4EST_DIM);
    interp_nm1.clear();
    interp_n  .clear();

    /* now find the departure point at time nm1 */
    idx = 0;
    for (unsigned int k = 0; k < n_lists; ++k) {
      (*xyz_nm1[k][0]).resize(ndata_in_list[k]);
      (*xyz_nm1[k][1]).resize(ndata_in_list[k]);
#ifdef P4_TO_P8
      (*xyz_nm1[k][2]).resize(ndata_in_list[k]);
#endif
      for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
        (*xyz_nm1[k][0])[lidx] = xyz_np1[k][0][lidx] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[0][idx] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[0][idx] );
        (*xyz_nm1[k][1])[lidx] = xyz_np1[k][1][lidx] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[1][idx] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[1][idx] );
#ifdef P4_TO_P8
        (*xyz_nm1[k][2])[lidx] = xyz_np1[k][2][lidx] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[2][idx] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[2][idx] );
#endif

        for(int dd=0; dd<P4EST_DIM; ++dd)
        {
          if      (is_periodic(p4est_n,dd) && (*xyz_nm1[k][dd])[lidx]<xyz_min[dd])  (*xyz_nm1[k][dd])[lidx] += xyz_max[dd]-xyz_min[dd];
          else if (is_periodic(p4est_n,dd) && (*xyz_nm1[k][dd])[lidx]>xyz_max[dd])  (*xyz_nm1[k][dd])[lidx] -= xyz_max[dd]-xyz_min[dd];
          else                                                                      (*xyz_nm1[k][dd])[lidx] = MAX(xyz_min[dd], MIN(xyz_max[dd], (*xyz_nm1[k][dd])[lidx]));
        }
        idx++;
      }
    }
    P4EST_ASSERT(idx == ndata); idx = 0;
  }

  ierr = PetscLogEventEnd(log_trajectory_from_np1_bunch_of_points, 0, 0, 0, 0); CHKERRXX(ierr);
}
