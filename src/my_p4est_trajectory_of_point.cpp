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
  bool  periodic[P4EST_DIM];

  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (unsigned char dir=0; dir < P4EST_DIM; dir++)
  {
    xyz_min[dir]  = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
    xyz_max[dir]  = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];
    periodic[dir] = is_periodic(p4est, dir);
  }

  my_p4est_interpolation_nodes_t interp(ngbd_n);
  bool use_second_derivatives_v = (second_derivatives_v!=NULL);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
      use_second_derivatives_v = use_second_derivatives_v && (second_derivatives_v[dd][dim] != NULL);

  const double *v_p[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) { ierr = VecGetArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr); }

  for (p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);

    /* find the intermediate point */
    double xyz_star[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_star[dir] = xyz[dir] - .5*dt*v_p[dir][n];
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

    interp.add_point(n, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vstar[P4EST_DIM];
  double *data_star[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
    vstar[dir].resize(nodes->num_owned_indeps);
    data_star[dir] = vstar[dir].data();
  }
  if(use_second_derivatives_v)
    interp.set_input(v, DIM(second_derivatives_v[0], second_derivatives_v[1], second_derivatives_v[2]), quadratic, P4EST_DIM);
  else
    interp.set_input(v, quadratic, P4EST_DIM);
  interp.interpolate(data_star);

  /* find the departure points */
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    xyz_d[dir].resize(nodes->num_owned_indeps);
  for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    double xyz_[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz_);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_[dir] -= dt*vstar[dir][n];
    clip_in_domain(xyz_, xyz_min, xyz_max, periodic);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_d[dir][n] = xyz_[dir];
  }

  ierr = PetscLogEventEnd(log_trajectory_from_np1_to_n, 0, 0, 0, 0); CHKERRXX(ierr);
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
  bool  periodic[P4EST_DIM];

  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (unsigned char dir=0; dir < P4EST_DIM; dir++)
  {
    xyz_min[dir]  = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
    xyz_max[dir]  = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];
    periodic[dir] = is_periodic(p4est_n, dir);
  }

  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
  bool use_second_derivatives_vnm1 = (second_derivatives_vnm1!=NULL);
  bool use_second_derivatives_vn = (second_derivatives_vn!=NULL);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
    {
      use_second_derivatives_vnm1 = use_second_derivatives_vnm1 && (second_derivatives_vnm1[dd][dim] != NULL);
      use_second_derivatives_vn   = use_second_derivatives_vn   && (second_derivatives_vn[dd][dim] != NULL);
    }

  const double *v_p[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir){ ierr = VecGetArrayRead(vn[dir], &v_p[dir]); CHKERRXX(ierr); }

  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);

    /* find the intermediate point */
    double xyz_star[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_star[dir] = xyz[dir] - .5*dt_n*v_p[dir][n];
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

    interp_n  .add_point(n, xyz_star);
    interp_nm1.add_point(n, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vnm1_star[P4EST_DIM];
  std::vector<double> vn_star  [P4EST_DIM];
  double *data_star_nm1[P4EST_DIM];
  double *data_star_n[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vnm1_star[dir].resize(nodes_n->num_owned_indeps);
    data_star_nm1[dir] = vnm1_star[dir].data();
    vn_star[dir].resize(nodes_n->num_owned_indeps);
    data_star_n[dir] = vn_star[dir].data();
  }
  if(use_second_derivatives_vnm1)
    interp_nm1.set_input(vnm1, DIM(second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2]), quadratic, P4EST_DIM);
  else
    interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
  interp_nm1.interpolate(data_star_nm1);
  if(use_second_derivatives_vn)
    interp_n.set_input(vn, DIM(second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2]), quadratic, P4EST_DIM);
  else
    interp_n.set_input(vn, quadratic, P4EST_DIM);
  interp_n.interpolate(data_star_n);
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    xyz_n[dim].resize(nodes_n->num_owned_indeps);
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz_[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_[dir] -= dt_n*((1+.5*dt_n/dt_nm1)*vn_star[dir][n] - .5*dt_n/dt_nm1*vnm1_star[dir][n]);
    clip_in_domain(xyz_, xyz_min, xyz_max, periodic);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_n[dir][n] = xyz_[dir];
  }

  /* proceed similarly for the departure point at time nm1 */
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz_star[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_star);

    /* find the intermediate point */
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_star[dir] -= .5*(dt_n+dt_nm1)*v_p[dir][n];
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

    interp_n  .add_point(n, xyz_star);
    interp_nm1.add_point(n, xyz_star);
  }

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vn[dir], &v_p[dir]); CHKERRXX(ierr); }

  /* compute the velocities at the intermediate point */
  if(use_second_derivatives_vnm1)
    interp_nm1.set_input(vnm1, DIM(second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2]), quadratic, P4EST_DIM);
  else
    interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
  interp_nm1.interpolate(data_star_nm1);
  if(use_second_derivatives_vn)
    interp_n.set_input(vn, DIM(second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2]), quadratic, P4EST_DIM);
  else
    interp_n.set_input(vn, quadratic, P4EST_DIM);
  interp_n.interpolate(data_star_n);
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time nm1 */
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    xyz_nm1[dim].resize(nodes_n->num_owned_indeps);
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz_[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_[dir] -= (dt_n+dt_nm1)*((1.0+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[dir][n] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[dir][n]);
    clip_in_domain(xyz_, xyz_min, xyz_max, periodic);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_nm1[dir][n] = xyz_[dir];
  }
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
  for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
  {
    xyz_np1[comp].resize(faces_n->num_local[dir], 0);
    pointers_to_xyz_n[0][comp]    = &xyz_n[comp];
    pointers_to_xyz_nm1[0][comp]  = ((xyz_nm1 != NULL)? &xyz_nm1[comp]: NULL);
  }
  for (unsigned int f_idx = 0; f_idx < (unsigned int)faces_n->num_local[dir]; ++f_idx) {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_np1[dir][f_idx] = xyz[dir];
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
                                    std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM],
                                    std::vector<double> xyz_nm1[P4EST_DIM][P4EST_DIM])
{
  std::vector<double> xyz_np1[P4EST_DIM][P4EST_DIM];
  std::vector<double> *pointers_to_xyz_n[P4EST_DIM][P4EST_DIM];
  std::vector<double> *pointers_to_xyz_nm1[P4EST_DIM][P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (unsigned char comp = 0; comp < P4EST_DIM; ++comp) {
      xyz_np1[dir][comp].resize(faces_n->num_local[dir], 0);
      pointers_to_xyz_n[dir][comp] = &xyz_n[dir][comp];
      pointers_to_xyz_nm1[dir][comp] = ((xyz_nm1 != NULL)? &xyz_nm1[dir][comp] : NULL);
    }
    for (unsigned int f_idx = 0; f_idx < (unsigned int)faces_n->num_local[dir]; ++f_idx) {
      double xyz[P4EST_DIM];
      faces_n->xyz_fr_f(f_idx, dir, xyz);
      for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
        xyz_np1[dir][dd][f_idx] = xyz[dd];
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
    P4EST_ASSERT((ndata_in_list[k]==xyz_np1[k][1].size()) ONLY3D(&& (ndata_in_list[k]==xyz_np1[k][2].size())));
    ndata += ndata_in_list[k];
  }
  P4EST_ASSERT(ndata > 0);

  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  bool  periodic[P4EST_DIM];

  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (unsigned char dd=0; dd < P4EST_DIM; dd++)
  {
    xyz_min[dd]   = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dd];
    xyz_max[dd]   = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dd];
    periodic[dd]  = is_periodic(p4est_n, dd);
  }

  /* first find the velocity at the np1 points */
  my_p4est_interpolation_nodes_t interp_np1(ngbd_n);
  bool use_second_derivatives_vnm1 = (second_derivatives_vnm1!=NULL);
  bool use_second_derivatives_vn = (second_derivatives_vn!=NULL);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      use_second_derivatives_vnm1 = use_second_derivatives_vnm1 && (second_derivatives_vnm1[dd][dim] != NULL);
      use_second_derivatives_vn   = use_second_derivatives_vn   && (second_derivatives_vn[dd][dim] != NULL);
    }

  double xyz_tmp[P4EST_DIM];
  unsigned int idx = 0;
  for (unsigned int k = 0; k < n_lists; ++k)
    for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
      for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        xyz_tmp[comp] = xyz_np1[k][comp][lidx];
      interp_np1.add_point(idx++, xyz_tmp);
    }

  P4EST_ASSERT(idx == ndata); idx = 0;
  std::vector<double> vnp1[P4EST_DIM];
  double *data[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vnp1[dir].resize(ndata);
    data[dir] = vnp1[dir].data();
  }
  if (use_second_derivatives_vn)
    interp_np1.set_input(vn, DIM(second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2]), quadratic, P4EST_DIM);
  else
    interp_np1.set_input(vn, quadratic, P4EST_DIM);
  interp_np1.interpolate(data);


  /* find xyz_star */
  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );

  for (unsigned int k = 0; k < n_lists; ++k)
    for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
      /* find the intermediate point */
      double xyz_star[P4EST_DIM];
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        xyz_star[dir] = xyz_np1[k][dir][lidx] - 0.5*dt_n*vnp1[dir][lidx];
      clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

      interp_nm1.add_point(idx, xyz_star);
      interp_n  .add_point(idx++, xyz_star);
    }

  P4EST_ASSERT(idx == ndata); idx = 0;

  /* compute the velocities at the intermediate point */
  std::vector<double> vn_star  [P4EST_DIM];
  std::vector<double> vnm1_star[P4EST_DIM];
  double *data_star_nm1[P4EST_DIM];
  double *data_star_n[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vnm1_star[dir].resize(ndata);
    data_star_nm1[dir] = vnm1_star[dir].data();
    vn_star[dir].resize(ndata);
    data_star_n[dir] = vn_star[dir].data();
  }
  if(use_second_derivatives_vnm1)
    interp_nm1.set_input(vnm1, DIM(second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2]), quadratic, P4EST_DIM);
  else
    interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
  interp_nm1.interpolate(data_star_nm1);
  if(use_second_derivatives_vn)
    interp_n.set_input(vn, DIM(second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2]), quadratic, P4EST_DIM);
  else
    interp_n.set_input(vn, quadratic, P4EST_DIM);
  interp_n.interpolate(data_star_n);
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (unsigned int k = 0; k < n_lists; ++k) {
    for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
      (*xyz_n[k][comp]).resize(ndata_in_list[k]);
    for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
      for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        xyz_tmp[comp] = xyz_np1[k][comp][lidx] - dt_n*((1.0+.5*dt_n/dt_nm1)*vn_star[comp][idx] - .5*dt_n/dt_nm1*vnm1_star[comp][idx]);
      clip_in_domain(xyz_tmp, xyz_min, xyz_max, periodic);
      for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        (*xyz_n[k][comp])[lidx] = xyz_tmp[comp];
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
        double xyz_star[P4EST_DIM];
        for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
          xyz_star[comp] = xyz_np1[k][comp][lidx] - 0.5*(dt_n+dt_nm1)*vnp1[comp][idx];
        clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

        interp_n  .add_point(idx, xyz_star);
        interp_nm1.add_point(idx++, xyz_star);
      }
    }
    P4EST_ASSERT(idx == ndata); idx = 0;

    /* compute the velocities at the intermediate point */
    if(use_second_derivatives_vnm1)
      interp_nm1.set_input(vnm1, DIM(second_derivatives_vnm1[0], second_derivatives_vnm1[1], second_derivatives_vnm1[2]), quadratic, P4EST_DIM);
    else
      interp_nm1.set_input(vnm1, quadratic, P4EST_DIM);
    interp_nm1.interpolate(data_star_nm1);
    if(use_second_derivatives_vn)
      interp_n.set_input(vn, DIM(second_derivatives_vn[0], second_derivatives_vn[1], second_derivatives_vn[2]), quadratic, P4EST_DIM);
    else
      interp_n.set_input(vn, quadratic, P4EST_DIM);
    interp_n.interpolate(data_star_n);
    interp_nm1.clear();
    interp_n  .clear();

    /* now find the departure point at time nm1 */
    idx = 0;
    for (unsigned int k = 0; k < n_lists; ++k) {
      for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        (*xyz_nm1[k][comp]).resize(ndata_in_list[k]);
      for (unsigned int lidx = 0; lidx < ndata_in_list[k]; ++lidx) {
        for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
          xyz_tmp[comp] = xyz_np1[k][comp][lidx] - (dt_n+dt_nm1)*((1.0+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[comp][idx] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[comp][idx]);

        clip_in_domain(xyz_tmp, xyz_min, xyz_max, periodic);

        for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
          (*xyz_nm1[k][comp])[lidx] = xyz_tmp[comp];

        idx++;
      }
    }
    P4EST_ASSERT(idx == ndata); idx = 0;
  }

  ierr = PetscLogEventEnd(log_trajectory_from_np1_bunch_of_points, 0, 0, 0, 0); CHKERRXX(ierr);
}
