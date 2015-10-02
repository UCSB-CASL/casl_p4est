#include "my_p4est_trajectory_of_point.h"

#ifdef P4_TO_P8
#include "my_p8est_interpolation_nodes.h"
#else
#include "my_p4est_interpolation_nodes.h"
#endif

extern PetscLogEvent log_trajectory_from_np1_to_n;
extern PetscLogEvent log_trajectory_from_np1_to_nm1;
extern PetscLogEvent log_trajectory_from_np1_to_nm1_faces;

void trajectory_from_np1_to_n( p4est_t *p4est, p4est_nodes_t *nodes,
                               my_p4est_node_neighbors_t *ngbd_n,
                               double dt,
                               Vec v[P4EST_DIM],
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

    xyz_star[0] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_star[0]));
    xyz_star[1] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_star[1]));
#ifdef P4_TO_P8
    xyz_star[2] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_star[2]));
#endif

    interp.add_point(n, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vstar[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
    vstar[dir].resize(nodes->num_owned_indeps);
    interp.set_input(v[dir], quadratic);
    interp.interpolate(vstar[dir].data());
  }

  /* find the departure points */
  for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);

    xyz_d[0][n] = xyz[0] - dt*vstar[0][n];
    xyz_d[1][n] = xyz[1] - dt*vstar[1][n];
#ifdef P4_TO_P8
    xyz_d[2][n] = xyz[2] - dt*vstar[2][n];
#endif

    xyz_d[0][n] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_d[0][n]));
    xyz_d[1][n] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_d[1][n]));
#ifdef P4_TO_P8
    xyz_d[2][n] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_d[2][n]));
#endif
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

    xyz_star[0] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_star[0]));
    xyz_star[1] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_star[1]));
#ifdef P4_TO_P8
    xyz_star[2] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_star[2]));
#endif

    interp_n  .add_point(n, xyz_star);
    interp_nm1.add_point(n, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vnm1_star[P4EST_DIM];
  std::vector<double> vn_star  [P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vn[dir], &v_p[dir]); CHKERRXX(ierr);

    vnm1_star[dir].resize(nodes_n->num_owned_indeps);
    interp_nm1.set_input(vnm1[dir], quadratic);
    interp_nm1.interpolate(vnm1_star[dir].data());

    vn_star[dir].resize(nodes_n->num_owned_indeps);
    interp_n.set_input(vn[dir], quadratic);
    interp_n.interpolate(vn_star[dir].data());
  }
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);

    xyz_n[0][n] = xyz[0] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[0][n] - .5*dt_n/dt_nm1*vnm1_star[0][n] );
    xyz_n[1][n] = xyz[1] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[1][n] - .5*dt_n/dt_nm1*vnm1_star[1][n] );
#ifdef P4_TO_P8
    xyz_n[2][n] = xyz[2] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[2][n] - .5*dt_n/dt_nm1*vnm1_star[2][n] );
#endif

    xyz_n[0][n] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_n[0][n]));
    xyz_n[1][n] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_n[1][n]));
#ifdef P4_TO_P8
    xyz_n[2][n] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_n[2][n]));
#endif
  }



  /* proceed similarly for the departure point at time nm1 */
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArrayRead(vn[dir], &v_p[dir]); CHKERRXX(ierr); }

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

    xyz_star[0] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_star[0]));
    xyz_star[1] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_star[1]));
#ifdef P4_TO_P8
    xyz_star[2] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_star[2]));
#endif

    interp_n  .add_point(n, xyz_star);
    interp_nm1.add_point(n, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vn[dir], &v_p[dir]); CHKERRXX(ierr);

    interp_nm1.set_input(vnm1[dir], quadratic);
    interp_nm1.interpolate(vnm1_star[dir].data());

    interp_n.set_input(vn[dir], quadratic);
    interp_n.interpolate(vn_star[dir].data());
  }
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);

    xyz_nm1[0][n] = xyz[0] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[0][n] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[0][n] );
    xyz_nm1[1][n] = xyz[1] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[1][n] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[1][n] );
#ifdef P4_TO_P8
    xyz_nm1[2][n] = xyz[2] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[2][n] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[2][n] );
#endif

    xyz_nm1[0][n] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_nm1[0][n]));
    xyz_nm1[1][n] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_nm1[1][n]));
#ifdef P4_TO_P8
    xyz_nm1[2][n] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_nm1[2][n]));
#endif
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
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_trajectory_from_np1_to_nm1_faces, 0, 0, 0, 0); CHKERRXX(ierr);

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

  /* first find the velocity at the faces */
  my_p4est_interpolation_nodes_t interp_np1(ngbd_n);
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);
    interp_np1.add_point(f_idx, xyz);
  }
  std::vector<double> vnp1[P4EST_DIM];
  for(int dd=0; dd<P4EST_DIM; ++dd)
  {
    vnp1[dd].resize(faces_n->num_local[dir]);
    interp_np1.set_input(vn[dd], quadratic);
    interp_np1.interpolate(vnp1[dd].data());
  }

  /* find xyz_star */
  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);

    /* find the intermediate point */
    double xyz_star[] =
    {
      xyz[0] - .5*dt_n*vnp1[0][f_idx],
      xyz[1] - .5*dt_n*vnp1[1][f_idx]
  #ifdef P4_TO_P8
      ,xyz[2] - .5*dt_n*vnp1[2][f_idx]
  #endif
    };

    xyz_star[0] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_star[0]));
    xyz_star[1] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_star[1]));
#ifdef P4_TO_P8
    xyz_star[2] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_star[2]));
#endif

    interp_nm1.add_point(f_idx, xyz_star);
    interp_n  .add_point(f_idx, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vn_star  [P4EST_DIM];
  std::vector<double> vnm1_star[P4EST_DIM];
  for(int dd=0; dd<P4EST_DIM; ++dd)
  {
    vnm1_star[dd].resize(faces_n->num_local[dir]);
    interp_nm1.set_input(vnm1[dd], quadratic);
    interp_nm1.interpolate(vnm1_star[dd].data());

    vn_star[dd].resize(faces_n->num_local[dir]);
    interp_n.set_input(vn[dd], quadratic);
    interp_n.interpolate(vn_star[dd].data());
  }
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);

    xyz_n[0][f_idx] = xyz[0] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[0][f_idx] - .5*dt_n/dt_nm1*vnm1_star[0][f_idx] );
    xyz_n[1][f_idx] = xyz[1] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[1][f_idx] - .5*dt_n/dt_nm1*vnm1_star[1][f_idx] );
#ifdef P4_TO_P8
    xyz_n[2][f_idx] = xyz[2] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[2][f_idx] - .5*dt_n/dt_nm1*vnm1_star[2][f_idx] );
#endif

    xyz_n[0][f_idx] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_n[0][f_idx]));
    xyz_n[1][f_idx] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_n[1][f_idx]));
#ifdef P4_TO_P8
    xyz_n[2][f_idx] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_n[2][f_idx]));
#endif
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
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_trajectory_from_np1_to_nm1_faces, 0, 0, 0, 0); CHKERRXX(ierr);

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

  /* first find the velocity at the faces */
  my_p4est_interpolation_nodes_t interp_np1(ngbd_n);
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);
    interp_np1.add_point(f_idx, xyz);
  }
  std::vector<double> vnp1[P4EST_DIM];
  for(int dd=0; dd<P4EST_DIM; ++dd)
  {
    vnp1[dd].resize(faces_n->num_local[dir]);
    interp_np1.set_input(vn[dd], quadratic);
    interp_np1.interpolate(vnp1[dd].data());
  }

  /* find xyz_star */
  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);

    /* find the intermediate point */
    double xyz_star[] =
    {
      xyz[0] - .5*dt_n*vnp1[0][f_idx],
      xyz[1] - .5*dt_n*vnp1[1][f_idx]
  #ifdef P4_TO_P8
      ,xyz[2] - .5*dt_n*vnp1[2][f_idx]
  #endif
    };

    xyz_star[0] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_star[0]));
    xyz_star[1] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_star[1]));
#ifdef P4_TO_P8
    xyz_star[2] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_star[2]));
#endif

    interp_nm1.add_point(f_idx, xyz_star);
    interp_n  .add_point(f_idx, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vn_star  [P4EST_DIM];
  std::vector<double> vnm1_star[P4EST_DIM];
  for(int dd=0; dd<P4EST_DIM; ++dd)
  {
    vnm1_star[dd].resize(faces_n->num_local[dir]);
    interp_nm1.set_input(vnm1[dd], quadratic);
    interp_nm1.interpolate(vnm1_star[dd].data());

    vn_star[dd].resize(faces_n->num_local[dir]);
    interp_n.set_input(vn[dd], quadratic);
    interp_n.interpolate(vn_star[dd].data());
  }
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);

    xyz_n[0][f_idx] = xyz[0] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[0][f_idx] - .5*dt_n/dt_nm1*vnm1_star[0][f_idx] );
    xyz_n[1][f_idx] = xyz[1] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[1][f_idx] - .5*dt_n/dt_nm1*vnm1_star[1][f_idx] );
#ifdef P4_TO_P8
    xyz_n[2][f_idx] = xyz[2] - dt_n* ( (1+.5*dt_n/dt_nm1)*vn_star[2][f_idx] - .5*dt_n/dt_nm1*vnm1_star[2][f_idx] );
#endif

    xyz_n[0][f_idx] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_n[0][f_idx]));
    xyz_n[1][f_idx] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_n[1][f_idx]));
#ifdef P4_TO_P8
    xyz_n[2][f_idx] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_n[2][f_idx]));
#endif
  }

  /* proceed similarly for the departure point at time nm1 */
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);

    /* find the intermediate point */
    double xyz_star[] =
    {
      xyz[0] - .5*(dt_n+dt_nm1)*vnp1[0][f_idx],
      xyz[1] - .5*(dt_n+dt_nm1)*vnp1[1][f_idx]
  #ifdef P4_TO_P8
      ,xyz[2] - .5*(dt_n+dt_nm1)*vnp1[2][f_idx]
  #endif
    };

    xyz_star[0] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_star[0]));
    xyz_star[1] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_star[1]));
#ifdef P4_TO_P8
    xyz_star[2] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_star[2]));
#endif

    interp_n  .add_point(f_idx, xyz_star);
    interp_nm1.add_point(f_idx, xyz_star);
  }

  /* compute the velocities at the intermediate point */
  for(int dd=0; dd<P4EST_DIM; ++dd)
  {
    interp_nm1.set_input(vnm1[dd], quadratic);
    interp_nm1.interpolate(vnm1_star[dd].data());

    interp_n.set_input(vn[dd], quadratic);
    interp_n.interpolate(vn_star[dd].data());
  }
  interp_nm1.clear();
  interp_n  .clear();

  /* now find the departure point at time n */
  for (p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM];
    faces_n->xyz_fr_f(f_idx, dir, xyz);

    xyz_nm1[0][f_idx] = xyz[0] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[0][f_idx] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[0][f_idx] );
    xyz_nm1[1][f_idx] = xyz[1] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[1][f_idx] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[1][f_idx] );
#ifdef P4_TO_P8
    xyz_nm1[2][f_idx] = xyz[2] - (dt_n+dt_nm1) * ( (1+.5*(dt_n-dt_nm1)/dt_nm1)*vn_star[2][f_idx] - .5*(dt_n-dt_nm1)/dt_nm1*vnm1_star[2][f_idx] );
#endif

    xyz_nm1[0][f_idx] = MAX(xyz_min[0], MIN(xyz_max[0], xyz_nm1[0][f_idx]));
    xyz_nm1[1][f_idx] = MAX(xyz_min[1], MIN(xyz_max[1], xyz_nm1[1][f_idx]));
#ifdef P4_TO_P8
    xyz_nm1[2][f_idx] = MAX(xyz_min[2], MIN(xyz_max[2], xyz_nm1[2][f_idx]));
#endif
  }
}
