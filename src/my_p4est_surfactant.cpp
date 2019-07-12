#include "my_p4est_surfactant.h"

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_vtk.h>
#include <p8est_extended.h>
#include <p8est_algorithms.h>
#else
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <p4est_extended.h>
#include <p4est_algorithms.h>
#endif

#include <algorithm>

//#ifndef CASL_LOG_EVENTS
//#undef PetscLogEventBegin
//#undef PetscLogEventEnd
//#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
//#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
//#else
//extern PetscLogEvent log_my_p4est_navier_stokes_viscosity;
//extern PetscLogEvent log_my_p4est_navier_stokes_projection;
//extern PetscLogEvent log_my_p4est_navier_stokes_update;
//#endif


#ifdef P4_TO_P8
double my_p4est_surfactant_t::band::operator ()( double x, double y, double z ) const
#else
double my_p4est_surfactant_t::band::operator ()( double x, double y ) const
#endif
{
#ifdef P4_TO_P8
  double phi_val = _prnt->interp_phi->operator()(x,y,z);
#else
  double phi_val = _prnt->interp_phi->operator()(x,y);
#endif
  return fabs(phi_val) - 0.5*band_width*(_prnt->dxyz_max);
}


void my_p4est_surfactant_t::splitting_criteria_surfactant_t::tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx,
                                                                          my_p4est_interpolation_nodes_t &phi, band *band_p)
{
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  if (quad->level < min_lvl)
    quad->p.user_int = REFINE_QUADRANT;

  else if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT;

  else
  {
    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
#ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (xmax-xmin) * dmin;
    double dy = (ymax-ymin) * dmin;
#ifdef P4_TO_P8
    double dz = (zmax-zmin) * dmin;
#endif

#ifdef P4_TO_P8
    double d = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
#else
    double d = sqrt(SQR(dx) + SQR(dy));
#endif

    double x = (xmax-xmin)*quad_x_fr_i(quad) + xmin;
    double y = (ymax-ymin)*quad_y_fr_j(quad) + ymin;
#ifdef P4_TO_P8
    double z = (zmax-zmin)*quad_z_fr_k(quad) + zmin;
#endif

    bool coar_band = true;
    bool coar_intf = true;
    for(int i=0; i<2; ++i)
      for(int j=0; j<2; ++j)
#ifdef P4_TO_P8
        for(int k=0; k<2; ++k)
        {
          coar_band = coar_band && fabs(band_p->operator ()(x+i*dx, y+j*dy, z+k*dz))>(band_p->lip)*2*dmin;
          coar_intf = coar_intf && fabs(phi(x+i*dx, y+j*dy, z+k*dz))>lip*2*d;
#else
        {
          coar_band = coar_band && fabs(band_p->operator ()(x+i*dx, y+j*dy))>(band_p->lip)*2*dmin;
          coar_intf = coar_intf && fabs(phi(x+i*dx, y+j*dy))>lip*2*d;
#endif
        }

    bool coarsen = true;
    coarsen = coar_intf && coar_band;
    coarsen = coarsen && quad->level > min_lvl;

    bool ref_band = false;
    bool ref_intf = false;
    for(int i=0; i<3; ++i)
      for(int j=0; j<3; ++j)
#ifdef P4_TO_P8
        for(int k=0; k<3; ++k)
        {
          ref_band = ref_band || fabs(band_p->operator ()(x+i*dx/2, y+j*dy/2, z+k*dz/2))<=(band_p->lip)*dmin;
          ref_intf = ref_intf || fabs(phi(x+i*dx/2, y+j*dy/2, z+k*dz/2))<=lip*d;
#else
        {
          ref_band = ref_band || fabs(band_p->operator ()(x+i*dx/2, y+j*dy/2))<=(band_p->lip)*dmin;
          ref_intf = ref_intf || fabs(phi(x+i*dx/2, y+j*dy/2))<=lip*d;
#endif
        }

    bool refine = false;
    refine = ref_intf || ref_band;
    refine = refine && quad->level < max_lvl;

    if (refine)
      quad->p.user_int = REFINE_QUADRANT;

    else if (coarsen)
      quad->p.user_int = COARSEN_QUADRANT;

    else
      quad->p.user_int = SKIP_QUADRANT;
  }
}


bool my_p4est_surfactant_t::splitting_criteria_surfactant_t::refine_and_coarsen(p4est_t* p4est, my_p4est_node_neighbors_t *ngbd_n, Vec phi)
{
    my_p4est_interpolation_nodes_t interp_phi(ngbd_n);
    interp_phi.set_input(phi, linear);

    /* tag the quadrants that need to be refined or coarsened */
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
            for (size_t q = 0; q <tree->quadrants.elem_count; ++q)
            {
                p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
                tag_quadrant(p4est, quad_idx, tree_idx, interp_phi, &(_prnt->phi_band_gen));
            }
    }

    my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_surfactant_t::coarsen_fn, splitting_criteria_surfactant_t::init_fn);
    my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_surfactant_t::refine_fn,  splitting_criteria_surfactant_t::init_fn);

    int is_grid_changed = false;
    for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it)
    {
        p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, it);
        for (size_t q = 0; q <tree->quadrants.elem_count; ++q)
        {
            p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
            if (quad->p.user_int == NEW_QUADRANT)
            {
                is_grid_changed = true;
                goto function_end;
            }
        }
    }

function_end:
    MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm);
    return is_grid_changed;
}


bool my_p4est_surfactant_t::is_in_domain(const double xyz_[]) const
{
  double threshold[P4EST_DIM];
  for (short dd = 0; dd < P4EST_DIM; ++dd)
    threshold[dd] = 0.1*dxyz[dd];
    return ( (((xyz_[0] - xyz_min[0] > -threshold[0]) && (xyz_[0] - xyz_max[0] < threshold[0])) || is_periodic(p4est_n, dir::x))
          && (((xyz_[1] - xyz_min[1] > -threshold[1]) && (xyz_[1] - xyz_max[1] < threshold[1])) || is_periodic(p4est_n, dir::y))
#ifdef P4_TO_P8
          && (((xyz_[2] - xyz_min[2] > -threshold[2]) && (xyz_[2] - xyz_max[2] < threshold[2])) || is_periodic(p4est_n, dir::z))
#endif
      );
}


void my_p4est_surfactant_t::update_phi_band_vector()
{
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_band); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, phi_band_gen, phi_band);
}


my_p4est_surfactant_t::my_p4est_surfactant_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, const double &band_width)
    : brick(ngbd_n->myb), conn(ngbd_n->p4est->connectivity),
      p4est_nm1(ngbd_nm1->p4est), ghost_nm1(ngbd_nm1->ghost), nodes_nm1(ngbd_nm1->nodes),
      hierarchy_nm1(ngbd_nm1->hierarchy), ngbd_nm1(ngbd_nm1),
      p4est_n(ngbd_n->p4est), ghost_n(ngbd_n->ghost), nodes_n(ngbd_n->nodes),
      hierarchy_n(ngbd_n->hierarchy), ngbd_n(ngbd_n), phi_band_gen(this, band_width)
{
    n_times_dt = 1;
    dt_updated = false;

    double *v2c = p4est_n->connectivity->vertices;
    p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
    p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count-1;
    p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

    splitting_criteria_t *data = (splitting_criteria_t*)p4est_n->user_pointer;

    for (int dir=0; dir<P4EST_DIM; dir++)
    {
        xyz_min[dir] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
        xyz_max[dir] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];

        double xyz_tmp = v2c[3*t2v[P4EST_CHILDREN*first_tree + last_vertex] + dir];
        dxyz[dir] = (xyz_tmp-xyz_min[dir]) / (1<<data->max_lvl);
        convert_to_xyz[dir] = xyz_tmp-xyz_min[dir];
    }

#ifdef P4_TO_P8
    dxyz_min = MIN(dxyz[0],dxyz[1],dxyz[2]);
    dxyz_max = MAX(dxyz[0],dxyz[1],dxyz[2]);
#else
    dxyz_min = MIN(dxyz[0],dxyz[1]);
    dxyz_max = MAX(dxyz[0],dxyz[1]);
#endif

#ifdef P4_TO_P8
    dt_nm1 = dt_n = .5 * MIN(dxyz[0], dxyz[1], dxyz[2]);
#else
    dt_nm1 = dt_n = .5 * MIN(dxyz[0], dxyz[1]);
#endif

    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
    Vec vec_loc;
    ierr = VecGhostGetLocalForm(phi, &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, -1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi, &vec_loc); CHKERRXX(ierr);

    phi_band = NULL;

    Gamma_np1 = NULL;
    Gamma_n   = NULL;
    Gamma_nm1 = NULL;

    str_n = NULL;
    str_nm1 = NULL;

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        vn_nodes[dir]   = NULL;
        vnm1_nodes[dir] = NULL;
        vn_s_nodes[dir]   = NULL;
        vnm1_s_nodes[dir] = NULL;

        for (unsigned short dd = 0; dd < P4EST_DIM; ++dd)
        {
            dd_vn_nodes[dd][dir]    = NULL;
            dd_vnm1_nodes[dd][dir]  = NULL;
            dd_vn_s_nodes[dd][dir]    = NULL;
            dd_vnm1_s_nodes[dd][dir]  = NULL;
        }

        normal_n  [dir] = NULL;
    }

    interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
    interp_phi->set_input(phi, linear);
}

#ifdef P4_TO_P8
my_p4est_surfactant_t::my_p4est_surfactant_t(const mpi_environment_t& mpi, my_p4est_brick_t *brick_input, p8est_connectivity *conn_input,
                                             const int& lmin, const int& lmax, CF_3 *ls_n, CF_3 *ls_nm1, const bool& with_reinit, const double& band_width)
#else
my_p4est_surfactant_t::my_p4est_surfactant_t(const mpi_environment_t& mpi, my_p4est_brick_t* brick_input, p4est_connectivity* conn_input,
                                             const int& lmin, const int& lmax, CF_2 *ls_n, CF_2 *ls_nm1, const bool& with_reinit, const double& band_width)
#endif
  : brick(brick_input), conn(conn_input), phi_band_gen(this, band_width)
{
  // Obtain geometric information from the brick and compute grid sizes
  for(unsigned short dd=0; dd<P4EST_DIM; ++dd)
  {
    xyz_min[dd] = brick->xyz_min[dd];
    xyz_max[dd] = brick->xyz_max[dd];
    dxyz[dd] = (brick->xyz_max[dd]-brick->xyz_min[dd])/(brick->nxyztrees[dd]*(1<<lmax));
    convert_to_xyz[dd] = (brick->xyz_max[dd]-brick->xyz_min[dd])/brick->nxyztrees[dd];
  }

#ifdef P4_TO_P8
  dxyz_min = MIN(dxyz[0],dxyz[1],dxyz[2]);
  dxyz_max = MAX(dxyz[0],dxyz[1],dxyz[2]);
#else
  dxyz_min = MIN(dxyz[0],dxyz[1]);
  dxyz_max = MAX(dxyz[0],dxyz[1]);
#endif

// Create trees and associated structures at time n (initial time)
  ref_data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, ls_n, 1+0.5*band_width, ls_n->lip);
  p4est_n = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);
  p4est_n->user_pointer = (void*) ref_data;
  for(int l=0; l<lmax; ++l)
  {
    my_p4est_refine(p4est_n, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
  }
  ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);

// Create trees and associated structures at time nm1 (previous time step)
  if(ls_nm1==NULL)
  {
    p4est_nm1 = my_p4est_copy(p4est_n,P4EST_FALSE);
    p4est_nm1->connectivity = conn;
    p4est_nm1->user_pointer = (void*) ref_data;
  }
  else
  {
    p4est_nm1 = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);
    delete ref_data;
    ref_data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, ls_nm1, 1+0.5*band_width, ls_nm1->lip);
    p4est_nm1->user_pointer = (void*) ref_data;
    for(int l=0; l<lmax; ++l)
    {
      my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
      my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
    }
  }
  ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  //  Sample and (optionally) reinitialize level-set function
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  set_phi(ls_n, with_reinit);

  // Tree for next time-step and associated classes are set to NULL for now...
  p4est_np1     = NULL;
  ghost_np1     = NULL;
  nodes_np1     = NULL;
  hierarchy_np1 = NULL;
  ngbd_np1      = NULL;

  // Set rest of parameters
  n_times_dt = 1;
  dt_updated = false;

#ifdef P4_TO_P8
  dt_nm1 = dt_n = 1.0 * MIN(dxyz[0], dxyz[1], dxyz[2]);
#else
  dt_nm1 = dt_n = 1.0 * MIN(dxyz[0], dxyz[1]);
#endif

// Initialize variables to NULL

  Gamma_np1 = NULL;
  Gamma_n   = NULL;
  Gamma_nm1 = NULL;

  str_n = NULL;
  str_nm1 = NULL;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vn_nodes[dir]     = NULL;
    vnm1_nodes[dir]   = NULL;
    vn_s_nodes[dir]   = NULL;
    vnm1_s_nodes[dir] = NULL;

    for (unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      dd_vn_nodes[dd][dir]      = NULL;
      dd_vnm1_nodes[dd][dir]    = NULL;
      dd_vn_s_nodes[dd][dir]    = NULL;
      dd_vnm1_s_nodes[dd][dir]  = NULL;
    }
  }

  // Compute normals, curvature, and sample the band l-s function
  for(unsigned short dd=0; dd<P4EST_DIM; ++dd)
  {
    normal_n  [dd] = NULL;
  }
  phi_band = NULL;
  compute_initial_normals_from_phi();
  update_phi_band_vector();
}


my_p4est_surfactant_t::~my_p4est_surfactant_t()
{
  if(phi!=NULL)      { ierr = VecDestroy(phi)      ; CHKERRXX(ierr); }
  if(phi_band!=NULL) { ierr = VecDestroy(phi_band) ; CHKERRXX(ierr); }

  if(Gamma_nm1!=NULL){ ierr = VecDestroy(Gamma_nm1); CHKERRXX(ierr); }
  if(Gamma_n  !=NULL){ ierr = VecDestroy(Gamma_n  ); CHKERRXX(ierr); }
  if(Gamma_np1!=NULL){ ierr = VecDestroy(Gamma_np1); CHKERRXX(ierr); }

  if(str_nm1!=NULL){ ierr = VecDestroy(str_nm1); CHKERRXX(ierr); }
  if(str_n  !=NULL){ ierr = VecDestroy(str_n  ); CHKERRXX(ierr); }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(vn_nodes[dir]  !=NULL)       { ierr = VecDestroy(vn_nodes[dir]  ); CHKERRXX(ierr); }
    if(vnm1_nodes[dir]!=NULL)       { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(vn_s_nodes[dir]  !=NULL)     { ierr = VecDestroy(vn_nodes[dir]  ); CHKERRXX(ierr); }
    if(vnm1_s_nodes[dir]!=NULL)     { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(normal_n[dir]!=NULL)         { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }

    for (unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vn_nodes[dd][dir]!= NULL) { ierr = VecDestroy(dd_vn_nodes[dd][dir]) ; CHKERRXX(ierr); };
      if(dd_vnm1_nodes[dd][dir] != NULL) { ierr = VecDestroy(dd_vnm1_nodes[dd][dir]) ; CHKERRXX(ierr); };
      if(dd_vn_s_nodes[dd][dir]!= NULL) { ierr = VecDestroy(dd_vn_s_nodes[dd][dir]) ; CHKERRXX(ierr); };
      if(dd_vnm1_s_nodes[dd][dir] != NULL) { ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]) ; CHKERRXX(ierr); };
    }
  }

  if(interp_phi!=NULL) delete interp_phi;

  if(ngbd_nm1!=NULL) { delete ngbd_nm1; }
  if(hierarchy_nm1!=NULL) { delete hierarchy_nm1; }
  if(nodes_nm1!=NULL) { p4est_nodes_destroy(nodes_nm1); }
  if(ghost_nm1!=NULL) { p4est_ghost_destroy(ghost_nm1); }
  if(p4est_nm1!=NULL) { p4est_destroy(p4est_nm1); }

  if(ngbd_n!=NULL) { delete ngbd_n; }
  if(hierarchy_n!=NULL) { delete hierarchy_n; }
  if(nodes_n!=NULL) { p4est_nodes_destroy(nodes_n); }
  if(ghost_n!=NULL) { p4est_ghost_destroy(ghost_n); }
  if(p4est_n!=NULL) { p4est_destroy(p4est_n); }

  if(ngbd_np1!=NULL) { delete ngbd_np1; }
  if(hierarchy_np1!=NULL) { delete hierarchy_np1; }
  if(nodes_np1!=NULL) { p4est_nodes_destroy(nodes_np1); }
  if(ghost_np1!=NULL) { p4est_ghost_destroy(ghost_np1); }
  if(p4est_np1!=NULL) { p4est_destroy(p4est_np1); }

  if(ref_data!=NULL) { delete ref_data; }
}


void my_p4est_surfactant_t::set_no_surface_diffusion(const bool &flag)
{
    NO_SURFACE_DIFFUSION = flag;
}


void my_p4est_surfactant_t::set_phi(Vec level_set, const bool& do_reinit)
{
  if(this->phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  this->phi = level_set;
  interp_phi->set_input(level_set, linear);

  if(do_reinit)
  {
    my_p4est_level_set_t lsn(ngbd_n);
    lsn.reinitialize_2nd_order(phi,20);
    lsn.perturb_level_set_function(phi, EPS*dxyz_min);
  }
}


#ifdef P4_TO_P8
void my_p4est_surfactant_t::set_phi(CF_3 *level_set, const bool& do_reinit)
{
#else
void my_p4est_surfactant_t::set_phi(CF_2 *level_set, const bool& do_reinit)
{
#endif
  if(phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, *level_set, phi);
  interp_phi->set_input(phi, linear);

  if(do_reinit)
  {
    my_p4est_level_set_t lsn(ngbd_n);
    lsn.reinitialize_2nd_order(phi);
    lsn.perturb_level_set_function(phi, EPS*dxyz_min);
  }
}


void my_p4est_surfactant_t::compute_initial_normals_from_phi()
{
  double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(normal_n[dir]!=NULL) { ierr = VecDestroy(normal_n[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &normal_n[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_n[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    normal_p[2][n] = qnnn.dz_central(phi_p);
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

    normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
    normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
#ifdef P4_TO_P8
    normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
#endif
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(normal_n[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    normal_p[2][n] = qnnn.dz_central(phi_p);
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

    normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
    normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
#ifdef P4_TO_P8
    normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
#endif
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(normal_n[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(normal_n[dir], &normal_p[dir]); CHKERRXX(ierr);
  }
}


//void my_p4est_surfactant_t::compute_normal_np1()
//{
//    double *normal_p[P4EST_DIM];
//    for(int dir=0; dir<P4EST_DIM; ++dir)
//    {
//        if(normal[dir]!=NULL) { ierr = VecDestroy(normal[dir]); CHKERRXX(ierr); }
//        ierr = VecCreateGhostNodes(p4est_n, nodes_n, &normal[dir]); CHKERRXX(ierr);
//        ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//    }

//    const double *phi_p;
//    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

//    quad_neighbor_nodes_of_node_t qnnn;
//    for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
//    {
//        p4est_locidx_t n = ngbd_n->get_layer_node(i);
//        ngbd_n->get_neighbors(n, qnnn);
//        normal_p[0][n] = qnnn.dx_central(phi_p);
//        normal_p[1][n] = qnnn.dy_central(phi_p);
//  #ifdef P4_TO_P8
//        normal_p[2][n] = qnnn.dz_central(phi_p);
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
//  #else
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
//  #endif

//        normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
//        normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
//  #ifdef P4_TO_P8
//        normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
//  #endif
//    }

//    for(int dir=0; dir<P4EST_DIM; ++dir)
//    {
//        ierr = VecGhostUpdateBegin(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }

//    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
//    {
//        p4est_locidx_t n = ngbd_n->get_local_node(i);
//        ngbd_n->get_neighbors(n, qnnn);
//        normal_p[0][n] = qnnn.dx_central(phi_p);
//        normal_p[1][n] = qnnn.dy_central(phi_p);
//  #ifdef P4_TO_P8
//        normal_p[2][n] = qnnn.dz_central(phi_p);
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
//  #else
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
//  #endif

//        normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
//        normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
//  #ifdef P4_TO_P8
//        normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
//  #endif
//    }
//    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

//    for(int dir=0; dir<P4EST_DIM; ++dir)
//    {
//        ierr = VecGhostUpdateEnd(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }

//    for(int dir=0; dir<P4EST_DIM; ++dir)
//    {
//        ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//    }
//}

#ifdef P4_TO_P8
void my_p4est_surfactant_t::set_Gamma(CF_3 *Gamma_nm1_input, CF_3 *Gamma_n_input)
#else
void my_p4est_surfactant_t::set_Gamma(CF_2 *Gamma_nm1_input, CF_2 *Gamma_n_input)
#endif
{
  if(Gamma_nm1!=NULL) { ierr = VecDestroy(Gamma_nm1); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &Gamma_nm1); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_nm1, nodes_nm1, *Gamma_nm1_input, Gamma_nm1);

  if(Gamma_n!=NULL) { ierr = VecDestroy(Gamma_n); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_n); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, *Gamma_n_input, Gamma_n);
}

//void my_p4est_surfactant_t::set_velocities(Vec *vnm1_nodes, Vec *vn_nodes)
//{
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    this->vn_nodes[dir]   = vn_nodes[dir];
//    this->vnm1_nodes[dir] = vnm1_nodes[dir];
//    ierr = VecDuplicate(vn_nodes[dir], &vnp1_nodes [dir]); CHKERRXX(ierr);

//    ierr = VecDuplicate(face_is_well_defined[dir], &vstar[dir]); CHKERRXX(ierr);
//    ierr = VecDuplicate(face_is_well_defined[dir], &vnp1[dir]); CHKERRXX(ierr);

//  ierr = VecDuplicate(phi, &vorticity); CHKERRXX(ierr);
//}

#ifdef P4_TO_P8
void my_p4est_surfactant_t::set_velocities(CF_3 **vnm1, CF_3 **vn)
#else
void my_p4est_surfactant_t::set_velocities(CF_2 **vnm1, CF_2 **vn)
#endif
{
  double *v_p;
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vnm1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vnm1_nodes[dir], &v_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes_nm1->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_nm1, nodes_nm1, xyz);
#ifdef P4_TO_P8
      v_p[n] = (*vnm1[dir])(xyz[0], xyz[1], xyz[2]);
#else
      v_p[n] = (*vnm1[dir])(xyz[0], xyz[1]);
#endif
    }
    ierr = VecRestoreArray(vnm1_nodes[dir], &v_p); CHKERRXX(ierr);

    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_nodes[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes_n->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
#ifdef P4_TO_P8
      v_p[n] = (*vn[dir])(xyz[0], xyz[1], xyz[2]);
#else
      v_p[n] = (*vn[dir])(xyz[0], xyz[1]);
#endif
    }
    ierr = VecRestoreArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);
  }

  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vnm1_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      }
      if(dd_vn_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vn_nodes[dd][dir]); CHKERRXX(ierr);
      }

      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_n  , nodes_n  , &dd_vn_nodes  [dd][dir]); CHKERRXX(ierr);
    }
#ifdef P4_TO_P8
  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_nodes[0], dd_vnm1_nodes[1], dd_vnm1_nodes[2], P4EST_DIM);
  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_nodes[0],   dd_vn_nodes[1],   dd_vn_nodes[2],   P4EST_DIM);
#else
  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_nodes[0], dd_vnm1_nodes[1], P4EST_DIM);
  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_nodes[0],   dd_vn_nodes[1],   P4EST_DIM);
#endif
}


#ifdef P4_TO_P8
void my_p4est_surfactant_t::compute_extended_velocities(CF_3 *ls_nm1, CF_3 *ls_n, const bool& do_reinit_nm1, const bool& do_reinit_n)
#else
void my_p4est_surfactant_t::compute_extended_velocities(CF_2 *ls_nm1, CF_2 *ls_n, const bool& do_reinit_nm1, const bool& do_reinit_n)
#endif
{
  my_p4est_level_set_t ls_extend_nm1(ngbd_nm1);
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir){
    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vnm1_s_nodes[dir]); CHKERRXX(ierr); }
  Vec phi_nm1_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &phi_nm1_temp); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_nm1, nodes_nm1, *ls_nm1, phi_nm1_temp);
  if(do_reinit_nm1)
  {
    ls_extend_nm1.reinitialize_2nd_order(phi_nm1_temp);
    ls_extend_nm1.perturb_level_set_function(phi_nm1_temp, EPS*dxyz_min);
  }
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
    ls_extend_nm1.extend_from_interface_to_whole_domain_TVD(phi_nm1_temp, vnm1_nodes[dir], vnm1_s_nodes[dir]);

  my_p4est_level_set_t ls_extend_n(ngbd_n);
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir){
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_s_nodes[dir]); CHKERRXX(ierr); }
  if(ls_n==NULL)
  {
    for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
    {
      ls_extend_n.extend_from_interface_to_whole_domain_TVD(phi, vn_nodes[dir], vn_s_nodes[dir]);
    }
  }
  else
  {
    Vec phi_n_temp = NULL;
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n_temp); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, *ls_n, phi_n_temp);
    if(do_reinit_n)
    {
      ls_extend_n.reinitialize_2nd_order(phi_n_temp);
      ls_extend_n.perturb_level_set_function(phi_n_temp, EPS*dxyz_min);
    }
    for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
    {
      ls_extend_n.extend_from_interface_to_whole_domain_TVD(phi_n_temp, vn_nodes[dir], vn_s_nodes[dir]);
    }
  }

  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vnm1_s_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
      }
      if(dd_vn_s_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vn_s_nodes[dd][dir]); CHKERRXX(ierr);
      }

      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_n  , nodes_n  , &dd_vn_s_nodes  [dd][dir]); CHKERRXX(ierr);
    }
#ifdef P4_TO_P8
  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], dd_vnm1_s_nodes[2], P4EST_DIM);
  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_s_nodes[0],   dd_vn_s_nodes[1],   dd_vn_s_nodes[2],   P4EST_DIM);
#else
  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], P4EST_DIM);
  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_s_nodes[0],   dd_vn_s_nodes[1],   P4EST_DIM);
#endif

  compute_stretching_term_nm1(ls_nm1);
  compute_stretching_term_n();
}


#ifdef P4_TO_P8
void my_p4est_surfactant_t::compute_stretching_term_nm1(CF_3 *ls_nm1)
#else
void my_p4est_surfactant_t::compute_stretching_term_nm1(CF_2 *ls_nm1)
#endif
{
  Vec str_nm1_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &str_nm1_temp); CHKERRXX(ierr);
  double *str_nm1_p;
  ierr = VecGetArray(str_nm1_temp, &str_nm1_p); CHKERRXX(ierr);

  const double *vnm1_s_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir){ ierr = VecGetArrayRead(vnm1_s_nodes[dir], &vnm1_s_p[dir]); CHKERRXX(ierr); }

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_nm1->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_nm1->get_layer_node(i);
    ngbd_nm1->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    str_nm1_p[n] = qnnn.dx_central(vnm1_s_p[0]) + qnnn.dy_central(vnm1_s_p[1]) + qnnn.dz_central(vnm1_s_p[2]);
#else
    str_nm1_p[n] = qnnn.dx_central(vnm1_s_p[0]) + qnnn.dy_central(vnm1_s_p[1]);
#endif
  }

  ierr = VecGhostUpdateBegin(str_nm1_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_nm1->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_nm1->get_local_node(i);
    ngbd_nm1->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    str_nm1_p[n] = qnnn.dx_central(vnm1_s_p[0]) + qnnn.dy_central(vnm1_s_p[1]) + qnnn.dz_central(vnm1_s_p[2]);
#else
    str_nm1_p[n] = qnnn.dx_central(vnm1_s_p[0]) + qnnn.dy_central(vnm1_s_p[1]);
#endif
  }

  ierr = VecGhostUpdateEnd(str_nm1_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vnm1_s_nodes[dir], &vnm1_s_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArray(str_nm1_temp, &str_nm1_p); CHKERRXX(ierr);

  Vec phi_nm1_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &phi_nm1_temp); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_nm1, nodes_nm1, *ls_nm1, phi_nm1_temp);
  my_p4est_level_set_t ls(ngbd_nm1);
  if(str_nm1!=NULL) { ierr = VecDestroy(str_nm1); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &str_nm1); CHKERRXX(ierr);
  ls.extend_from_interface_to_whole_domain_TVD(phi_nm1_temp, str_nm1_temp, str_nm1);
}

#ifdef P4_TO_P8
void my_p4est_surfactant_t::compute_stretching_term_n(CF_3 *ls_n)
#else
void my_p4est_surfactant_t::compute_stretching_term_n(CF_2 *ls_n)
#endif
{
  Vec str_n_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_n_temp); CHKERRXX(ierr);
  double *str_n_p;
  ierr = VecGetArray(str_n_temp, &str_n_p); CHKERRXX(ierr);

  const double *vn_s_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir){ ierr = VecGetArrayRead(vn_s_nodes[dir], &vn_s_p[dir]); CHKERRXX(ierr); }

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]) + qnnn.dz_central(vn_s_p[2]);
#else
    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]);
#endif
  }

  ierr = VecGhostUpdateBegin(str_n_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]) + qnnn.dz_central(vn_s_p[2]);
#else
    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]);
#endif
  }

  ierr = VecGhostUpdateEnd(str_n_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vn_s_nodes[dir], &vn_s_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArray(str_n_temp, &str_n_p); CHKERRXX(ierr);

  if(str_n!=NULL) { ierr = VecDestroy(str_n); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_n); CHKERRXX(ierr);
  my_p4est_level_set_t ls(ngbd_n);

  if(ls_n==NULL)
  {
    ls.extend_from_interface_to_whole_domain_TVD(phi, str_n_temp, str_n);
  }
  else
  {
    Vec phi_n_temp = NULL;
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n_temp); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, *ls_n, phi_n_temp);
    ls.extend_from_interface_to_whole_domain_TVD(phi_n_temp, str_n_temp, str_n);
  }
}

// NOTE: To be cleaned
//void my_p4est_surfactant_t::compute_second_derivatives_for_interface_advection(Vec *dd_phi)
//{
//    const double *phi_read_p;
//    double *dd_phi_p[P4EST_DIM];

//    // get all pointers
//    ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
//    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    {
//        ierr = VecGetArray(dd_phi[dir], &dd_phi_p[dir]); CHKERRXX(ierr);
//    }

//    quad_neighbor_nodes_of_node_t qnnn;

//    // loop over the layer nodes of trees n and nm1
//    for (size_t n = 0; n < ngbd_n->get_layer_size(); ++n)
//    {
//        p4est_locidx_t node_idx = ngbd_n->get_layer_node(n);
//        ngbd_n->get_neighbors(node_idx, qnnn);
//        dd_phi_p[0][node_idx] = qnnn.dxx_central(phi_read_p);
//        dd_phi_p[1][node_idx] = qnnn.dyy_central(phi_read_p);
//#ifdef P4_TO_P8
//        dd_phi_p[2][node_idx] = qnnn.dzz_central(phi_read_p);
//#endif
//    }

//    // begin updating the ghost layers of trees n and nm1
//    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    {
//        ierr = VecGhostUpdateBegin(dd_phi[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }

//    // loop over the local nodes of trees n and nm1
//    for (size_t n = 0; n < ngbd_n->get_local_size(); ++n)
//    {
//        p4est_locidx_t node_idx = ngbd_n->get_local_node(n);
//        ngbd_n->get_neighbors(node_idx, qnnn);
//        dd_phi_p[0][node_idx] = qnnn.dxx_central(phi_read_p);
//        dd_phi_p[1][node_idx] = qnnn.dyy_central(phi_read_p);
//#ifdef P4_TO_P8
//        dd_phi_p[2][node_idx] = qnnn.dzz_central(phi_read_p);
//#endif
//    }

//    // end updating the ghost layers
//    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    {
//        ierr = VecGhostUpdateEnd(dd_phi[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }

//    // restore pointers
//    ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
//    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    {
//        ierr = VecRestoreArrayRead(vn_nodes[dir], &vn_nodes_read_p[dir]); CHKERRXX(ierr);
//        ierr = VecRestoreArrayRead(vnm1_nodes[dir], &vnm1_nodes_read_p[dir]); CHKERRXX(ierr);
//        ierr = VecRestoreArray(dd_phi[dir], &dd_phi_p[dir]); CHKERRXX(ierr);

//        for (short der = 0; der < P4EST_DIM; ++der)
//        {
//            ierr = VecRestoreArray(dd_vn_nodes[dir][der], &dd_vn_nodes_p[dir][der]); CHKERRXX(ierr);
//            ierr = VecRestoreArray(dd_vnm1_nodes[dir][der], &dd_vnm1_nodes_p[dir][der]); CHKERRXX(ierr);
//        }
//    }
//}

// NOTE: To be cleaned when efficient interface advection is implemented
void my_p4est_surfactant_t::compute_second_derivatives_for_interface_advection(Vec *dd_phi, Vec **dd_vn_nodes, Vec **dd_vnm1_nodes)
{
    const double *phi_read_p, *vn_nodes_read_p[P4EST_DIM], *vnm1_nodes_read_p[P4EST_DIM];
    double *dd_phi_p[P4EST_DIM], *dd_vn_nodes_p[P4EST_DIM][P4EST_DIM], *dd_vnm1_nodes_p[P4EST_DIM][P4EST_DIM];

    // get all pointers
    ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    {
        ierr = VecGetArrayRead(vn_nodes[dir], &vn_nodes_read_p[dir]); CHKERRXX(ierr);
        ierr = VecGetArrayRead(vnm1_nodes[dir], &vnm1_nodes_read_p[dir]); CHKERRXX(ierr);
        ierr = VecGetArray(dd_phi[dir], &dd_phi_p[dir]); CHKERRXX(ierr);

        for (unsigned short der = 0; der < P4EST_DIM; ++der)
        {
            ierr = VecGetArray(dd_vn_nodes[dir][der], &dd_vn_nodes_p[dir][der]); CHKERRXX(ierr);
            ierr = VecGetArray(dd_vnm1_nodes[dir][der], &dd_vnm1_nodes_p[dir][der]); CHKERRXX(ierr);
        }
    }

    quad_neighbor_nodes_of_node_t qnnn;

    // loop over the layer nodes of trees n and nm1
    for (size_t n = 0; n < ngbd_n->get_layer_size(); ++n)
    {
        p4est_locidx_t node_idx = ngbd_n->get_layer_node(n);
        ngbd_n->get_neighbors(node_idx, qnnn);
        dd_phi_p[0][node_idx] = qnnn.dxx_central(phi_read_p);
        dd_phi_p[1][node_idx] = qnnn.dyy_central(phi_read_p);
#ifdef P4_TO_P8
        dd_phi_p[2][node_idx] = qnnn.dzz_central(phi_read_p);
#endif
        for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
        {
            dd_vn_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vn_nodes_read_p[dir]);
            dd_vn_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vn_nodes_read_p[dir]);
#ifdef P4_TO_P8
            dd_vn_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vn_nodes_read_p[dir]);
#endif
        }
    }
    for (size_t n = 0; n < ngbd_nm1->get_layer_size(); ++n)
    {
        p4est_locidx_t node_idx = ngbd_nm1->get_layer_node(n);
        ngbd_nm1->get_neighbors(node_idx, qnnn);
        for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
        {
            dd_vnm1_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vnm1_nodes_read_p[dir]);
            dd_vnm1_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vnm1_nodes_read_p[dir]);
#ifdef P4_TO_P8
            dd_vnm1_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vnm1_nodes_read_p[dir]);
#endif
        }
    }

    // begin updating the ghost layers of trees n and nm1
    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    {
        ierr = VecGhostUpdateBegin(dd_phi[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        for (unsigned short der = 0; der < P4EST_DIM; ++der)
        {
            ierr = VecGhostUpdateBegin(dd_vn_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecGhostUpdateBegin(dd_vnm1_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
    }

    // loop over the local nodes of trees n and nm1
    for (size_t n = 0; n < ngbd_n->get_local_size(); ++n)
    {
        p4est_locidx_t node_idx = ngbd_n->get_local_node(n);
        ngbd_n->get_neighbors(node_idx, qnnn);
        dd_phi_p[0][node_idx] = qnnn.dxx_central(phi_read_p);
        dd_phi_p[1][node_idx] = qnnn.dyy_central(phi_read_p);
#ifdef P4_TO_P8
        dd_phi_p[2][node_idx] = qnnn.dzz_central(phi_read_p);
#endif
        for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
        {
            dd_vn_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vn_nodes_read_p[dir]);
            dd_vn_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vn_nodes_read_p[dir]);
#ifdef P4_TO_P8
            dd_vn_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vn_nodes_read_p[dir]);
#endif
        }
    }
    for (size_t n = 0; n < ngbd_nm1->get_local_size(); ++n)
    {
        p4est_locidx_t node_idx = ngbd_nm1->get_local_node(n);
        ngbd_nm1->get_neighbors(node_idx, qnnn);
        for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
        {
            dd_vnm1_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vnm1_nodes_read_p[dir]);
            dd_vnm1_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vnm1_nodes_read_p[dir]);
#ifdef P4_TO_P8
            dd_vnm1_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vnm1_nodes_read_p[dir]);
#endif
        }
    }

    // end updating the ghost layers
    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    {
        ierr = VecGhostUpdateEnd(dd_phi[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        for (unsigned short der = 0; der < P4EST_DIM; ++der)
        {
            ierr = VecGhostUpdateEnd(dd_vn_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecGhostUpdateEnd(dd_vnm1_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
    }

    // restore pointers
    ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
    for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    {
        ierr = VecRestoreArrayRead(vn_nodes[dir], &vn_nodes_read_p[dir]); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(vnm1_nodes[dir], &vnm1_nodes_read_p[dir]); CHKERRXX(ierr);
        ierr = VecRestoreArray(dd_phi[dir], &dd_phi_p[dir]); CHKERRXX(ierr);

        for (short der = 0; der < P4EST_DIM; ++der)
        {
            ierr = VecRestoreArray(dd_vn_nodes[dir][der], &dd_vn_nodes_p[dir][der]); CHKERRXX(ierr);
            ierr = VecRestoreArray(dd_vnm1_nodes[dir][der], &dd_vnm1_nodes_p[dir][der]); CHKERRXX(ierr);
        }
    }
}


void my_p4est_surfactant_t::advect_interface_one_step_TEST(bool second_order_SL)
{
  if(second_order_SL==false) { throw std::runtime_error("my_p4est_surfactant_t::advect_interface_one_step_TEST: first-order semi-Lagrangian not available for the moment."); }

  Vec dd_phi[P4EST_DIM];
  for(unsigned short dd = 0; dd < P4EST_DIM; ++dd) { ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_phi[dd]); CHKERRXX(ierr); }
  ngbd_n->second_derivatives_central(phi, dd_phi);

  // create np1 tree, initially copying it from the n tree
  p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_np1->connectivity = p4est_n->connectivity;
  p4est_np1->user_pointer = p4est_n->user_pointer;

  ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

  // trace back characteristics USING THE EXTENDED VELOCITY and store departure points
  trajectory_from_np1_to_nm1(p4est_np1, nodes_np1, ngbd_nm1, ngbd_n,
                             vnm1_s_nodes, dd_vnm1_s_nodes,
                             vn_s_nodes, dd_vn_s_nodes,
                             dt_nm1, dt_n,
                             (second_order_SL? xyz_dep_s_nm1 : NULL), xyz_dep_s_n);

  // find the value of the level-set function at x_dep_n
  Vec phi_np1 = NULL;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t interp_n(ngbd_n);

  for(size_t n=0; n< nodes_np1->indep_nodes.elem_count/*num_owned_indeps*/; ++n)
  {
#ifdef P4_TO_P8
    double xyz_d_[] = {xyz_dep_s_n[0][n], xyz_dep_s_n[1][n], xyz_dep_s_n[2][n]};
#else
    double xyz_d_[] = {xyz_dep_s_n[0][n], xyz_dep_s_n[1][n]};
#endif
    interp_n.add_point(n, xyz_d_);
  }
  interp_n.set_input(phi, dd_phi[0], dd_phi[1],
#ifdef P4_TO_P8
                     dd_phi[2],
#endif
                     quadratic_non_oscillatory);
  interp_n.interpolate(phi_np1);
  interp_n.clear();

  // reset interpolation tool for new LS (otherwise the band cannot be generated!)
  delete interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_np1);
  interp_phi->set_input(phi_np1, linear);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_np1->user_pointer;
  splitting_criteria_surfactant_t criteria(this,data->min_lvl,data->max_lvl,data->lip);
  p4est_np1->user_pointer = (void*)&criteria;

  // repeat iteratively
  bool grid_is_changing = criteria.refine_and_coarsen(p4est_np1, ngbd_np1, phi_np1);
  //int iter = 0;

  if(grid_is_changing)
  {
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
    p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
    delete hierarchy_np1; hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
    delete ngbd_np1; ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

    for(unsigned int dir=0; dir<P4EST_DIM; ++dir)
    {
      xyz_dep_s_nm1[dir].clear();
      xyz_dep_s_n[dir].clear();
    }
    trajectory_from_np1_to_nm1(p4est_np1, nodes_np1, ngbd_nm1, ngbd_n,
                               vnm1_s_nodes, dd_vnm1_s_nodes,
                               vn_s_nodes, dd_vn_s_nodes,
                               dt_nm1, dt_n,
                               (second_order_SL? xyz_dep_s_nm1 : NULL), xyz_dep_s_n);

//    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

//    for(size_t n=0; n< nodes_np1->indep_nodes.elem_count/*num_owned_indeps*/; ++n)
//    {
//#ifdef P4_TO_P8
//      double xyz_d_[] = {xyz_dep_s_n[0][n], xyz_dep_s_n[1][n], xyz_dep_s_n[2][n]};
//#else
//      double xyz_d_[] = {xyz_dep_s_n[0][n], xyz_dep_s_n[1][n]};
//#endif
//      interp_n.add_point(n, xyz_d_);
//    }
//    interp_n.set_input(phi, dd_phi[0], dd_phi[1],
//#ifdef P4_TO_P8
//                       dd_phi[2],
//#endif
//                       quadratic_non_oscillatory);
//    interp_n.interpolate(phi_np1);
//    interp_n.clear();

  }

//  //TEMPORAL
//  const double *phi_p;
//  ierr = VecGetArrayRead(phi_np1  , &phi_p  ); CHKERRXX(ierr);
//  my_p4est_vtk_write_all(p4est_np1, nodes_np1, ghost_np1,
//                         P4EST_TRUE, P4EST_TRUE,
//                         1, /* number of VTK_POINT_DATA (nodes) */
//                         0, /* number of VTK_CELL_DATA  (cells) */
//  #ifdef P4_TO_P8
//                         "/home/temprano/Output/p4est_surfactant/tests/3d/advection_expansion/3_8/vtu/snapshot_extra_np1",
//  #else
//                         "/home/temprano/Output/p4est_surfactant/tests/2d/advection_expansion/3_8/vtu/snapshot_extra_np1",
//  #endif
//                         VTK_POINT_DATA, "phi_np1"     , phi_p);
//  //TEMPORAL

  // blah blah blah

  for(unsigned short dd=0; dd<P4EST_DIM; ++dd) { ierr = VecDestroy(dd_phi[dd]); CHKERRXX(ierr); }
}

void my_p4est_surfactant_t::advect_interface_one_step(bool second_order_SL)
{
  // initialize and compute second derivatives of phi for the interpolation in the semi-lagrangian advection of phi
  Vec dd_phi[P4EST_DIM], *dd_vn_nodes_[P4EST_DIM], *dd_vnm1_nodes_[P4EST_DIM];
  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_phi[dir]); CHKERRXX(ierr);
    dd_vn_nodes_[dir] = new Vec[P4EST_DIM];
    dd_vnm1_nodes_[dir] = new Vec[P4EST_DIM];
    for (unsigned short der = 0; der < P4EST_DIM; ++der)
    {
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_vn_nodes_[dir][der]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_nodes_[dir][der]); CHKERRXX(ierr);
    }
  }
  compute_second_derivatives_for_interface_advection(dd_phi, dd_vn_nodes_, dd_vnm1_nodes_);

  // create np1 tree, initially copying it from the n tree
  p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_np1->connectivity = p4est_n->connectivity;
  p4est_np1->user_pointer = p4est_n->user_pointer;

  ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

  // create the temporary phi vector for phi_np1
  Vec phi_np1 = NULL;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  // advect the level-set function and fill phi_np1 with the updated values
  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n, ngbd_nm1);
  double *phi_np1_p;
  ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
  if(second_order_SL)
    sl.advect_from_n_to_np1(dt_nm1, dt_n, vnm1_nodes, dd_vnm1_nodes_, vn_nodes, dd_vn_nodes_, phi, dd_phi, phi_np1_p);
  else
    sl.advect_from_n_to_np1(dt_n, vn_nodes, dd_vn_nodes_, phi, dd_phi, phi_np1_p);
  ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

  // reset interpolation tool for new LS (otherwise the band cannot be generated!)
  delete interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_np1);
  interp_phi->set_input(phi_np1, linear);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_np1->user_pointer;
  splitting_criteria_surfactant_t criteria(this,data->min_lvl,data->max_lvl,data->lip);
  p4est_np1->user_pointer = (void*)&criteria;

  // update grid iteratively
  bool grid_is_changing = criteria.refine_and_coarsen(p4est_np1, ngbd_np1, phi_np1);
  int iter = 0;
  while(grid_is_changing)
  {
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
    p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
    delete hierarchy_np1; hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
    delete ngbd_np1; ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    my_p4est_semi_lagrangian_t updated_sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n, ngbd_nm1);
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    if(second_order_SL)
      updated_sl.advect_from_n_to_np1(dt_nm1, dt_n, vnm1_nodes, dd_vnm1_nodes_, vn_nodes, dd_vn_nodes_, phi, dd_phi, phi_np1_p);
    else
      updated_sl.advect_from_n_to_np1(dt_n, vn_nodes, dd_vn_nodes_, phi, dd_phi, phi_np1_p);
    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    // reset interpolation tool for new LS (otherwise the band cannot be generated!)
    delete interp_phi;
    interp_phi = new my_p4est_interpolation_nodes_t(ngbd_np1);
    interp_phi->set_input(phi_np1, linear);

    grid_is_changing = criteria.refine_and_coarsen(p4est_np1, ngbd_np1, phi_np1);

    iter++;
    if(iter>1+data->max_lvl-data->min_lvl)
    {
      ierr = PetscPrintf(p4est_n->mpicomm, "The grid update did not converge\n"); CHKERRXX(ierr);
      break;
    }
  }

  p4est_np1->user_pointer = data;
  my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
  p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  delete hierarchy_np1; hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  delete ngbd_np1; ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

  ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  my_p4est_semi_lagrangian_t updated_sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n, ngbd_nm1);
  ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
  if(second_order_SL)
    updated_sl.advect_from_n_to_np1(dt_nm1, dt_n, vnm1_nodes, dd_vnm1_nodes_, vn_nodes, dd_vn_nodes_, phi, dd_phi, phi_np1_p);
  else
    updated_sl.advect_from_n_to_np1(dt_n, vn_nodes, dd_vn_nodes_, phi, dd_phi, phi_np1_p);
  ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

  // reset interpolation tool for new LS (otherwise the band cannot be generated!)
  delete interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_np1);
  interp_phi->set_input(phi_np1, linear);

  // reinitialize final level-set function
  my_p4est_level_set_t lsn(ngbd_np1);
  lsn.reinitialize_2nd_order(phi_np1,20);
  lsn.perturb_level_set_function(phi_np1, EPS*dxyz_min);

  //TEMPORARY
//  const double *phi_p;
//  ierr = VecGetArrayRead(phi_np1  , &phi_p  ); CHKERRXX(ierr);
//  my_p4est_vtk_write_all(p4est_np1, nodes_np1, ghost_np1,
//                         P4EST_TRUE, P4EST_TRUE,
//                         1, /* number of VTK_POINT_DATA (nodes) */
//                         0, /* number of VTK_CELL_DATA  (cells) */
//#ifdef P4_TO_P8
//                         "/home/temprano/Output/p4est_surfactant/tests/3d/advection_expansion/3_8/vtu/snapshot_extra_np1",
//#else
//                         "/home/temprano/Output/p4est_surfactant/tests/2d/advection_expansion/3_8/vtu/snapshot_extra_np1",
//#endif
//                         VTK_POINT_DATA, "phi_np1"     , phi_p);
  //TEMPORARY

  // update phi
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_np1;

  // destroy second derivatives created within the function
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(dd_phi[dir]); CHKERRXX(ierr);
    for (unsigned short der = 0; der < P4EST_DIM; ++der)
    {
        ierr = VecDestroy(dd_vnm1_nodes_[dir][der]); CHKERRXX(ierr);
        ierr = VecDestroy(dd_vn_nodes_  [dir][der]); CHKERRXX(ierr);
    }
    delete[] dd_vnm1_nodes_[dir];
    delete[] dd_vn_nodes_[dir];
  }
}


void my_p4est_surfactant_t::compute_one_step_Gamma()
{
  // backtrack characteristics using the EXTENDED velocity and obtain departure points:
  trajectory_from_np1_to_nm1(p4est_n, nodes_n, ngbd_nm1, ngbd_n,
                             vnm1_s_nodes, dd_vnm1_s_nodes,
                             vn_s_nodes, dd_vn_s_nodes,
                             dt_nm1, dt_n,
                             xyz_dep_s_nm1, xyz_dep_s_n);

  // obtain Gamma and the stretching term at departure points from interpolation:
  my_p4est_interpolation_nodes_t interp_n(ngbd_n);
  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);

  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz_tmp[P4EST_DIM];

    xyz_tmp[0] = xyz_dep_s_n[0][n];
    xyz_tmp[1] = xyz_dep_s_n[1][n];
#ifdef P4_TO_P8
    xyz_tmp[2] = xyz_dep_s_n[2][n];
#endif
    interp_n.add_point(n, xyz_tmp);

    xyz_tmp[0] = xyz_dep_s_nm1[0][n];
    xyz_tmp[1] = xyz_dep_s_nm1[1][n];
#ifdef P4_TO_P8
    xyz_tmp[2] = xyz_dep_s_nm1[2][n];
#endif
    interp_nm1.add_point(n, xyz_tmp);
  }

  Vec Gamma_dep_n = NULL;
  Vec str_dep_n = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_dep_n); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_dep_n); CHKERRXX(ierr);
  std::vector<Vec> interp_inputs_n; interp_inputs_n.resize(0);
  interp_inputs_n.push_back(Gamma_n);
  interp_inputs_n.push_back(str_n);
  std::vector<Vec> interp_outputs_n; interp_outputs_n.resize(0);
  interp_outputs_n.push_back(Gamma_dep_n);
  interp_outputs_n.push_back(str_dep_n);
  interp_n.set_input(interp_inputs_n,linear);
  interp_n.interpolate(interp_outputs_n);
  interp_inputs_n.resize(0);
  interp_outputs_n.resize(0);
  interp_n.clear();

  Vec Gamma_dep_nm1 = NULL;
  Vec str_dep_nm1 = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_dep_nm1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_dep_nm1); CHKERRXX(ierr);
  std::vector<Vec> interp_inputs_nm1; interp_inputs_nm1.resize(0);
  interp_inputs_nm1.push_back(Gamma_nm1);
  interp_inputs_nm1.push_back(str_nm1);
  std::vector<Vec> interp_outputs_nm1; interp_outputs_nm1.resize(0);
  interp_outputs_nm1.push_back(Gamma_dep_nm1);
  interp_outputs_nm1.push_back(str_dep_nm1);
  interp_nm1.set_input(interp_inputs_nm1,linear);
  interp_nm1.interpolate(interp_outputs_nm1);
  interp_inputs_nm1.resize(0);
  interp_outputs_nm1.resize(0);
  interp_nm1.clear();

  // interpolation happened only for local nodes, so we need to update the ghost layer:
  ierr = VecGhostUpdateBegin(Gamma_dep_n,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(Gamma_dep_nm1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(str_dep_n,     INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(str_dep_nm1,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(Gamma_dep_n,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(Gamma_dep_nm1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(str_dep_n,     INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(str_dep_nm1,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // compute the right-hand side of the linear system of equations from data at departure points:
  double alpha = (2*dt_n + dt_nm1)/(dt_n+dt_nm1);
  double beta  = -dt_n/(dt_n+dt_nm1);
  double eta   = (dt_nm1+dt_n)/dt_nm1;
  double zeta  = -dt_n/dt_nm1;

  Vec rhs_Gamma      = NULL;
  Vec rhs_Gamma_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma_temp); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma);      CHKERRXX(ierr);

  const double *Gamma_dep_nm1_p, *Gamma_dep_n_p, *str_dep_nm1_p, *str_dep_n_p;
  ierr = VecGetArrayRead(Gamma_dep_nm1 , &Gamma_dep_nm1_p  ); CHKERRXX(ierr);
  ierr = VecGetArrayRead(Gamma_dep_n   , &Gamma_dep_n_p  );   CHKERRXX(ierr);
  ierr = VecGetArrayRead(str_dep_nm1   , &str_dep_nm1_p  );   CHKERRXX(ierr);
  ierr = VecGetArrayRead(str_dep_n     , &str_dep_n_p  );     CHKERRXX(ierr);

  double *rhs_temp_p;
  ierr = VecGetArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    rhs_temp_p[n]    = Gamma_dep_n_p[n]  *( (alpha/dt_n) - (beta/dt_nm1) - eta *(str_dep_n_p[n])   ) +
                       Gamma_dep_nm1_p[n]*(                (beta/dt_nm1) - zeta*(str_dep_nm1_p[n]) );
  }
  ierr = VecGhostUpdateBegin(rhs_Gamma_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    rhs_temp_p[n]    = Gamma_dep_n_p[n]  *( (alpha/dt_n) - (beta/dt_nm1) - eta *(str_dep_n_p[n])   ) +
                       Gamma_dep_nm1_p[n]*(                (beta/dt_nm1) - zeta*(str_dep_nm1_p[n]) );
  }
  ierr = VecGhostUpdateEnd  (rhs_Gamma_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(Gamma_dep_nm1 , &Gamma_dep_nm1_p ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Gamma_dep_n   , &Gamma_dep_n_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(str_dep_nm1   , &str_dep_nm1_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(str_dep_n     , &str_dep_n_p     ); CHKERRXX(ierr);

  // constant extension of the right-hand side:
  my_p4est_level_set_t ls(ngbd_n);
  ls.extend_from_interface_to_whole_domain_TVD(phi, rhs_Gamma_temp, rhs_Gamma);

  // solve for Gamma at t_np1 (at the t_n grid!)
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_np1); CHKERRXX(ierr);
  if(NO_SURFACE_DIFFUSION)
  {
    // with no diffusion, multiply rhs by dt_n/alpha and copy it to Gamma_np1:
    Vec from_loc, to_loc;
    ierr = VecGhostGetLocalForm(rhs_Gamma, &from_loc);     CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(Gamma_np1, &to_loc);       CHKERRXX(ierr);
    ierr = VecScale(from_loc, (PetscScalar)dt_n/alpha);    CHKERRXX(ierr);
    ierr = VecCopy(from_loc, to_loc);
    ierr = VecGhostRestoreLocalForm(rhs_Gamma, &from_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(Gamma_np1, &to_loc);   CHKERRXX(ierr);
  }
  else
  {
    throw std::runtime_error("The solver is not setup for surface diffusion yet");
  }

//  //TEMPORARY
//  if(p4est_n->mpirank==0)
//  {
//    std::cout << nodes_n->num_owned_indeps << std::endl;
//    std::cout << ngbd_n->get_local_size() << std::endl;
//  }
//  const double *phi_band_p, *rhs_p, *Gamma_np1_p, *Gamma_dep_nm1_vtk_p, *Gamma_dep_n_vtk_p, *str_dep_nm1_vtk_p, *str_dep_n_vtk_p;
//  ierr = VecGetArrayRead(phi_band      , &phi_band_p);          CHKERRXX(ierr);
//  ierr = VecGetArrayRead(rhs_Gamma     , &rhs_p     );          CHKERRXX(ierr);
//  ierr = VecGetArrayRead(Gamma_np1     , &Gamma_np1_p);         CHKERRXX(ierr);
//  ierr = VecGetArrayRead(Gamma_dep_nm1 , &Gamma_dep_nm1_vtk_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(Gamma_dep_n   , &Gamma_dep_n_vtk_p);   CHKERRXX(ierr);
//  ierr = VecGetArrayRead(str_dep_nm1   , &str_dep_nm1_vtk_p);   CHKERRXX(ierr);
//  ierr = VecGetArrayRead(str_dep_n     , &str_dep_n_vtk_p);     CHKERRXX(ierr);
//  my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
//                         P4EST_TRUE, P4EST_TRUE,
//                         7, /* number of VTK_POINT_DATA (nodes) */
//                         0, /* number of VTK_CELL_DATA  (cells) */
//  #ifdef P4_TO_P8
//                         "/home/temprano/Output/p4est_surfactant/tests/3d/advection_expansion/3_8/vtu/snapshot_extra",
//  #else
//                         "/home/temprano/Output/p4est_surfactant/tests/2d/advection_expansion/3_8/vtu/snapshot_extra",
//  #endif
//                         VTK_POINT_DATA, "phi_band",      phi_band_p,
//                         VTK_POINT_DATA, "Gamma_dep_nm1", Gamma_dep_nm1_vtk_p,
//                         VTK_POINT_DATA, "Gamma_dep_n",   Gamma_dep_n_vtk_p,
//                         VTK_POINT_DATA, "str_dep_nm1",   str_dep_nm1_vtk_p,
//                         VTK_POINT_DATA, "str_dep_n",     str_dep_n_vtk_p,
//                         VTK_POINT_DATA, "rhs",           rhs_p,
//                         VTK_POINT_DATA, "Gamma_np1",     Gamma_np1_p    );
//  ierr = VecRestoreArrayRead(phi_band      , &phi_band_p);          CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(rhs_Gamma     , &rhs_p     );          CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(Gamma_np1     , &Gamma_np1_p);         CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(Gamma_dep_nm1 , &Gamma_dep_nm1_vtk_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(Gamma_dep_n   , &Gamma_dep_n_vtk_p);   CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(str_dep_nm1   , &str_dep_nm1_vtk_p);   CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(str_dep_n     , &str_dep_n_vtk_p);     CHKERRXX(ierr);
//  //TEMPORARY

//  //TEMPORARY
//  const double *str_nm1_print_p, *Gamma_nm1_print_p;
//  ierr = VecGetArrayRead(Gamma_nm1, &Gamma_nm1_print_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(str_nm1  , &str_nm1_print_p);   CHKERRXX(ierr);
//  my_p4est_vtk_write_all(p4est_nm1, nodes_nm1, ghost_nm1,
//                         P4EST_TRUE, P4EST_TRUE,
//                         2, /* number of VTK_POINT_DATA (nodes) */
//                         0, /* number of VTK_CELL_DATA  (cells) */
//  #ifdef P4_TO_P8
//                         "/home/temprano/Output/p4est_surfactant/tests/3d/advection_expansion/3_8/vtu/snapshot_extra_nm1",
//  #else
//                         "/home/temprano/Output/p4est_surfactant/tests/2d/advection_expansion/3_8/vtu/snapshot_extra_nm1",
//  #endif
//                         VTK_POINT_DATA, "str_nm1",   str_nm1_print_p,
//                         VTK_POINT_DATA, "Gamma_nm1", Gamma_nm1_print_p);
//  ierr = VecRestoreArrayRead(Gamma_nm1, &Gamma_nm1_print_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(str_nm1  , &str_nm1_print_p);   CHKERRXX(ierr);
//  //TEMPORARY
}


void my_p4est_surfactant_t::set_dt(const double& dt_nm1, const double& dt_n)
{
  this->dt_nm1 = dt_nm1;
  this->dt_n = dt_n;
}


void my_p4est_surfactant_t::set_dt_n(const double& dt_n)
{
  this->dt_n = dt_n;
}


void my_p4est_surfactant_t::save_vtk(const char* name)
{
  unsigned short count_nodes = 0;

  const double *phi_p;
  const double *phi_band_p;
  const double *Gamma_n_p;
  const double *str_n_p;
  ierr = VecGetArrayRead(phi  , &phi_p  );       CHKERRXX(ierr); ++count_nodes;
  ierr = VecGetArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr); ++count_nodes;
  ierr = VecGetArrayRead(Gamma_n , &Gamma_n_p);  CHKERRXX(ierr); ++count_nodes;
  ierr = VecGetArrayRead(str_n , &str_n_p);      CHKERRXX(ierr); ++count_nodes;

  const double *vn_p[P4EST_DIM];
  const double *vn_s_p[P4EST_DIM];
  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(vn_nodes[dir],   &vn_p[dir]);     CHKERRXX(ierr); ++count_nodes;
    ierr = VecGetArrayRead(vn_s_nodes[dir], &vn_s_p[dir]);   CHKERRXX(ierr); ++count_nodes;
    ierr = VecGetArrayRead(normal_n[dir],   &normal_p[dir]); CHKERRXX(ierr); ++count_nodes;
  }

  my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                         P4EST_TRUE, P4EST_TRUE,
                         count_nodes, /* number of VTK_POINT_DATA (nodes) */
                         0, /* number of VTK_CELL_DATA  (cells) */
                         name,
                         VTK_POINT_DATA, "phi"     , phi_p,
                         VTK_POINT_DATA, "phi_band", phi_band_p,
                         VTK_POINT_DATA, "Gamma"   , Gamma_n_p,
                         VTK_POINT_DATA, "n_x", normal_p[0],
                         VTK_POINT_DATA, "n_y", normal_p[1],
#ifdef P4_TO_P8
                         VTK_POINT_DATA, "n_z", normal_p[2],
#endif
                         VTK_POINT_DATA, "v_x", vn_p[0],
                         VTK_POINT_DATA, "v_y", vn_p[1],
#ifdef P4_TO_P8
                         VTK_POINT_DATA, "v_z", vn_p[2],
#endif
                         VTK_POINT_DATA, "vs_x", vn_s_p[0],
                         VTK_POINT_DATA, "vs_y", vn_s_p[1],
#ifdef P4_TO_P8
                         VTK_POINT_DATA, "vs_z", vn_s_p[2],
#endif
                         VTK_POINT_DATA, "str", str_n_p);

  ierr = VecRestoreArrayRead(phi  , &phi_p  );       CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Gamma_n , &Gamma_n_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(str_n , &str_n_p);      CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vn_nodes[dir],   &vn_p[dir]);     CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vn_s_nodes[dir], &vn_s_p[dir]);   CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(normal_n[dir],   &normal_p[dir]); CHKERRXX(ierr);
  }
}
