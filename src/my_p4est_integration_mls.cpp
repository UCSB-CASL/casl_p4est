#ifdef P4_TO_P8
//#include <src/my_p8est_utils.h>
//#include <src/my_p8est_tools.h>
//#include <p8est_connectivity.h>
#include "my_p8est_integration_mls.h"
#include <src/my_p8est_utils.h>
#else
//#include <src/my_p4est_utils.h>
//#include <p4est_connectivity.h>
#include "my_p4est_integration_mls.h"
#include <src/my_p4est_utils.h>
#endif

#include <mpi.h>
//#include <vector>
//#include <set>
//#include <sstream>
//#include <petsclog.h>
//#include <src/CASL_math.h>
//#include <src/petsc_compatibility.h>

void my_p4est_integration_mls_t::initialize()
{
  initialized = true;

  PetscErrorCode ierr;
  const p4est_locidx_t *q2n = nodes->local_nodes;

  int n_phis = action->size();

  std::vector< std::vector<double> > P_interpolation(n_phis, std::vector<double> (P4EST_CHILDREN, -10.));
  std::vector< std::vector<double> > Pdd_interpolation(n_phis, std::vector<double> (P4EST_CHILDREN*P4EST_DIM, 0.));

  std::vector<double *> P(n_phis, NULL);
  std::vector<double *> Pxx(n_phis, NULL);
  std::vector<double *> Pyy(n_phis, NULL);
#ifdef P4_TO_P8
  std::vector<double *> Pzz(n_phis, NULL);
#endif

  if (phi_cf == NULL)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
    if (!linear_integration)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
    }
  }

  cubes.clear();
  cubes.reserve(p4est->local_num_quadrants);

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

      /* get location and size of a quadrant */
      double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = (tree_xmax-tree_xmin)*dmin; double x0 = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;  double x1 = x0 + dx;
      double dy = (tree_ymax-tree_ymin)*dmin; double y0 = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;  double y1 = y0 + dy;
#ifdef P4_TO_P8
      double dz = (tree_zmax-tree_zmin)*dmin; double z0 = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;  double z1 = z0 + dz;
#endif

#ifdef P4_TO_P8
      double xyz_min[] = {x0, y0, z0};
      double xyz_max[] = {x1, y1, z1};
      int mnk[] = {1, 1, 1};
#else
      double xyz_min[] = {x0, y0};
      double xyz_max[] = {x1, y1};
      int mnk[] = {1, 1};
#endif

      int order = linear_integration ? 1 : 2;

#ifdef P4_TO_P8
      cubes.push_back(cube3_mls_t(xyz_min, xyz_max, mnk, order));
#else
      cubes.push_back(cube2_mls_t(xyz_min, xyz_max, mnk, order));
#endif

      std::vector<double> x; cubes.back().get_x_coord(x);
      std::vector<double> y; cubes.back().get_y_coord(y);
#ifdef P4_TO_P8
      std::vector<double> z; cubes.back().get_z_coord(z);
#endif

      // get values of LSFs
      int s = quad_idx_forest*P4EST_CHILDREN;

      int points_total = x.size();

      std::vector<double> phi_cube(n_phis*points_total, -1);

      if (phi_cf != NULL)
      {
        for (short p = 0; p < n_phis; ++p)
        {
          for (int pnt_idx = 0; pnt_idx < points_total; ++pnt_idx)
#ifdef P4_TO_P8
            phi_cube[p*points_total + pnt_idx] = (*phi_cf->at(p))(x[pnt_idx], y[pnt_idx], z[pnt_idx]);
#else
            phi_cube[p*points_total + pnt_idx] = (*phi_cf->at(p))(x[pnt_idx], y[pnt_idx]);
#endif
        }
      } else {
        for (short p = 0; p < n_phis; ++p)
        {
          for (short j = 0; j < P4EST_CHILDREN; ++j)
          {
            P_interpolation[p][j] = P[p][ q2n[ s + j ] ];

            if (!linear_integration)
            {
              Pdd_interpolation[p][j*P4EST_DIM+0] = Pxx[p][ q2n[ s + j ] ];
              Pdd_interpolation[p][j*P4EST_DIM+1] = Pyy[p][ q2n[ s + j ] ];
#ifdef P4_TO_P8
              Pdd_interpolation[p][j*P4EST_DIM+2] = Pzz[p][ q2n[ s + j ] ];
#endif
            }
          }

          quadrant_interp_t phi_interpolation(p4est, tree_idx, quad, linear_integration ? linear : quadratic, &P_interpolation[p], linear_integration ? NULL : &Pdd_interpolation[p]);

          for (int pnt_idx = 0; pnt_idx < points_total; ++pnt_idx)
#ifdef P4_TO_P8
            phi_cube[p*points_total + pnt_idx] = phi_interpolation(x[pnt_idx], y[pnt_idx], z[pnt_idx]);
#else
            phi_cube[p*points_total + pnt_idx] = phi_interpolation(x[pnt_idx], y[pnt_idx]);
#endif
        }
      }

//      std::cout << "[ERROR]: here " << quad_idx << "\n";

      cubes.back().reconstruct(phi_cube, *action, *color);

//      std::cout << "[ERROR]: here " << quad_idx << "\n";
    }
  }

  if (phi_cf == NULL)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi->at(i), &P[i]); CHKERRXX(ierr);}

    if (!linear_integration)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
    }
  }
}

double my_p4est_integration_mls_t::perform(int_type_t int_type, int n0, int n1, int n2, Vec f, Vec *fdd)
{
  PetscErrorCode ierr;
  double sum = 0.;

  int n_phis = action->size();

  std::vector< std::vector<double> > P_interpolation(n_phis, std::vector<double> (P4EST_CHILDREN, -10.));
  std::vector< std::vector<double> > Pdd_interpolation(n_phis, std::vector<double> (P4EST_CHILDREN*P4EST_DIM, 0.));

  std::vector<double> F_interpolation(P4EST_CHILDREN, 1);
  std::vector<double> Fdd_interpolation(P4EST_CHILDREN*P4EST_DIM, 0);

  double *F;
  std::vector<double *> Fdd(P4EST_DIM, NULL);

  if (f != NULL)  {ierr = VecGetArray(f, &F); CHKERRXX(ierr);}
  if (fdd != NULL)
    for (short dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecGetArray(fdd[dir], &Fdd[dir]); CHKERRXX(ierr);
    }

  std::vector<double> x_node(3, 0);
  std::vector<double> y_node(3, 0);
#ifdef P4_TO_P8
  std::vector<double> z_node(3, 0);
#endif

  const p4est_locidx_t *q2n = nodes->local_nodes;

  if (initialized)
  {
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

        /* get location and size of a quadrant */
        double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
        double dx = (tree_xmax-tree_xmin)*dmin; double x0 = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;  double x1 = x0 + dx;
        double dy = (tree_ymax-tree_ymin)*dmin; double y0 = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;  double y1 = y0 + dy;
#ifdef P4_TO_P8
        double dz = (tree_zmax-tree_zmin)*dmin; double z0 = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;  double z1 = z0 + dz;
#endif

        // integrate function
        int s = quad_idx_forest*P4EST_CHILDREN;

        if (f != NULL)
        {
          for (short j = 0; j < P4EST_CHILDREN; ++j)
          {
            F_interpolation[j] = F[ q2n[ s + j ] ];
          }

          if (fdd != NULL)
            for (short j = 0; j < P4EST_CHILDREN; ++j)
            {
              Fdd_interpolation[j*P4EST_DIM+0] = Fdd[0][ q2n[ s + j ] ];
              Fdd_interpolation[j*P4EST_DIM+1] = Fdd[1][ q2n[ s + j ] ];
#ifdef P4_TO_P8
              Fdd_interpolation[j*P4EST_DIM+2] = Fdd[2][ q2n[ s + j ] ];
#endif
            }
        }

        quadrant_interp_t f_interp(p4est, tree_idx, quad, quadratic, &F_interpolation, &Fdd_interpolation);

        std::vector<double> W, X, Y, Z;

#ifdef P4_TO_P8
        switch (int_type)
        {
          case DOM: cubes[quad_idx].quadrature_over_domain      (         W,X,Y,Z);  break;
          case FC1: cubes[quad_idx].quadrature_over_interface   (n0,      W,X,Y,Z);  break;
          case FC2: cubes[quad_idx].quadrature_over_intersection(n0,n1,   W,X,Y,Z);  break;
          case FC3: cubes[quad_idx].quadrature_over_intersection(n0,n1,n2,W,X,Y,Z);  break;
        }
#else
        switch (int_type)
        {
          case DOM: cubes[quad_idx].quadrature_over_domain      (         W,X,Y);  break;
          case FC1: cubes[quad_idx].quadrature_over_interface   (n0,      W,X,Y);  break;
          case FC2: cubes[quad_idx].quadrature_over_intersection(n0,n1,   W,X,Y);  break;
        }
#endif

        for (int i = 0; i < W.size(); i++)
#ifdef P4_TO_P8
          sum += W[i]*f_interp(X[i],Y[i],Z[i]);
#else
          sum += W[i]*f_interp(X[i],Y[i]);
#endif

      }
    }

  } else {

    std::vector<double *> P(n_phis, NULL);
    std::vector<double *> Pxx(n_phis, NULL);
    std::vector<double *> Pyy(n_phis, NULL);
#ifdef P4_TO_P8
    std::vector<double *> Pzz(n_phis, NULL);
#endif

    if (phi_cf == NULL)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
      if (!linear_integration)
      {
        for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
        for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
        for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
      }
    }

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

        int test = quad->level;

        /* get location and size of a quadrant */
        double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
        double dx = (tree_xmax-tree_xmin)*dmin; double x0 = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;  double x1 = x0 + dx;
        double dy = (tree_ymax-tree_ymin)*dmin; double y0 = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;  double y1 = y0 + dy;
#ifdef P4_TO_P8
        double dz = (tree_zmax-tree_zmin)*dmin; double z0 = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;  double z1 = z0 + dz;
#endif

#ifdef P4_TO_P8
        double xyz_min[] = {x0, y0, z0};
        double xyz_max[] = {x1, y1, z1};
        int mnk[] = {1, 1, 1};
#else
        double xyz_min[] = {x0, y0};
        double xyz_max[] = {x1, y1};
        int mnk[] = {1, 1};
#endif
        int order = linear_integration ? 1 : 2;

#ifdef P4_TO_P8
        cube3_mls_t cube(xyz_min, xyz_max, mnk, order);
#else
        cube2_mls_t cube(xyz_min, xyz_max, mnk, order);
#endif

        std::vector<double> x; cube.get_x_coord(x);
        std::vector<double> y; cube.get_y_coord(y);
#ifdef P4_TO_P8
        std::vector<double> z; cube.get_z_coord(z);
#endif

        // get values of LSFs
        int s = quad_idx_forest*P4EST_CHILDREN;

        int points_total = x.size();

        std::vector<double> phi_cube(n_phis*points_total, -1);

        if (phi_cf != NULL)
        {
          for (short p = 0; p < n_phis; ++p)
          {
            for (int pnt_idx = 0; pnt_idx < points_total; ++pnt_idx)
#ifdef P4_TO_P8
              phi_cube[p*points_total + pnt_idx] = (*phi_cf->at(p))(x[pnt_idx], y[pnt_idx], z[pnt_idx]);
#else
              phi_cube[p*points_total + pnt_idx] = (*phi_cf->at(p))(x[pnt_idx], y[pnt_idx]);
#endif
          }
        } else {
          for (short p = 0; p < n_phis; ++p)
          {
            for (short j = 0; j < P4EST_CHILDREN; ++j)
            {
              P_interpolation[p][j] = P[p][ q2n[ s + j ] ];

              if (!linear_integration)
              {
                Pdd_interpolation[p][j*P4EST_DIM+0] = Pxx[p][ q2n[ s + j ] ];
                Pdd_interpolation[p][j*P4EST_DIM+1] = Pyy[p][ q2n[ s + j ] ];
#ifdef P4_TO_P8
                Pdd_interpolation[p][j*P4EST_DIM+2] = Pzz[p][ q2n[ s + j ] ];
#endif
              }
            }
            quadrant_interp_t phi_interpolation(p4est, tree_idx, quad, linear_integration ? linear : quadratic, &P_interpolation[p], linear_integration ? NULL : &Pdd_interpolation[p]);

            for (int pnt_idx = 0; pnt_idx < points_total; ++pnt_idx)
#ifdef P4_TO_P8
              phi_cube[p*points_total + pnt_idx] = phi_interpolation(x[pnt_idx], y[pnt_idx], z[pnt_idx]);
#else
              phi_cube[p*points_total + pnt_idx] = phi_interpolation(x[pnt_idx], y[pnt_idx]);
#endif
          }
        }

        cube.reconstruct(phi_cube, *action, *color);

        std::vector<double> W, X, Y, Z;

#ifdef P4_TO_P8
        switch (int_type)
        {
          case DOM: cube.quadrature_over_domain      (         W,X,Y,Z);  break;
          case FC1: cube.quadrature_over_interface   (n0,      W,X,Y,Z);  break;
          case FC2: cube.quadrature_over_intersection(n0,n1,   W,X,Y,Z);  break;
          case FC3: cube.quadrature_over_intersection(n0,n1,n2,W,X,Y,Z);  break;
        }
#else
        switch (int_type)
        {
          case DOM: cube.quadrature_over_domain      (         W,X,Y);  break;
          case FC1: cube.quadrature_over_interface   (n0,      W,X,Y);  break;
          case FC2: cube.quadrature_over_intersection(n0,n1,   W,X,Y);  break;
        }
#endif


        // integrate function
        if (f != NULL)
        {
          for (short j = 0; j < P4EST_CHILDREN; ++j)
          {
            F_interpolation[j] = F[ q2n[ s + j ] ];
          }

          if (fdd != NULL)
            for (short j = 0; j < P4EST_CHILDREN; ++j)
            {
              Fdd_interpolation[j*P4EST_DIM+0] = Fdd[0][ q2n[ s + j ] ];
              Fdd_interpolation[j*P4EST_DIM+1] = Fdd[1][ q2n[ s + j ] ];
#ifdef P4_TO_P8
              Fdd_interpolation[j*P4EST_DIM+2] = Fdd[2][ q2n[ s + j ] ];
#endif
            }
        }

        quadrant_interp_t f_interp(p4est, tree_idx, quad, quadratic, &F_interpolation, &Fdd_interpolation);

        for (int i = 0; i < W.size(); i++)
#ifdef P4_TO_P8
          sum += W[i]*f_interp(X[i],Y[i],Z[i]);
#else
          sum += W[i]*f_interp(X[i],Y[i]);
#endif

      }
    }

    if (phi_cf == NULL)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
      if (!linear_integration)
      {
        for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
        for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
        for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
      }
    }

  }

  if (f != NULL)  {ierr = VecRestoreArray(f, &F); CHKERRXX(ierr);}
  if (fdd != NULL)
    for (short dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecRestoreArray(fdd[dir], &Fdd[dir]); CHKERRXX(ierr);
    }

  /* compute global sum */
  double sum_global = 0;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}


double my_p4est_integration_mls_t::integrate_everywhere(Vec f)
{
  PetscErrorCode ierr;
  double sum = 0.;

  double *F;

  if (f != NULL)  {ierr = VecGetArray(f, &F); CHKERRXX(ierr);}

  const p4est_locidx_t *q2n = nodes->local_nodes;

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

        /* get location and size of a quadrant */
        double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
        double dx = (tree_xmax-tree_xmin)*dmin; double x0 = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;  double x1 = x0 + dx;
        double dy = (tree_ymax-tree_ymin)*dmin; double y0 = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;  double y1 = y0 + dy;
#ifdef P4_TO_P8
        double dz = (tree_zmax-tree_zmin)*dmin; double z0 = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;  double z1 = z0 + dz;
#endif

        double tmp = 0;

        // integrate function
        if (f != NULL) {
          int s = quad_idx_forest*P4EST_CHILDREN;
          for (int j = 0; j < P4EST_CHILDREN; j++)
            tmp += F[ q2n[ s + j ] ];
        }

#ifdef P4_TO_P8
        sum += dx*dy*dz*tmp/(double)(P4EST_CHILDREN);
#else
        sum += dx*dy*tmp/(double)(P4EST_CHILDREN);
#endif
      }
    }

  if (f != NULL)  {ierr = VecRestoreArray(f, &F); CHKERRXX(ierr);}

  /* compute global sum */
  double sum_global;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}


