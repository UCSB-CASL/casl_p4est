#ifdef P4_TO_P8
//#include <src/my_p8est_utils.h>
//#include <src/my_p8est_tools.h>
//#include <p8est_connectivity.h>
#include "my_p8est_integration_mls.h"
#else
//#include <src/my_p4est_utils.h>
//#include <p4est_connectivity.h>
#include "my_p4est_integration_mls.h"
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

  std::vector< std::vector<double> > phi_values   (n_phis, std::vector<double> (P4EST_CHILDREN, -1.));
  std::vector< std::vector<double> > phi_xx_values(n_phis, std::vector<double> (P4EST_CHILDREN, 0.));
  std::vector< std::vector<double> > phi_yy_values(n_phis, std::vector<double> (P4EST_CHILDREN, 0.));
#ifdef P4_TO_P8
  std::vector< std::vector<double> > phi_zz_values(n_phis, std::vector<double> (P4EST_CHILDREN, 0.));
#endif

  std::vector<double *> P(n_phis, NULL);
  std::vector<double *> Pxx(n_phis, NULL);
  std::vector<double *> Pyy(n_phis, NULL);
#ifdef P4_TO_P8
  std::vector<double *> Pzz(n_phis, NULL);
#endif

  bool only_linear = false;
  if (phi_cf == NULL && (phi_xx == NULL || phi_yy == NULL)) only_linear = true;

  if (phi_cf == NULL)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
    if (!only_linear)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
    }
  }

  if (use_cube_refined)
  {
    cubes_refined.clear();
    cubes_refined.reserve(p4est->local_num_quadrants);
  } else {
    cubes.clear();
    cubes.reserve(p4est->local_num_quadrants);
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

      /* get location and size of a quadrant */
      double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = (tree_xmax-tree_xmin)*dmin; double x0 = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;  double x1 = x0 + dx;
      double dy = (tree_ymax-tree_ymin)*dmin; double y0 = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;  double y1 = y0 + dy;
#ifdef P4_TO_P8
      double dz = (tree_zmax-tree_zmin)*dmin; double z0 = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;  double z1 = z0 + dz;
#endif

      // get values of LSFs
      int s = quad_idx_forest*P4EST_CHILDREN;

      if (phi_cf == NULL)
      {
        for (int i = 0; i < n_phis; i++)
          for (int j = 0; j < P4EST_CHILDREN; j++)
            phi_values    [i][j] = P  [i][ q2n[ s + j ] ];

        // get values of second derivatives of LSFs
        if (!only_linear)
          for (int i = 0; i < n_phis; i++)
            for (int j = 0; j < P4EST_CHILDREN; j++)
            {
              phi_xx_values [i][j] = Pxx[i][ q2n[ s + j ] ];
              phi_yy_values [i][j] = Pyy[i][ q2n[ s + j ] ];
#ifdef P4_TO_P8
              phi_zz_values [i][j] = Pzz[i][ q2n[ s + j ] ];
#endif
            }

      } else {

        double dx_min = 0.1*dx;
        double dy_min = 0.1*dy;
#ifdef P4_TO_P8
        double dz_min = 0.1*dz;
        double x_coord[P4EST_CHILDREN] = {x0, x1, x0, x1, x0, x1, x0, x1};
        double y_coord[P4EST_CHILDREN] = {y0, y0, y1, y1, y0, y0, y1, y1};
        double z_coord[P4EST_CHILDREN] = {z0, z0, z0, z0, z1, z1, z1, z1};
#else
        double x_coord[P4EST_CHILDREN] = {x0, x1, x0, x1};
        double y_coord[P4EST_CHILDREN] = {y0, y1, y0, y1};
#endif

        for (int i = 0; i < n_phis; i++)
          for (int j = 0; j < P4EST_CHILDREN; j++)
          {
#ifdef P4_TO_P8
            double phi_000 = (*phi_cf->at(i))(x_coord[j], y_coord[j], z_coord[j]);
            double phi_m00 = (*phi_cf->at(i))(x_coord[j]-dx_min, y_coord[j], z_coord[j]);
            double phi_p00 = (*phi_cf->at(i))(x_coord[j]+dx_min, y_coord[j], z_coord[j]);
            double phi_0m0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]-dy_min, z_coord[j]);
            double phi_0p0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]+dy_min, z_coord[j]);
            double phi_00m = (*phi_cf->at(i))(x_coord[j], y_coord[j], z_coord[j]-dz_min);
            double phi_00p = (*phi_cf->at(i))(x_coord[j], y_coord[j], z_coord[j]+dz_min);
            phi_zz_values[i][j] = (phi_00p+phi_00m-2.0*phi_000)/dz_min/dz_min;
#else
            double phi_000 = (*phi_cf->at(i))(x_coord[j], y_coord[j]);
            double phi_m00 = (*phi_cf->at(i))(x_coord[j]-dx_min, y_coord[j]);
            double phi_p00 = (*phi_cf->at(i))(x_coord[j]+dx_min, y_coord[j]);
            double phi_0m0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]-dy_min);
            double phi_0p0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]+dy_min);
#endif

            phi_values    [i][j] = phi_000;
            phi_xx_values [i][j] = (phi_p00+phi_m00-2.0*phi_000)/dx_min/dx_min;
            phi_yy_values [i][j] = (phi_0p0+phi_0m0-2.0*phi_000)/dy_min/dy_min;
          }

      }

      // reconstruct interface
      if (use_cube_refined)
      {

#ifdef P4_TO_P8
        cubes_refined.push_back(cube3_refined_mls_t(x0, x1, y0, y1, z0, z1));
#else
        cubes_refined.push_back(cube2_refined_mls_t(x0, x1, y0, y1));
#endif

        if (phi_cf != NULL) cubes_refined.back().set_phi(*phi_cf, *action, *color);
        if (only_linear)    cubes_refined.back().set_phi(phi_values, *action, *color);
#ifdef P4_TO_P8
        else                cubes_refined.back().set_phi(phi_values, phi_xx_values, phi_yy_values, phi_zz_values, *action, *color);
#else
        else                cubes_refined.back().set_phi(phi_values, phi_xx_values, phi_yy_values, *action, *color);
#endif

#ifdef P4_TO_P8
        cubes_refined.back().construct_domain(1,1,1,level);
#else
        cubes_refined.back().construct_domain(1,1,level);
#endif

      } else {

#ifdef P4_TO_P8
        cubes.push_back(cube3_mls_t(x0, x1, y0, y1, z0, z1));

        if (only_linear) cubes.back().set_phi(phi_values, *action, *color);
        else             cubes.back().set_phi(phi_values, phi_xx_values, phi_yy_values, phi_zz_values, *action, *color);
#else
        cubes.push_back(cube2_mls_t(x0, x1, y0, y1));

        if (only_linear) cubes.back().set_phi(phi_values, *action, *color);
        else             cubes.back().set_phi(phi_values, phi_xx_values, phi_yy_values, *action, *color);
#endif
        cubes.back().construct_domain();

      }
    }
  }

  if (phi_cf == NULL)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi->at(i), &P[i]); CHKERRXX(ierr);}

    if (!only_linear)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
    }
  }
}

double my_p4est_integration_mls_t::perform(int_type_t int_type, Vec f, int n0, int n1, int n2)
{
  PetscErrorCode ierr;
  double sum = 0.;

  std::vector<double> fun_values(P4EST_CHILDREN, 1.);
  double *F;

  if (f != NULL)  {ierr = VecGetArray(f, &F); CHKERRXX(ierr);}

  const p4est_locidx_t *q2n = nodes->local_nodes;

  if (initialized)
  {
    for (p4est_topidx_t quad_idx = 0; quad_idx < p4est->local_num_quadrants; quad_idx++)
    {
      /* get values of a function to integrate */
      if (f != NULL) {
        int s = quad_idx*P4EST_CHILDREN;
        for (int j = 0; j < P4EST_CHILDREN; j++)
          fun_values[j] = F[ q2n[ s + j ] ];
      }

      if (use_cube_refined)
      {
        switch (int_type){
        case DOM: sum += cubes_refined[quad_idx].integrate_over_domain      (fun_values);           break;
        case FC1: sum += cubes_refined[quad_idx].integrate_over_interface   (fun_values,n0);        break;
        case FC2: sum += cubes_refined[quad_idx].integrate_over_intersection(fun_values,n0,n1);     break;
#ifdef P4_TO_P8
        case FC3: sum += cubes_refined[quad_idx].integrate_over_intersection(fun_values,n0,n1,n2);  break;
#endif
        }
      } else {
        switch (int_type){
        case DOM: sum += cubes[quad_idx].integrate_over_domain      (fun_values.data());           break;
        case FC1: sum += cubes[quad_idx].integrate_over_interface   (fun_values.data(),n0);        break;
        case FC2: sum += cubes[quad_idx].integrate_over_intersection(fun_values.data(),n0,n1);     break;
#ifdef P4_TO_P8
        case FC3: sum += cubes[quad_idx].integrate_over_intersection(fun_values.data(),n0,n1,n2);  break;
#endif
        }
      }
    }
  }
  else
  {
    int n_phis = action->size();

    std::vector< std::vector<double> > phi_values   (n_phis, std::vector<double> (P4EST_CHILDREN, -1.));
    std::vector< std::vector<double> > phi_xx_values(n_phis, std::vector<double> (P4EST_CHILDREN, 0.));
    std::vector< std::vector<double> > phi_yy_values(n_phis, std::vector<double> (P4EST_CHILDREN, 0.));
  #ifdef P4_TO_P8
    std::vector< std::vector<double> > phi_zz_values(n_phis, std::vector<double> (P4EST_CHILDREN, 0.));
  #endif

    std::vector<double *> P(n_phis, NULL);
    std::vector<double *> Pxx(n_phis, NULL);
    std::vector<double *> Pyy(n_phis, NULL);
  #ifdef P4_TO_P8
    std::vector<double *> Pzz(n_phis, NULL);
  #endif

    bool only_linear = false;
    if (phi_cf == NULL && (phi_xx == NULL || phi_yy == NULL)) only_linear = true;

    if (phi_cf == NULL)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
      if (!only_linear)
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

        /* get location and size of a quadrant */
        double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
        double dx = (tree_xmax-tree_xmin)*dmin; double x0 = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;  double x1 = x0 + dx;
        double dy = (tree_ymax-tree_ymin)*dmin; double y0 = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;  double y1 = y0 + dy;
#ifdef P4_TO_P8
        double dz = (tree_zmax-tree_zmin)*dmin; double z0 = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;  double z1 = z0 + dz;
#endif

        // get values of LSFs
        int s = quad_idx_forest*P4EST_CHILDREN;

        if (phi_cf == NULL)
        {
          for (int i = 0; i < n_phis; i++)
            for (int j = 0; j < P4EST_CHILDREN; j++)
            {
              phi_values    [i][j] = P  [i][ q2n[ s + j ] ];
              if (phi_values[i][j] != phi_values[i][j]) std::cout << phi_values[i][j] << std::endl;
            }

          if (!only_linear) // get values of second derivatives of LSFs
            for (int i = 0; i < n_phis; i++)
              for (int j = 0; j < P4EST_CHILDREN; j++)
              {
                phi_xx_values [i][j] = Pxx[i][ q2n[ s + j ] ];
                phi_yy_values [i][j] = Pyy[i][ q2n[ s + j ] ];
#ifdef P4_TO_P8
                phi_zz_values [i][j] = Pzz[i][ q2n[ s + j ] ];
#endif
              }

        } else {

          double dx_min = 0.1*dx;
          double dy_min = 0.1*dy;
#ifdef P4_TO_P8
          double dz_min = 0.1*dz;
          double x_coord[P4EST_CHILDREN] = {x0, x1, x0, x1, x0, x1, x0, x1};
          double y_coord[P4EST_CHILDREN] = {y0, y0, y1, y1, y0, y0, y1, y1};
          double z_coord[P4EST_CHILDREN] = {z0, z0, z0, z0, z1, z1, z1, z1};
#else
          double x_coord[P4EST_CHILDREN] = {x0, x1, x0, x1};
          double y_coord[P4EST_CHILDREN] = {y0, y1, y0, y1};
#endif

          for (int i = 0; i < n_phis; i++)
            for (int j = 0; j < P4EST_CHILDREN; j++)
            {
#ifdef P4_TO_P8
              double phi_000 = (*phi_cf->at(i))(x_coord[j], y_coord[j], z_coord[j]);
              double phi_m00 = (*phi_cf->at(i))(x_coord[j]-dx_min, y_coord[j], z_coord[j]);
              double phi_p00 = (*phi_cf->at(i))(x_coord[j]+dx_min, y_coord[j], z_coord[j]);
              double phi_0m0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]-dy_min, z_coord[j]);
              double phi_0p0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]+dy_min, z_coord[j]);
              double phi_00m = (*phi_cf->at(i))(x_coord[j], y_coord[j], z_coord[j]-dz_min);
              double phi_00p = (*phi_cf->at(i))(x_coord[j], y_coord[j], z_coord[j]+dz_min);
              phi_zz_values[i][j] = (phi_00p+phi_00m-2.0*phi_000)/dz_min/dz_min;
#else
              double phi_000 = (*phi_cf->at(i))(x_coord[j], y_coord[j]);
              double phi_m00 = (*phi_cf->at(i))(x_coord[j]-dx_min, y_coord[j]);
              double phi_p00 = (*phi_cf->at(i))(x_coord[j]+dx_min, y_coord[j]);
              double phi_0m0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]-dy_min);
              double phi_0p0 = (*phi_cf->at(i))(x_coord[j], y_coord[j]+dy_min);
#endif
              phi_values    [i][j] = phi_000;
              phi_xx_values [i][j] = (phi_p00+phi_m00-2.0*phi_000)/dx_min/dx_min;
              phi_yy_values [i][j] = (phi_0p0+phi_0m0-2.0*phi_000)/dy_min/dy_min;
            }

        }

        // reconstruct interface
#ifdef P4_TO_P8
        cube3_refined_mls_t cube_refined(x0, x1, y0, y1, z0, z1);
        cube3_mls_t cube(x0, x1, y0, y1, z0, z1);
#else
        cube2_refined_mls_t cube_refined(x0, x1, y0, y1);
        cube2_mls_t cube(x0, x1, y0, y1);
//        cube.set_interpolation_grid(x0, x1, y0, y1, 1, 1);
#endif

        if (use_cube_refined)
        {
          if (phi_cf != NULL) cube_refined.set_phi(*phi_cf, *action, *color);
          if (only_linear)    cube_refined.set_phi(phi_values, *action, *color);
#ifdef P4_TO_P8
          else                cube_refined.set_phi(phi_values, phi_xx_values, phi_yy_values, phi_zz_values, *action, *color);
#else
          else                cube_refined.set_phi(phi_values, phi_xx_values, phi_yy_values, *action, *color);
#endif

#ifdef P4_TO_P8
          cube_refined.construct_domain(1,1,1,level);
#else
          cube_refined.construct_domain(1,1,level);
#endif

        } else {

#ifdef P4_TO_P8
          if (only_linear) cube.set_phi(phi_values, *action, *color);
          else             cube.set_phi(phi_values, phi_xx_values, phi_yy_values, phi_zz_values, *action, *color);
#else
          if (only_linear) cube.set_phi(phi_values, *action, *color);
          else             cube.set_phi(phi_values, phi_xx_values, phi_yy_values, *action, *color);
#endif
          cube.construct_domain();

        }

        // integrate function
        if (f != NULL) {
          int s = quad_idx_forest*P4EST_CHILDREN;
          for (int j = 0; j < P4EST_CHILDREN; j++)
            fun_values[j] = F[ q2n[ s + j ] ];
        }

        if (use_cube_refined)
        {
          switch (int_type){
          case DOM: sum += cube_refined.integrate_over_domain      (fun_values);           break;
          case FC1: sum += cube_refined.integrate_over_interface   (fun_values,n0);        break;
          case FC2: sum += cube_refined.integrate_over_intersection(fun_values,n0,n1);     break;
  #ifdef P4_TO_P8
          case FC3: sum += cube_refined.integrate_over_intersection(fun_values,n0,n1,n2);  break;
  #endif
          }
        } else {
          switch (int_type){
          case DOM: sum += cube.integrate_over_domain      (fun_values.data());           break;
          case FC1: sum += cube.integrate_over_interface   (fun_values.data(),n0);        break;
          case FC2: sum += cube.integrate_over_intersection(fun_values.data(),n0,n1);     break;
  #ifdef P4_TO_P8
          case FC3: sum += cube.integrate_over_intersection(fun_values.data(),n0,n1,n2);  break;
  #endif
          }
        }

      }
    }

    if (phi_cf == NULL)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
      if (!only_linear)
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


