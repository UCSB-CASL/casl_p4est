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

  double *phi_values = new double [P4EST_CHILDREN*n_phis];

  double **P = new double* [n_phis];
  for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}

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

      /* get values of level-set functions */
      int s = quad_idx_forest*P4EST_CHILDREN;
      for (int i = 0; i < n_phis; i++)
        for (int j = 0; j < P4EST_CHILDREN; j++)
          phi_values[i*P4EST_CHILDREN + j] = P[i][ q2n[ s + j ] ];

      if (use_cube_refined)
      {
#ifdef P4_TO_P8
        cubes_refined.push_back(cube3_refined_mls_t(x0, x1, y0, y1, z0, z1));
#else
        cubes_refined.push_back(cube2_refined_mls_t(x0, x1, y0, y1));
#endif
      } else {
#ifdef P4_TO_P8
        cubes.push_back(cube3_mls_t(x0, x1, y0, y1, z0, z1));
#else
        cubes.push_back(cube2_mls_t(x0, x1, y0, y1));
        cubes.back().construct_domain(phi_values, phi_xx_values, phi_yy_values, *action, *color);
#endif
      }

    }
  }

  for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi->at(i), &P[i]); CHKERRXX(ierr);} delete[] P;
  delete[] phi_values;
}

double my_p4est_integration_mls_t::perform(int_type_t int_type, Vec f, int n0, int n1, int n2)
{
  PetscErrorCode ierr;
  double sum = 0.;

  double fun_values[P4EST_CHILDREN];
  double *F;

  if (f != NULL)  {ierr = VecGetArray(f, &F); CHKERRXX(ierr);}
  else            {for (int i = 0; i < P4EST_CHILDREN; i++) fun_values[i] = 1.;}

  const p4est_locidx_t *q2n = nodes->local_nodes;

  if (initialized)
  {
    for (p4est_topidx_t quad_idx = 0; quad_idx < p4est->local_num_quadrants; quad_idx++)
    {
      /* get values of a function to integrate */
      if (f != NULL) {int s = quad_idx*P4EST_CHILDREN; for (int j = 0; j < P4EST_CHILDREN; j++) fun_values[j] = F[ q2n[ s + j ] ];}

      switch (int_type){
      case DOM: sum += cubes[quad_idx].integrate_over_domain      (fun_values);           break;
      case FC1: sum += cubes[quad_idx].integrate_over_interface   (fun_values,n0);        break;
      case FC2: sum += cubes[quad_idx].integrate_over_intersection(fun_values,n0,n1);     break;
#ifdef P4_TO_P8
      case FC3: sum += cubes[quad_idx].integrate_over_intersection(fun_values,n0,n1,n2);  break;
#endif
      }
    }
  }
  else
  {
    int n_phis = action->size();

    double *phi_values = new double [P4EST_CHILDREN*n_phis];

    double **P = new double* [n_phis]; for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}

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

        /* get values of level-set functions */
        int s = quad_idx_forest*P4EST_CHILDREN;
        for (int i = 0; i < n_phis; i++)
          for (int j = 0; j < P4EST_CHILDREN; j++) phi_values[i*P4EST_CHILDREN + j] = P[i][ q2n[ s + j ] ];

        /* get values of a function to integrate */
        if (f != NULL) {int s = quad_idx_forest*P4EST_CHILDREN; for (int j = 0; j < P4EST_CHILDREN; j++) fun_values[j] = F[ q2n[ s + j ] ];}

#ifdef P4_TO_P8
        cube3_mls_t cube(x0, x1, y0, y1, z0, z1);
#else
        cube2_mls_t cube(x0, x1, y0, y1);
#endif

        cube.construct_domain(phi_values, *action, *color);

        switch (int_type){
        case DOM: sum += cube.integrate_over_domain      (fun_values);          break;
        case FC1: sum += cube.integrate_over_interface   (fun_values,n0);       break;
        case FC2: sum += cube.integrate_over_intersection(fun_values,n0,n1);    break;
#ifdef P4_TO_P8
        case FC3: sum += cube.integrate_over_intersection(fun_values,n0,n1,n2); break;
#endif
        }
      }
    }

    for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi->at(i), &P[i]); CHKERRXX(ierr);} delete[] P;
    delete[] phi_values;
  }

  if (f != NULL)  {ierr = VecRestoreArray(f, &F); CHKERRXX(ierr);}

  /* compute global sum */
  double sum_global;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}


