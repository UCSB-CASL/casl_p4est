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

  int n_nodes = linear_integration ? P4EST_CHILDREN : pow(3, P4EST_DIM);

  double P_interpolation[P4EST_CHILDREN];
  double Pdd_interpolation[P4EST_CHILDREN*P4EST_DIM];

  std::vector< std::vector<double> > phi_values   (n_phis, std::vector<double> (n_nodes));

  std::vector<double *> P(n_phis, NULL);
  std::vector<double *> Pxx(n_phis, NULL);
  std::vector<double *> Pyy(n_phis, NULL);
#ifdef P4_TO_P8
  std::vector<double *> Pzz(n_phis, NULL);
#endif

  for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
  if (!linear_integration)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
    for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
    for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
  }

  if (linear_integration)
  {
    cubes_linear.clear();
    cubes_linear.reserve(p4est->local_num_quadrants);
  } else {
    cubes_quadratic.clear();
    cubes_quadratic.reserve(p4est->local_num_quadrants);
  }

  std::vector<double> x_node(3, 0);
  std::vector<double> y_node(3, 0);
#ifdef P4_TO_P8
  std::vector<double> z_node(3, 0);
#endif

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

      if (linear_integration)
      {
        for (int i = 0; i < n_phis; i++)
          for (int j = 0; j < n_nodes; j++)
            phi_values[i][j] = P[i][ q2n[ s + j ] ];

      } else {

        x_node[0] = x0; x_node[1] = 0.5*(x0+x1); x_node[2] = x1;
        y_node[0] = y0; y_node[1] = 0.5*(y0+y1); y_node[2] = y1;
#ifdef P4_TO_P8
        z_node[0] = z0; z_node[1] = 0.5*(z0+z1); z_node[2] = z1;
#endif

        for (short p = 0; p < n_phis; ++p)
        {

          for (short j = 0; j < P4EST_CHILDREN; ++j)
          {
            P_interpolation[j] = P[p][ q2n[ s + j ] ];
            Pdd_interpolation[j*P4EST_DIM+0] = Pxx[p][ q2n[ s + j ] ];
            Pdd_interpolation[j*P4EST_DIM+1] = Pyy[p][ q2n[ s + j ] ];
#ifdef P4_TO_P8
            Pdd_interpolation[j*P4EST_DIM+2] = Pzz[p][ q2n[ s + j ] ];
#endif
          }

          int m = 0;
#ifdef P4_TO_P8
          for (short k = 0; k < 3; k++)
#endif
            for (short j = 0; j < 3; j++)
              for (short i = 0; i < 3; i++)
              {
#ifdef P4_TO_P8
                double xyz_node[P4EST_DIM] = {x_node[i], y_node[j], z_node[k]};
#else
                double xyz_node[P4EST_DIM] = {x_node[i], y_node[j]};
#endif
//                phi_values[p][m] = quadratic_non_oscillatory_interpolation(p4est, tree_idx, *quad, P_interpolation, Pdd_interpolation, xyz_node);
                phi_values[p][m] = quadratic_interpolation(p4est, tree_idx, *quad, P_interpolation, Pdd_interpolation, xyz_node);
                ++m;
              }
        }
      }

      // reconstruct interface
      if (linear_integration)
      {
#ifdef P4_TO_P8
        cubes_linear.push_back(cube3_mls_t(x0, x1, y0, y1, z0, z1));
#else
        cubes_linear.push_back(cube2_mls_t(x0, x1, y0, y1));
#endif
        cubes_linear.back().set_phi(phi_values, *action, *color);
        cubes_linear.back().construct_domain();

      } else {

#ifdef P4_TO_P8
        cubes_quadratic.push_back(cube3_mls_quadratic_t(x0, x1, y0, y1, z0, z1));
#else
        cubes_quadratic.push_back(cube2_mls_quadratic_t(x0, x1, y0, y1));
#endif
        cubes_quadratic.back().construct_domain(phi_values, *action, *color);
      }
    }
  }

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

double my_p4est_integration_mls_t::perform(int_type_t int_type, int n0, int n1, int n2, Vec f, Vec *fdd)
{
  PetscErrorCode ierr;
  double sum = 0.;

  int n_nodes = linear_integration ? P4EST_CHILDREN : pow(3, P4EST_DIM);

  double P_interpolation[P4EST_CHILDREN];
  double Pdd_interpolation[P4EST_CHILDREN*P4EST_DIM];

  std::vector<double> fun_values(n_nodes, 1.);
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
          if (linear_integration)
          {
            for (int j = 0; j < n_nodes; j++)
              fun_values[j] = F[ q2n[ s + j ] ];
          } else {

            x_node[0] = x0; x_node[1] = 0.5*(x0+x1); x_node[2] = x1;
            y_node[0] = y0; y_node[1] = 0.5*(y0+y1); y_node[2] = y1;
  #ifdef P4_TO_P8
            z_node[0] = z0; z_node[1] = 0.5*(z0+z1); z_node[2] = z1;
  #endif

            for (short j = 0; j < P4EST_CHILDREN; ++j)
            {
              P_interpolation[j] = F[ q2n[ s + j ] ];
            }

            if (fdd != NULL)
              for (short j = 0; j < P4EST_CHILDREN; ++j)
              {
                Pdd_interpolation[j*P4EST_DIM+0] = Fdd[0][ q2n[ s + j ] ];
                Pdd_interpolation[j*P4EST_DIM+1] = Fdd[1][ q2n[ s + j ] ];
#ifdef P4_TO_P8
                Pdd_interpolation[j*P4EST_DIM+2] = Fdd[2][ q2n[ s + j ] ];
#endif
              }

            int m = 0;
#ifdef P4_TO_P8
            for (short k = 0; k < 3; k++)
#endif
              for (short j = 0; j < 3; j++)
                for (short i = 0; i < 3; i++)
                {
#ifdef P4_TO_P8
                  double xyz_node[P4EST_DIM] = {x_node[i], y_node[j], z_node[k]};
#else
                  double xyz_node[P4EST_DIM] = {x_node[i], y_node[j]};
#endif
                  if (fdd != NULL)  fun_values[m] = quadratic_non_oscillatory_interpolation(p4est, tree_idx, *quad, P_interpolation, Pdd_interpolation, xyz_node);
                  else              fun_values[m] = linear_interpolation                   (p4est, tree_idx, *quad, P_interpolation, xyz_node);
                  ++m;
                }

          }

        if (linear_integration)
        {
          switch (int_type){
            case DOM: sum += cubes_linear[quad_idx].integrate_over_domain      (fun_values.data());           break;
            case FC1: sum += cubes_linear[quad_idx].integrate_over_interface   (fun_values.data(),n0);        break;
            case FC2: sum += cubes_linear[quad_idx].integrate_over_intersection(fun_values.data(),n0,n1);     break;
#ifdef P4_TO_P8
            case FC3: sum += cubes_linear[quad_idx].integrate_over_intersection(fun_values.data(),n0,n1,n2);  break;
#endif
          }
        } else {
          switch (int_type){
            case DOM: sum += cubes_quadratic[quad_idx].integrate_over_domain      (fun_values);           break;
            case FC1: sum += cubes_quadratic[quad_idx].integrate_over_interface   (fun_values,n0);        break;
            case FC2: sum += cubes_quadratic[quad_idx].integrate_over_intersection(fun_values,n0,n1);     break;
#ifdef P4_TO_P8
            case FC3: sum += cubes_quadratic[quad_idx].integrate_over_intersection(fun_values,n0,n1,n2);  break;
#endif
          }
        }

      }
    }

  } else {

    int n_phis = action->size();

//    std::vector< std::vector<double> > test (10, std::vector<double> (10,-1));

    std::vector< std::vector<double> > phi_values   (n_phis, std::vector<double> (n_nodes, -1.));

    std::vector<double *> P(n_phis, NULL);
    std::vector<double *> Pxx(n_phis, NULL);
    std::vector<double *> Pyy(n_phis, NULL);
#ifdef P4_TO_P8
    std::vector<double *> Pzz(n_phis, NULL);
#endif

    for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i), &P[i]); CHKERRXX(ierr);}
    if (!linear_integration)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_xx->at(i), &Pxx[i]); CHKERRXX(ierr);}
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_yy->at(i), &Pyy[i]); CHKERRXX(ierr);}
#ifdef P4_TO_P8
      for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi_zz->at(i), &Pzz[i]); CHKERRXX(ierr);}
#endif
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

        if (linear_integration)
        {
          for (int i = 0; i < n_phis; i++)
            for (int j = 0; j < n_nodes; j++)
              phi_values[i][j] = P[i][ q2n[ s + j ] ];

        } else {

          x_node[0] = x0; x_node[1] = 0.5*(x0+x1); x_node[2] = x1;
          y_node[0] = y0; y_node[1] = 0.5*(y0+y1); y_node[2] = y1;
#ifdef P4_TO_P8
          z_node[0] = z0; z_node[1] = 0.5*(z0+z1); z_node[2] = z1;
#endif

          for (short p = 0; p < n_phis; ++p)
          {

            for (short j = 0; j < P4EST_CHILDREN; ++j)
            {
              P_interpolation[j] = P[p][ q2n[ s + j ] ];
              Pdd_interpolation[j*P4EST_DIM+0] = Pxx[p][ q2n[ s + j ] ];
              Pdd_interpolation[j*P4EST_DIM+1] = Pyy[p][ q2n[ s + j ] ];
  #ifdef P4_TO_P8
              Pdd_interpolation[j*P4EST_DIM+2] = Pzz[p][ q2n[ s + j ] ];
  #endif
            }

            int m = 0;
#ifdef P4_TO_P8
            for (short k = 0; k < 3; k++)
#endif
              for (short j = 0; j < 3; j++)
                for (short i = 0; i < 3; i++)
                {
#ifdef P4_TO_P8
                  double xyz_node[P4EST_DIM] = {x_node[i], y_node[j], z_node[k]};
#else
                  double xyz_node[P4EST_DIM] = {x_node[i], y_node[j]};
#endif
//                  phi_values[p][m] = quadratic_non_oscillatory_interpolation(p4est, tree_idx, *quad, P_interpolation, Pdd_interpolation, xyz_node);
                  phi_values[p][m] = quadratic_interpolation(p4est, tree_idx, *quad, P_interpolation, Pdd_interpolation, xyz_node);
//                  phi_values[p][m] = linear_interpolation(p4est, tree_idx, *quad, P_interpolation, xyz_node);
                  ++m;
                }
          }
        }


#ifdef P4_TO_P8
        cube3_mls_t           cube_linear   (x0, x1, y0, y1, z0, z1);
        cube3_mls_quadratic_t cube_quadratic(x0, x1, y0, y1, z0, z1);
#else
        cube2_mls_t           cube_linear   (x0, x1, y0, y1);
        cube2_mls_quadratic_t cube_quadratic(x0, x1, y0, y1);
#endif

        // reconstruct interface
        if (linear_integration)
        {
          cube_linear.set_phi(phi_values, *action, *color);
          cube_linear.construct_domain();
        } else {
          cube_quadratic.construct_domain(phi_values, *action, *color);
        }

        // integrate function
        if (f != NULL)
          if (linear_integration)
          {
            int s = quad_idx_forest*P4EST_CHILDREN;
            for (int j = 0; j < n_nodes; j++)
              fun_values[j] = F[ q2n[ s + j ] ];
          } else {

            for (short j = 0; j < P4EST_CHILDREN; ++j)
            {
              P_interpolation[j] = F[ q2n[ s + j ] ];
            }

            if (fdd != NULL)
              for (short j = 0; j < P4EST_CHILDREN; ++j)
              {
                Pdd_interpolation[j*P4EST_DIM+0] = Fdd[0][ q2n[ s + j ] ];
                Pdd_interpolation[j*P4EST_DIM+1] = Fdd[1][ q2n[ s + j ] ];
#ifdef P4_TO_P8
                Pdd_interpolation[j*P4EST_DIM+2] = Fdd[2][ q2n[ s + j ] ];
#endif
              }

            int m = 0;
#ifdef P4_TO_P8
            for (short k = 0; k < 3; k++)
#endif
              for (short j = 0; j < 3; j++)
                for (short i = 0; i < 3; i++)
                {
#ifdef P4_TO_P8
                  double xyz_node[P4EST_DIM] = {x_node[i], y_node[j], z_node[k]};
#else
                  double xyz_node[P4EST_DIM] = {x_node[i], y_node[j]};
#endif
                  if (fdd != NULL)
//                    fun_values[m] = quadratic_non_oscillatory_interpolation(p4est, tree_idx, *quad, P_interpolation, Pdd_interpolation, xyz_node);
                    fun_values[m] = quadratic_interpolation(p4est, tree_idx, *quad, P_interpolation, Pdd_interpolation, xyz_node);
                  else              fun_values[m] = linear_interpolation                   (p4est, tree_idx, *quad, P_interpolation, xyz_node);
                  ++m;
                }

          }

        if (linear_integration)
        {
          switch (int_type){
            case DOM: sum += cube_linear.integrate_over_domain      (fun_values.data());           break;
            case FC1: sum += cube_linear.integrate_over_interface   (fun_values.data(),n0);        break;
            case FC2: sum += cube_linear.integrate_over_intersection(fun_values.data(),n0,n1);     break;
#ifdef P4_TO_P8
            case FC3: sum += cube_linear.integrate_over_intersection(fun_values.data(),n0,n1,n2);  break;
#endif
          }
        } else {
          switch (int_type){
            case DOM: sum += cube_quadratic.integrate_over_domain      (fun_values);           break;
            case FC1: sum += cube_quadratic.integrate_over_interface   (fun_values,n0);        break;
            case FC2: sum += cube_quadratic.integrate_over_intersection(fun_values,n0,n1);     break;
#ifdef P4_TO_P8
            case FC3: sum += cube_quadratic.integrate_over_intersection(fun_values,n0,n1,n2);  break;
#endif
          }
        }

      }
    }

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


