// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/simplex3_mls_vtk.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/simplex2_mls_vtk.h>
#endif

#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

//#include "geometry_one_circle.cpp"
//#include "geometry_two_circles_union.cpp"
//#include "geometry_two_circles_intersection.cpp"
//#include "geometry_two_circles_coloration.cpp"
#include "geometry_four_flowers.cpp"

class Result
{
public:
  vector<double> ID, IB, IDr2, IBr2;
  vector< vector<double> > ISB, ISBr2, IXr2, IX;
  Result()
  {
    for (int i = 0; i < exact.n_subs; i++)
    {
      ISB.push_back(vector<double>());
      ISBr2.push_back(vector<double>());
    }
    for (int i = 0; i < exact.n_Xs; i++)
    {
      IX.push_back(vector<double>());
      IXr2.push_back(vector<double>());
    }
  }
} res_mlt;

vector<double> level, h;
void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi,
              int compt)
{
  PetscErrorCode ierr;
#ifdef STAMPEDE
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#else
  char out_dir[10000];
  sprintf(out_dir, OUTPUT_DIR);
#endif

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/nodes_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  parStopWatch w;
  w.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const int n_xyz[] = {1, 1, 1};
  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};
  const int periodic[] = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &ls_ref, 1.2);
    p4est->user_pointer = (void*)(&data);

    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    /* functions to integrate */
    Vec f_r2;
    ierr = VecCreateGhostNodes(p4est, nodes, &f_r2); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, func_r2, f_r2);
    Vec f_x;
    ierr = VecCreateGhostNodes(p4est, nodes, &f_x); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, func_x, f_x);
    Vec f_y;
    ierr = VecCreateGhostNodes(p4est, nodes, &f_y); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, func_y, f_y);

    my_p4est_level_set_t ls(&ngbd_n);

    /* level-set functions */
    vector<Vec> phi_vec, phi_xx_vec, phi_yy_vec;
#ifdef P4_TO_P8
    vector<Vec> phi_zz_vec;
#endif

    for (int i = 0; i < geometry.LSF.size(); i++)
    {
      phi_vec.push_back(Vec());     ierr = VecCreateGhostNodes(p4est, nodes, &phi_vec[i]); CHKERRXX(ierr);
      phi_xx_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_vec[i]); CHKERRXX(ierr);
      phi_yy_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_vec[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
      phi_zz_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz_vec[i]); CHKERRXX(ierr);
#endif

      sample_cf_on_nodes(p4est, nodes, *geometry.LSF[i], phi_vec[i]);

#ifdef P4_TO_P8
      ngbd_n.second_derivatives_central(phi_vec[i], phi_xx_vec[i], phi_yy_vec[i], phi_zz_vec[i]);
#else
      ngbd_n.second_derivatives_central(phi_vec[i], phi_xx_vec[i], phi_yy_vec[i]);
#endif
    }

    my_p4est_integration_mls_t integration;
    integration.set_p4est(p4est, nodes);
//    integration.set_phi(phi_vec, geometry.action, geometry.color);
//    integration.set_phi(geometry.LSF, geometry.action, geometry.color);
//#ifdef P4_TO_P8
//    integration.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, phi_zz_vec, geometry.action, geometry.color);
//#else
//    integration.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, geometry.action, geometry.color);
//#endif
#ifdef P4_TO_P8
    integration.set_phi(phi_vec, geometry.action, geometry.color);
#else
    integration.set_phi(phi_vec, geometry.action, geometry.color);
#endif
//    integration.set_use_cube_refined(0);


    if (save_vtk)
    {
      integration.initialize();
#ifdef P4_TO_P8
      vector<simplex3_mls_t *> simplices;
      int n_sps = NTETS;
#else
      vector<simplex2_mls_t *> simplices;
      int n_sps = 2;
#endif

      for (int k = 0; k < integration.cubes.size(); k++)
        if (integration.cubes[k].loc == FCE)
          for (int l = 0; l < n_sps; l++)
            simplices.push_back(&integration.cubes[k].simplex[l]);

#ifdef P4_TO_P8
      simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#else
      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#endif
      save_VTK(p4est, ghost, nodes, &brick, phi_vec[0], iter);
    }


    /* Calculate and store results */
    if (exact.provided || iter < nb_splits-1)
    {
      level.push_back(lmax+iter);
      h.push_back((xmax-xmin)/pow(2.0,(double)(lmax+iter)));

      res_mlt.ID.push_back(integration.measure_of_domain    ());
//      res_mlt.IB.push_back(integration.measure_of_interface (-1));

      res_mlt.IDr2.push_back(integration.integrate_over_domain    (f_r2));
//      res_mlt.IBr2.push_back(integration.integrate_over_interface (f_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        res_mlt.ISB[i].push_back(integration.measure_of_interface(geometry.color[i]));
        res_mlt.ISBr2[i].push_back(integration.integrate_over_interface(f_r2, geometry.color[i]));
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
#ifdef P4_TO_P8
        res_mlt.IX[i].push_back(integration.measure_of_intersection( exact.IXc0[i], exact.IXc1[i]));
        res_mlt.IXr2[i].push_back(integration.integrate_over_intersection(f_r2, exact.IXc0[i], exact.IXc1[i]));
#else
        res_mlt.IX[i].push_back(integration.integrate_over_intersection(f_x, exact.IXc0[i], exact.IXc1[i]));
        res_mlt.IXr2[i].push_back(integration.integrate_over_intersection(f_y, exact.IXc0[i], exact.IXc1[i]));
#endif
      }
    }
    else if (iter == nb_splits-1)
    {
      exact.ID    = (integration.measure_of_domain        ());
//      exact.IB    = (integration.measure_of_interface     (-1));
      exact.IDr2  = (integration.integrate_over_domain    (f_r2));
//      exact.IBr2  = (integration.integrate_over_interface (f_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        exact.ISB.push_back(integration.measure_of_interface(geometry.color[i]));
        exact.ISBr2.push_back(integration.integrate_over_interface(f_r2, geometry.color[i]));
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
#ifdef P4_TO_P8
        exact.IX.push_back(integration.measure_of_intersection(exact.IXc0[i], exact.IXc1[i]));
        exact.IXr2.push_back(integration.integrate_over_intersection(f_r2, exact.IXc0[i], exact.IXc1[i]));
#else
        exact.IX.push_back(integration.integrate_over_intersection(f_x, exact.IXc0[i], exact.IXc1[i]));
        exact.IXr2.push_back(integration.integrate_over_intersection(f_y, exact.IXc0[i], exact.IXc1[i]));
#endif
      }
    }

    ierr = VecDestroy(f_r2); CHKERRXX(ierr);

    for (int i = 0; i < phi_vec.size(); i++) {ierr = VecDestroy(phi_vec[i]); CHKERRXX(ierr);}
    phi_vec.clear();

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  if (mpi.rank() == 0)
  {
    Gnuplot plot_ID;
    print_Table("Domain", exact.ID, level, h, "MLT", res_mlt.ID, 2, &plot_ID);

    //  Gnuplot plot_IB;
    //  print_Table("Interface", exact.IB, level, h, "MLT", res_mlt.IB, 2, &plot_IB);

    Gnuplot plot_IDr2;
    print_Table("2nd moment of domain", exact.IDr2, level, h, "MLT", res_mlt.IDr2, 2, &plot_IDr2);

    //  Gnuplot plot_IBr2;
    //  print_Table("2nd moment of interface", exact.IBr2, level, h, "MLT", res_mlt.IBr2, 2, &plot_IBr2);

    vector<Gnuplot *> plot_ISB;
    vector<Gnuplot *> plot_ISBr2;
    for (int i = 0; i < exact.n_subs; i++)
    {
      plot_ISB.push_back(new Gnuplot());
      print_Table("Interface #"+to_string(i), exact.ISB[i], level, h, "MLT", res_mlt.ISB[i], 2, plot_ISB[i]);

      plot_ISBr2.push_back(new Gnuplot());
      print_Table("2nd moment of interface #"+to_string(i), exact.ISBr2[i], level, h, "MLT", res_mlt.ISBr2[i], 2, plot_ISBr2[i]);
    }

    vector<Gnuplot *> plot_IXr2;
    for (int i = 0; i < exact.n_Xs; i++)
    {
      plot_IXr2.push_back(new Gnuplot());
      print_Table("Intersection of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), exact.IX[i], level, h, "MLT", res_mlt.IX[i], 1, plot_IXr2[i]);
      print_Table("Intersection of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), exact.IXr2[i], level, h, "MLT", res_mlt.IXr2[i], 2, plot_IXr2[i]);
    }

//    // print short table
//    for (int i = 0; i < h.size(); i++)
//    {
//      cout << h[i] << ", "
//           << fabs(res_mlt.ID[i]-exact.ID) << ", "
//           << fabs(res_mlt.IDr2[i]-exact.IDr2) << ", "
//           << fabs(res_mlt.ISB[0][i]-exact.ISB[0]) << ", "
//           << fabs(res_mlt.ISBr2[0][i]-exact.ISBr2[0]) << ", "
//           << fabs(res_mlt.ISB[1][i]-exact.ISB[1]) << ", "
//           << fabs(res_mlt.ISBr2[1][i]-exact.ISBr2[1]) <<  ", "
//           << fabs(res_mlt.IX[0][i]-exact.IX[0]) <<  ", "
//           << fabs(res_mlt.IXr2[0][i]-exact.IXr2[0]) << ";" << endl;
//    }
//    cout.precision(32);
//    for (int i = 0; i < h.size(); i++)
//    {
//      cout << h[i] << ", "
//           << res_mlt.ID[i] << ", "
//           << res_mlt.IDr2[i] << ", ";

//      for (int j = 0; j < exact.n_subs; j++)
//      cout << res_mlt.ISB[j][i] << ", "
//           << res_mlt.ISBr2[j][i] << ", ";

//      for (int j = 0; j < exact.n_Xs; j++)
//      cout << res_mlt.IX[j][i] <<  ", "
//           << res_mlt.IXr2[j][i] << ";" << endl;
//    }
//    // print names
//    cout << "string('$I_{\Omega}$');"
//         << "string('$I_{\Omega}^{[\mathbf{r}^2]}$');"
//         << "string('$I_{\Gamma_1}$');"
//         << "string('$I_{\Gamma_1}$^{[\mathbf{r}^2]}$');"
//         << "string('$I_{\Gamma_2}$');"
//         << "string('$I_{\Gamma_2}$^{[\mathbf{r}^2]}$');"
//         << "string('$I_{\Gamma_1 \cap \Gamma_2}^{[x]}$');"
//         << "string('$I_{\Gamma_1 \cap \Gamma_2}^{[y]}$');\n";
//    std::cin.get();


    for (int i = 0; i < exact.n_subs; i++)
    {
      delete plot_ISB[i];
      delete plot_ISBr2[i];
    }

    for (int i = 0; i < exact.n_Xs; i++)
    {
      delete plot_IXr2[i];
    }
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
