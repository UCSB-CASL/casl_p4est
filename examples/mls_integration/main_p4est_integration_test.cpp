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

/* discretization */
int lmin = 3;
int lmax = 3;
#ifdef P4_TO_P8
int nb_splits = 3;
#else
int nb_splits = 6;
#endif

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

bool save_vtk = false;

//#include "geometry_one_circle.cpp"
#include "geometry_two_circles_union.cpp"

class Result
{
public:
  vector<double> ID, IB, IDr2, IBr2;
  vector< vector<double> > ISB, ISBr2, IXr2;
  Result()
  {
    for (int i = 0; i < exact.n_subs; i++)
    {
      ISB.push_back(vector<double>());
      ISBr2.push_back(vector<double>());
    }
    for (int i = 0; i < exact.n_Xs; i++)
    {
      IXr2.push_back(vector<double>());
    }
  }
} res_mlt;

vector<double> level, h;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz,
                                    xmin, xmax, ymin, ymax, zmin, zmax,
                                    &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny,
                                    xmin, xmax, ymin, ymax,
                                    &brick);
#endif

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &ls_circle_0, 1.2);
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
    integration.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, phi_zz_vec, geometry.action, geometry.color);
//    integration.set_use_cube_refined(4);

    integration.initialize();

    if (save_vtk)
    {
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
    }

    /* Calculate and store results */
    if (exact.provided || iter < nb_splits-1)
    {
      level.push_back(lmax+iter);
      h.push_back((xmax-xmin)/pow(2.0,(double)(lmax+iter)));

      res_mlt.ID.push_back(integration.measure_of_domain    ());
      res_mlt.IB.push_back(integration.measure_of_interface (-1));

      res_mlt.IDr2.push_back(integration.integrate_over_domain    (f_r2));
      res_mlt.IBr2.push_back(integration.integrate_over_interface (f_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        res_mlt.ISB[i].push_back(integration.measure_of_interface(geometry.color[i]));
        res_mlt.ISBr2[i].push_back(integration.integrate_over_interface(f_r2, geometry.color[i]));
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
        res_mlt.IXr2[i].push_back(integration.integrate_over_intersection(f_r2, exact.IXc0[i], exact.IXc1[i]));
      }
    }
    else if (iter == nb_splits-1)
    {
      exact.ID    = (integration.measure_of_domain        ());
      exact.IB    = (integration.measure_of_interface     (-1));
      exact.IDr2  = (integration.integrate_over_domain    (f_r2));
      exact.IBr2  = (integration.integrate_over_interface (f_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        exact.ISB.push_back(integration.measure_of_interface(geometry.color[i]));
        exact.ISBr2.push_back(integration.integrate_over_interface(f_r2, geometry.color[i]));
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
        exact.IXr2.push_back(integration.integrate_over_intersection(f_r2, exact.IXc0[i], exact.IXc1[i]));
      }
    }

    ierr = VecDestroy(f_r2); CHKERRXX(ierr);

    for (int i = 0; i < phi_vec.size(); i++) {ierr = VecDestroy(phi_vec[i]); CHKERRXX(ierr);}
    phi_vec.clear();

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  Gnuplot plot_ID;
  print_Table("Domain", exact.ID, level, h, "MLT", res_mlt.ID, 2, &plot_ID);

  Gnuplot plot_IB;
  print_Table("Interface", exact.IB, level, h, "MLT", res_mlt.IB, 2, &plot_IB);

//  Gnuplot plot_IDr2;
//  print_Table("2nd moment of domain", exact.IDr2, level, h, "MLT", res_mlt.IDr2, 2, &plot_IDr2);

//  Gnuplot plot_IBr2;
//  print_Table("2nd moment of interface", exact.IBr2, level, h, "MLT", res_mlt.IBr2, 2, &plot_IBr2);

  vector<Gnuplot *> plot_ISB;
  vector<Gnuplot *> plot_ISBr2;
  for (int i = 0; i < exact.n_subs; i++)
  {
    plot_ISB.push_back(new Gnuplot());
    print_Table("Interface #"+to_string(i), exact.ISB[i], level, h, "MLT", res_mlt.ISB[i], 2, plot_ISB[i]);

//    plot_ISBr2.push_back(new Gnuplot());
//    print_Table("2nd moment of interface #"+to_string(i), exact.ISBr2[i], level, h, "MLT", res_mlt.ISBr2[i], 2, plot_ISBr2[i]);
  }

  vector<Gnuplot *> plot_IXr2;
  for (int i = 0; i < exact.n_Xs; i++)
  {
//    plot_IXr2.push_back(new Gnuplot());
//    print_Table("Intersection of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), exact.IXr2[i], level, h, "MLT", res_mlt.IXr2[i], 2, plot_IXr2[i]);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  std::cin.get();

  for (int i = 0; i < exact.n_subs; i++)
  {
    delete plot_ISB[i];
    delete plot_ISBr2[i];
  }

  for (int i = 0; i < exact.n_Xs; i++)
  {
    delete plot_IXr2[i];
  }

  return 0;
}
