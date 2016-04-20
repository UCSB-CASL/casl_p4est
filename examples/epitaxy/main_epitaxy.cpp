#include <src/Parser.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_epitaxy.h>


int main(int argc, char **argv)
{
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);
  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nx", "number of blox in x-dimension, default = 2");
  cmd.add_option("ny", "number of blox in y-dimension, default = 2");
  cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
  cmd.add_option("save_every_n", "save vtk every n iteration, default = 1");
  cmd.add_option("tf", "final time");
  cmd.add_option("L", "set box_size, default = 180");
  cmd.add_option("D", "the diffusion coefficient, default = 1e5");
  cmd.add_option("F", "the deposition flux, default = 1");
  cmd.add_option("coverage", "end the simulation when the coverage theta is reached, default = .2");
  cmd.add_option("save_stats", "compute statistics for the final state. 0 or 1, default = 0");
  cmd.add_option("a", "set the lattice spacing, default = 180/300 = .6");
  cmd.add_option("one_level", "set to 1 to restrict the islands to one level only, default = 0");
  cmd.parse(argc, argv);

  int nx = cmd.get("nx", 2);
  int ny = cmd.get("ny", 2);
  double L = cmd.get("L", 180);
  double D = cmd.get("D", 1e5);
  double F = cmd.get("F", 1);
  double coverage = cmd.get("coverage", .2);
  int lmin = cmd.get("lmin", 4);
  int lmax = cmd.get("lmax", 7);
  double tf = cmd.get("tf", DBL_MAX);
  int save_stats = cmd.get("save_stats", 0);
  double a = cmd.get("a", .6);
  int one_level_only = cmd.get("one_level", 0);

  bool save_vtk = cmd.get("save_vtk", 1);
  int save_every_n = cmd.get("save_every_n", 1);

  if(0)
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }

  /* create the p4est */
  my_p4est_brick_t brick;
  p4est_connectivity_t *connectivity = my_p4est_brick_new(nx, ny, 0, L, 0, L, &brick, 1, 1);

  splitting_criteria_t data(lmin, lmax);
  p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, (void*)(&data));
  for(int lvl=0; lvl<lmin; ++lvl)
  {
    my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }
  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);

  my_p4est_epitaxy_t epitaxy(ngbd);
  epitaxy.set_parameters(D, F, 1.05, a);
  epitaxy.set_one_level_only(one_level_only);

  double tn = 0;
  int iter = 0;
  PetscErrorCode ierr;

  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  FILE *fp = NULL;
  char name[1000];
  if(out_dir == NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "you need to set the environment variable OUT_DIR to save coverage vs. time information\n"); CHKERRXX(ierr);
  }
  else if(p4est->mpirank==0)
  {
    snprintf(name, 1000, "%s/%d-%d_DF_%1.2e.dat", out_dir, lmin, lmax, D/F);
    fp = fopen(name, "w");
    if(fp==NULL)
      throw std::invalid_argument("could not open file for coverage vs. time output");
    fprintf(fp, "%%time | coverage | Nuc | dt | nb_nodes\n");
    fclose(fp);
  }

  while(tn<tf && epitaxy.compute_coverage()<coverage)
  {
    p4est = epitaxy.get_p4est();
    ierr = PetscPrintf(p4est->mpicomm, "###########################################\n"); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Iteration #%d, tn = %e, coverage theta = %2.1f%%\n", iter, tn, epitaxy.compute_coverage()*100); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "###########################################\n"); CHKERRXX(ierr);
    if(iter!=0)
    {
      epitaxy.compute_velocity();
//      epitaxy.compute_average_islands_velocity();
      epitaxy.compute_dt();
      epitaxy.update_grid();
      epitaxy.nucleate_new_island();
    }

    do
    {
      epitaxy.solve_rho();
      epitaxy.update_nucleation();
    } while(!epitaxy.check_time_step());
    epitaxy.compute_islands_numbers();

    if(save_vtk==true && iter%save_every_n==0)
    {
      epitaxy.save_vtk(iter/save_every_n);
    }

    double coverage_n = epitaxy.compute_coverage();
    p4est = epitaxy.get_p4est();
    nodes = epitaxy.get_nodes();
    if(p4est->mpirank==0)
    {
      p4est_locidx_t nb_nodes_global = 0;
      for(int r=0; r<p4est->mpisize; ++r)
        nb_nodes_global += nodes->global_owned_indeps[r];

      printf("The p4est has %d nodes.\n", nb_nodes_global);

      fp = fopen(name, "a");
      if(fp==NULL)
      {
        throw std::invalid_argument("could not open file for coverage vs. time output");
      }
      fprintf(fp, "%.15e %.15e %.15e %.15e %d\n", tn, coverage_n, epitaxy.get_Nuc(), epitaxy.get_dt(), nb_nodes_global);
      fclose(fp);
    }

    iter++;
    tn += epitaxy.get_dt();
  }

  if(save_stats==1)
    epitaxy.compute_statistics();

  return 0;
}
