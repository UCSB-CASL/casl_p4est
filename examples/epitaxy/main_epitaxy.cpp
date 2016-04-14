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
  cmd.add_option("nx", "number of blox in x-dimension");
  cmd.add_option("ny", "number of blox in y-dimension");
  cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
  cmd.add_option("save_every_n", "save vtk every n iteration");
  cmd.add_option("tf", "final time");
  cmd.add_option("box_size", "set box_size");
  cmd.add_option("D", "the diffusion coefficient, ");
  cmd.parse(argc, argv);

  int nx = cmd.get("nx", 2);
  int ny = cmd.get("ny", 2);
  double L = cmd.get("box_size", 180);
  int lmin = cmd.get("lmin", 4);
  int lmax = cmd.get("lmax", 7);
  double tf = cmd.get("tf", DBL_MAX);

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
  epitaxy.set_parameters(1e5, 1, 1);

  double tn = 0;
  int iter = 0;
  PetscErrorCode ierr;

  while(tn<tf)
  {
    p4est = epitaxy.get_p4est();
    ierr = PetscPrintf(p4est->mpicomm, "###########################################\n"); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Iteration #%d, tn = %e\n", iter, tn); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "###########################################\n"); CHKERRXX(ierr);
    if(iter!=0)
    {
      epitaxy.compute_velocity();
      epitaxy.compute_average_islands_velocity();
      epitaxy.compute_dt();
      epitaxy.update_grid();
      epitaxy.nucleate_new_island();
    }

    do
    {
      epitaxy.solve_rho();
      epitaxy.update_nucleation();
    } while(!epitaxy.check_time_step());

    if(save_vtk==true && iter%save_every_n==0)
    {
      epitaxy.save_vtk(iter/save_every_n);
    }

    iter++;
    tn += epitaxy.get_dt();
//    if(iter==6) break;
  }

  return 0;
}
