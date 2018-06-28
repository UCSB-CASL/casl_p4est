#include <src/Parser.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_epitaxy.h>

#include <ctime>


int main(int argc, char **argv)
{
  using namespace std;
  clock_t begin = clock();

  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nx", "number of blox in x-dimension, default = 2");
  cmd.add_option("ny", "number of blox in y-dimension, default = 2");
  cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
  cmd.add_option("save_every_n", "save vtk every n iteration, default = 1");
  cmd.add_option("tf", "final time");
  cmd.add_option("maxTimeStep", "defines the largest time step allowed, default = 2.5e-4");
  cmd.add_option("maxTimeStep", "defines the largest time step allowed, default = 6.25e-5");
  cmd.add_option("L", "set box_size, default = 180");
  cmd.add_option("D", "the diffusion coefficient, default = 1e5");
  cmd.add_option("F", "the deposition flux, default = 1");
  cmd.add_option("finalCoverage", "end the simulation when the final coverage theta is reached, default = .2");
  cmd.add_option("save_stats", "compute statistics for the final state. 0 or 1, default = 0");
  cmd.add_option("a", "set the lattice spacing, default = 1");
  cmd.add_option("one_level", "set to 1 to restrict the islands to one level only, default = 0");
  cmd.add_option("bc", "the boundary condition for rho at the island, either Dirichlet or Robin, default = Robin");
  cmd.add_option("barrier", "if robin bc, set the ratio D'/D. ~1 -> Dirichlet, <<1 -> steep islands. default = .5");
  cmd.parse(argc, argv);

  int nx                    = cmd.get("nx", 2);     // 2    MACROMESH
  int ny                    = cmd.get("ny", 2);     // 2    MACROMESH
  int lmin                  = cmd.get("lmin", 4);   // 4
  int lmax                  = cmd.get("lmax", 7);   // 9
  int save_stats            = cmd.get("save_stats", 1);
  int one_level_only        = cmd.get("one_level", 0);  //LEAVE AS IS

  double L                  = cmd.get("L", 180);
  double D                  = cmd.get("D", 1e7);    // D/F
  double F                  = cmd.get("F", 1);  // ALWAYS 1
  double a                  = cmd.get("a", 1);  // ALWAYS 1
  double tf                 = cmd.get("tf", DBL_MAX);
  double maxTimeStep        = cmd.get("maxTimeStep", 0.001); // 6.25e-5  // For D/F=10^p, shall we take maxTimeStep = 2.5*10^{-(p-1)}?
  double finalCoverage      = cmd.get("finalCoverage", .2); // .02 CORRESPONDS TO 2%. 1 LAYER IS finalCoverages=1
  double barrier            = cmd.get("barrier", .2);   // D'/D as in Papac et al

  BoundaryConditionType bc  = cmd.get("bc", DIRICHLET);

  bool save_vtk     = cmd.get("save_vtk", 1);
  int  save_every_n = cmd.get("save_every_n", 1);

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
  int n_xyz[] = {nx, ny};
  double xyz_min[] = {0, 0};
  double xyz_max[] = {L, L};
  int periodic[] = {1, 1};
  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  splitting_criteria_t data(lmin, lmax);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, (void*)(&data));
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
  epitaxy.set_parameters(D, F, 1.05, a, bc, barrier);
  epitaxy.set_one_level_only(one_level_only);

  double tn = 0;
  int iter = 0;
  PetscErrorCode ierr;

  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  FILE *fp = NULL, *fpTiming = NULL;
  char name[1000], nameTimeElapsed[1000];
  if(out_dir == NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save information into files\n"); CHKERRXX(ierr);
  }
  else if(p4est->mpirank==0)
  {
    if(bc==DIRICHLET){
      snprintf(name           , 1000, "%s/       Dirichlet_L=%d_Levels=(%d,%d)_D=%1.0e.txt", out_dir, (int)L, lmin, lmax, D/F);
      snprintf(nameTimeElapsed, 1000, "%s/Timing_Dirichlet_L=%d_Levels=(%d,%d)_D=%1.0e.txt", out_dir, (int)L, lmin, lmax, D/F);
    }
    else{
      snprintf(name           , 1000, "%s/       Robin_L=%d_Levels=(%d,%d)_D=%1.0e_Barrier=%1.0e.txt", out_dir, (int)L, lmin, lmax, D/F, barrier);
      snprintf(nameTimeElapsed, 1000, "%s/Timing_Robin_L=%d_Levels=(%d,%d)_D=%1.0e_Barrier=%1.0e.txt", out_dir, (int)L, lmin, lmax, D/F, barrier);
    }
    fp       = fopen(name, "w");
    if(fp==NULL)
      throw std::invalid_argument("Could not open file for time \t| coverage \t| Nuc \t| dt \t| nb_nodes \t| rho average");
    fprintf(fp, "%%time \t| coverage \t| Nuc \t| dt \t| nb_nodes \t| rho average \n");
    fclose(fp);

  }

  while(tn < tf && epitaxy.compute_coverage() < finalCoverage)
  {

    p4est = epitaxy.get_p4est();
    ierr = PetscPrintf(p4est->mpicomm, "\n###########################################\n\n"); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Iteration #%d, time tn = %e, coverage theta = %1.2f%% \n", iter, tn, epitaxy.compute_coverage()*100); CHKERRXX(ierr);

    if(iter!=0)
    {
//PAM
      epitaxy.compute_velocity();
//      epitaxy.compute_average_islands_velocity();
      epitaxy.compute_dt();
      epitaxy.update_grid();
      epitaxy.nucleate_new_island();
      
    }
    epitaxy.set_dt(maxTimeStep);
    epitaxy.compute_capture_zone();
    //epitaxy.stochastic_reversibility();
    do  // Update nucleation ODE for one time step, decreasing the time step if rho becomes negative or if we are about to seed
        // more than one island.
        // Note: this does not mean that N necessarily crosses an integer value.
    {
      epitaxy.solve_rho();
      epitaxy.update_nucleation();
    } while(!epitaxy.check_time_step());
    //epitaxy.compute_capture_zone();
    if(save_vtk==true && iter%save_every_n==0)
    {
      epitaxy.save_vtk(iter/save_every_n);

//PAM
    //std::ostringstream hierarchy_name; hierarchy_name << P4EST_DIM << "d_hierrchy";
  //  hierarchy->write_vtk("hierarchy");
    }
    double coverage_n = epitaxy.compute_coverage();
    p4est = epitaxy.get_p4est();
    nodes = epitaxy.get_nodes();
    if(p4est->mpirank==0)
    {
      p4est_locidx_t nb_nodes_global = 0;
      for(int r=0; r<p4est->mpisize; ++r)
        nb_nodes_global += nodes->global_owned_indeps[r];

      fp = fopen(name, "a");
      if(fp==NULL)
      {
        throw std::invalid_argument("Could not open file for time \t| coverage \t| Nuc \t| dt \t| nb_nodes \t| rho average");
      }
      fprintf(fp, "%.15e %.15e %.15e %.15e %d %.15e\n", tn, coverage_n, epitaxy.get_Nuc(), epitaxy.get_dt(), nb_nodes_global, epitaxy.getRhoAverage());
      fclose(fp);
    }

    tn += epitaxy.get_dt();
    iter++;
  }

  if(save_stats==1)
    epitaxy.compute_statistics();   // used to compute the cluster size distribution

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  if(p4est->mpirank==0) {
      fpTiming = fopen(nameTimeElapsed, "a");
      if(fpTiming==NULL)
      {
        throw std::invalid_argument("could not open file for timing output");
      }
      fprintf(fpTiming, "----------------------------------------------------\n");
      fprintf(fpTiming, "The simulation parameters are: \n");
      fprintf(fpTiming, "\t D/F:\t %1.0e \n", D/F);
      fprintf(fpTiming, "\t L:\t %d \n", (int) L);
      fprintf(fpTiming, "\t maxTimeStep:\t %1.0e \n", maxTimeStep);
      fprintf(fpTiming, "\t finalCoverage:\t %1.0f%% \n", finalCoverage*100);
      if (tf == DBL_MAX) fprintf(fpTiming, "\t tf:\t infinite, i.e. no constraint on final time. \n");
      else               fprintf(fpTiming, "\t tf:\t %e \n", tf);
      fprintf(fpTiming, "\t nx:\t %d \n", nx);
      fprintf(fpTiming, "\t ny:\t %d \n", ny);
      fprintf(fpTiming, "\t lmin:\t %d \n", lmin);
      fprintf(fpTiming, "\t lmax:\t %d \n", lmax);
      if (bc == ROBIN) fprintf(fpTiming, "\t ROBIN with a barrier of: %1.2f \n", barrier);
      else             fprintf(fpTiming, "\t DIRICHLET \n");
      fprintf(fpTiming, "\t Number of processors: %d \n", p4est->mpisize);
      fprintf(fpTiming, "\nThe total run time is: %e seconds \n\n", elapsed_secs);
      fclose(fpTiming);
  }

  return 0;
}
