/*
 * Title: petsc_memleak_check
 * Description: testing if your installed version of PetSc leaks memory in two basic features:
 * 1) creation and destruction of vectors
 * 2) nonblocking updates of ghost values
 * (some compilers may make PetSc's basic functions leak memory)
 * Author: Raphael Egan
 * Date Created: 07-07-2020
 */

#include <src/Parser.h>
#include <src/petsc_compatibility.h>
#include <vector>
#include <stdlib.h>
#include <time.h>

const static std::string main_description =
    std::string("In this example, we test the local installation of Petsc for possible memory leak within the two elementary features:\n")
    + std::string("1) creation and destruction of PetSc vectors; \n")
    + std::string("2) nonblocking updates of ghost values in such vectors. \n")
    + std::string("The program first goes through a loop of niter steps creating and destructing a (parallel) vector of nloc local entries and nghost ghost entries. \n")
    + std::string("Then the program creates one single such vector and goes through a loop of niter steps updating the ghost values using asynchronous PetSc methods. \n")
    + std::string("The memory usage is assessed at every step of either of these loops and exported in a .dat file on disk in the work_dir directory. \n")
    + std::string("(Gnuplot elementary plotting files are exported along these.dat files for easier visualization of the results).\n")
    + std::string("Developers: Raphael Egan (raphaelegan@ucsb.edu), Summer 2020\n");

const static int default_niter  = 10000;
const static int default_nghost = 10000;
const static int default_nloc   = 1024*1024;

int main(int argc, char** argv) {

  PetscErrorCode ierr;
  const MPI_Comm mpicomm = MPI_COMM_WORLD;
  int mpirank;
  int mpisize;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(mpicomm, &mpisize);
  MPI_Comm_rank(mpicomm, &mpirank);
  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);
  cmdParser cmd;
  cmd.add_option("niter",     "number of iterations for both loops, default is " + std::to_string(default_niter));
  cmd.add_option("nghost",    "number of ghost entries per process in the vectors, default is " + std::to_string(default_nghost));
  cmd.add_option("nloc",      "number of local (owned) entries per process in the vectors, default is " + std::to_string(default_nloc));
  cmd.add_option("work_dir",  "exportation directory, this is required for saving .dat and gnuplot files. Default is the value defined in the environmnent variable OUT_DIR or './' if not defined so...");
  if(cmd.parse(argc, argv, main_description))
    return 0;

  const PetscInt niter  = cmd.get<PetscInt>("niter",  default_niter);
  const PetscInt nghost = cmd.get<PetscInt>("nghost", (mpisize == 1 ? 0 : default_nghost));
  const PetscInt nloc   = cmd.get<PetscInt>("nloc",   default_nloc);
  std::string out_folder = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? "./" : getenv("OUT_DIR")));
  if(out_folder.back() != '/')
    out_folder += "/";
  std::ostringstream oss;
  oss << out_folder;
  std::ostringstream command;
  command << "mkdir -p " << oss.str();
  if (mpirank == 0)
  {
    std::cout << "Creating a folder in " << oss.str() << std::endl;
    int ret = system(command.str().c_str());
    if(ret != 0)
    {
      std::cout << "failed to create the desired folder... Aborting! " << std::endl;
      MPI_Abort(mpicomm, 42);
    }
  }
  int mpiret = MPI_Barrier(mpicomm); CHKERRMPI(mpiret);

  if(niter < 0 || nghost < 0 || nloc < 0 || (mpisize > 1 && nghost > (mpisize - 1)*nloc))
    throw std::invalid_argument("petsc_memleak_check : invalid input parameter niter, nghost or nloc...");

  srand (time(NULL));
  std::vector<PetscInt> ghost_node_global_indices(nghost);
  for (int k = 0; k < nghost; ++k) {
    PetscInt ghost_glo_idx = rand() % (mpisize * nloc); // between 0 and (mpisize * nloc - 1)
    while (mpirank*nloc <= ghost_glo_idx && ghost_glo_idx < (mpirank + 1)*nloc) // take another one ifnot an actual ghost index
      ghost_glo_idx = rand() % (mpisize * nloc);
    ghost_node_global_indices[k] = ghost_glo_idx;
  }

  // creation - destruction
  const std::string create_destroy_memory = out_folder + "create_destroy.dat";
  const std::string liveplot_memory_create_destroy= out_folder + "live_creation_destruction.gnu";

  FILE* fp_liveplot;
  ierr = PetscFOpen(mpicomm, liveplot_memory_create_destroy.c_str(), "w", &fp_liveplot); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp_liveplot, "set term wxt noraise\n"); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp_liveplot, "set key bottom right Left font \"Arial,14\"\n"); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp_liveplot, "set xlabel \"Iteration [-]\" font \"Arial,14\"\n"); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp_liveplot, "set ylabel \"Memory usage [Mb]\" font \"Arial,14\"\n"); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp_liveplot, "plot \"%s\" using 1:2 title 'Memory usage for creation-destruction loop' with lines lw 3\n", create_destroy_memory.c_str()); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp_liveplot, "pause 4\n"); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp_liveplot, "reread"); CHKERRXX(ierr);
  ierr = PetscFClose(mpicomm,  fp_liveplot); CHKERRXX(ierr);

  FILE* fp;
  ierr = PetscFOpen(mpicomm, create_destroy_memory.c_str(), "w", &fp); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpicomm, fp, "%% iter | memory (Mb) \n"); CHKERRXX(ierr);
  for (int iter = 0; iter < niter; ++iter) {
    Vec test;
    ierr = VecCreateGhost(mpicomm, nloc, mpisize*nloc, nghost, ghost_node_global_indices.data(), &test); CHKERRXX(ierr);

    ierr = VecSetFromOptions(test); CHKERRXX(ierr);
    ierr = VecDestroy(test); CHKERRXX(ierr);

    PetscLogDouble mem_petsc = 0;
    ierr = PetscMemoryGetCurrentUsage(&mem_petsc); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp, "%d %e\n", iter, mem_petsc/1024./1024.); CHKERRXX(ierr);
  }
  ierr = PetscFClose(mpicomm,  fp); CHKERRXX(ierr);

  if(nghost > 0 && mpisize > 1)
  {
    // ghost update
    const std::string ghost_update_memory = out_folder + "ghost_update.dat";
    const std::string liveplot_memory_ghost_update = out_folder + "live_ghost_update.gnu";

    ierr = PetscFOpen(mpicomm, liveplot_memory_ghost_update.c_str(), "w", &fp_liveplot); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp_liveplot, "set term wxt noraise\n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp_liveplot, "set key bottom right Left font \"Arial,14\"\n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp_liveplot, "set xlabel \"Iteration [-]\" font \"Arial,14\"\n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp_liveplot, "set ylabel \"Memory usage [Mb]\" font \"Arial,14\"\n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp_liveplot, "plot \"%s\" using 1:2 title 'Memory usage for ghost-update loop' with lines lw 3\n", ghost_update_memory.c_str()); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp_liveplot, "pause 4\n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp_liveplot, "reread"); CHKERRXX(ierr);
    ierr = PetscFClose(mpicomm,  fp_liveplot); CHKERRXX(ierr);

    ierr = PetscFOpen(mpicomm, ghost_update_memory.c_str(), "w", &fp); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpicomm, fp, "%% iter | memory (Mb) \n"); CHKERRXX(ierr);
    Vec test;
    ierr = VecCreateGhost(mpicomm, nloc, mpisize*nloc, nghost, ghost_node_global_indices.data(), &test); CHKERRXX(ierr);
    ierr = VecSetFromOptions(test); CHKERRXX(ierr);
    for (int iter = 0; iter < niter; ++iter) {
      ierr = VecGhostUpdateBegin(test, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(test, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      PetscLogDouble mem_petsc = 0;
      ierr = PetscMemoryGetCurrentUsage(&mem_petsc); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpicomm, fp, "%d %e\n", iter, mem_petsc/1024./1024.); CHKERRXX(ierr);
    }
    ierr = VecDestroy(test); CHKERRXX(ierr);
    ierr = PetscFClose(mpicomm,  fp); CHKERRXX(ierr);
  }

  ierr = PetscFinalize(); CHKERRXX(ierr);
  MPI_Finalize();
}

