// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// My files for this project

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>

// casl_p4est
#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/semi_lagrangian.h>
#include <src/refine_coarsen.h>

using namespace std;

int main (int argc, char* argv[]){

  mpi_context_t       mpi_context, *mpi = &mpi_context;
  mpi->mpicomm = MPI_COMM_WORLD;
  p4est_connectivity_t *connectivity;
  p4est_t            *p4est;

  struct phi:CF_2{
    phi(double r_)
      : r(r_)
    {}
    void update(double r_){
      r = r_;
    }

    double operator()(double x, double y) const {
      return r - sqrt(SQR(x-1.0) + SQR(y-1.0));
    }

  private:
    double r;
  };

  double rmin = 0.0, rmax = 1.5;
  phi circle(rmin);
  grid_continous_data_t data = {&circle, 6, 0, 1.0};

  Session session(argc, argv);
  session.init(mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);


  // Create the connectivity object
  w2.start("connectivity");
  connectivity = p4est_connectivity_new_brick (2, 2, 0, 0);
  w2.stop(); w2.read_duration();

  // Now create the forest
  w2.start("p4est generation");
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();
  p4est->user_pointer = (void*)(&data);

  double dr = (rmax-rmin)/30;
  int tc = 0;
  std::ostringstream oss;
  for (double r = rmin; r<=rmax; r += dr, tc++)
  {
    circle.update(r);
    oss << "grid refining/coarrsening step-" << tc;

    w2.start(oss.str());
    p4est_refine (p4est, P4EST_TRUE, refine_levelset_continous , NULL);
    p4est_coarsen(p4est, P4EST_TRUE, coarsen_levelset_continous, NULL);
    w2.stop(); w2.read_duration();

    oss.str(""); oss << "grid partitioning step-" << tc;
    w2.start(oss.str());
    p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    oss.str(""); oss << "grid." << tc;
    my_p4est_vtk_write_all(p4est, NULL, 1.0,
                           0, 0, oss.str().c_str());
  }

  // destroy the p4est and its connectivity structure
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);

  w1.stop(); w1.read_duration();

  return 0;
}

