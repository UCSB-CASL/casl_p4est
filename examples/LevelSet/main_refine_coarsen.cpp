// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// My files for this project

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <list>

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
  mpi->mpicomm = MPI_COMM_WORLD;        //Use all processors for task
  p4est_connectivity_t *connectivity;   //Sets up the macro mesh, aka the entire forest itself
  p4est_t            *p4est;            //Sets up the micro mesh, the individual trees

  struct phi:CF_2{
    phi(double r_)
      : r(r_)
    {}
    void update(double r_){
      r = r_;
    }

    /*double operator()(double x, double y) const {   //Creates the circle with center at X, Y
      return r - sqrt(SQR(x-2.0) + SQR(y-2.0));     //Defines a level set of a circle. Return the level set!
    }
    */

    double operator()(double x, double y) const{
        double c1 = -1 - sqrt(SQR(x-0.5) + SQR(y-0.5));
        double c2 = .5 - sqrt(SQR(x-1.5) + SQR(y-0.5));
        double c3 = 1.5 - sqrt(SQR(x-0.5) + SQR(y-1.5));
        double c4 = 1 - sqrt(SQR(x-1.5) + SQR(y-1.5));

        return max(max(max(c1, c2), c3), c4);
    }

    private:
        double r;
  };

  double rmin = 0, rmax = 3.5;
  phi circle(rmin);
  refine_coarsen_data_t data = {&circle, 6, 0, 1.0};

  Session session(argc, argv);  //Used to initialize mpi
  session.init(mpi->mpicomm);   //Used to initialize mpi

  parStopWatch w1, w2;      //Stop watch for timing functions
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);  //Size of group of processors
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);  //Rank of the group


  // Create the connectivity object
  w2.start("connectivity");     //Start timer with "message"
  connectivity = p4est_connectivity_new_brick (4, 4, 0, 0); //(Col, Rows) Determines the number of macro meshes, or the initial amount of "quadrants".
  w2.stop(); w2.read_duration();    //End timer

  /*
  w2.start("connectivity");
  connectivity = p4est_connectivity_new_star();
  w2.stop(); w2.read_duration();
  */

  // Now create the forest
  w2.start("p4est generation");
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();
  p4est->user_pointer = (void*)(&data);

  double dr = (rmax-rmin)/30;
  int tc = 0;
  std::ostringstream oss;   //Used to output to console

  for (double r = rmin; r<=rmax; r += dr, tc++)
  {
    circle.update(r);   //Update the boundary
    oss << "grid refining/coarrsening step-" << tc;     //Help output steps

    w2.start(oss.str());
    p4est_refine (p4est, P4EST_TRUE, refine_levelset , NULL); //Moves the childen/leave/divisions/cells close to the boundary
    p4est_coarsen(p4est, P4EST_TRUE, coarsen_levelset, NULL); //Removes the children/leaves further away from the boundary
    w2.stop(); w2.read_duration();

    oss.str(""); oss << "grid partitioning step-" << tc;    //Help output steps
    w2.start(oss.str());
    p4est_partition(p4est, NULL);   //Partition the forest to equal (enough) quadrants and distribute them to each processor
    w2.stop(); w2.read_duration();

    //Will want to add code to do more interesting things here at one point in time.    

    oss.str(""); oss << "grid." << tc;  //Export data to a .pvtu file for use in paraView
    my_p4est_vtk_write_all(p4est, NULL, 1.0,
                           0, 0, oss.str().c_str());
  }


  // destroy the p4est and its connectivity structure. Free up memory
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);

  w1.stop(); w1.read_duration();

  cout << "Hello World!@?" << endl;
  return 0;
}

