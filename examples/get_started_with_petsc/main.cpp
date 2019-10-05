/* 
 * Title: get_started_with_petsc
 * --------------------------------------------------------------------------------------------------------
 * Short description: example project to illustrate the use of Petsc (vectors, matrices, index sets and
 * re-mapping)
 * --------------------------------------------------------------------------------------------------------
 * Exhaustive description: this example illustrates fundamental Petsc functions and features. We consider a
 * one-dimensional domain discretized with nnodes_per_proc grid nodes per process, uniformly spaced. The domain
 * can be either periodic or not. If not periodic, the domain is [xmin, xmax[, otherwise it is [xmin, xmax].
 * The grid node of global index j_glo = 0, 1, ...,  nprocs*nnodes_per_proc-1 is located at
 * x = xmin + ((double) j_glo)*(xmax-xmin)/((double) (nprocs*nnodes_per_proc-1)), in absence of periodicity
 * x = xmin + ((double) j_glo)*(xmax-xmin)/((double) (nprocs*nnodes_per_proc)),   in presence of periodicity
 *
 * This example first shows how to fill a Petsc Vector with appropriate values, including ghost values.
 *
 * Then a relevant explicit calculation is executed (calculation of second derivatives using standard finite
 * difference formula). This step illustrates the methodology used to hide communications as much as possible:
 * the relevant explicit calculations are done for layer nodes first (nodes locally owned but that are ghost
 * nodes for other processes), non-blocking communications are then initiated, then the relevant calculations
 * are executed for all the inner nodes before the communications are completed.
 *
 * Then, this example program calculates the first derivatives of a node-sampled field using compact finite
 * differences with a user-defined order of accuracy chosen between 4 or 6 (see Journal of Computational
 * Physics, Volume 103, Issue 1, November 1992, Pages 16-42). This step involves the resolution of a linear
 * system of equations, which requires the definition of a (sparse) matrix and the use of a built-in Petsc
 * iterative solver.
 *
 * Finally, we randomly redistribute the grid nodes: the grid changes from an equal number of nodes per process
 * to an unbalanced number of grid nodes per process. A relevant Petsc vector is then re-distributed (or
 * "re-scattered") to math the new grid partition.
 *
 * Author: Raphael Egan
 * Date Created: 09-26-2019
 */

#include "my_mpi_world.h"
#include <petsc.h>
#include <src/Parser.h>
#include "my_petsc_utils.h"
#include "one_dimensional_uniform_grid.h"
#include <math.h>

#define SQR(a) (a*a)

// dummy identity function returning the x-coordinate
class identity_function : public cont_function
{
public:
  double operator()(double x) const {return x;}
} identity_function;

// definition of a smooth function to be sampled at the nodes of the one-dimensional grid
// (feel free to change this definition, just bare in mind that periodicity must be respected
// if running the program with the "-periodic" argument)
class smooth_function : public cont_function
{
private:
  const double xmin, xmax;
public:
  smooth_function(double xmin_, double xmax_) : xmin(xmin_), xmax(xmax_) {}
  double operator()(double x) const { return log(1.0+SQR(sin(2.0*M_PI*x/(xmax-xmin)))); }
};

int main(int argc, char** argv) {
  // create and initialize the mpi and Petsc environment --> finalization is taken care of in the class' destructor!
  my_mpi_world mpi_environment(argc, argv);
  try {
    PetscErrorCode ierr;
    // create the argument parser and define the parameters that the user can set
    // (This is not a purely illustrative class, it is the casl_p4est's parser)
    cmdParser cmd;
    cmd.add_option("nnodes_proc",   "number of grid nodes per processor, a strict minimum of twice the number of ghost nodes is required (default is 128)");
    cmd.add_option("xmin",          "left boundary of the one-dimensional domain (default is 0.0)");
    cmd.add_option("xmax",          "right boundary of the one-dimensional domain (default is 1.0)");
    cmd.add_option("periodic",      "consider periodic boundary conditions if present");
    cmd.add_option("OOA",           "desired order of accuracy, 4 or 6 (defaukt is 4)");
    cmd.add_option("output_folder", "desired folder for outputs (defaukt is '.')");
    // read the user's input
    cmd.parse(argc, argv);
    // now read and check the user-provided parameters or set default values
    const unsigned int nnodes_per_proc  = cmd.get<unsigned int>("nnodes_proc", 128);
    const double xmin                   = cmd.get<double>("xmin", 0.0);
    const double xmax                   = cmd.get<double>("xmax", 1.0);
    const bool is_periodic              = cmd.contains("periodic");
    const unsigned int OOA              = cmd.get<unsigned int >("OOA", 4);
    const std::string out_dir           = cmd.get<std::string>("output_folder", ".");
    std::string filename;

    ierr = (OOA!=4 && OOA!=6); CHKERRXX(ierr);          // The order of accuracy (OOA) must be either 4 or 6
    ierr = (xmax <= xmin); CHKERRXX(ierr);              // xmax must be strictly greater than xmin.
    ierr = (nnodes_per_proc <= OOA-2); CHKERRXX(ierr);  // we prevent overlaps of ghost layers (it is technically possible to handle but not the purpose of the current illustration)

    // let's build a dummy parallel one-dimensional grid with uniform spacing
    // and an equal number of locally owned nodes on each process:
    one_dimensional_uniform_grid grid(mpi_environment, xmin, xmax, is_periodic);
    grid.set_partition_and_ghosts(nnodes_per_proc, OOA/2-1);

    // let's create two parallel vectors that samples the x-coordinates of the grid nodes
    // and the values of a continuous, well-behaved function and export them
    Vec coordinates = NULL;
    Vec node_sampled_function = NULL;                                                             // nothing but a (bunch of) pointer(s) are created at this stage
    ierr = vec_create_on_one_dimensional_grid(grid, &coordinates); CHKERRXX(ierr);                // this builds the parallel Petsc vector, but the assigned values are undetermined yet
    ierr = vec_create_on_one_dimensional_grid(grid, &node_sampled_function); CHKERRXX(ierr);      // this builds the parallel Petsc vector, but the assigned values are undetermined yet
    ierr = sample_vector_on_grid(coordinates, grid, identity_function); CHKERRXX(ierr);           // this assigns the values in the Petsc vector (including ghost values)
    smooth_function my_smooth_function(xmin, xmax);
    ierr = sample_vector_on_grid(node_sampled_function, grid, my_smooth_function); CHKERRXX(ierr);// this assigns the values in the Petsc vector (including ghost values)
    // let's export all that
    filename = out_dir+"/coordinates.mat";
    ierr = export_in_binary_format(coordinates, filename.c_str()); CHKERRXX(ierr);
    filename = out_dir+"/my_smooth_function.mat";
    ierr = export_in_binary_format(node_sampled_function, filename.c_str()); CHKERRXX(ierr);

    // now, let's calculate the second derivative using standard finite differences, node by
    // node and export it as well. This routine illustrates how to hide communications: the
    // calculations are first done for layer nodes then communications are initiated, then inner
    // node calculations are done and finally, communications are completed afterwards.
    Vec second_derivative_of_node_sampled_function = NULL;
    ierr = vec_create_on_one_dimensional_grid(grid, &second_derivative_of_node_sampled_function); CHKERRXX(ierr);
    grid.calculate_second_derivative_of_field(node_sampled_function, second_derivative_of_node_sampled_function);
    filename = out_dir+"/second_derivative_of_smooth_function.mat";
    ierr = export_in_binary_format(second_derivative_of_node_sampled_function, filename.c_str()); CHKERRXX(ierr);

    // now, let's evaluate the first derivative using compact finite differences. This method requires
    // the resolution of a linear system of equation, so the following method illustrates how to build a
    // linear system of equation A*x=b and how to solve for x for a fairly simple problem.
    Vec first_derivative_compact_finite_differences = NULL;
    ierr = vec_create_on_one_dimensional_grid(grid, &first_derivative_compact_finite_differences); CHKERRXX(ierr);
    grid.calculate_first_derivative_compact_fd(node_sampled_function, first_derivative_compact_finite_differences, OOA);
    filename = out_dir+"/first_derivative_of_smooth_function.mat";
    ierr = export_in_binary_format(first_derivative_compact_finite_differences, filename.c_str()); CHKERRXX(ierr);

    // now let's randomly redistribute the one-dimensional grid and let's rescatter the node values of the
    // node-sampled first derivative to the new layout dictated by the newly defined grid.
    // The function remap used here below shows two rather equivalent approaches, check out the source code
    // if you are interested.
    grid.shuffle();
    Vec first_derivative_compact_finite_differences_on_new_grid = NULL;
    ierr = vec_create_on_one_dimensional_grid(grid, &first_derivative_compact_finite_differences_on_new_grid); CHKERRXX(ierr);
    grid.remap(first_derivative_compact_finite_differences, first_derivative_compact_finite_differences_on_new_grid);
    filename = out_dir+"/first_derivative_of_smooth_function_on_new_grid.mat";
    ierr = export_in_binary_format(first_derivative_compact_finite_differences_on_new_grid, filename.c_str()); CHKERRXX(ierr);

    // never forget to destroy whatever you have created.
    if (coordinates != NULL){
      ierr = VecDestroy(coordinates); CHKERRXX(ierr); }
    if (node_sampled_function != NULL){
      ierr = VecDestroy(node_sampled_function); CHKERRXX(ierr); }
    if (second_derivative_of_node_sampled_function != NULL) {
      ierr = VecDestroy(second_derivative_of_node_sampled_function); CHKERRXX(ierr); }
    if (first_derivative_compact_finite_differences != NULL) {
      ierr = VecDestroy(first_derivative_compact_finite_differences); CHKERRXX(ierr); }
    if(first_derivative_compact_finite_differences_on_new_grid!=NULL){
      ierr = VecDestroy(first_derivative_compact_finite_differences_on_new_grid); CHKERRXX(ierr); }
  } catch (const std::exception& e) {
    if(mpi_environment.rank()==0)
      std::cerr << e.what() << std::endl;
  }

  return 0;
}

