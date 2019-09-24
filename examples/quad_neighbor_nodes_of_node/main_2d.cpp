/*
 * Title: quad_neighbor_nodes_of_node
 * Description: testing new features added to the my_p4est_quad_neighbor_nodes_of_nodes_t class
 * added features: block-structured vectors, possibility to store elementary operators for derivatives
 * Author: Raphael Egan
 * Date Created: 09-20-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif

#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/casl_math.h>
#include <stdlib.h>

using namespace std;

#ifdef P4_TO_P8
class test_function : public CF_3 {
#else
class test_function : public CF_2 {
#endif
public:
#ifdef P4_TO_P8
  virtual double dx   (double x, double y, double z) const=0;
  virtual double dy   (double x, double y, double z) const=0;
  virtual double dz   (double x, double y, double z) const=0;
  virtual double ddxx (double x, double y, double z) const=0;
  virtual double ddyy (double x, double y, double z) const=0;
  virtual double ddzz (double x, double y, double z) const=0;
#else
  virtual double dx   (double x, double y) const=0;
  virtual double dy   (double x, double y) const=0;
  virtual double ddxx (double x, double y) const=0;
  virtual double ddyy (double x, double y) const=0;
#endif
  virtual ~test_function() {}
};


#ifdef P4_TO_P8
struct circle : CF_3 {
  circle(double x0_, double y0_, double z0_, double r_): x0(x0_), y0(y0_), z0(z0_), r(r_) {}
#else
struct circle : CF_2 {
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
#else
  double operator()(double x, double y) const {
#endif
    return r - sqrt(SQR(x-x0) + SQR(y-y0)
                #ifdef P4_TO_P8
                    + SQR(z-z0)
                #endif
                    );
  }

private:
  double x0, y0;
#ifdef P4_TO_P8
  double z0;
#endif
  double r;
};

#ifdef P4_TO_P8
class uex : public test_function {
private:
  double a, b, c;
public:
  uex(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
#else
class uex : public test_function {
private:
  double a, b;
public:
  uex(double a_, double b_) : a(a_), b(b_) {}
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
    return 1.0/(cos(a*x*x + b*y*y + c*z*z)+1.5);
#else
  double operator()(double x, double y) const {
    return 1.0/(cos(a*x*x + b*y*y)+1.5);
#endif
  }

#ifdef P4_TO_P8
  double dx(double x, double y, double z) const {
    return 2.0*a*x*sin(a*x*x + b*y*y + c*z*z)/(SQR(cos(a*x*x + b*y*y + c*z*z)+1.5));
#else
  double dx(double x, double y) const {
    return 2.0*a*x*sin(a*x*x + b*y*y)/(SQR(cos(a*x*x + b*y*y)+1.5));
#endif
  }

#ifdef P4_TO_P8
  double dy(double x, double y, double z) const {
    return 2.0*b*y*sin(a*x*x + b*y*y + c*z*z)/(SQR(cos(a*x*x + b*y*y + c*z*z)+1.5));
#else
  double dy(double x, double y) const {
    return 2.0*b*y*sin(a*x*x + b*y*y)/(SQR(cos(a*x*x + b*y*y)+1.5));
#endif
  }

#ifdef P4_TO_P8
  double dz(double x, double y, double z) const {
    return 2.0*c*z*sin(a*x*x + b*y*y + c*z*z)/(SQR(cos(a*x*x + b*y*y + c*z*z)+1.5));
  }
#endif

#ifdef P4_TO_P8
  double ddxx(double x, double y, double z) const {
    return (4.0*a*a*x*x*cos(a*x*x + b*y*y + c*z*z)/SQR(1.5 + cos(a*x*x + b*y*y + c*z*z)) + 8.0*a*a*x*x*SQR(sin(a*x*x + b*y*y + c*z*z))/pow((1.5 + cos(a*x*x + b*y*y + c*z*z)), 3.0) + 2.0*a*sin(a*x*x + b*y*y + c*z*z)/SQR(1.5+cos(a*x*x + b*y*y + c*z*z)));
#else
  double ddxx(double x, double y) const {
    return (4.0*a*a*x*x*cos(a*x*x + b*y*y)/SQR(1.5 + cos(a*x*x + b*y*y)) + 8.0*a*a*x*x*SQR(sin(a*x*x + b*y*y))/pow((1.5 + cos(a*x*x + b*y*y)), 3.0) + 2.0*a*sin(a*x*x + b*y*y)/SQR(1.5+cos(a*x*x + b*y*y)));
#endif
  }

#ifdef P4_TO_P8
  double ddyy(double x, double y, double z) const {
    return (4.0*b*b*y*y*cos(a*x*x + b*y*y + c*z*z)/SQR(1.5 + cos(a*x*x + b*y*y + c*z*z)) + 8.0*b*b*y*y*SQR(sin(a*x*x + b*y*y + c*z*z))/pow((1.5 + cos(a*x*x + b*y*y + c*z*z)), 3.0) + 2.0*b*sin(a*x*x + b*y*y + c*z*z)/SQR(1.5+cos(a*x*x + b*y*y + c*z*z)));
#else
  double ddyy(double x, double y) const {
    return (4.0*b*b*y*y*cos(a*x*x + b*y*y)/SQR(1.5 + cos(a*x*x + b*y*y)) + 8.0*b*b*y*y*SQR(sin(a*x*x + b*y*y))/pow((1.5 + cos(a*x*x + b*y*y)), 3.0) + 2.0*b*sin(a*x*x + b*y*y)/SQR(1.5+cos(a*x*x + b*y*y)));
#endif
  }

#ifdef P4_TO_P8
  double ddzz(double x, double y, double z) const {
    return (4.0*c*c*z*z*cos(a*x*x + b*y*y + c*z*z)/SQR(1.5 + cos(a*x*x + b*y*y + c*z*z)) + 8.0*c*c*z*z*SQR(sin(a*x*x + b*y*y + c*z*z))/pow((1.5 + cos(a*x*x + b*y*y + c*z*z)), 3.0) + 2.0*c*sin(a*x*x + b*y*y + c*z*z)/SQR(1.5+cos(a*x*x + b*y*y + c*z*z)));
  }
#endif
};

#ifdef P4_TO_P8
class vex : public test_function {
private:
  double a, b, c;
public:
  vex(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
#else
class vex : public test_function {
private:
  double a, b;
public:
  vex(double a_, double b_) : a(a_), b(b_) {}
#endif

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
    return cos(a*x+y)*sin(b*y-x)*atan(c*z);
#else
  double operator()(double x, double y) const {
    return cos(a*x+y)*sin(b*y-x);
#endif
  }

#ifdef P4_TO_P8
  double dx(double x, double y, double z) const {
    return (-a*sin(a*x+y)*sin(b*y-x) - cos(a*x+y)*cos(b*y-x))*atan(c*z);
#else
  double dx(double x, double y) const {
    return (-a*sin(a*x+y)*sin(b*y-x) - cos(a*x+y)*cos(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double ddxx(double x, double y, double z) const {
    return (-a*a*cos(a*x+y)*sin(b*y-x) + 2.0*a*sin(a*x+y)*cos(b*y-x) - cos(a*x+y)*sin(b*y-x))*atan(c*z);
#else
  double ddxx(double x, double y) const {
    return (-a*a*cos(a*x+y)*sin(b*y-x) + 2.0*a*sin(a*x+y)*cos(b*y-x) - cos(a*x+y)*sin(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double dy(double x, double y, double z) const {
    return (-sin(a*x+y)*sin(b*y-x) + b*cos(a*x+y)*cos(b*y-x))*atan(c*z);
#else
  double dy(double x, double y) const {
    return (-sin(a*x+y)*sin(b*y-x) + b*cos(a*x+y)*cos(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double ddyy(double x, double y, double z) const {
    return (-cos(a*x+y)*sin(b*y-x) - 2.0*b*sin(a*x+y)*cos(b*y-x) - b*b*cos(a*x+y)*sin(b*y-x))*atan(c*z);
#else
  double ddyy(double x, double y) const {
    return (-cos(a*x+y)*sin(b*y-x) - 2.0*b*sin(a*x+y)*cos(b*y-x) - b*b*cos(a*x+y)*sin(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double dz(double x, double y, double z) const {
    return  c*cos(a*x+y)*sin(b*y-x)/(1+SQR(c*z));
  }

  double ddzz(double x, double y, double z) const {
    return  -2.0*c*c*c*z*cos(a*x+y)*sin(b*y-x)/(SQR(1+SQR(c*z)));
  }
#endif
};

#ifdef P4_TO_P8
class wex : public test_function {
private:
  double a, b, c;
public:
  wex(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
  double operator()(double x, double y, double z) const {
    return log(a*x*x+1)*atan(y)*sin(c*z+b*y);
  }

  double dx(double x, double y, double z) const {
    return atan(y)*sin(c*z+b*y)*2.0*a*x/(1.0+a*x*x);
  }

  double ddxx(double x, double y, double z) const {
    return atan(y)*sin(c*z+b*y)*(2.0*a*(1-a*x*x))/SQR(1.0+a*x*x);
  }

  double dy(double x, double y, double z) const {
    return log(a*x*x+1)*(sin(c*z+b*y)/(1 + y*y) + atan(y)*b*cos(c*z+b*y));
  }
  double ddyy(double x, double y, double z) const {
    return log(a*x*x+1)*(-2.0*y*sin(c*z+b*y)/(SQR(1 + y*y)) + 2.0*b*cos(c*z+b*y)/(1 + y*y) - atan(y)*b*b*sin(c*z+b*y));
  }

  double dz(double x, double y, double z) const {
    return log(a*x*x+1)*atan(y)*c*cos(c*z+b*y);
  }

  double ddzz(double x, double y, double z) const {
    return -log(a*x*x+1)*atan(y)*c*c*sin(c*z+b*y);
  }
};
#endif

static void sample_test_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const test_function* cf_array[], Vec f)
{
  double *f_p;
  PetscInt bs;
  PetscErrorCode ierr;
  ierr = VecGetBlockSize(f, &bs); CHKERRXX(ierr);
  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i) {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(i, p4est, nodes, xyz);

    for (PetscInt j = 0; j<bs; j++) {
      const test_function &cf = *cf_array[j];
#ifdef P4_TO_P8
      f_p[i*bs + j] = cf(xyz[0], xyz[1], xyz[2]);
#else
      f_p[i*bs + j] = cf(xyz[0], xyz[1]);
#endif
    }
  }
  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}

static double random_generator(const double &min=0.0, const double &max=1.0)
{
  return (min+(max-min)*((double) rand())/((double) RAND_MAX));
}

int main (int argc, char* argv[]){

  try{
    mpi_environment_t mpi;
    mpi.init(argc, argv);

    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("seed", "seed for random number generator (default is 163)");
    cmd.add_option("ntree", "number of trees per dimensions (default is 2)");
    cmd.add_option("lmin", "min level of the original tree (default is 4)");
    cmd.add_option("lmax", "max level of the original tree (default is 6)");
    cmd.add_option("method", "0==original capability of the code, calculating derivatives component by component one after each other; 1==considering component by component as well but calculating all derivatives all at once; 2==storing data by block, calculating all at once as well. (default is 0).");
    cmd.add_option("timing_off", "disables timing for calculation of derivatives if present");
    cmd.add_option("n_fields", "number of fields to calculate first and second derivatives of (default is number of dimensions P4EST_DIM)");
    cmd.parse(argc, argv);

    const unsigned int method = cmd.get<unsigned int>("method", 0);
    unsigned int seed = cmd.get<unsigned int>("seed", 163);
    bool timing_off = cmd.contains("timing_off");
    int lmin_original = cmd.get<int>("lmin", 4);
    int lmax_original = cmd.get<int>("lmax", 6);
    int ntree             = cmd.get<int>("ntree", 2);
    unsigned int n_fields = cmd.get<unsigned int>("n_fields", P4EST_DIM);
    srand(seed);

    // sphere of random center in [r, 2-r]^P4EST_DIM or random radius in [r/2, r]
    const double r = 0.3;
#ifdef P4_TO_P8
    circle circ(random_generator(r, 2.0-r), random_generator(r, 2.0-r), random_generator(r, 2.0-r), random_generator(r/2, r));
#else
    circle circ(random_generator(r, 2.0-r), random_generator(r, 2.0-r), random_generator(r/2, r));
#endif
    splitting_criteria_cf_t cf_data(lmin_original, lmax_original, &circ, 1);

    parStopWatch w_total, w_sub;

    const test_function *cf_field[n_fields];
    for (unsigned int k = 0; k < n_fields; ++k) {
#ifdef P4_TO_P8
      if(k%P4EST_DIM==0)
        cf_field[k] = new uex(random_generator(0.0, 1.0), random_generator(0.0, 1.0), random_generator(0.0, 1.0));
      else if (k%P4EST_DIM==1)
        cf_field[k] = new vex(random_generator(0.0, 2.0), random_generator(0.0, 0.7), random_generator(0.0, 0.5));
      else
        cf_field[k] = new wex(random_generator(0.0, 1.0), random_generator(-1.0, 1.0), random_generator(-1.0, 1.0));
#else
      if(k%P4EST_DIM==0)
        cf_field[k] = new uex(random_generator(0.0, 1.0), random_generator(0.0, 1.0));
      else
        cf_field[k] = new vex(random_generator(0.0, 2.0), random_generator(0.0, 0.7));
#endif
    }

    // Create the connectivity object
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t my_brick, *brick = &my_brick;
    int n_xyz [] = {ntree, ntree, ntree};
    double xyz_min [] = {0, 0, 0};
    double xyz_max [] = {2, 2, 2};
    int periodic []   = {0, 0, 0};
    connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, brick, periodic);

    // Now create the forest
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    // Now refine the tree
    p4est->user_pointer = (void*)(&cf_data);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    // Finally re-partition
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    // Create the ghost structure
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    // generate the node data structure
    nodes = my_p4est_nodes_new(p4est, ghost);

    // compute the node-sampled vector field
    Vec field_comp[n_fields], *grad_field_comp[n_fields], *second_derivatives_field_comp[n_fields];
    for (unsigned int comp = 0; comp < n_fields; ++comp){
      grad_field_comp[comp]               = P4EST_ALLOC(Vec, P4EST_DIM);
      second_derivatives_field_comp[comp] = P4EST_ALLOC(Vec, P4EST_DIM);
    }
    Vec field_block, grad_field_block, second_derivatives_field_block;
    if(method==2)
    {
      ierr = VecCreateGhostNodesBlock(p4est, nodes, n_fields, &field_block); CHKERRXX(ierr);
      ierr = VecCreateGhostNodesBlock(p4est, nodes, n_fields*P4EST_DIM, &grad_field_block); CHKERRXX(ierr);
      ierr = VecCreateGhostNodesBlock(p4est, nodes, n_fields*P4EST_DIM, &second_derivatives_field_block); CHKERRXX(ierr);
      for (unsigned int comp = 0; comp < n_fields; ++comp) {
        field_comp[comp] = NULL;
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          grad_field_comp[comp][der] = NULL;
          second_derivatives_field_comp[comp][der] = NULL;
        }
      }
      sample_test_cf_on_nodes(p4est, nodes, cf_field, field_block);
    }
    else
    {
      field_block = NULL;
      grad_field_block = NULL;
      second_derivatives_field_block = NULL;
      for (unsigned int comp = 0; comp < n_fields; ++comp) {
        ierr = VecCreateGhostNodes(p4est, nodes, &field_comp[comp]); CHKERRXX(ierr);
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          ierr = VecCreateGhostNodes(p4est, nodes, &grad_field_comp[comp][der]); CHKERRXX(ierr);
          ierr = VecCreateGhostNodes(p4est, nodes, &second_derivatives_field_comp[comp][der]); CHKERRXX(ierr);
        }
      }
      for (unsigned int k = 0; k < n_fields; ++k)
        sample_cf_on_nodes(p4est, nodes, *cf_field[k], field_comp[k]);
    }

    // set up the qnnn neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy, nodes);

    if(!timing_off){
      w_total.start("total time spent (including initialization of neighbors)");
      w_sub.start("initializing the node neighbors");
    }
    ngbd_n.init_neighbors();
    if(!timing_off){
      w_sub.stop(); w_sub.print_duration();
      w_sub.start("calculating gradient and second derivatives");
    }

    if(method==2)
    {
      ngbd_n.first_derivatives_central(field_block, grad_field_block, n_fields);
      ngbd_n.second_derivatives_central(field_block, second_derivatives_field_block, n_fields);
    }
    else
    {
      if(method==0)
      {
        for (unsigned int comp = 0; comp < n_fields; ++comp) {
          ngbd_n.first_derivatives_central(field_comp[comp], grad_field_comp[comp]);
          ngbd_n.second_derivatives_central(field_comp[comp], second_derivatives_field_comp[comp]);
        }
      }
      else
      {
        // need to "transpose" it for consistency with original version
        Vec *grad_field_comp_remap[P4EST_DIM];
        Vec *second_derivatives_field_comp_remap[P4EST_DIM];
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          grad_field_comp_remap[der] = P4EST_ALLOC(Vec, n_fields);
          second_derivatives_field_comp_remap[der] = P4EST_ALLOC(Vec, n_fields);
        }
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          for (unsigned int k = 0; k < n_fields; ++k)
          {
            grad_field_comp_remap[der][k] = grad_field_comp[k][der];
            second_derivatives_field_comp_remap[der][k] = second_derivatives_field_comp[k][der];
          }
        ngbd_n.first_derivatives_central(field_comp, grad_field_comp_remap, n_fields);
        ngbd_n.second_derivatives_central(field_comp, second_derivatives_field_comp_remap, n_fields);
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          P4EST_FREE(grad_field_comp_remap[der]);
          P4EST_FREE(second_derivatives_field_comp_remap[der]);
        }
      }
    }

    if(!timing_off){
      w_sub.stop(); w_sub.print_duration();
      w_total.stop(); w_total.print_duration();
    }

    const double *field_comp_p[n_fields], *grad_field_comp_p[n_fields][P4EST_DIM], *second_derivatives_field_comp_p[n_fields][P4EST_DIM];
    const double *field_block_p, *grad_field_block_p, *second_derivatives_field_block_p;
    if(method==2){
      ierr = VecGetArrayRead(field_block, &field_block_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(grad_field_block, &grad_field_block_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(second_derivatives_field_block, &second_derivatives_field_block_p); CHKERRXX(ierr);
    }
    else
    {
      for (unsigned char comp = 0; comp < n_fields; ++comp) {
        ierr = VecGetArrayRead(field_comp[comp], &field_comp_p[comp]); CHKERRXX(ierr);
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          ierr = VecGetArrayRead(grad_field_comp[comp][der], &grad_field_comp_p[comp][der]); CHKERRXX(ierr);
          ierr = VecGetArrayRead(second_derivatives_field_comp[comp][der], &second_derivatives_field_comp_p[comp][der]); CHKERRXX(ierr);
        }
      }
    }

    if(mpi.rank() == 0)
      std::cout << std::endl << "Errors in infinity norm disregarding wall nodes: " << std::endl << std::endl;
    double err_gradient[n_fields][P4EST_DIM];
    double err_second_derivatives[n_fields][P4EST_DIM];
    for (unsigned int comp = 0; comp < n_fields; ++comp)
      for (unsigned char der = 0; der < P4EST_DIM; ++der) {
        err_gradient[comp][der] = 0.0;
        err_second_derivatives[comp][der] = 0.0;
      }

    for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i)
    {
      p4est_indep_t* node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes, i);
      if(is_node_Wall(p4est, node))
        continue;
      double xyz [P4EST_DIM];
      node_xyz_fr_n(i, p4est, nodes, xyz);
      if(method!=2)
      {
        for (unsigned int k = 0; k < n_fields; ++k) {
          err_gradient[k][0] = MAX(err_gradient[k][0], fabs(grad_field_comp_p[k][0][i]-cf_field[k]->dx(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
          err_gradient[k][1] = MAX(err_gradient[k][1], fabs(grad_field_comp_p[k][1][i]-cf_field[k]->dy(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
#ifdef P4_TO_P8
          err_gradient[k][2] = MAX(err_gradient[k][2], fabs(grad_field_comp_p[k][2][i]-cf_field[k]->dz(xyz[0], xyz[1], xyz[2])));
#endif
          err_second_derivatives[k][0] = MAX(err_second_derivatives[k][0], fabs(second_derivatives_field_comp_p[k][0][i]-cf_field[k]->ddxx(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
          err_second_derivatives[k][1] = MAX(err_second_derivatives[k][1], fabs(second_derivatives_field_comp_p[k][1][i]-cf_field[k]->ddyy(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
#ifdef P4_TO_P8
          err_second_derivatives[k][2] = MAX(err_second_derivatives[k][2], fabs(second_derivatives_field_comp_p[k][2][i]-cf_field[k]->ddzz(xyz[0], xyz[1], xyz[2])));
#endif
        }
      }
      else
      {
        for (unsigned int k = 0; k < n_fields; ++k) {
          err_gradient[k][0] = MAX(err_gradient[k][0], fabs(grad_field_block_p[i*n_fields*P4EST_DIM+k*P4EST_DIM+0]-cf_field[k]->dx(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
          err_gradient[k][1] = MAX(err_gradient[k][1], fabs(grad_field_block_p[i*n_fields*P4EST_DIM+k*P4EST_DIM+1]-cf_field[k]->dy(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
#ifdef P4_TO_P8
          err_gradient[k][2] = MAX(err_gradient[k][2], fabs(grad_field_block_p[i*n_fields*P4EST_DIM+k*P4EST_DIM+2]-cf_field[k]->dz(xyz[0], xyz[1], xyz[2])));
#endif
          err_second_derivatives[k][0] = MAX(err_second_derivatives[k][0], fabs(second_derivatives_field_block_p[i*n_fields*P4EST_DIM+k*P4EST_DIM+0]-cf_field[k]->ddxx(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
          err_second_derivatives[k][1] = MAX(err_second_derivatives[k][1], fabs(second_derivatives_field_block_p[i*n_fields*P4EST_DIM+k*P4EST_DIM+1]-cf_field[k]->ddyy(xyz[0], xyz[1]
    #ifdef P4_TO_P8
              , xyz[2]
    #endif
              )));
#ifdef P4_TO_P8
          err_second_derivatives[k][2] = MAX(err_second_derivatives[k][2], fabs(second_derivatives_field_block_p[i*n_fields*P4EST_DIM+k*P4EST_DIM+2]-cf_field[k]->ddzz(xyz[0], xyz[1], xyz[2])));
#endif
        }
      }
    }
    int mpiret  = MPI_Allreduce(MPI_IN_PLACE, err_gradient,           n_fields*P4EST_DIM,  MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
    mpiret      = MPI_Allreduce(MPI_IN_PLACE, err_second_derivatives, n_fields*P4EST_DIM,  MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
    if(mpi.rank() == 0)
    {
      std::cout << "The errors in gradient are: " << std::endl;
      for (unsigned int comp = 0; comp < n_fields; ++comp) {
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          std::cout << err_gradient[comp][der] << "  ";
        std::cout << std::endl;
      }
      std::cout << std::endl;
      std::cout << "The errors in second derivatives are: " << std::endl;
      for (unsigned int comp = 0; comp < n_fields; ++comp) {
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          std::cout << err_second_derivatives[comp][der] << "  ";
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    if(method==2){
      ierr = VecRestoreArrayRead(field_block, &field_block_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(grad_field_block, &grad_field_block_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(second_derivatives_field_block, &second_derivatives_field_block_p); CHKERRXX(ierr);
      ierr = VecDestroy(field_block); CHKERRXX(ierr);
      ierr = VecDestroy(grad_field_block); CHKERRXX(ierr);
      ierr = VecDestroy(second_derivatives_field_block); CHKERRXX(ierr);
    }
    else
    {
      for (unsigned int comp = 0; comp < n_fields; ++comp) {
        ierr = VecRestoreArrayRead(field_comp[comp], &field_comp_p[comp]); CHKERRXX(ierr);
        ierr = VecDestroy(field_comp[comp]); CHKERRXX(ierr);
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          ierr = VecRestoreArrayRead(grad_field_comp[comp][der], &grad_field_comp_p[comp][der]); CHKERRXX(ierr);
          ierr = VecRestoreArrayRead(second_derivatives_field_comp[comp][der], &second_derivatives_field_comp_p[comp][der]); CHKERRXX(ierr);
          ierr = VecDestroy(grad_field_comp[comp][der]); CHKERRXX(ierr);
          ierr = VecDestroy(second_derivatives_field_comp[comp][der]); CHKERRXX(ierr);
        }
      }
    }
    for (unsigned int comp = 0; comp < n_fields; ++comp){
      P4EST_FREE(grad_field_comp[comp]);
      P4EST_FREE(second_derivatives_field_comp[comp]);
    }

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);

    my_p4est_brick_destroy(connectivity, brick);
    for (unsigned int k = 0; k < n_fields; ++k)
      delete cf_field[k];
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

