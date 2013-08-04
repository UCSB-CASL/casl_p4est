#ifndef PARALLEL_SEMI_LAGRANGIAN_H
#define PARALLEL_SEMI_LAGRANGIAN_H

#include <vector>
#include <algorithm>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <iostream>

namespace parallel{

struct quad_information{       //Used to hold info of quadrants for parallelization
    std::vector<double>              xy;//XY Departure points [0] = X, [1] = Y
    std::vector<p4est_topidx_t>      tree_idx;    //Index of Tree
    std::vector<p4est_locidx_t>      quad_idx;    //Index of Quadrant
    std::vector<p4est_quadrant_t*>   quad;        //Quadrant pointers
    std::vector<p4est_locidx_t>      ni;          //Node index

    void push (double *xy_dep, p4est_topidx_t t_idx, p4est_locidx_t q_idx, p4est_quadrant_t *q, p4est_locidx_t ni_){
        xy.push_back(xy_dep[0]);  //Pushback x_dep points
        xy.push_back(xy_dep[1]);  //Pushback y_dep points
        tree_idx.push_back(t_idx);
        quad_idx.push_back(q_idx);
        quad.push_back(q);
        ni.push_back(ni_);
    }

    void print (){
        std::cout << "Printing xy_dep vector" << endl;
        for (int it = 0; it < xy.size(); ++it){
            std::cout << "[" << xy[it] << "] ";
        }
        std::cout << "\nPrinting tree_idx vector" << endl;
        for (int it = 0; it < tree_idx.size(); ++it){
            std::cout << "[" << tree_idx[it] << "] ";
        }
        std::cout << "\nPrinting quad_idx vector" << endl;
        for (int it = 0; it < quad_idx.size(); ++it){
            std::cout << "[" << quad_idx[it] << "] ";
        }
        std::cout << "\nPrinting quad vector" << endl;
        for (int it = 0; it < quad.size(); ++it){
            std::cout << "[" << quad[it] << "] ";
        }
        std::cout << "\nPrinting node index vector" << endl;
        for (int it = 0; it < ni.size(); ++it){
            std::cout << "[" << ni[it] << "] ";
        }
    }
};

struct non_local_point_buffer{      //Sending buffer used to hold xy_departure and node index.
    std::vector<double>     xy;     //XY Departure points [0] = X, [1] = Y
    std::vector<p4est_locidx_t> ni; //Node Index

    void push (double *xy_,p4est_locidx_t ni_){
        xy.push_back(xy_[0]);
        xy.push_back(xy_[1]);
        ni.push_back(ni_);
    }
};

class SemiLagrangian
{
  p4est_t **p_p4est_, *p4est_;
  p4est_nodes_t **p_nodes_, *nodes_;
  my_p4est_brick_t *myb_;

  double xmin, xmax, ymin, ymax;

  std::vector<double> local_xy_departure_dep, non_local_xy_departure_dep;   //Buffers to hold local and non-local departure points
  PetscErrorCode ierr;

  inline double compute_dt(const CF_2& vx, const CF_2& vy){
    double dt = 1000;

    // loop over trees
    for (p4est_topidx_t tr_it = p4est_->first_local_tree; tr_it <=p4est_->last_local_tree; ++tr_it){
      p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tr_it);
      p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
      double *v2c = p4est_->connectivity->vertices;

      double tr_xmin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 0];
      double tr_ymin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 1];

      // loop over quadrants
      for (size_t qu_it=0; qu_it<tree->quadrants.elem_count; ++qu_it){
        p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

        double dx = int2double_coordinate_transform(P4EST_QUADRANT_LEN(quad->level));
        double x  = int2double_coordinate_transform(quad->x) + 0.5*dx + tr_xmin;
        double y  = int2double_coordinate_transform(quad->y) + 0.5*dx + tr_ymin;
        double vn = SQRT(SQR(vx(x,y)) + SQR(vy(x,y)));
        dt = MIN(dt, dx/vn);
      }
    }

    double dt_min;
    MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est_->mpicomm);

    return dt_min;
  }

  double linear_interpolation(const double *F, const double xy[], p4est_topidx_t tree_idx = 0);
  void update_p4est(Vec& phi, p4est_ghost_t *ghost);

public:
  SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, my_p4est_brick_t *myb);

  double advect(const CF_2& vx, const CF_2& vy, Vec &phi);
};
} // namespace parallel


#endif // PARALLEL_SEMI_LAGRANGIAN_H
