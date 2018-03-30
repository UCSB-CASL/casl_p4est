#ifndef MLS_INTEGRATION_TYPES
#define MLS_INTEGRATION_TYPES
enum loc_t {INS, OUT, FCE, LNE, PNT};
enum action_t {INTERSECTION, ADDITION, COLORATION};
#endif

#ifndef simplex2_mls_q_ref_H
#define simplex2_mls_q_ref_H

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cfloat>
#include <climits>
#include "simplex2_mls_q.h"

#define simplex2_mls_q_ref_DEBUG

class simplex2_mls_q_ref_t
{
  friend class simplex2_mls_q_ref_vtk;
  friend class cube2_mls_q_t;

public:

  //--------------------------------------------------
  // Class Constructors
  //--------------------------------------------------
  simplex2_mls_q_ref_t(double x0, double y0,
                       double x1, double y1,
                       double x2, double y2,
                       double x3, double y3,
                       double x4, double y4,
                       double x5, double y5);

  //--------------------------------------------------
  // Domain Reconstruction
  //--------------------------------------------------
  void construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr);

  //--------------------------------------------------
  // Quadrature Points
  //--------------------------------------------------
  void quadrature_over_domain       (                    std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_interface    (int num,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_intersection (int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_in_dir            (int dir,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);

private:

  // some geometric info
  const static int nodes_per_tri_ = 6;

  std::vector<simplex2_mls_q_t> simplices_;

  //--------------------------------------------------
  // Interpolation
  //--------------------------------------------------
  double interpolate_from_parent(std::vector<double> &f, double x, double y);
  double interpolate_from_parent(double x, double y);
};

#endif // simplex2_mls_q_ref_H
