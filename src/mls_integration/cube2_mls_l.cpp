#include "cube2_mls_l.h"

void cube2_mls_l_t::construct_domain(const std::vector<double> &phi_all, const std::vector<action_t> &acn, const std::vector<int> &clr)
{
  bool all_positive, all_negative;

  num_of_lsfs = acn.size();

#ifdef CASL_THROWS
  if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
          throw std::domain_error("[CASL_ERROR]: (cube2_mls_l_t::construct_domain) sizes of phi, acn and clr do not coincide.");
#endif

  /* Eliminate unnecessary splitting */
  loc = INS;
  for (unsigned int p = 0; p < num_of_lsfs; p++)
  {
    all_negative = true;
    all_positive = true;

    for (unsigned short j = 0; j < n_nodes_dir; ++j)
      for (unsigned short i = 0; i < n_nodes_dir; ++i)
      {
        double value = phi_all[n_nodes*p + n_nodes_dir*j + i];
        all_negative = (all_negative && (value < 0.));
        all_positive = (all_positive && (value > 0.));
      }

    if (all_positive)
    {
      if (acn[p] == CUBE_MLS_INTERSECTION) loc = OUT;
    }
    else if (all_negative)
    {
      if (acn[p] == CUBE_MLS_ADDITION) loc = INS;
    }
    else if ((loc == INS && acn[p] == CUBE_MLS_INTERSECTION) || (loc == OUT && acn[p] == CUBE_MLS_ADDITION))
    {
      loc = FCE;
    }
  }

//  num_non_trivial = non_trivial.size();

  if (loc == FCE)
  {
//    if (non_trivial_action[0] == CUBE_MLS_ADDITION) // the first action always has to be CUBE_MLS_INTERSECTION
//      non_trivial_action[0] = CUBE_MLS_INTERSECTION;

    /* Split the cube into 2 simplices */
    double x[4] = {x0, x1, x0, x1}; double y[4] = {y0, y0, y1, y1};

    simplex.clear();
    simplex.reserve(2);
    simplex.push_back(simplex2_mls_l_t(x[l2_t0p0], y[l2_t0p0], x[l2_t0p1], y[l2_t0p1], x[l2_t0p2], y[l2_t0p2])); // simplex.back().set_use_linear(use_linear);
    simplex.push_back(simplex2_mls_l_t(x[l2_t1p0], y[l2_t1p0], x[l2_t1p1], y[l2_t1p1], x[l2_t1p2], y[l2_t1p2])); // simplex.back().set_use_linear(use_linear);

    // TODO: mark appropriate edges for integrate_in_dir
//    simplex[0].edgs_[0].dir = 1; simplex[0].edgs_[2].dir = 2;
//    simplex[1].edgs_[0].dir = 3; simplex[1].edgs_[2].dir = 0;


    std::vector<double> phi_s0(n_nodes_simplex*num_of_lsfs, -1);
    std::vector<double> phi_s1(n_nodes_simplex*num_of_lsfs, -1);

    for (unsigned int i = 0; i < num_of_lsfs; ++i)
    {
//      int phi_idx = nt_idx[i];
      int phi_idx = i;
      phi_s0[n_nodes_simplex*i+0] = phi_all[n_nodes*phi_idx+l2_t0p0]; phi_s1[n_nodes_simplex*i+0] = phi_all[n_nodes*phi_idx+l2_t1p0];
      phi_s0[n_nodes_simplex*i+1] = phi_all[n_nodes*phi_idx+l2_t0p1]; phi_s1[n_nodes_simplex*i+1] = phi_all[n_nodes*phi_idx+l2_t1p1];
      phi_s0[n_nodes_simplex*i+2] = phi_all[n_nodes*phi_idx+l2_t0p2]; phi_s1[n_nodes_simplex*i+2] = phi_all[n_nodes*phi_idx+l2_t1p2];
    }

    simplex[0].construct_domain(phi_s0, acn, clr);
    simplex[1].construct_domain(phi_s1, acn, clr);
  }

}


/* Quadrature points */

void cube2_mls_l_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
//  weights.clear();
//  X.clear();
//  Y.clear();

  switch (loc){
    case INS:
      {
        static double alpha = 1./sqrt(3.);

        double w = (x1-x0)*(y1-y0)/4.0;

        double xm = x0+.5*(-alpha+1.)*(x1-x0);
        double xp = x0+.5*( alpha+1.)*(x1-x0);

        double ym = y0+.5*(-alpha+1.)*(y1-y0);
        double yp = y0+.5*( alpha+1.)*(y1-y0);

        X.push_back( xm ); Y.push_back( ym );  weights.push_back(w);
        X.push_back( xp ); Y.push_back( ym );  weights.push_back(w);
        X.push_back( xm ); Y.push_back( yp );  weights.push_back(w);
        X.push_back( xp ); Y.push_back( yp );  weights.push_back(w);
      } break;
    case OUT: break;
    case FCE:
      {
        simplex[0].quadrature_over_domain(weights, X, Y);
        simplex[1].quadrature_over_domain(weights, X, Y);
      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#else
      break;
#endif
  }
}

void cube2_mls_l_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
//  weights.clear();
//  X.clear();
//  Y.clear();

  if (loc == FCE)
  {
    simplex[0].quadrature_over_interface(num, weights, X, Y);
    simplex[1].quadrature_over_interface(num, weights, X, Y);
  }
}

void cube2_mls_l_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
//  weights.clear();
//  X.clear();
//  Y.clear();

  if (loc == FCE && num_of_lsfs > 1)
  {
    simplex[0].quadrature_over_intersection(num0, num1, weights, X, Y);
    simplex[1].quadrature_over_intersection(num0, num1, weights, X, Y);
  }
}

void cube2_mls_l_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
//  weights.clear();
//  X.clear();
//  Y.clear();

  switch (loc){
    case OUT: break;
    case INS:
      {
        static double alpha = 1./sqrt(3.);

        double xm = x0+.5*(-alpha+1.)*(x1-x0);
        double xp = x0+.5*( alpha+1.)*(x1-x0);

        double ym = y0+.5*(-alpha+1.)*(y1-y0);
        double yp = y0+.5*( alpha+1.)*(y1-y0);

        double w;

        switch (dir) {
          case 0:
            w = .5*(y1-y0);
            X.push_back(x0); Y.push_back(ym); weights.push_back(w);
            X.push_back(x0); Y.push_back(yp); weights.push_back(w);
            break;
          case 1:
            w = .5*(y1-y0);
            X.push_back(x1); Y.push_back(ym); weights.push_back(w);
            X.push_back(x1); Y.push_back(yp); weights.push_back(w);
            break;
          case 2:
            w = .5*(x1-x0);
            X.push_back(xm); Y.push_back(y0); weights.push_back(w);
            X.push_back(xp); Y.push_back(y0); weights.push_back(w);
            break;
          case 3:
            w = .5*(x1-x0);
            X.push_back(xm); Y.push_back(y1); weights.push_back(w);
            X.push_back(xp); Y.push_back(y1); weights.push_back(w);
            break;
        }
      } break;
    case FCE:
      {
      /* integration in dir
       * 0: s1->e2
       * 1: s0->e0
       * 2: s0->e2
       * 3: s1->e0
       */

      switch (dir)
      {
        case 0: simplex[1].quadrature_in_dir(2, weights, X, Y); break;
        case 1: simplex[0].quadrature_in_dir(0, weights, X, Y); break;
        case 2: simplex[0].quadrature_in_dir(2, weights, X, Y); break;
        case 3: simplex[1].quadrature_in_dir(0, weights, X, Y); break;
      }

//        simplex[0].quadrature_in_dir(dir, weights, X, Y);
//        simplex[1].quadrature_in_dir(dir, weights, X, Y);
      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#else
      break;
#endif
  }
}
