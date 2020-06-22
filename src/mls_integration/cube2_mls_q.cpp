#include "cube2_mls_q.h"

void cube2_mls_q_t::construct_domain(std::vector<double> &phi_all, const std::vector<action_t> &acn, const std::vector<int> &clr)
{
  bool all_positive, all_negative;

  num_of_lsfs = acn.size();

  // 3-by-3 grid
  double xc = .5*(x0+x1);
  double yc = .5*(y0+y1);

#ifdef CASL_THROWS
  if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
          throw std::domain_error("[CASL_ERROR]: (cube2_mls_q_t::construct_domain) sizes of phi, acn and clr do not coincide.");
#endif

  /* Eliminate unnecessary splitting */

  loc = INS;

  double value;
  double p0, p1, p2;
  double c0, c1, c2;
  int I, J;

  std::vector<action_t> nt_acn;
  std::vector<int>      nt_clr;
  std::vector<int>      nt_idx;

  for (unsigned int p = 0; p < num_of_lsfs; p++)
  {
    // sample level-set function and check for intersections
    all_negative = true;
    all_positive = true;

    double phi_max = 0;
    for (unsigned short j = 0; j < n_nodes_dir; ++j)
      for (unsigned short i = 0; i < n_nodes_dir; ++i)
      {
        value = phi_all[n_nodes*p + n_nodes_dir*j + i];
        all_negative = (all_negative && (value < 0.));
        all_positive = (all_positive && (value > 0.));
        phi_max = phi_max > fabs(value) ? phi_max : fabs(value);
      }

    double phi_eps = phi_max*eps_rel_;

    for (unsigned short j = 0; j < n_nodes_dir; ++j)
      for (unsigned short i = 0; i < n_nodes_dir; ++i)
        perturb(phi_all[n_nodes*p + n_nodes_dir*j + i], phi_eps);

    // check edges for complex intersections
    if (all_negative || all_positive)
    {
      int dim;
      for (dim = 0; dim < 2*2; ++dim)
      {
        switch (dim)  {
          case 0: // -x
            I = 0; J = 0; p0 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 0; J = 1; p1 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 0; J = 2; p2 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            break;
          case 1: // +x
            I = 2; J = 0; p0 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 2; J = 1; p1 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 2; J = 2; p2 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            break;
          case 2: // -y
            I = 0; J = 0; p0 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 1; J = 0; p1 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 2; J = 0; p2 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            break;
          case 3: // +y
            I = 0; J = 2; p0 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 1; J = 2; p1 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            I = 2; J = 2; p2 = phi_all[n_nodes*p + n_nodes_dir*J + I];
            break;
        }

        c0 = p0;
        c1 = -3.*p0 + 4.*p1 -    p2;
        c2 =  2.*p0 - 4.*p1 + 2.*p2;

//        if (fabs(c2) > 1.e-15)
        if (c1*(2.*c2+c1) < 0.)
        {
          double a_ext = -.5*c1/c2;
          if (a_ext > 0. && a_ext < 1.)
          {
            double phi_ext = c0 + c1*a_ext + c2*a_ext*a_ext;

            if (p0*phi_ext < 0)
            {
              all_negative = false;
              all_positive = false;
              break;
            }
          }
        }

      }
    }


    if (all_positive)
    {
      if (acn[p] == CUBE_MLS_INTERSECTION)
      {
        loc = OUT;
//        nt_phi.clear();
        nt_acn.clear();
        nt_clr.clear();
        nt_idx.clear();
      }
    }
    else if (all_negative)
    {
      if (acn[p] == CUBE_MLS_ADDITION)
      {
        loc = INS;
//        nt_phi.clear();
        nt_acn.clear();
        nt_clr.clear();
        nt_idx.clear();
      }
    }
    else if (loc == FCE || (loc == INS && acn[p] == CUBE_MLS_INTERSECTION) || (loc == OUT && acn[p] == CUBE_MLS_ADDITION))
    {
      loc = FCE;
//      nt_phi.push_back(phi[p]);
      nt_acn.push_back(acn[p]);
      nt_clr.push_back(clr[p]);
      nt_idx.push_back(p);
    }
  }

  /* uncomment this to use all level-set functions */
//  nt_phi = phi;
//  nt_acn = acn;
//  nt_clr = clr;
//  loc = FCE;

  if (loc == FCE)
  {
    // the first action always has to be CUBE_MLS_INTERSECTION
    if (nt_acn[0] == CUBE_MLS_ADDITION) nt_acn[0] = CUBE_MLS_INTERSECTION;

    /* Split the cube into 2 simplices */
    double x[] = { x0, xc, x1, x0, xc, x1, x0, xc, x1 };
    double y[] = { y0, y0, y0, yc, yc, yc, y1, y1, y1 };

    simplex.clear();
    simplex.reserve(2);

    simplex.push_back(simplex2_mls_q_t(x[q2_t0p0], y[q2_t0p0],
                                       x[q2_t0p1], y[q2_t0p1],
                                       x[q2_t0p2], y[q2_t0p2],
                                       x[q2_t0p3], y[q2_t0p3],
                                       x[q2_t0p4], y[q2_t0p4],
                                       x[q2_t0p5], y[q2_t0p5]));

    simplex.push_back(simplex2_mls_q_t(x[q2_t1p0], y[q2_t1p0],
                                       x[q2_t1p1], y[q2_t1p1],
                                       x[q2_t1p2], y[q2_t1p2],
                                       x[q2_t1p3], y[q2_t1p3],
                                       x[q2_t1p4], y[q2_t1p4],
                                       x[q2_t1p5], y[q2_t1p5]));

    // mark appropriate edges for integrate_in_dir
//    simplex[0].edgs_[0].dir = 1; simplex[0].edgs_[2].dir = 2;
//    simplex[1].edgs_[0].dir = 3; simplex[1].edgs_[2].dir = 0;

    num_of_lsfs = nt_acn.size();

    std::vector<double> phi_s0(n_nodes_simplex*num_of_lsfs, -1);
    std::vector<double> phi_s1(n_nodes_simplex*num_of_lsfs, -1);

    for (unsigned int i = 0; i < num_of_lsfs; ++i)
    {
      int phi_idx = nt_idx[i];
      phi_s0[n_nodes_simplex*i+0] = phi_all[n_nodes*phi_idx+q2_t0p0]; phi_s1[n_nodes_simplex*i+0] = phi_all[n_nodes*phi_idx+q2_t1p0];
      phi_s0[n_nodes_simplex*i+1] = phi_all[n_nodes*phi_idx+q2_t0p1]; phi_s1[n_nodes_simplex*i+1] = phi_all[n_nodes*phi_idx+q2_t1p1];
      phi_s0[n_nodes_simplex*i+2] = phi_all[n_nodes*phi_idx+q2_t0p2]; phi_s1[n_nodes_simplex*i+2] = phi_all[n_nodes*phi_idx+q2_t1p2];
      phi_s0[n_nodes_simplex*i+3] = phi_all[n_nodes*phi_idx+q2_t0p3]; phi_s1[n_nodes_simplex*i+3] = phi_all[n_nodes*phi_idx+q2_t1p3];
      phi_s0[n_nodes_simplex*i+4] = phi_all[n_nodes*phi_idx+q2_t0p4]; phi_s1[n_nodes_simplex*i+4] = phi_all[n_nodes*phi_idx+q2_t1p4];
      phi_s0[n_nodes_simplex*i+5] = phi_all[n_nodes*phi_idx+q2_t0p5]; phi_s1[n_nodes_simplex*i+5] = phi_all[n_nodes*phi_idx+q2_t1p5];
    }
    simplex[0].construct_domain(phi_s0, nt_acn, nt_clr);
    simplex[1].construct_domain(phi_s1, nt_acn, nt_clr);
  }


}


/* Quadrature points */

void cube2_mls_q_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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

void cube2_mls_q_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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

void cube2_mls_q_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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

void cube2_mls_q_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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
      }
      break;
    case FCE:
      switch (dir)
      {
        case 0: simplex[1].quadrature_in_dir(2, weights, X, Y); break;
        case 1: simplex[0].quadrature_in_dir(0, weights, X, Y); break;
        case 2: simplex[0].quadrature_in_dir(2, weights, X, Y); break;
        case 3: simplex[1].quadrature_in_dir(0, weights, X, Y); break;
      }
      break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#else
      break;
#endif
  }
}
