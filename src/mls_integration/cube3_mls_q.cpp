#include "cube3_mls_q.h"

void cube3_mls_q_t::construct_domain(std::vector<double> &phi_all, std::vector<action_t> &acn, std::vector<int> &clr)
{
  num_of_lsfs = acn.size();

  // 3-by-3-by-3 grid
  double xc = .5*(x0+x1);
  double yc = .5*(y0+y1);
  double zc = .5*(z0+z1);

  double x[] = { x0, xc, x1 };
  double y[] = { y0, yc, y1 };
  double z[] = { z0, zc, z1 };

#ifdef CASL_THROWS
  if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
          throw std::domain_error("[CASL_ERROR]: (cube3_mls_q_t::construct_domain) sizes of phi, acn and clr do not coincide.");
#endif

  /* Eliminate unnecessary splitting */

  loc = INS;

  double value;
  double p0, p1, p2, p3, p4, p5;
  double c0, c1, c2;
  int I, J, K;

  std::vector<action_t> nt_acn;
  std::vector<int>      nt_clr;
  std::vector<int>      nt_idx;

  double band = lip*diag;

//  if(0)
  for (unsigned int p = 0; p < num_of_lsfs; p++)
  {
    bool all_negative = false;
    bool all_positive = false;

    // check how far cube is from interface
    double value = phi_all[n_nodes*p + 0];

    if (value > band)
    {
      all_negative = false;
      all_positive = true;
    }
    else if (value < -band)
    {
      all_negative = true;
      all_positive = false;
    }

    if (all_positive)
    {
      if (acn[p] == INTERSECTION)
      {
        loc = OUT;
      }
    }
    else if (all_negative)
    {
      if (acn[p] == ADDITION)
      {
        loc = INS;
      }
    }
    else //if (loc == FCE || (loc == INS && acn[p] == INTERSECTION) || (loc == OUT && acn[p] == ADDITION))
    {
      loc = FCE;
    }
  }

  if (loc == FCE)
  {
    loc = INS;
  for (unsigned int p = 0; p < num_of_lsfs; p++)
  {
    bool all_negative = true;
    bool all_positive = true;

    // check how far cube is from interface
    double value = phi_all[n_nodes*p + 0];

    if (value > band)
    {
      all_negative = false;
      all_positive = true;
    }
    else if (value < -band)
    {
      all_negative = true;
      all_positive = false;
    }
    else
    {
      // sample level-set function and check for intersections
      for (short k = 0; k < 3; ++k)
        for (short j = 0; j < 3; ++j)
          for (short i = 0; i < 3; ++i)
          {
            value = phi_all[n_nodes*p + n_nodes_dir*n_nodes_dir*k + n_nodes_dir*j + i];
            all_negative = (all_negative && (value < 0.));
            all_positive = (all_positive && (value > 0.));
          }

      // check edges for complex intersections
//      if(0)
      if (all_negative || all_positive)
      {
        for (int idx_edge = 0; idx_edge < num_edges; ++idx_edge)
        {
          p0 = phi_all[n_nodes*p + ep[idx_edge][0]];
          p1 = phi_all[n_nodes*p + ep[idx_edge][1]];
          p2 = phi_all[n_nodes*p + ep[idx_edge][2]];

          c0 = p0;
          c1 = -3.*p0 + 4.*p1 -    p2;
          c2 =  2.*p0 - 4.*p1 + 2.*p2;

          if (c1*(2.*c2+c1) < 0.)
          {
            double a_ext = -.5*c1/c2;
            if (a_ext >= 0. && a_ext <= 1.)
            {
              double phi_ext = c0 + c1*a_ext + c2*a_ext*a_ext;

              if (p0*phi_ext < 0)
              {
                //              std::cout << "cube: " << a_ext << " " << phi_ext << "\n";
                all_negative = false;
                all_positive = false;
                break;
              }
            }
          }

        }
      }

      // check faces for complex intersections
//      if(0)
      if (all_negative || all_positive)
      {
        for (int idx_face = 0; idx_face < num_faces; ++idx_face)
        {
          p0 = phi_all[n_nodes*p + fp[idx_face][0]];
          p1 = phi_all[n_nodes*p + fp[idx_face][1]];
          p2 = phi_all[n_nodes*p + fp[idx_face][2]];
          p3 = phi_all[n_nodes*p + fp[idx_face][3]];
          p4 = phi_all[n_nodes*p + fp[idx_face][4]];
          p5 = phi_all[n_nodes*p + fp[idx_face][5]];

          if (p0*p1 > 0 && p1*p2 > 0 && p2*p3 > 0 && p3*p4 > 0 && p4*p5 > 0)
          {
            double paa = 2.*p0 + 2.*p1 - 4.*p3;
            double pab = 4.*p0 - 4.*p3 + 4.*p4 - 4.*p5;
            double pbb = 2.*p0 + 2.*p2 - 4.*p5;
            double pa = -3.*p0 -    p1 + 4.*p3;
            double pb = -3.*p0 -    p2 + 4.*p5;

            double det = 4.*paa*pbb - pab*pab;

            if (fabs(det) > 1.e-15)
            {
              double a = (pb*pab - 2.*pa*pbb)/det;
              double b = (pa*pab - 2.*pb*paa)/det;

              if (a >= 0 && b >= 0 && a+b <= 1.)
              {
                double phi_extremum = paa*a*a + pab*a*b + pbb*b*b + pa*a + pb*b + p0;

                if (phi_extremum*p0 < 0)
                {
//                  std::cout << "face " << a << " " << b << "\n";
                  all_negative = false;
                  all_positive = false;
                  break;
                }
              }
            }
          }
        }
      }
    }

    if (all_positive)
    {
      if (acn[p] == INTERSECTION)
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
      if (acn[p] == ADDITION)
      {
        loc = INS;
//        nt_phi.clear();
        nt_acn.clear();
        nt_clr.clear();
        nt_idx.clear();
      }
    }
    else if (loc == FCE || (loc == INS && acn[p] == INTERSECTION) || (loc == OUT && acn[p] == ADDITION))
    {
      loc = FCE;
//      nt_phi.push_back(phi[p]);
      nt_acn.push_back(acn[p]);
      nt_clr.push_back(clr[p]);
      nt_idx.push_back(p);
    }
  }
  }

  /* uncomment this to use all level-set functions */
//  nt_phi = phi;
//  nt_acn = acn;
//  nt_clr = clr;
//  loc = FCE;

//  if (nt_idx.size() > 1) loc = OUT;

  if (loc == FCE)
  {
    if (nt_acn[0] == ADDITION) nt_acn[0] = INTERSECTION;

    double x[27] = { x0, xc, x1, x0, xc, x1, x0, xc, x1,
                     x0, xc, x1, x0, xc, x1, x0, xc, x1,
                     x0, xc, x1, x0, xc, x1, x0, xc, x1 };

    double y[27] = { y0, y0, y0, yc, yc, yc, y1, y1, y1,
                     y0, y0, y0, yc, yc, yc, y1, y1, y1,
                     y0, y0, y0, yc, yc, yc, y1, y1, y1 };

    double z[27] = { z0, z0, z0, z0, z0, z0, z0, z0, z0,
                     zc, zc, zc, zc, zc, zc, zc, zc, zc,
                     z1, z1, z1, z1, z1, z1, z1, z1, z1 };

    /* Split a cube into 6 simplices */
    simplex.clear();
    simplex.reserve(NUM_TETS);
    simplex.push_back(simplex3_mls_q_t(x[t0p0],y[t0p0],z[t0p0],
                                               x[t0p1],y[t0p1],z[t0p1],
                                               x[t0p2],y[t0p2],z[t0p2],
                                               x[t0p3],y[t0p3],z[t0p3],
                                               x[t0p4],y[t0p4],z[t0p4],
                                               x[t0p5],y[t0p5],z[t0p5],
                                               x[t0p6],y[t0p6],z[t0p6],
                                               x[t0p7],y[t0p7],z[t0p7],
                                               x[t0p8],y[t0p8],z[t0p8],
                                               x[t0p9],y[t0p9],z[t0p9]));

    simplex.push_back(simplex3_mls_q_t(x[t1p0],y[t1p0],z[t1p0],
                                               x[t1p1],y[t1p1],z[t1p1],
                                               x[t1p2],y[t1p2],z[t1p2],
                                               x[t1p3],y[t1p3],z[t1p3],
                                               x[t1p4],y[t1p4],z[t1p4],
                                               x[t1p5],y[t1p5],z[t1p5],
                                               x[t1p6],y[t1p6],z[t1p6],
                                               x[t1p7],y[t1p7],z[t1p7],
                                               x[t1p8],y[t1p8],z[t1p8],
                                               x[t1p9],y[t1p9],z[t1p9]));

    simplex.push_back(simplex3_mls_q_t(x[t2p0],y[t2p0],z[t2p0],
                                               x[t2p1],y[t2p1],z[t2p1],
                                               x[t2p2],y[t2p2],z[t2p2],
                                               x[t2p3],y[t2p3],z[t2p3],
                                               x[t2p4],y[t2p4],z[t2p4],
                                               x[t2p5],y[t2p5],z[t2p5],
                                               x[t2p6],y[t2p6],z[t2p6],
                                               x[t2p7],y[t2p7],z[t2p7],
                                               x[t2p8],y[t2p8],z[t2p8],
                                               x[t2p9],y[t2p9],z[t2p9]));

    simplex.push_back(simplex3_mls_q_t(x[t3p0],y[t3p0],z[t3p0],
                                               x[t3p1],y[t3p1],z[t3p1],
                                               x[t3p2],y[t3p2],z[t3p2],
                                               x[t3p3],y[t3p3],z[t3p3],
                                               x[t3p4],y[t3p4],z[t3p4],
                                               x[t3p5],y[t3p5],z[t3p5],
                                               x[t3p6],y[t3p6],z[t3p6],
                                               x[t3p7],y[t3p7],z[t3p7],
                                               x[t3p8],y[t3p8],z[t3p8],
                                               x[t3p9],y[t3p9],z[t3p9]));

    simplex.push_back(simplex3_mls_q_t(x[t4p0],y[t4p0],z[t4p0],
                                               x[t4p1],y[t4p1],z[t4p1],
                                               x[t4p2],y[t4p2],z[t4p2],
                                               x[t4p3],y[t4p3],z[t4p3],
                                               x[t4p4],y[t4p4],z[t4p4],
                                               x[t4p5],y[t4p5],z[t4p5],
                                               x[t4p6],y[t4p6],z[t4p6],
                                               x[t4p7],y[t4p7],z[t4p7],
                                               x[t4p8],y[t4p8],z[t4p8],
                                               x[t4p9],y[t4p9],z[t4p9]));

    simplex.push_back(simplex3_mls_q_t(x[t5p0],y[t5p0],z[t5p0],
                                               x[t5p1],y[t5p1],z[t5p1],
                                               x[t5p2],y[t5p2],z[t5p2],
                                               x[t5p3],y[t5p3],z[t5p3],
                                               x[t5p4],y[t5p4],z[t5p4],
                                               x[t5p5],y[t5p5],z[t5p5],
                                               x[t5p6],y[t5p6],z[t5p6],
                                               x[t5p7],y[t5p7],z[t5p7],
                                               x[t5p8],y[t5p8],z[t5p8],
                                               x[t5p9],y[t5p9],z[t5p9]));

//    // mark appropriate edges for integrate_in_dir
//    simplex[0].tris_[0].dir = 1; simplex[0].tris_[3].dir = 4;
//    simplex[1].tris_[0].dir = 3; simplex[1].tris_[3].dir = 4;
//    simplex[2].tris_[0].dir = 1; simplex[2].tris_[3].dir = 2;
//    simplex[3].tris_[0].dir = 3; simplex[3].tris_[3].dir = 0;
//    simplex[4].tris_[0].dir = 5; simplex[4].tris_[3].dir = 2;
//    simplex[5].tris_[0].dir = 5; simplex[5].tris_[3].dir = 0;

    num_of_lsfs = nt_idx.size();

    std::vector<double> phi_s(n_nodes_simplex*num_of_lsfs, -1);

    for (int s = 0; s < NUM_TETS; ++s)
    {
      for (int i = 0; i < num_of_lsfs; ++i)
      {
        int phi_idx = nt_idx[i];
        for (int n = 0; n < n_nodes_simplex; ++n)
          phi_s[n_nodes_simplex*i + n] = phi_all[n_nodes*phi_idx + tp_q[s][n]];
      }
      simplex[s].set_check_for_curvature(check_for_curvature_);
      simplex[s].construct_domain(phi_s, nt_acn, nt_clr);
    }
  }
}

/* Quadrature points */

void cube3_mls_q_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
//  weights.clear();
//  X.clear();
//  Y.clear();
//  Z.clear();

  switch (loc){
    case INS:
      {
        static double alpha = 1./sqrt(3.);

        double w = (x1-x0)*(y1-y0)*(z1-z0)/8.0;

        double xm = x0+.5*(-alpha+1.)*(x1-x0);
        double xp = x0+.5*( alpha+1.)*(x1-x0);

        double ym = y0+.5*(-alpha+1.)*(y1-y0);
        double yp = y0+.5*( alpha+1.)*(y1-y0);

        double zm = z0+.5*(-alpha+1.)*(z1-z0);
        double zp = z0+.5*( alpha+1.)*(z1-z0);

        X.push_back( xm ); Y.push_back( ym ); Z.push_back( zm ); weights.push_back(w);
        X.push_back( xp ); Y.push_back( ym ); Z.push_back( zm ); weights.push_back(w);
        X.push_back( xm ); Y.push_back( yp ); Z.push_back( zm ); weights.push_back(w);
        X.push_back( xp ); Y.push_back( yp ); Z.push_back( zm ); weights.push_back(w);
        X.push_back( xm ); Y.push_back( ym ); Z.push_back( zp ); weights.push_back(w);
        X.push_back( xp ); Y.push_back( ym ); Z.push_back( zp ); weights.push_back(w);
        X.push_back( xm ); Y.push_back( yp ); Z.push_back( zp ); weights.push_back(w);
        X.push_back( xp ); Y.push_back( yp ); Z.push_back( zp ); weights.push_back(w);
      } break;
    case OUT: break;
    case FCE:
      {
        simplex[0].quadrature_over_domain(weights, X, Y, Z);
        simplex[1].quadrature_over_domain(weights, X, Y, Z);
        simplex[2].quadrature_over_domain(weights, X, Y, Z);
        simplex[3].quadrature_over_domain(weights, X, Y, Z);
        simplex[4].quadrature_over_domain(weights, X, Y, Z);
        simplex[5].quadrature_over_domain(weights, X, Y, Z);
      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
  }
}

void cube3_mls_q_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
//  weights.clear();
//  X.clear();
//  Y.clear();
//  Z.clear();

  if (loc == FCE)
  {
    simplex[0].quadrature_over_interface(num, weights, X, Y, Z);
    simplex[1].quadrature_over_interface(num, weights, X, Y, Z);
    simplex[2].quadrature_over_interface(num, weights, X, Y, Z);
    simplex[3].quadrature_over_interface(num, weights, X, Y, Z);
    simplex[4].quadrature_over_interface(num, weights, X, Y, Z);
    simplex[5].quadrature_over_interface(num, weights, X, Y, Z);
  }
}

void cube3_mls_q_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
//  weights.clear();
//  X.clear();
//  Y.clear();
//  Z.clear();

  if (loc == FCE && num_of_lsfs > 1)
  {
    simplex[0].quadrature_over_intersection(num0, num1, weights, X, Y, Z);
    simplex[1].quadrature_over_intersection(num0, num1, weights, X, Y, Z);
    simplex[2].quadrature_over_intersection(num0, num1, weights, X, Y, Z);
    simplex[3].quadrature_over_intersection(num0, num1, weights, X, Y, Z);
    simplex[4].quadrature_over_intersection(num0, num1, weights, X, Y, Z);
    simplex[5].quadrature_over_intersection(num0, num1, weights, X, Y, Z);
  }
}

void cube3_mls_q_t::quadrature_over_intersection(int num0, int num1, int num2, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
//  weights.clear();
//  X.clear();
//  Y.clear();
//  Z.clear();

  if (loc == FCE && num_of_lsfs > 2)
  {
    simplex[0].quadrature_over_intersection(num0, num1, num2, weights, X, Y, Z);
    simplex[1].quadrature_over_intersection(num0, num1, num2, weights, X, Y, Z);
    simplex[2].quadrature_over_intersection(num0, num1, num2, weights, X, Y, Z);
    simplex[3].quadrature_over_intersection(num0, num1, num2, weights, X, Y, Z);
    simplex[4].quadrature_over_intersection(num0, num1, num2, weights, X, Y, Z);
    simplex[5].quadrature_over_intersection(num0, num1, num2, weights, X, Y, Z);
  }
}

void cube3_mls_q_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
//  weights.clear();
//  X.clear();
//  Y.clear();
//  Z.clear();

  switch (loc){
    case OUT: break;
    case INS:
      {
        static double alpha = 1./sqrt(3.);

        double xm = x0+.5*(-alpha+1.)*(x1-x0);
        double xp = x0+.5*( alpha+1.)*(x1-x0);

        double ym = y0+.5*(-alpha+1.)*(y1-y0);
        double yp = y0+.5*( alpha+1.)*(y1-y0);

        double zm = z0+.5*(-alpha+1.)*(z1-z0);
        double zp = z0+.5*( alpha+1.)*(z1-z0);

        double w;

        switch (dir) {
          case 0:
            w = .25*(y1-y0)*(z1-z0);
            X.push_back(x0); Y.push_back(ym); Z.push_back(zm); weights.push_back(w);
            X.push_back(x0); Y.push_back(yp); Z.push_back(zm); weights.push_back(w);
            X.push_back(x0); Y.push_back(ym); Z.push_back(zp); weights.push_back(w);
            X.push_back(x0); Y.push_back(yp); Z.push_back(zp); weights.push_back(w);
            break;
          case 1:
            w = .25*(y1-y0)*(z1-z0);
            X.push_back(x1); Y.push_back(ym); Z.push_back(zm); weights.push_back(w);
            X.push_back(x1); Y.push_back(yp); Z.push_back(zm); weights.push_back(w);
            X.push_back(x1); Y.push_back(ym); Z.push_back(zp); weights.push_back(w);
            X.push_back(x1); Y.push_back(yp); Z.push_back(zp); weights.push_back(w);
            break;
          case 2:
            w = .25*(x1-x0)*(z1-z0);
            X.push_back(xm); Y.push_back(y0); Z.push_back(zm); weights.push_back(w);
            X.push_back(xp); Y.push_back(y0); Z.push_back(zm); weights.push_back(w);
            X.push_back(xm); Y.push_back(y0); Z.push_back(zp); weights.push_back(w);
            X.push_back(xp); Y.push_back(y0); Z.push_back(zp); weights.push_back(w);
            break;
          case 3:
            w = .25*(x1-x0)*(z1-z0);
            X.push_back(xm); Y.push_back(y1); Z.push_back(zm); weights.push_back(w);
            X.push_back(xp); Y.push_back(y1); Z.push_back(zm); weights.push_back(w);
            X.push_back(xm); Y.push_back(y1); Z.push_back(zp); weights.push_back(w);
            X.push_back(xp); Y.push_back(y1); Z.push_back(zp); weights.push_back(w);
            break;
          case 4:
            w = .25*(x1-x0)*(y1-y0);
            X.push_back(xm); Y.push_back(ym); Z.push_back(z0); weights.push_back(w);
            X.push_back(xp); Y.push_back(ym); Z.push_back(z0); weights.push_back(w);
            X.push_back(xm); Y.push_back(yp); Z.push_back(z0); weights.push_back(w);
            X.push_back(xp); Y.push_back(yp); Z.push_back(z0); weights.push_back(w);
            break;
          case 5:
            w = .25*(x1-x0)*(y1-y0);
            X.push_back(xm); Y.push_back(ym); Z.push_back(z1); weights.push_back(w);
            X.push_back(xp); Y.push_back(ym); Z.push_back(z1); weights.push_back(w);
            X.push_back(xm); Y.push_back(yp); Z.push_back(z1); weights.push_back(w);
            X.push_back(xp); Y.push_back(yp); Z.push_back(z1); weights.push_back(w);
            break;
        }
      } break;
    case FCE:
      {
      /* integration in dir
       * 0: s3->f3 + s5->f3
       * 1: s0->f0 + s2->f0
       * 2: s2->f3 + s4->f3
       * 3: s1->f0 + s3->f0
       * 4: s0->f3 + s1->f3
       * 5: s4->f0 + s5->f0
       */
      switch (dir)
      {
        case 0:
          simplex[3].quadrature_in_dir(3, weights, X, Y, Z);
          simplex[5].quadrature_in_dir(3, weights, X, Y, Z);
          break;
        case 1:
          simplex[0].quadrature_in_dir(0, weights, X, Y, Z);
          simplex[2].quadrature_in_dir(0, weights, X, Y, Z);
          break;
        case 2:
          simplex[2].quadrature_in_dir(3, weights, X, Y, Z);
          simplex[4].quadrature_in_dir(3, weights, X, Y, Z);
          break;
        case 3:
          simplex[1].quadrature_in_dir(0, weights, X, Y, Z);
          simplex[3].quadrature_in_dir(0, weights, X, Y, Z);
          break;
        case 4:
          simplex[0].quadrature_in_dir(3, weights, X, Y, Z);
          simplex[1].quadrature_in_dir(3, weights, X, Y, Z);
          break;
        case 5:
          simplex[4].quadrature_in_dir(0, weights, X, Y, Z);
          simplex[5].quadrature_in_dir(0, weights, X, Y, Z);
          break;
      }
//        simplex[0].quadrature_in_dir(dir, weights, X, Y, Z);
//        simplex[1].quadrature_in_dir(dir, weights, X, Y, Z);
//        simplex[2].quadrature_in_dir(dir, weights, X, Y, Z);
//        simplex[3].quadrature_in_dir(dir, weights, X, Y, Z);
//        simplex[4].quadrature_in_dir(dir, weights, X, Y, Z);
//        simplex[5].quadrature_in_dir(dir, weights, X, Y, Z);
      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
  }
}
