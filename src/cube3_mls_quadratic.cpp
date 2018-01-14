#include "cube3_mls_quadratic.h"

void cube3_mls_quadratic_t::construct_domain(std::vector<CF_3 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  num_of_lsfs = phi.size();

  // 3-by-3-by-3 grid
  double xc = .5*(x0+x1);
  double yc = .5*(y0+y1);
  double zc = .5*(z0+z1);

  double x[] = { x0, xc, x1 };
  double y[] = { y0, yc, y1 };
  double z[] = { z0, zc, z1 };

#ifdef CASL_THROWS
  if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
          throw std::domain_error("[CASL_ERROR]: (cube3_mls_quadratic_t::construct_domain) sizes of phi, acn and clr do not coincide.");
#endif

  std::vector<double> phi_all(num_of_lsfs*n_nodes, -1.);

  /* Eliminate unnecessary splitting */

  loc = INS;

  double value;
  double p0, p1, p2, p3, p4, p5;
  double c0, c1, c2;
  int I, J, K;

  std::vector<CF_3 *>   nt_phi;
  std::vector<action_t> nt_acn;
  std::vector<int>      nt_clr;
  std::vector<int>      nt_idx;

  double band = lip*diag;

//  if(0)
//  for (unsigned int p = 0; p < num_of_lsfs; p++)
//  {
//    bool all_negative = false;
//    bool all_positive = false;

//    // check how far cube is from interface
//    double value = (*phi[p]) ( xc, yc, zc );

//    if (value > band)
//    {
//      all_negative = false;
//      all_positive = true;
//    }
//    else if (value < -band)
//    {
//      all_negative = true;
//      all_positive = false;
//    }

//    if (all_positive)
//    {
//      if (acn[p] == INTERSECTION)
//      {
//        loc = OUT;
//      }
//    }
//    else if (all_negative)
//    {
//      if (acn[p] == ADDITION)
//      {
//        loc = INS;
//      }
//    }
//    else if (loc == FCE || (loc == INS && acn[p] == INTERSECTION) || (loc == OUT && acn[p] == ADDITION))
//    {
//      loc = FCE;
//    }
//  }

//  if (loc == FCE)
  for (unsigned int p = 0; p < num_of_lsfs; p++)
  {
    bool all_negative = true;
    bool all_positive = true;

    // check how far cube is from interface
    double value = (*phi[p]) ( xc, yc, zc );

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
            value = (*phi[p]) ( x[i], y[j], z[k] );
//            value = -1;
            phi_all[n_nodes*p + n_nodes_dir*n_nodes_dir*k + n_nodes_dir*j + i] = value;
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

          if (fabs(c2) > EPS)
          {
            double a_ext = -.5*c1/c2;
            if (a_ext > 0. && a_ext < 1.)
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

            if (fabs(det) > EPS)
            {
              double a = (pb*pab - 2.*pa*pbb)/det;
              double b = (pa*pab - 2.*pb*paa)/det;

              if (a > 0 && b > 0 && a+b < 1.)
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
        nt_phi.clear();
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
        nt_phi.clear();
        nt_acn.clear();
        nt_clr.clear();
        nt_idx.clear();
      }
    }
    else if (loc == FCE || (loc == INS && acn[p] == INTERSECTION) || (loc == OUT && acn[p] == ADDITION))
    {
      loc = FCE;
      nt_phi.push_back(phi[p]);
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
    simplex.push_back(simplex3_mls_quadratic_t(x[t0p0],y[t0p0],z[t0p0],
                                               x[t0p1],y[t0p1],z[t0p1],
                                               x[t0p2],y[t0p2],z[t0p2],
                                               x[t0p3],y[t0p3],z[t0p3],
                                               x[t0p4],y[t0p4],z[t0p4],
                                               x[t0p5],y[t0p5],z[t0p5],
                                               x[t0p6],y[t0p6],z[t0p6],
                                               x[t0p7],y[t0p7],z[t0p7],
                                               x[t0p8],y[t0p8],z[t0p8],
                                               x[t0p9],y[t0p9],z[t0p9]));

    simplex.push_back(simplex3_mls_quadratic_t(x[t1p0],y[t1p0],z[t1p0],
                                               x[t1p1],y[t1p1],z[t1p1],
                                               x[t1p2],y[t1p2],z[t1p2],
                                               x[t1p3],y[t1p3],z[t1p3],
                                               x[t1p4],y[t1p4],z[t1p4],
                                               x[t1p5],y[t1p5],z[t1p5],
                                               x[t1p6],y[t1p6],z[t1p6],
                                               x[t1p7],y[t1p7],z[t1p7],
                                               x[t1p8],y[t1p8],z[t1p8],
                                               x[t1p9],y[t1p9],z[t1p9]));

    simplex.push_back(simplex3_mls_quadratic_t(x[t2p0],y[t2p0],z[t2p0],
                                               x[t2p1],y[t2p1],z[t2p1],
                                               x[t2p2],y[t2p2],z[t2p2],
                                               x[t2p3],y[t2p3],z[t2p3],
                                               x[t2p4],y[t2p4],z[t2p4],
                                               x[t2p5],y[t2p5],z[t2p5],
                                               x[t2p6],y[t2p6],z[t2p6],
                                               x[t2p7],y[t2p7],z[t2p7],
                                               x[t2p8],y[t2p8],z[t2p8],
                                               x[t2p9],y[t2p9],z[t2p9]));

    simplex.push_back(simplex3_mls_quadratic_t(x[t3p0],y[t3p0],z[t3p0],
                                               x[t3p1],y[t3p1],z[t3p1],
                                               x[t3p2],y[t3p2],z[t3p2],
                                               x[t3p3],y[t3p3],z[t3p3],
                                               x[t3p4],y[t3p4],z[t3p4],
                                               x[t3p5],y[t3p5],z[t3p5],
                                               x[t3p6],y[t3p6],z[t3p6],
                                               x[t3p7],y[t3p7],z[t3p7],
                                               x[t3p8],y[t3p8],z[t3p8],
                                               x[t3p9],y[t3p9],z[t3p9]));

    simplex.push_back(simplex3_mls_quadratic_t(x[t4p0],y[t4p0],z[t4p0],
                                               x[t4p1],y[t4p1],z[t4p1],
                                               x[t4p2],y[t4p2],z[t4p2],
                                               x[t4p3],y[t4p3],z[t4p3],
                                               x[t4p4],y[t4p4],z[t4p4],
                                               x[t4p5],y[t4p5],z[t4p5],
                                               x[t4p6],y[t4p6],z[t4p6],
                                               x[t4p7],y[t4p7],z[t4p7],
                                               x[t4p8],y[t4p8],z[t4p8],
                                               x[t4p9],y[t4p9],z[t4p9]));

    simplex.push_back(simplex3_mls_quadratic_t(x[t5p0],y[t5p0],z[t5p0],
                                               x[t5p1],y[t5p1],z[t5p1],
                                               x[t5p2],y[t5p2],z[t5p2],
                                               x[t5p3],y[t5p3],z[t5p3],
                                               x[t5p4],y[t5p4],z[t5p4],
                                               x[t5p5],y[t5p5],z[t5p5],
                                               x[t5p6],y[t5p6],z[t5p6],
                                               x[t5p7],y[t5p7],z[t5p7],
                                               x[t5p8],y[t5p8],z[t5p8],
                                               x[t5p9],y[t5p9],z[t5p9]));

    // mark appropriate edges for integrate_in_dir
    simplex[0].tris[0].dir = 1; simplex[0].tris[3].dir = 4;
    simplex[1].tris[0].dir = 3; simplex[1].tris[3].dir = 4;
    simplex[2].tris[0].dir = 1; simplex[2].tris[3].dir = 2;
    simplex[3].tris[0].dir = 3; simplex[3].tris[3].dir = 0;
    simplex[4].tris[0].dir = 5; simplex[4].tris[3].dir = 2;
    simplex[5].tris[0].dir = 5; simplex[5].tris[3].dir = 0;

    num_of_lsfs = nt_idx.size();

    std::vector<double> phi_s(n_nodes_simplex*num_of_lsfs, -1);

    for (int s = 0; s < NUM_TETS; ++s)
    {
      for (int i = 0; i < num_of_lsfs; ++i)
      {
        int phi_idx = nt_idx[i];
        for (int n = 0; n < n_nodes_simplex; ++n)
          phi_s[n_nodes_simplex*i + n] = phi_all[n_nodes*phi_idx + tp[s][n]];
      }
      simplex[s].construct_domain(phi_s, nt_acn, nt_clr);

//      simplex[s].construct_domain(nt_phi, nt_acn, nt_clr);
    }
  }

//  if (simplex[0].tets.size() == 1 &&
//      simplex[1].tets.size() == 1 &&
//      simplex[2].tets.size() == 1 &&
//      simplex[3].tets.size() == 1 &&
//      simplex[4].tets.size() == 1 &&
//      simplex[5].tets.size() == 1)
//  {
//    loc = loc_prelim;
//    simplex.clear();
//  }
}

double cube3_mls_quadratic_t::integrate_over_domain(CF_3 &f)
{
//  return 0;
  switch (loc){
    case INS:
      {
        double alpha = 1./sqrt(3.);

        double F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
        double F1 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
        double F2 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
        double F3 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
        double F4 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );
        double F5 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );
        double F6 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );
        double F7 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );

        return (x1-x0)*(y1-y0)*(z1-z0)*(F0+F1+F2+F3+F4+F5+F6+F7)/8.0;
      } break;
    case OUT: return 0.0; break;
    case FCE:
      {
        double result = simplex[0].integrate_over_domain(f)
             + simplex[1].integrate_over_domain(f)
             + simplex[2].integrate_over_domain(f)
             + simplex[3].integrate_over_domain(f)
             + simplex[4].integrate_over_domain(f)
             + simplex[5].integrate_over_domain(f);
#ifdef CASL_THROWS
        if (result != result)
          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
        return result;
      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0;
  }

#ifdef CASL_THROWS
  throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
  return 0;
}

double cube3_mls_quadratic_t::integrate_over_interface(CF_3& f, int num)
{
//  return 0;
  if (loc == FCE)
  {
//    double f0 = simplex[0].integrate_over_interface(f, num);
//    double f1 = simplex[1].integrate_over_interface(f, num);
//    double f2 = simplex[2].integrate_over_interface(f, num);
//    double f3 = simplex[3].integrate_over_interface(f, num);
//    double f4 = simplex[4].integrate_over_interface(f, num);
//    double f5 = simplex[5].integrate_over_interface(f, num);
//    return MAX(f0, f1, MAX(f2, f3, MAX(f4, f5)));
    double result = simplex[0].integrate_over_interface(f, num)
         + simplex[1].integrate_over_interface(f, num)
         + simplex[2].integrate_over_interface(f, num)
         + simplex[3].integrate_over_interface(f, num)
         + simplex[4].integrate_over_interface(f, num)
         + simplex[5].integrate_over_interface(f, num);
#ifdef CASL_THROWS
        if (result != result)
          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
        return result;
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_over_colored_interface(CF_3 &f, int num0, int num1)
{
  if (loc == FCE)
  {
    double result = simplex[0].integrate_over_colored_interface(f, num0, num1)
         + simplex[1].integrate_over_colored_interface(f, num0, num1)
         + simplex[2].integrate_over_colored_interface(f, num0, num1)
         + simplex[3].integrate_over_colored_interface(f, num0, num1)
         + simplex[4].integrate_over_colored_interface(f, num0, num1)
         + simplex[5].integrate_over_colored_interface(f, num0, num1);

#ifdef CASL_THROWS
        if (result != result)
          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
        return result;
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_over_intersection(CF_3 &f, int num0, int num1)
{
  if (loc == FCE && num_of_lsfs > 1)
  {
    double result = simplex[0].integrate_over_intersection(f, num0, num1)
         + simplex[1].integrate_over_intersection(f, num0, num1)
         + simplex[2].integrate_over_intersection(f, num0, num1)
         + simplex[3].integrate_over_intersection(f, num0, num1)
         + simplex[4].integrate_over_intersection(f, num0, num1)
         + simplex[5].integrate_over_intersection(f, num0, num1);

#ifdef CASL_THROWS
        if (result != result)
          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
        return result;
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_over_intersection(CF_3 &f, int num0, int num1, int num2)
{
  if (loc == FCE && num_of_lsfs > 2)
  {
    double result = simplex[0].integrate_over_intersection(f, num0, num1, num2)
         + simplex[1].integrate_over_intersection(f, num0, num1, num2)
         + simplex[2].integrate_over_intersection(f, num0, num1, num2)
         + simplex[3].integrate_over_intersection(f, num0, num1, num2)
         + simplex[4].integrate_over_intersection(f, num0, num1, num2)
         + simplex[5].integrate_over_intersection(f, num0, num1, num2);

#ifdef CASL_THROWS
        if (result != result)
          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
        return result;
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_in_dir(CF_3 &f, int dir)
{
  switch (loc){
    case OUT: return 0;
    case INS:
      {
        double alpha = 1./sqrt(3.);
        double F0 = 1, F1 = 1, F2 = 1, F3 = 1, s = 0;

        switch (dir) {
          case 0: s = (y1-y0)*(z1-z0);
            F0 = f( x0, y0+.5*(-alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
            F1 = f( x0, y0+.5*( alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
            F2 = f( x0, y0+.5*(-alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );
            F3 = f( x0, y0+.5*( alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );
            break;
          case 1: s = (y1-y0)*(z1-z0);
            F0 = f( x1, y0+.5*(-alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
            F1 = f( x1, y0+.5*( alpha+1.)*(y1-y0), z0+.5*(-alpha+1.)*(z1-z0) );
            F2 = f( x1, y0+.5*(-alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );
            F3 = f( x1, y0+.5*( alpha+1.)*(y1-y0), z0+.5*( alpha+1.)*(z1-z0) );
            break;
          case 2: s = (x1-x0)*(z1-z0);
            F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y0, z0+.5*(-alpha+1.)*(z1-z0) );
            F1 = f( x0+.5*( alpha+1.)*(x1-x0), y0, z0+.5*(-alpha+1.)*(z1-z0) );
            F2 = f( x0+.5*(-alpha+1.)*(x1-x0), y0, z0+.5*( alpha+1.)*(z1-z0) );
            F3 = f( x0+.5*( alpha+1.)*(x1-x0), y0, z0+.5*( alpha+1.)*(z1-z0) );
            break;
          case 3: s = (x1-x0)*(z1-z0);
            F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y1, z0+.5*(-alpha+1.)*(z1-z0) );
            F1 = f( x0+.5*( alpha+1.)*(x1-x0), y1, z0+.5*(-alpha+1.)*(z1-z0) );
            F2 = f( x0+.5*(-alpha+1.)*(x1-x0), y1, z0+.5*( alpha+1.)*(z1-z0) );
            F3 = f( x0+.5*( alpha+1.)*(x1-x0), y1, z0+.5*( alpha+1.)*(z1-z0) );
            break;
          case 4: s = (x1-x0)*(y1-y0);
            F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z0 );
            F1 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z0 );
            F2 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z0 );
            F3 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z0 );
            break;
          case 5: s = (x1-x0)*(y1-y0);
            F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z1 );
            F1 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0), z1 );
            F2 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z1 );
            F3 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0), z1 );
            break;
        }

        return 0.25*s*(F0+F1+F2+F3);
      }
    case FCE:
      {
        double result = ( simplex[0].integrate_in_dir(f, dir) +
            simplex[1].integrate_in_dir(f, dir) +
            simplex[2].integrate_in_dir(f, dir) +
            simplex[3].integrate_in_dir(f, dir) +
            simplex[4].integrate_in_dir(f, dir) +
            simplex[5].integrate_in_dir(f, dir) );
#ifdef CASL_THROWS
        if (result != result)
          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
        return result;
      }
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0;
  }
}

//void cube3_mls_quadratic_t::interpolate_to_cube(double *in, double *out)
//{
//  out[0] = interp.linear(in, x0, y0, z0);
//  out[1] = interp.linear(in, x1, y0, z0);
//  out[2] = interp.linear(in, x0, y1, z0);
//  out[3] = interp.linear(in, x1, y1, z0);
//  out[4] = interp.linear(in, x0, y0, z1);
//  out[5] = interp.linear(in, x1, y0, z1);
//  out[6] = interp.linear(in, x0, y1, z1);
//  out[7] = interp.linear(in, x1, y1, z1);
//}

double cube3_mls_quadratic_t::interpolate_quad(std::vector<double> f, double X, double Y, double Z)
{
  // Note that X and Y are scaled coordinates, i.e. in [-1,1]

//  double X = 2.*(x-x0)/(x1-x0) - 1.;
//  double Y = 2.*(y-y0)/(y1-y0) - 1.;

  std::vector<double> wx(3, 0);
  std::vector<double> wy(3, 0);
  std::vector<double> wz(3, 0);

  wx[0] = .5*X*(X-1.);
  wx[1] = -(X*X-1.);
  wx[2] = .5*X*(X+1.);

  wy[0] = .5*Y*(Y-1.);
  wy[1] = -(Y*Y-1.);
  wy[2] = .5*Y*(Y+1.);

  wz[0] = .5*Z*(Z-1.);
  wz[1] = -(Z*Z-1.);
  wz[2] = .5*Z*(Z+1.);

  double result = 0;

  for (short k = 0; k < 3; ++k)
    for (short j = 0; j < 3; ++j)
      for (short i = 0; i < 3; ++i)
        result += f[k*9+j*3+i]*wx[i]*wy[j]*wz[k];

  return result;
}


//double cube3_mls_quadratic_t::measure_of_domain()
//{
//  switch (loc){
//    case INS:
//      {
//        return (x1-x0)*(y1-y0)*(z1-z0);
//      } break;
//    case OUT: return 0.0; break;
//    case FCE:
//      {
//        std::vector<double> F(n_nodes_simplex, 1);

//        return simplex[0].integrate_over_domain(F)
//            +  simplex[1].integrate_over_domain(F)
//            +  simplex[2].integrate_over_domain(F)
//            +  simplex[3].integrate_over_domain(F)
//            +  simplex[4].integrate_over_domain(F)
//            +  simplex[5].integrate_over_domain(F);
//      } break;
//    default:
//#ifdef CASL_THROWS
//      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0;
//  }
//}

//double cube3_mls_quadratic_t::measure_of_interface(int num)
//{
//  if (loc == FCE)
//  {
//    std::vector<double> F(n_nodes_simplex, 1);

//    return simplex[0].integrate_over_interface(F, num) +
//           simplex[1].integrate_over_interface(F, num) +
//           simplex[2].integrate_over_interface(F, num) +
//           simplex[3].integrate_over_interface(F, num) +
//           simplex[4].integrate_over_interface(F, num) +
//           simplex[5].integrate_over_interface(F, num);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_quadratic_t::measure_of_colored_interface(int num0, int num1)
//{
//  if (loc == FCE)
//  {
//    std::vector<double> F(n_nodes_simplex, 1);

//    return simplex[0].integrate_over_colored_interface(F, num0, num1) +
//           simplex[1].integrate_over_colored_interface(F, num0, num1) +
//           simplex[2].integrate_over_colored_interface(F, num0, num1) +
//           simplex[3].integrate_over_colored_interface(F, num0, num1) +
//           simplex[5].integrate_over_colored_interface(F, num0, num1) +
//           simplex[4].integrate_over_colored_interface(F, num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_quadratic_t::measure_of_intersection(int num0, int num1)
//{
//  if (loc == FCE && num_of_lsfs > 1)
//  {
//    std::vector<double> F(n_nodes_simplex, 1);

//    return simplex[0].integrate_over_intersection(F, num0, num1) +
//           simplex[1].integrate_over_intersection(F, num0, num1) +
//           simplex[2].integrate_over_intersection(F, num0, num1) +
//           simplex[3].integrate_over_intersection(F, num0, num1) +
//           simplex[4].integrate_over_intersection(F, num0, num1) +
//           simplex[5].integrate_over_intersection(F, num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_quadratic_t::measure_in_dir(int dir)
//{
//  switch (loc){
//    case OUT: return 0;
//    case INS:
//      {
//        switch (dir) {
//          case 0: return (y1-y0)*(z1-z0);
//          case 1: return (y1-y0)*(z1-z0);
//          case 2: return (x1-x0)*(z1-z0);
//          case 3: return (x1-x0)*(z1-z0);
//          case 4: return (x1-x0)*(y1-y0);
//          case 5: return (x1-x0)*(y1-y0);
//        }
//      }
//    case FCE:
//      {
//        std::vector<double> F(n_nodes_simplex, 1);

//        return simplex[0].integrate_in_dir(F, dir) +
//               simplex[1].integrate_in_dir(F, dir) +
//               simplex[2].integrate_in_dir(F, dir) +
//               simplex[3].integrate_in_dir(F, dir) +
//               simplex[4].integrate_in_dir(F, dir) +
//               simplex[5].integrate_in_dir(F, dir);
//      }
//    default:
//#ifdef CASL_THROWS
//      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0;
//  }
//}

//double cube3_mls_quadratic_t::interpolate_linear(double *f, double x, double y, double z)
//{
//  double eps = 1.e-12;

//  // calculate relative distances to edges
//  double d_m00 = (x-x0)/(x1-x0);
//  if (d_m00 < 0.) d_m00 = 0.;
//  if (d_m00 > 1.) d_m00 = 1.;
//  double d_p00 = 1.-d_m00;

//  double d_0m0 = (y-y0)/(y1-y0);
//  if (d_0m0 < 0.) d_0m0 = 0.;
//  if (d_0m0 > 1.) d_0m0 = 1.;
//  double d_0p0 = 1.-d_0m0;

//  double d_00m = (z-z0)/(z1-z0);
//  if (d_00m < 0.) d_00m = 0.;
//  if (d_00m > 1.) d_00m = 1.;
//  double d_00p = 1.-d_00m;

//  // check if a point is a corner point
//  if (d_m00 < eps && d_0m0 < eps && d_00m < eps) return f[0];
//  if (d_p00 < eps && d_0m0 < eps && d_00m < eps) return f[1];
//  if (d_m00 < eps && d_0p0 < eps && d_00m < eps) return f[2];
//  if (d_p00 < eps && d_0p0 < eps && d_00m < eps) return f[3];
//  if (d_m00 < eps && d_0m0 < eps && d_00p < eps) return f[4];
//  if (d_p00 < eps && d_0m0 < eps && d_00p < eps) return f[5];
//  if (d_m00 < eps && d_0p0 < eps && d_00p < eps) return f[6];
//  if (d_p00 < eps && d_0p0 < eps && d_00p < eps) return f[7];

//  // do trilinear interpolation
//  double w_mmm = d_p00*d_0p0*d_00p;
//  double w_pmm = d_m00*d_0p0*d_00p;
//  double w_mpm = d_p00*d_0m0*d_00p;
//  double w_ppm = d_m00*d_0m0*d_00p;
//  double w_mmp = d_p00*d_0p0*d_00m;
//  double w_pmp = d_m00*d_0p0*d_00m;
//  double w_mpp = d_p00*d_0m0*d_00m;
//  double w_ppp = d_m00*d_0m0*d_00m;

//  return
//      w_mmm*f[0] +
//      w_pmm*f[1] +
//      w_mpm*f[2] +
//      w_ppm*f[3] +
//      w_mmp*f[4] +
//      w_pmp*f[5] +
//      w_mpp*f[6] +
//      w_ppp*f[7];
//}

//double cube3_mls_quadratic_t::interpolate_quadratic(double *f, double *fxx, double *fyy, double *fzz, double x, double y, double z)
//{
//  double eps = 1.e-12;

//  // calculate relative distances to edges
//  double d_m00 = (x-x0)/(x1-x0);
//  if (d_m00 < 0.) d_m00 = 0.;
//  if (d_m00 > 1.) d_m00 = 1.;
//  double d_p00 = 1.-d_m00;

//  double d_0m0 = (y-y0)/(y1-y0);
//  if (d_0m0 < 0.) d_0m0 = 0.;
//  if (d_0m0 > 1.) d_0m0 = 1.;
//  double d_0p0 = 1.-d_0m0;

//  double d_00m = (z-z0)/(z1-z0);
//  if (d_00m < 0.) d_00m = 0.;
//  if (d_00m > 1.) d_00m = 1.;
//  double d_00p = 1.-d_00m;

//  // check if a point is a corner point
//  if (d_m00 < eps && d_0m0 < eps && d_00m < eps) return f[0];
//  if (d_p00 < eps && d_0m0 < eps && d_00m < eps) return f[1];
//  if (d_m00 < eps && d_0p0 < eps && d_00m < eps) return f[2];
//  if (d_p00 < eps && d_0p0 < eps && d_00m < eps) return f[3];
//  if (d_m00 < eps && d_0m0 < eps && d_00p < eps) return f[4];
//  if (d_p00 < eps && d_0m0 < eps && d_00p < eps) return f[5];
//  if (d_m00 < eps && d_0p0 < eps && d_00p < eps) return f[6];
//  if (d_p00 < eps && d_0p0 < eps && d_00p < eps) return f[7];

//  // do trilinear interpolation
//  double w_mmm = d_p00*d_0p0*d_00p;
//  double w_pmm = d_m00*d_0p0*d_00p;
//  double w_mpm = d_p00*d_0m0*d_00p;
//  double w_ppm = d_m00*d_0m0*d_00p;
//  double w_mmp = d_p00*d_0p0*d_00m;
//  double w_pmp = d_m00*d_0p0*d_00m;
//  double w_mpp = d_p00*d_0m0*d_00m;
//  double w_ppp = d_m00*d_0m0*d_00m;

//  double F =
//      w_mmm*f[0] +
//      w_pmm*f[1] +
//      w_mpm*f[2] +
//      w_ppm*f[3] +
//      w_mmp*f[4] +
//      w_pmp*f[5] +
//      w_mpp*f[6] +
//      w_ppp*f[7];

//  double Fxx =
//      w_mmm*fxx[0] +
//      w_pmm*fxx[1] +
//      w_mpm*fxx[2] +
//      w_ppm*fxx[3] +
//      w_mmp*fxx[4] +
//      w_pmp*fxx[5] +
//      w_mpp*fxx[6] +
//      w_ppp*fxx[7];

//  double Fyy =
//      w_mmm*fyy[0] +
//      w_pmm*fyy[1] +
//      w_mpm*fyy[2] +
//      w_ppm*fyy[3] +
//      w_mmp*fyy[4] +
//      w_pmp*fyy[5] +
//      w_mpp*fyy[6] +
//      w_ppp*fyy[7];

//  double Fzz =
//      w_mmm*fzz[0] +
//      w_pmm*fzz[1] +
//      w_mpm*fzz[2] +
//      w_ppm*fzz[3] +
//      w_mmp*fzz[4] +
//      w_pmp*fzz[5] +
//      w_mpp*fzz[6] +
//      w_ppp*fzz[7];

//  F -= 0.5*((x1-x0)*(x1-x0)*d_p00*d_m00*Fxx + (y1-y0)*(y1-y0)*d_0p0*d_0m0*Fyy + (z1-z0)*(z1-z0)*d_00p*d_00m*Fzz);

//  return F;
//}
