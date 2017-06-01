#include "cube2_mls_quadratic.h"

void cube2_mls_quadratic_t::construct_domain(std::vector<CF_2 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  bool all_positive, all_negative;

  num_of_lsfs = phi.size();

  double xc = .5*(x0+x1);
  double yc = .5*(y0+y1);

#ifdef CASL_THROWS
  if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
          throw std::domain_error("[CASL_ERROR]: (cube2_mls_quadratic_t::construct_domain) sizes of phi, acn and clr do not coincide.");
#endif

  /* Eliminate unnecessary splitting */
  loc = INS;
  for (unsigned int p = 0; p < num_of_lsfs; p++)
  {
    all_negative = true;
    all_positive = true;

    double x[3] = { x0, xc, x1 };
    double y[3] = { y0, yc, y1 };

    for (short j = 0; j < 3; ++j)
      for (short i = 0; i < 3; ++i)
      {
        double value = (*phi[p]) ( x[i], y[j]);
        all_negative = (all_negative && (value < 0.));
        all_positive = (all_positive && (value > 0.));
      }

    if (all_positive)
    {
      if (acn[p] == INTERSECTION) loc = OUT;
    }
    else if (all_negative)
    {
      if (acn[p] == ADDITION) loc = INS;
    }
    else if ((loc == INS && acn[p] == INTERSECTION) || (loc == OUT && acn[p] == ADDITION))
    {
      loc = FCE;
    }
  }


  if (loc == FCE)
  {
//    if (acn[0] == ADDITION) // the first action always has to be INTERSECTION
//      acn[0] = INTERSECTION;

    /* Split the cube into 2 simplices */
    double x[9] = { x0, xc, x1, x0, xc, x1, x0, xc, x1 };
    double y[9] = { y0, y0, y0, yc, yc, yc, y1, y1, y1 };

    simplex.clear();
    simplex.reserve(2);
    simplex.push_back(simplex2_mls_quadratic_t(x[t0p0], y[t0p0], x[t0p1], y[t0p1], x[t0p2], y[t0p2],
                                               x[t0p3], y[t0p3], x[t0p4], y[t0p4], x[t0p5], y[t0p5]));// simplex.back().set_use_linear(use_linear);
    simplex.push_back(simplex2_mls_quadratic_t(x[t1p0], y[t1p0], x[t1p1], y[t1p1], x[t1p2], y[t1p2],
                                               x[t1p3], y[t1p3], x[t1p4], y[t1p4], x[t1p5], y[t1p5]));// simplex.back().set_use_linear(use_linear);

    // mark appropriate edges for integrate_in_dir
    simplex[0].edgs[0].dir = 1; simplex[0].edgs[2].dir = 2;
    simplex[1].edgs[0].dir = 3; simplex[1].edgs[2].dir = 0;

    simplex[0].construct_domain(phi, acn, clr);
    simplex[1].construct_domain(phi, acn, clr);
  }

}

double cube2_mls_quadratic_t::integrate_over_domain(CF_2 &f)
{
  switch (loc){
    case INS: // use quadrature formula for rectangles
      {
        double alpha = 1./sqrt(3.);

        double F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0) );
        double F1 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*(-alpha+1.)*(y1-y0) );
        double F2 = f( x0+.5*(-alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0) );
        double F3 = f( x0+.5*( alpha+1.)*(x1-x0), y0+.5*( alpha+1.)*(y1-y0) );

        return (x1-x0)*(y1-y0)*(F0+F1+F2+F3)/4.0;
      } break;
    case OUT: return 0.; break;
    case FCE:
      {
        return simplex[0].integrate_over_domain(f)
             + simplex[1].integrate_over_domain(f);

      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0.;
  }
}

double cube2_mls_quadratic_t::integrate_over_interface(CF_2 &f, int num)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_interface(f, num)
         + simplex[1].integrate_over_interface(f, num);
  }
  else
    return 0.0;
}

double cube2_mls_quadratic_t::integrate_over_colored_interface(CF_2 &f, int num0, int num1)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_colored_interface(f, num0, num1)
         + simplex[1].integrate_over_colored_interface(f, num0, num1);
  }
  else
    return 0.0;
}

double cube2_mls_quadratic_t::integrate_over_intersection(CF_2 &f, int num0, int num1)
{
  if (loc == FCE && num_of_lsfs > 1)
  {
    return simplex[0].integrate_over_intersection(f, num0, num1)
         + simplex[1].integrate_over_intersection(f, num0, num1);
  }
  else
    return 0.0;
}

double cube2_mls_quadratic_t::integrate_in_dir(CF_2 &f, int dir)
{
  switch (loc){
    case INS:
      {
        double alpha = 1./sqrt(3.);
        double F0 = 1, F1 = 1, s = 0;
        switch (dir){
          case 0: s = (y1-y0);
            F0 = f( x0, y0+.5*(-alpha+1.)*(y1-y0) );
            F1 = f( x0, y0+.5*( alpha+1.)*(y1-y0) );
            break;
          case 1: s = (y1-y0);
            F0 = f( x1, y0+.5*(-alpha+1.)*(y1-y0) );
            F1 = f( x1, y0+.5*( alpha+1.)*(y1-y0) );
            break;
          case 2: s = (x1-x0);
            F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y0 );
            F1 = f( x0+.5*( alpha+1.)*(x1-x0), y0 );
            break;
          case 3: s = (x1-x0);
            F0 = f( x0+.5*(-alpha+1.)*(x1-x0), y1 );
            F1 = f( x0+.5*( alpha+1.)*(x1-x0), y1 );
            break;
        }

        return s*0.5*(F0+F1);
      } break;
    case OUT:
      return 0; break;
    case FCE:
      {
        return simplex[0].integrate_in_dir(f, dir)
             + simplex[1].integrate_in_dir(f, dir);
      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0.;
  }
#ifdef CASL_THROWS
  throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
  return 0.;
}

//double cube2_mls_quadratic_t::measure_of_domain()
//{
//  switch (loc){
//    case INS:
//      {
//        return (x1-x0)*(y1-y0);
//      } break;
//    case OUT: return 0.; break;
//    case FCE:
//      {
//        std::vector<double> F(n_nodes_simplex, 1);

//        return simplex[0].integrate_over_domain(F)
//             + simplex[1].integrate_over_domain(F);

//      } break;
//    default:
//#ifdef CASL_THROWS
//      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0.;
//  }
//}

//double cube2_mls_quadratic_t::measure_of_interface(int num)
//{
//  if (loc == FCE)
//  {
//    std::vector<double> F(n_nodes_simplex, 1);
//    return simplex[0].integrate_over_interface(F, num)
//         + simplex[1].integrate_over_interface(F, num);
//  }
//  else
//    return 0.0;
//}

//double cube2_mls_quadratic_t::measure_of_colored_interface(int num0, int num1)
//{
//  if (loc == FCE)
//  {
//    std::vector<double> F(n_nodes_simplex, 1);
//    return simplex[0].integrate_over_colored_interface(F, num0, num1)
//         + simplex[1].integrate_over_colored_interface(F, num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube2_mls_quadratic_t::measure_of_intersection(int num0, int num1)
//{
//  if (loc == FCE && num_of_lsfs > 1)
//  {
//    std::vector<double> F(n_nodes_simplex, 1);
//    return simplex[0].integrate_over_intersection(F, num0, num1)
//         + simplex[1].integrate_over_intersection(F, num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube2_mls_quadratic_t::measure_in_dir(int dir)
//{
//  switch (loc){
//    case INS:
//      {
//        switch (dir){
//          case 0: return (y1-y0); break;
//          case 1: return (y1-y0); break;
//          case 2: return (x1-x0); break;
//          case 3: return (x1-x0); break;
//        }
//      } break;
//    case OUT:
//      return 0; break;
//    case FCE:
//      {
//        std::vector<double> F(n_nodes_simplex, 1);
//        return simplex[0].integrate_in_dir(F, dir)
//             + simplex[1].integrate_in_dir(F, dir);
//      } break;
//    default:
//#ifdef CASL_THROWS
//      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0.;
//  }
//#ifdef CASL_THROWS
//  throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//  return 0.;
//}

//double cube2_mls_quadratic_t::integrate_in_non_cart_dir(double *f, int dir)
//{
//  if (loc == FCE)
//  {
//    double F[4];
//    F[0] = interp.linear(f,x0,y0);
//    F[1] = interp.linear(f,x1,y0);
//    F[2] = interp.linear(f,x0,y1);
//    F[3] = interp.linear(f,x1,y1);
//    return simplex[0].integrate_in_non_cart_dir(F[t0p0], F[t0p1], F[t0p2], dir)
//         + simplex[1].integrate_in_non_cart_dir(F[t1p0], F[t1p1], F[t1p2], dir);
//  }
//  else
//    return 0.0;
//}

//double cube2_mls_t::interpolate_linear(double *f, double x, double y)
//{
//  double d_m00 = (x-x0)/(x1-x0);
//  double d_p00 = (x1-x)/(x1-x0);
//  double d_0m0 = (y-y0)/(y1-y0);
//  double d_0p0 = (y1-y)/(y1-y0);

//  double w_mm0 = d_p00*d_0p0;
//  double w_pm0 = d_m00*d_0p0;
//  double w_mp0 = d_p00*d_0m0;
//  double w_pp0 = d_m00*d_0m0;

//  return (w_mm0*f[0] + w_pm0*f[1] + w_mp0*f[2] + w_pp0*f[3]);
//}

//double cube2_mls_t::interpolate_quadratic(double *f, double *fxx, double *fyy, double x, double y)
//{
//  double d_m00 = (x-x0)/(x1-x0);
//  double d_p00 = (x1-x)/(x1-x0);
//  double d_0m0 = (y-y0)/(y1-y0);
//  double d_0p0 = (y1-y)/(y1-y0);

//  double w_mm0 = d_p00*d_0p0;
//  double w_pm0 = d_m00*d_0p0;
//  double w_mp0 = d_p00*d_0m0;
//  double w_pp0 = d_m00*d_0m0;

//  double Fxx = w_mm0*fxx[0] + w_pm0*fxx[1] + w_mp0*fxx[2] + w_pp0*fxx[3];
//  double Fyy = w_mm0*fyy[0] + w_pm0*fyy[1] + w_mp0*fyy[2] + w_pp0*fyy[3];

//  double F = w_mm0*f[0] + w_pm0*f[1] + w_mp0*f[2] + w_pp0*f[3]
//             - 0.5*((x1-x0)*(x1-x0)*d_p00*d_m00*Fxx + (y1-y0)*(y1-y0)*d_0p0*d_0m0*Fyy);
//  return F;
//}

double cube2_mls_quadratic_t::interpolate_quad(std::vector<double> f, double X, double Y)
{
  // Note that X and Y are scaled coordinates, i.e. in [-1,1]

//  double X = 2.*(x-x0)/(x1-x0) - 1.;
//  double Y = 2.*(y-y0)/(y1-y0) - 1.;

  std::vector<double> wx(3, 0);
  std::vector<double> wy(3, 0);

  wx[0] = .5*X*(X-1.);
  wx[1] = -(X*X-1.);
  wx[2] = .5*X*(X+1.);

  wy[0] = .5*Y*(Y-1.);
  wy[1] = -(Y*Y-1.);
  wy[2] = .5*Y*(Y+1.);

  double result = 0;

  for (short j = 0; j < 3; ++j)
    for (short i = 0; i < 3; ++i)
      result += f[j*3+i]*wx[i]*wy[j];

  return result;
}
