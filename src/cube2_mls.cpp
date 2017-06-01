#include "cube2_mls.h"

void cube2_mls_t::construct_domain(std::vector<CF_2 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  bool all_positive, all_negative;

  num_of_lsfs = phi.size();

#ifdef CASL_THROWS
  if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
          throw std::domain_error("[CASL_ERROR]: (cube2_mls_t::construct_domain) sizes of phi, acn and clr do not coincide.");
#endif

  /* Eliminate unnecessary splitting */
  loc = INS;
  for (unsigned int p = 0; p < num_of_lsfs; p++)
  {
    all_negative = true;
    all_positive = true;

    double x[2] = { x0, x1 };
    double y[2] = { y0, y1 };

    for (short j = 0; j < 2; ++j)
      for (short i = 0; i < 2; ++i)
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

//  num_non_trivial = non_trivial.size();

  if (loc == FCE)
  {
//    if (non_trivial_action[0] == ADDITION) // the first action always has to be INTERSECTION
//      non_trivial_action[0] = INTERSECTION;

    /* Split the cube into 2 simplices */
    double x[4] = {x0, x1, x0, x1}; double y[4] = {y0, y0, y1, y1};

    simplex.clear();
    simplex.reserve(2);
    simplex.push_back(simplex2_mls_t(x[t0p0], y[t0p0], x[t0p1], y[t0p1], x[t0p2], y[t0p2])); // simplex.back().set_use_linear(use_linear);
    simplex.push_back(simplex2_mls_t(x[t1p0], y[t1p0], x[t1p1], y[t1p1], x[t1p2], y[t1p2])); // simplex.back().set_use_linear(use_linear);

    // TODO: mark appropriate edges for integrate_in_dir
    simplex[0].edgs[0].dir = 1; simplex[0].edgs[2].dir = 2;
    simplex[1].edgs[0].dir = 3; simplex[1].edgs[2].dir = 0;

    simplex[0].construct_domain(phi, acn, clr);
    simplex[1].construct_domain(phi, acn, clr);
  }

}

double cube2_mls_t::integrate_over_domain(CF_2 &f)
{
  switch (loc){
    case INS:
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
            +  simplex[1].integrate_over_domain(f);

      } break;
    default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0.;
  }
}

double cube2_mls_t::integrate_over_interface(CF_2 &f, int num)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_interface(f, num)
         + simplex[1].integrate_over_interface(f, num);
  }
  else
    return 0.0;
}

double cube2_mls_t::integrate_over_colored_interface(CF_2 &f, int num0, int num1)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_colored_interface(f, num0, num1)
         + simplex[1].integrate_over_colored_interface(f, num0, num1);
  }
  else
    return 0.0;
}

double cube2_mls_t::integrate_over_intersection(CF_2 &f, int num0, int num1)
{
  if (loc == FCE && num_of_lsfs > 1)
  {
    return simplex[0].integrate_over_intersection(f, num0, num1)
         + simplex[1].integrate_over_intersection(f, num0, num1);
  }
  else
    return 0.0;
}

double cube2_mls_t::integrate_in_dir(CF_2 &f, int dir)
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

//double cube2_mls_t::measure_of_domain()
//{
//  switch (loc){
//  case INS:
//  {
//    return (x1-x0)*(y1-y0);
//  } break;
//  case OUT: return 0.; break;
//  case FCE:
//  {
//    return simplex[0].integrate_over_domain(1., 1., 1.)
//         + simplex[1].integrate_over_domain(1., 1., 1.);

//  } break;
//    default:
//#ifdef CASL_THROWS
//          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0.;
//  }
//}

//double cube2_mls_t::measure_of_interface(int num)
//{
//  if (loc == FCE)
//  {
//    return simplex[0].integrate_over_interface(1., 1., 1., num)
//         + simplex[1].integrate_over_interface(1., 1., 1., num);
//  }
//  else
//    return 0.0;
//}

//double cube2_mls_t::measure_of_colored_interface(int num0, int num1)
//{
//  if (loc == FCE)
//  {
//    return simplex[0].integrate_over_colored_interface(1.,1.,1., num0, num1)
//         + simplex[1].integrate_over_colored_interface(1.,1.,1., num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube2_mls_t::measure_of_intersection(int num0, int num1)
//{
//  if (loc == FCE && num_non_trivial > 1)
//  {
//    return simplex[0].integrate_over_intersection(1.,1.,1., num0, num1)
//         + simplex[1].integrate_over_intersection(1.,1.,1., num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube2_mls_t::measure_in_dir(int dir)
//{
//  switch (loc){
//  case INS:
//  {
//    switch (dir){
//    case 0: return (y1-y0); break;
//    case 1: return (y1-y0); break;
//    case 2: return (x1-x0); break;
//    case 3: return (x1-x0); break;
//    }
//  } break;
//  case OUT:
//    return 0; break;
//  case FCE:
//  {
//    return simplex[0].integrate_in_dir(1.,1.,1., dir)
//         + simplex[1].integrate_in_dir(1.,1.,1., dir);
//  } break;
//    default:
//#ifdef CASL_THROWS
//          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0.;
//  }
//#ifdef CASL_THROWS
//          throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//  return 0.;
//}

//double cube2_mls_t::integrate_in_non_cart_dir(double *f, int dir)
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
