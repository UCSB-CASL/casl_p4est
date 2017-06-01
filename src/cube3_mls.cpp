#include "cube3_mls.h"

void cube3_mls_t::construct_domain(std::vector<CF_3 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  bool all_positive, all_negative;

   num_of_lsfs = phi.size();

 #ifdef CASL_THROWS
   if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
           throw std::domain_error("[CASL_ERROR]: (cube3_mls_quadratic_t::construct_domain) sizes of phi, acn and clr do not coincide.");
 #endif

   /* Eliminate unnecessary splitting */
   loc = INS;
   for (unsigned int p = 0; p < num_of_lsfs; p++)
   {
     all_negative = true;
     all_positive = true;

     double x[2] = { x0, x1 };
     double y[2] = { y0, y1 };
     double z[2] = { z0, z1 };

     for (short k = 0; k < 2; ++k)
       for (short j = 0; j < 2; ++j)
         for (short i = 0; i < 2; ++i)
         {
           double value = (*phi[p]) ( x[i], y[j], z[k] );
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
    double x[8] = {x0,x1,x0,x1,x0,x1,x0,x1};
    double y[8] = {y0,y0,y1,y1,y0,y0,y1,y1};
    double z[8] = {z0,z0,z0,z0,z1,z1,z1,z1};

    /* Split a cube into 5 simplices */
    simplex.clear();
    simplex.reserve(NTETS);
    simplex.push_back(simplex3_mls_t(x[t0p0],y[t0p0],z[t0p0], x[t0p1],y[t0p1],z[t0p1], x[t0p2],y[t0p2],z[t0p2], x[t0p3],y[t0p3],z[t0p3])); //simplex.back().set_use_linear(use_linear);
    simplex.push_back(simplex3_mls_t(x[t1p0],y[t1p0],z[t1p0], x[t1p1],y[t1p1],z[t1p1], x[t1p2],y[t1p2],z[t1p2], x[t1p3],y[t1p3],z[t1p3])); //simplex.back().set_use_linear(use_linear);
    simplex.push_back(simplex3_mls_t(x[t2p0],y[t2p0],z[t2p0], x[t2p1],y[t2p1],z[t2p1], x[t2p2],y[t2p2],z[t2p2], x[t2p3],y[t2p3],z[t2p3])); //simplex.back().set_use_linear(use_linear);
    simplex.push_back(simplex3_mls_t(x[t3p0],y[t3p0],z[t3p0], x[t3p1],y[t3p1],z[t3p1], x[t3p2],y[t3p2],z[t3p2], x[t3p3],y[t3p3],z[t3p3])); //simplex.back().set_use_linear(use_linear);
    simplex.push_back(simplex3_mls_t(x[t4p0],y[t4p0],z[t4p0], x[t4p1],y[t4p1],z[t4p1], x[t4p2],y[t4p2],z[t4p2], x[t4p3],y[t4p3],z[t4p3])); //simplex.back().set_use_linear(use_linear);
#ifdef CUBE3_MLS_KUHN
    simplex.push_back(simplex3_mls_t(x[t5p0],y[t5p0],z[t5p0], x[t5p1],y[t5p1],z[t5p1], x[t5p2],y[t5p2],z[t5p2], x[t5p3],y[t5p3],z[t5p3])); //simplex.back().set_use_linear(use_linear);
#endif

    // TODO: mark appropriate edges for integrate_in_dir
#ifdef CUBE3_MLS_KUHN
    simplex[0].tris[0].dir = 1; simplex[0].tris[3].dir = 4;
    simplex[1].tris[0].dir = 3; simplex[1].tris[3].dir = 4;
    simplex[2].tris[0].dir = 1; simplex[2].tris[3].dir = 2;
    simplex[3].tris[0].dir = 3; simplex[3].tris[3].dir = 0;
    simplex[4].tris[0].dir = 5; simplex[4].tris[3].dir = 2;
    simplex[5].tris[0].dir = 5; simplex[5].tris[3].dir = 0;
#endif
    // it doesn't make sense to do it for the MIDDLE_CUT triangulation

    simplex[0].construct_domain(phi, acn, clr);
    simplex[1].construct_domain(phi, acn, clr);
    simplex[2].construct_domain(phi, acn, clr);
    simplex[3].construct_domain(phi, acn, clr);
    simplex[4].construct_domain(phi, acn, clr);
    simplex[5].construct_domain(phi, acn, clr);
  }
}

double cube3_mls_t::integrate_over_domain(CF_3 &f)
{
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
        return simplex[0].integrate_over_domain(f)
            +  simplex[1].integrate_over_domain(f)
            +  simplex[2].integrate_over_domain(f)
            +  simplex[3].integrate_over_domain(f)
    #ifdef CUBE3_MLS_KUHN
            +  simplex[5].integrate_over_domain(f)
    #endif
            +  simplex[4].integrate_over_domain(f);
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

double cube3_mls_t::integrate_over_interface(CF_3& f, int num)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_interface(f, num) +
           simplex[1].integrate_over_interface(f, num) +
           simplex[2].integrate_over_interface(f, num) +
           simplex[3].integrate_over_interface(f, num) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_interface(f, num) +
#endif
           simplex[4].integrate_over_interface(f, num);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_colored_interface(CF_3& f, int num0, int num1)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_colored_interface(f, num0, num1) +
           simplex[1].integrate_over_colored_interface(f, num0, num1) +
           simplex[2].integrate_over_colored_interface(f, num0, num1) +
           simplex[3].integrate_over_colored_interface(f, num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_colored_interface(f, num0, num1) +
#endif
           simplex[4].integrate_over_colored_interface(f, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_intersection(CF_3& f, int num0, int num1)
{
  if (loc == FCE && num_of_lsfs > 1)
  {
    return simplex[0].integrate_over_intersection(f, num0, num1) +
           simplex[1].integrate_over_intersection(f, num0, num1) +
           simplex[2].integrate_over_intersection(f, num0, num1) +
           simplex[3].integrate_over_intersection(f, num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_intersection(f, num0, num1) +
#endif
           simplex[4].integrate_over_intersection(f, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_intersection(CF_3& f, int num0, int num1, int num2)
{
  if (loc == FCE && num_of_lsfs > 2)
  {
    return simplex[0].integrate_over_intersection(f, num0, num1, num2) +
           simplex[1].integrate_over_intersection(f, num0, num1, num2) +
           simplex[2].integrate_over_intersection(f, num0, num1, num2) +
           simplex[3].integrate_over_intersection(f, num0, num1, num2) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_intersection(f, num0, num1, num2) +
#endif
           simplex[4].integrate_over_intersection(f, num0, num1, num2);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_in_dir(CF_3& f, int dir)
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
    return simplex[0].integrate_in_dir(f, dir) +
           simplex[1].integrate_in_dir(f, dir) +
           simplex[2].integrate_in_dir(f, dir) +
           simplex[3].integrate_in_dir(f, dir) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_in_dir(f, dir) +
#endif
           simplex[4].integrate_in_dir(f, dir);
  }
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0;
  }
}

//void cube3_mls_t::interpolate_to_cube(double *in, double *out)
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


//double cube3_mls_t::measure_of_domain()
//{
//  switch (loc){
//  case INS:
//  {
//      return (x1-x0)*(y1-y0)*(z1-z0);
//  } break;
//  case OUT: return 0.0; break;
//  case FCE:
//  {
//    return simplex[0].integrate_over_domain(1.,1.,1.,1.)
//        +  simplex[1].integrate_over_domain(1.,1.,1.,1.)
//        +  simplex[2].integrate_over_domain(1.,1.,1.,1.)
//        +  simplex[3].integrate_over_domain(1.,1.,1.,1.)
//    #ifdef CUBE3_MLS_KUHN
//        +  simplex[5].integrate_over_domain(1.,1.,1.,1.)
//    #endif
//        +  simplex[4].integrate_over_domain(1.,1.,1.,1.);
//  } break;
//    default:
//#ifdef CASL_THROWS
//      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0;
//  }
//}

//double cube3_mls_t::measure_of_interface(int num)
//{
//  if (loc == FCE)
//  {
//    return simplex[0].integrate_over_interface(1.,1.,1.,1., num) +
//           simplex[1].integrate_over_interface(1.,1.,1.,1., num) +
//           simplex[2].integrate_over_interface(1.,1.,1.,1., num) +
//           simplex[3].integrate_over_interface(1.,1.,1.,1., num) +
//#ifdef CUBE3_MLS_KUHN
//           simplex[5].integrate_over_interface(1.,1.,1.,1., num) +
//#endif
//           simplex[4].integrate_over_interface(1.,1.,1.,1., num);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_t::measure_of_colored_interface(int num0, int num1)
//{
//  if (loc == FCE)
//  {
//    return simplex[0].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//           simplex[1].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//           simplex[2].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//           simplex[3].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//#ifdef CUBE3_MLS_KUHN
//           simplex[5].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//#endif
//           simplex[4].integrate_over_colored_interface(1.,1.,1.,1., num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_t::measure_of_intersection(int num0, int num1)
//{
//  if (loc == FCE && num_non_trivial > 1)
//  {
//    return simplex[0].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//           simplex[1].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//           simplex[2].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//           simplex[3].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//#ifdef CUBE3_MLS_KUHN
//           simplex[5].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//#endif
//           simplex[4].integrate_over_intersection(1.,1.,1.,1., num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_t::measure_in_dir(int dir)
//{
//  switch (loc){
//  case OUT: return 0;
//  case INS:
//  {
//    switch (dir) {
//    case 0: return (y1-y0)*(z1-z0);
//    case 1: return (y1-y0)*(z1-z0);
//    case 2: return (x1-x0)*(z1-z0);
//    case 3: return (x1-x0)*(z1-z0);
//    case 4: return (x1-x0)*(y1-y0);
//    case 5: return (x1-x0)*(y1-y0);
//    }
//  }
//  case FCE:
//  {
//    return simplex[0].integrate_in_dir(1.,1.,1.,1., dir) +
//           simplex[1].integrate_in_dir(1.,1.,1.,1., dir) +
//           simplex[2].integrate_in_dir(1.,1.,1.,1., dir) +
//           simplex[3].integrate_in_dir(1.,1.,1.,1., dir) +
//#ifdef CUBE3_MLS_KUHN
//           simplex[5].integrate_in_dir(1.,1.,1.,1., dir) +
//#endif
//           simplex[4].integrate_in_dir(1.,1.,1.,1., dir);
//  }
//    default:
//#ifdef CASL_THROWS
//      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif
//      return 0;
//  }
//}

//double cube3_mls_t::interpolate_linear(double *f, double x, double y, double z)
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

//double cube3_mls_t::interpolate_quadratic(double *f, double *fxx, double *fyy, double *fzz, double x, double y, double z)
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
