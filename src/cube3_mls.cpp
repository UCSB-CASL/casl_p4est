#include "cube3_mls.h"

void cube3_mls_t::construct_domain()
{
  bool use_linear = false;
  if (phi_xx == NULL || phi_yy == NULL || phi_zz == NULL) use_linear = true;

  bool all_positive, all_negative;

  std::vector<int>      non_trivial;
  std::vector<action_t> non_trivial_action;
  std::vector<int>      non_trivial_color;

  /* Eliminate unnecessary splitting */
  loc = INS;
  double F[8];
  for (int i = 0; i < action->size(); i++)
  {
    all_negative = true;
    all_positive = true;

    F[0] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x0, y0, z0);
    F[1] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x1, y0, z0);
    F[2] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x0, y1, z0);
    F[3] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x1, y1, z0);
    F[4] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x0, y0, z1);
    F[5] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x1, y0, z1);
    F[6] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x0, y1, z1);
    F[7] = interp.quadratic(phi->at(i).data(), phi_xx->at(i).data(), phi_yy->at(i).data(), phi_zz->at(i).data(), x1, y1, z1);

    for (int j = 0; j < 8; j++)
    {
      all_negative = (all_negative && (F[j] < 0.0));
      all_positive = (all_positive && (F[j] > 0.0));
    }

    if (all_positive)
    {
      if (action->at(i) == INTERSECTION)
      {
        loc = OUT;
        non_trivial.clear();
        non_trivial_action.clear();
        non_trivial_color.clear();
      }
    }
    else if (all_negative)
    {
      if (action->at(i) == ADDITION)
      {
        loc = INS;
        non_trivial.clear();
        non_trivial_action.clear();
        non_trivial_color.clear();
      }
      else if (action->at(i) == COLORATION && loc == FCE)
      {
        non_trivial.push_back(i);
        non_trivial_action.push_back(action->at(i));
        non_trivial_color.push_back(color->at(i));
//        for (int j = 0; j < color.size(); j++)
//          non_trivial_color[j] = color[i];
      }
    }
    else if (loc == FCE || (loc == INS && action->at(i) == INTERSECTION) || (loc == OUT && action->at(i) == ADDITION))
    {
      loc = FCE;
      non_trivial.push_back(i);
      non_trivial_action.push_back(action->at(i));
      non_trivial_color.push_back(color->at(i));
    }
  }

  num_non_trivial = non_trivial.size();

  if (num_non_trivial > 0)
  {

    if (non_trivial_action[0] == ADDITION) // the first action is always has to be INTERSECTION
      non_trivial_action[0] = INTERSECTION;

    double x[8] = {x0,x1,x0,x1,x0,x1,x0,x1};
    double y[8] = {y0,y0,y1,y1,y0,y0,y1,y1};
    double z[8] = {z0,z0,z0,z0,z1,z1,z1,z1};

    /* Split a cube into 5 simplices */
    simplex.clear();
    simplex.reserve(NTETS);
    simplex.push_back(simplex3_mls_t(x[t0p0],y[t0p0],z[t0p0], x[t0p1],y[t0p1],z[t0p1], x[t0p2],y[t0p2],z[t0p2], x[t0p3],y[t0p3],z[t0p3]));
    simplex.push_back(simplex3_mls_t(x[t1p0],y[t1p0],z[t1p0], x[t1p1],y[t1p1],z[t1p1], x[t1p2],y[t1p2],z[t1p2], x[t1p3],y[t1p3],z[t1p3]));
    simplex.push_back(simplex3_mls_t(x[t2p0],y[t2p0],z[t2p0], x[t2p1],y[t2p1],z[t2p1], x[t2p2],y[t2p2],z[t2p2], x[t2p3],y[t2p3],z[t2p3]));
    simplex.push_back(simplex3_mls_t(x[t3p0],y[t3p0],z[t3p0], x[t3p1],y[t3p1],z[t3p1], x[t3p2],y[t3p2],z[t3p2], x[t3p3],y[t3p3],z[t3p3]));
    simplex.push_back(simplex3_mls_t(x[t4p0],y[t4p0],z[t4p0], x[t4p1],y[t4p1],z[t4p1], x[t4p2],y[t4p2],z[t4p2], x[t4p3],y[t4p3],z[t4p3]));
#ifdef CUBE3_MLS_KUHN
    simplex.push_back(simplex3_mls_t(x[t5p0],y[t5p0],z[t5p0], x[t5p1],y[t5p1],z[t5p1], x[t5p2],y[t5p2],z[t5p2], x[t5p3],y[t5p3],z[t5p3]));
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

    /* Apply non trivial actions to every simplex */
    for (int j = 0; j < num_non_trivial; j++)
    {
      int i_phi = non_trivial[j];
//      int s = non_trivial[j]*8;

      for (int k = 0; k < NTETS; k++) // loop over simplices
      {
        int n_vtxs = simplex[k].vtxs.size();

        if (use_linear)
        {
          //vertices
          for (int i_vtx = 0; i_vtx < n_vtxs; i_vtx++)
            simplex[k].vtxs[i_vtx].value = interp.linear(phi->at(i_phi).data(),
                                                         simplex[k].vtxs[i_vtx].x, simplex[k].vtxs[i_vtx].y, simplex[k].vtxs[i_vtx].z);

          // edges
          double xyz[3] = {0.,0.,0.};
          for (int i_edg = 0; i_edg < simplex[k].edgs.size(); i_edg++)
            if (!simplex[k].edgs[i_edg].is_split)
            {
              simplex[k].get_edge_coords(i_edg,xyz);
              simplex[k].edgs[i_edg].value = interp.linear(phi->at(i_phi).data(), xyz[0], xyz[1], xyz[2]);
            }
        } else {
          //vertices
          for (int i_vtx = 0; i_vtx < n_vtxs; i_vtx++)
            simplex[k].vtxs[i_vtx].value = interp.quadratic(phi->at(i_phi).data(),
                                                            phi_xx->at(i_phi).data(),
                                                            phi_yy->at(i_phi).data(),
                                                            phi_zz->at(i_phi).data(),
                                                            simplex[k].vtxs[i_vtx].x,
                                                            simplex[k].vtxs[i_vtx].y,
                                                            simplex[k].vtxs[i_vtx].z);
          // edges
          double xyz[3] = {0.,0.,0.};
          for (int i_edg = 0; i_edg < simplex[k].edgs.size(); i_edg++)
            if (!simplex[k].edgs[i_edg].is_split)
            {
              simplex[k].get_edge_coords(i_edg,xyz);
              simplex[k].edgs[i_edg].value = interp.quadratic(phi->at(i_phi).data(),
                                                              phi_xx->at(i_phi).data(),
                                                              phi_yy->at(i_phi).data(),
                                                              phi_zz->at(i_phi).data(),
                                                              xyz[0], xyz[1], xyz[2]);
            }

        }
        simplex[k].do_action(non_trivial_color[j], non_trivial_action[j]);
      }
    }
  }
}

double cube3_mls_t::integrate_over_domain(double* F)
{
  switch (loc){
  case INS:
  {
    double f[8]; interpolate_to_cube(F, f);
      return (x1-x0)*(y1-y0)*(z1-z0)*(f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7])/8.0;
  } break;
  case OUT: return 0.0;                                                                     break;
  case FCE:
  {
    double f[8]; interpolate_to_cube(F, f);
    return simplex[0].integrate_over_domain(f[t0p0], f[t0p1], f[t0p2], f[t0p3])
        +  simplex[1].integrate_over_domain(f[t1p0], f[t1p1], f[t1p2], f[t1p3])
        +  simplex[2].integrate_over_domain(f[t2p0], f[t2p1], f[t2p2], f[t2p3])
        +  simplex[3].integrate_over_domain(f[t3p0], f[t3p1], f[t3p2], f[t3p3])
    #ifdef CUBE3_MLS_KUHN
        +  simplex[5].integrate_over_domain(f[t5p0], f[t5p1], f[t5p2], f[t5p3])
    #endif
        +  simplex[4].integrate_over_domain(f[t4p0], f[t4p1], f[t4p2], f[t4p3]);
  } break;
  }
}

double cube3_mls_t::integrate_over_interface(double *F, int num)
{
  if (loc == FCE)
  {
    double f[8]; interpolate_to_cube(F, f);
    return simplex[0].integrate_over_interface(f[t0p0], f[t0p1], f[t0p2], f[t0p3], num) +
           simplex[1].integrate_over_interface(f[t1p0], f[t1p1], f[t1p2], f[t1p3], num) +
           simplex[2].integrate_over_interface(f[t2p0], f[t2p1], f[t2p2], f[t2p3], num) +
           simplex[3].integrate_over_interface(f[t3p0], f[t3p1], f[t3p2], f[t3p3], num) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_interface(f[t5p0], f[t5p1], f[t5p2], f[t5p3], num) +
#endif
           simplex[4].integrate_over_interface(f[t4p0], f[t4p1], f[t4p2], f[t4p3], num);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_colored_interface(double *F, int num0, int num1)
{
  if (loc == FCE)
  {
    double f[8]; interpolate_to_cube(F, f);
    return simplex[0].integrate_over_colored_interface(f[t0p0], f[t0p1], f[t0p2], f[t0p3], num0, num1) +
           simplex[1].integrate_over_colored_interface(f[t1p0], f[t1p1], f[t1p2], f[t1p3], num0, num1) +
           simplex[2].integrate_over_colored_interface(f[t2p0], f[t2p1], f[t2p2], f[t2p3], num0, num1) +
           simplex[3].integrate_over_colored_interface(f[t3p0], f[t3p1], f[t3p2], f[t3p3], num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_colored_interface(f[t5p0], f[t5p1], f[t5p2], f[t5p3], num0, num1) +
#endif
           simplex[4].integrate_over_colored_interface(f[t4p0], f[t4p1], f[t4p2], f[t4p3], num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_intersection(double *F, int num0, int num1)
{
  if (loc == FCE && num_non_trivial > 1)
  {
    double f[8]; interpolate_to_cube(F, f);
    return simplex[0].integrate_over_intersection(f[t0p0], f[t0p1], f[t0p2], f[t0p3], num0, num1) +
           simplex[1].integrate_over_intersection(f[t1p0], f[t1p1], f[t1p2], f[t1p3], num0, num1) +
           simplex[2].integrate_over_intersection(f[t2p0], f[t2p1], f[t2p2], f[t2p3], num0, num1) +
           simplex[3].integrate_over_intersection(f[t3p0], f[t3p1], f[t3p2], f[t3p3], num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_intersection(f[t5p0], f[t5p1], f[t5p2], f[t5p3], num0, num1) +
#endif
           simplex[4].integrate_over_intersection(f[t4p0], f[t4p1], f[t4p2], f[t4p3], num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_intersection(double *F, int num0, int num1, int num2)
{
  if (loc == FCE && num_non_trivial > 2)
  {
    double f[8]; interpolate_to_cube(F, f);
    return simplex[0].integrate_over_intersection(f[t0p0], f[t0p1], f[t0p2], f[t0p3], num0, num1, num2) +
           simplex[1].integrate_over_intersection(f[t1p0], f[t1p1], f[t1p2], f[t1p3], num0, num1, num2) +
           simplex[2].integrate_over_intersection(f[t2p0], f[t2p1], f[t2p2], f[t2p3], num0, num1, num2) +
           simplex[3].integrate_over_intersection(f[t3p0], f[t3p1], f[t3p2], f[t3p3], num0, num1, num2) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_intersection(f[t5p0], f[t5p1], f[t5p2], f[t5p3], num0, num1, num2) +
#endif
           simplex[4].integrate_over_intersection(f[t4p0], f[t4p1], f[t4p2], f[t4p3], num0, num1, num2);
  }
  else
    return 0.0;
}

double cube3_mls_t::integrate_in_dir(double *F, int dir)
{
  switch (loc){
  case OUT: return 0;
  case INS:
  {
    double f[8]; interpolate_to_cube(F, f);
    switch (dir) {
    case 0: return (y1-y0)*(z1-z0)*0.25*(f[0]+f[2]+f[4]+f[6]);
    case 1: return (y1-y0)*(z1-z0)*0.25*(f[1]+f[3]+f[5]+f[7]);
    case 2: return (x1-x0)*(z1-z0)*0.25*(f[0]+f[1]+f[4]+f[5]);
    case 3: return (x1-x0)*(z1-z0)*0.25*(f[2]+f[3]+f[6]+f[7]);
    case 4: return (x1-x0)*(y1-y0)*0.25*(f[0]+f[1]+f[2]+f[3]);
    case 5: return (x1-x0)*(y1-y0)*0.25*(f[4]+f[5]+f[6]+f[7]);
    }
  }
  case FCE:
  {
    double f[8]; interpolate_to_cube(F, f);
    return simplex[0].integrate_in_dir(f[t0p0], f[t0p1], f[t0p2], f[t0p3], dir) +
           simplex[1].integrate_in_dir(f[t1p0], f[t1p1], f[t1p2], f[t1p3], dir) +
           simplex[2].integrate_in_dir(f[t2p0], f[t2p1], f[t2p2], f[t2p3], dir) +
           simplex[3].integrate_in_dir(f[t3p0], f[t3p1], f[t3p2], f[t3p3], dir) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_in_dir(f[t5p0], f[t5p1], f[t5p2], f[t5p3], dir) +
#endif
           simplex[4].integrate_in_dir(f[t4p0], f[t4p1], f[t4p2], f[t4p3], dir);
  }
  }
}

double cube3_mls_t::interpolate_to_cube(double *in, double *out)
{
  out[0] = interp.linear(in, x0, y0, z0);
  out[1] = interp.linear(in, x1, y0, z0);
  out[2] = interp.linear(in, x0, y1, z0);
  out[3] = interp.linear(in, x1, y1, z0);
  out[4] = interp.linear(in, x0, y0, z1);
  out[5] = interp.linear(in, x1, y0, z1);
  out[6] = interp.linear(in, x0, y1, z1);
  out[7] = interp.linear(in, x1, y1, z1);
}


double cube3_mls_t::measure_of_domain()
{
  switch (loc){
  case INS:
  {
      return (x1-x0)*(y1-y0)*(z1-z0);
  } break;
  case OUT: return 0.0; break;
  case FCE:
  {
    return simplex[0].integrate_over_domain(1.,1.,1.,1.)
        +  simplex[1].integrate_over_domain(1.,1.,1.,1.)
        +  simplex[2].integrate_over_domain(1.,1.,1.,1.)
        +  simplex[3].integrate_over_domain(1.,1.,1.,1.)
    #ifdef CUBE3_MLS_KUHN
        +  simplex[5].integrate_over_domain(1.,1.,1.,1.)
    #endif
        +  simplex[4].integrate_over_domain(1.,1.,1.,1.);
  } break;
  }
}

double cube3_mls_t::measure_of_interface(int num)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_interface(1.,1.,1.,1., num) +
           simplex[1].integrate_over_interface(1.,1.,1.,1., num) +
           simplex[2].integrate_over_interface(1.,1.,1.,1., num) +
           simplex[3].integrate_over_interface(1.,1.,1.,1., num) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_interface(1.,1.,1.,1., num) +
#endif
           simplex[4].integrate_over_interface(1.,1.,1.,1., num);
  }
  else
    return 0.0;
}

double cube3_mls_t::measure_of_colored_interface(int num0, int num1)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
           simplex[1].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
           simplex[2].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
           simplex[3].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
#endif
           simplex[4].integrate_over_colored_interface(1.,1.,1.,1., num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_t::measure_of_intersection(int num0, int num1)
{
  if (loc == FCE && num_non_trivial > 1)
  {
    return simplex[0].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
           simplex[1].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
           simplex[2].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
           simplex[3].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
#endif
           simplex[4].integrate_over_intersection(1.,1.,1.,1., num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_t::measure_in_dir(int dir)
{
  switch (loc){
  case OUT: return 0;
  case INS:
  {
    switch (dir) {
    case 0: return (y1-y0)*(z1-z0);
    case 1: return (y1-y0)*(z1-z0);
    case 2: return (x1-x0)*(z1-z0);
    case 3: return (x1-x0)*(z1-z0);
    case 4: return (x1-x0)*(y1-y0);
    case 5: return (x1-x0)*(y1-y0);
    }
  }
  case FCE:
  {
    return simplex[0].integrate_in_dir(1.,1.,1.,1., dir) +
           simplex[1].integrate_in_dir(1.,1.,1.,1., dir) +
           simplex[2].integrate_in_dir(1.,1.,1.,1., dir) +
           simplex[3].integrate_in_dir(1.,1.,1.,1., dir) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_in_dir(1.,1.,1.,1., dir) +
#endif
           simplex[4].integrate_in_dir(1.,1.,1.,1., dir);
  }
  }
}

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
