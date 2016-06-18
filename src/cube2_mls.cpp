#include "cube2_mls.h"

void cube2_mls_t::construct_domain()
{
  bool use_linear = false;
  if (phi_xx == NULL || phi_yy == NULL) use_linear = true;

  bool all_positive, all_negative;

  std::vector<int>      non_trivial;
  std::vector<action_t> non_trivial_action;
  std::vector<int>      non_trivial_color;

  // TO ADD: check if sizes of action, color, phi, phi_dd do coincide

  /* Eliminate unnecessary splitting */
  loc = INS;
  for (int i = 0; i < action->size(); i++)
  {
    all_negative = true;
    all_positive = true;

    for (int j = 0; j < 4; j++)
    {
      all_negative = (all_negative && (phi->at(i)[j] < 0.0));
      all_positive = (all_positive && (phi->at(i)[j] > 0.0));
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
//        for (int j = 0; j < color.size(); j++)
//          non_trivial_color[j] = color[i];
        non_trivial.push_back(i);
        non_trivial_action.push_back(action->at(i));
        non_trivial_color.push_back(color->at(i));
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
    if (non_trivial_action[0] == ADDITION) // the first action always has to be INTERSECTION
      non_trivial_action[0] = INTERSECTION;

    /* Split the cube into 2 simplices */
    double x[4] = {x0, x1, x0, x1}; double y[4] = {y0, y0, y1, y1};

    simplex.clear();
    simplex.reserve(2);
    simplex.push_back(simplex2_mls_t(x[t0p0], y[t0p0], x[t0p1], y[t0p1], x[t0p2], y[t0p2]));
    simplex.push_back(simplex2_mls_t(x[t1p0], y[t1p0], x[t1p1], y[t1p1], x[t1p2], y[t1p2]));

    // TODO: mark appropriate edges for integrate_in_dir
    simplex[0].edgs[0].dir = 1; simplex[0].edgs[2].dir = 2;
    simplex[1].edgs[0].dir = 3; simplex[1].edgs[2].dir = 0;

    /* Apply non trivial actions to every simplex */
    for (int j = 0; j < num_non_trivial; j++)
    {
      int i_phi = non_trivial[j];
//      int s = non_trivial[j]*4;

      for (int k = 0; k < 2; k++) // loop over simplices
      {
        int n_vtxs = simplex[k].vtxs.size();

        // fetch main nodes
        switch (k)
        {
        case 0:
          simplex[k].vtxs[0].value = phi->at(i_phi)[t0p0];
          simplex[k].vtxs[1].value = phi->at(i_phi)[t0p1];
          simplex[k].vtxs[2].value = phi->at(i_phi)[t0p2];
          break;
        case 1:
          simplex[k].vtxs[0].value = phi->at(i_phi)[t1p0];
          simplex[k].vtxs[1].value = phi->at(i_phi)[t1p1];
          simplex[k].vtxs[2].value = phi->at(i_phi)[t1p2];
          break;
        }

        // interpolate to the rest of vertices
        if (use_linear)
        {
          // vertices
          for (int i_vtx = 3; i_vtx < n_vtxs; i_vtx++)
            simplex[k].vtxs[i_vtx].value = interpolate_linear(phi->at(i_phi).data(), simplex[k].vtxs[i_vtx].x, simplex[k].vtxs[i_vtx].y);

          // edges
          double xyz[2] = {0.,0.};
          for (int i_edg = 0; i_edg < simplex[k].edgs.size(); i_edg++)
            if (!simplex[k].edgs[i_edg].is_split)
            {
              simplex[k].get_edge_coords(i_edg,xyz);
              simplex[k].edgs[i_edg].value = interpolate_linear(phi->at(i_phi).data(), xyz[0], xyz[1]);
            }
        } else {
          // vertices
          for (int i_vtx = 3; i_vtx < n_vtxs; i_vtx++)
            simplex[k].vtxs[i_vtx].value = interpolate_quadratic(phi->at(i_phi).data(),
                                                                 phi_xx->at(i_phi).data(),
                                                                 phi_yy->at(i_phi).data(),
                                                                 simplex[k].vtxs[i_vtx].x,
                                                                 simplex[k].vtxs[i_vtx].y);

          // edges
          double xyz[2] = {0.,0.};
          for (int i_edg = 0; i_edg < simplex[k].edgs.size(); i_edg++)
            if (!simplex[k].edgs[i_edg].is_split)
            {
              simplex[k].get_edge_coords(i_edg,xyz);
              simplex[k].edgs[i_edg].value = interpolate_quadratic(phi->at(i_phi).data(),
                                                                   phi_xx->at(i_phi).data(),
                                                                   phi_yy->at(i_phi).data(),
                                                                   xyz[0], xyz[1]);
            }
        }
        simplex[k].do_action(non_trivial_color[j], non_trivial_action[j]);
      }
    }

  }

}

double cube2_mls_t::integrate_over_domain(double* f)
{
  switch (loc){
  case INS: return (x1-x0)*(y1-y0)*(f[0]+f[1]+f[2]+f[3])/4.0;           break;
  case OUT: return 0.0;                                                         break;
  case FCE: return simplex[0].integrate_over_domain(f[t0p0], f[t0p1], f[t0p2])
                 + simplex[1].integrate_over_domain(f[t1p0], f[t1p1], f[t1p2]); break;
  }
}

double cube2_mls_t::integrate_over_interface(double *f, int num)
{
  if (loc == FCE)
    return simplex[0].integrate_over_interface(f[t0p0], f[t0p1], f[t0p2], num)
         + simplex[1].integrate_over_interface(f[t1p0], f[t1p1], f[t1p2], num);
  else
    return 0.0;
}

double cube2_mls_t::integrate_over_colored_interface(double *f, int num0, int num1)
{
  if (loc == FCE)
    return simplex[0].integrate_over_colored_interface(f[t0p0], f[t0p1], f[t0p2], num0, num1)
         + simplex[1].integrate_over_colored_interface(f[t1p0], f[t1p1], f[t1p2], num0, num1);
  else
    return 0.0;
}

double cube2_mls_t::integrate_over_intersection(double *f, int num0, int num1)
{
  if (loc == FCE && num_non_trivial > 1)
    return simplex[0].integrate_over_intersection(f[t0p0], f[t0p1], f[t0p2], num0, num1)
         + simplex[1].integrate_over_intersection(f[t1p0], f[t1p1], f[t1p2], num0, num1);
  else
    return 0.0;
}

double cube2_mls_t::integrate_in_dir(double *f, int dir)
{
  switch (loc){
  case INS:
    switch (dir){
    case 0: return (y1-y0)*0.5*(f[0]+f[2]); break;
    case 1: return (y1-y0)*0.5*(f[1]+f[3]); break;
    case 2: return (x1-x0)*0.5*(f[0]+f[1]); break;
    case 3: return (x1-x0)*0.5*(f[2]+f[3]); break;
    }
    break;
  case OUT:
    return 0; break;
  case FCE:
    return simplex[0].integrate_in_dir(f[t0p0], f[t0p1], f[t0p2], dir)
         + simplex[1].integrate_in_dir(f[t1p0], f[t1p1], f[t1p2], dir);
  }
}

double cube2_mls_t::integrate_in_non_cart_dir(double *f, int dir)
{
  if (loc == FCE)
    return simplex[0].integrate_in_non_cart_dir(f[t0p0], f[t0p1], f[t0p2], dir)
         + simplex[1].integrate_in_non_cart_dir(f[t1p0], f[t1p1], f[t1p2], dir);
  else
    return 0.0;
}

double cube2_mls_t::interpolate_linear(double *f, double x, double y)
{
  double d_m00 = (x-x0)/(x1-x0);
  double d_p00 = (x1-x)/(x1-x0);
  double d_0m0 = (y-y0)/(y1-y0);
  double d_0p0 = (y1-y)/(y1-y0);

  double w_mm0 = d_p00*d_0p0;
  double w_pm0 = d_m00*d_0p0;
  double w_mp0 = d_p00*d_0m0;
  double w_pp0 = d_m00*d_0m0;

  return (w_mm0*f[0] + w_pm0*f[1] + w_mp0*f[2] + w_pp0*f[3]);
}

double cube2_mls_t::interpolate_quadratic(double *f, double *fxx, double *fyy, double x, double y)
{
  double d_m00 = (x-x0)/(x1-x0);
  double d_p00 = (x1-x)/(x1-x0);
  double d_0m0 = (y-y0)/(y1-y0);
  double d_0p0 = (y1-y)/(y1-y0);

  double w_mm0 = d_p00*d_0p0;
  double w_pm0 = d_m00*d_0p0;
  double w_mp0 = d_p00*d_0m0;
  double w_pp0 = d_m00*d_0m0;

  double Fxx = w_mm0*fxx[0] + w_pm0*fxx[1] + w_mp0*fxx[2] + w_pp0*fxx[3];
  double Fyy = w_mm0*fyy[0] + w_pm0*fyy[1] + w_mp0*fyy[2] + w_pp0*fyy[3];

  double F = w_mm0*f[0] + w_pm0*f[1] + w_mp0*f[2] + w_pp0*f[3]
             - 0.5*((x1-x0)*(x1-x0)*d_p00*d_m00*Fxx + (y1-y0)*(y1-y0)*d_0p0*d_0m0*Fyy);
  return F;
}
