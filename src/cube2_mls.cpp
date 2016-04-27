#include "cube2_mls.h"

void cube2_mls_t::construct_domain(double *phi, std::vector<action_t> &action, std::vector<int> &color)
{
  bool all_positive, all_negative;

  std::vector<int>      non_trivial;
  std::vector<action_t> non_trivial_action;
  std::vector<int>      non_trivial_color;

  /* Eliminate unnecessary splitting */
  loc = INS;
  for (int i = 0; i < action.size(); i++)
  {
    all_negative = true;
    all_positive = true;

    for (int j = 0; j < 4; j++)
    {
      all_negative = (all_negative && (phi[i*4+j] < 0.0));
      all_positive = (all_positive && (phi[i*4+j] > 0.0));
    }

    if (all_positive)
    {
      if (action[i] == INTERSECTION)
      {
        loc = OUT;
        non_trivial.clear();
        non_trivial_action.clear();
        non_trivial_color.clear();
      }
    }
    else if (all_negative)
    {
      if (action[i] == ADDITION)
      {
        loc = INS;
        non_trivial.clear();
        non_trivial_action.clear();
        non_trivial_color.clear();
      }
      else if (action[i] == COLORATION && loc == FCE)
      {
//        for (int j = 0; j < color.size(); j++)
//          non_trivial_color[j] = color[i];
        non_trivial.push_back(i);
        non_trivial_action.push_back(action[i]);
        non_trivial_color.push_back(color[i]);
      }
    }
    else if (loc == FCE || (loc == INS && action[i] == INTERSECTION) || (loc == OUT && action[i] == ADDITION))
    {
      loc = FCE;
      non_trivial.push_back(i);
      non_trivial_action.push_back(action[i]);
      non_trivial_color.push_back(color[i]);
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
      int s = non_trivial[j]*4;
      simplex[0].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t0p0], phi[s+t0p1], phi[s+t0p2]);
      simplex[1].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t1p0], phi[s+t1p1], phi[s+t1p2]);
    }

//    if (measure_of_domain() < 1.e-15) loc = OUT;
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
    return simplex[0].integrate_over_interface(num, f[t0p0], f[t0p1], f[t0p2])
         + simplex[1].integrate_over_interface(num, f[t1p0], f[t1p1], f[t1p2]);
  else
    return 0.0;
}

double cube2_mls_t::integrate_over_colored_interface(double *f, int num0, int num1)
{
  if (loc == FCE)
    return simplex[0].integrate_over_colored_interface(num0, num1, f[t0p0], f[t0p1], f[t0p2])
         + simplex[1].integrate_over_colored_interface(num0, num1, f[t1p0], f[t1p1], f[t1p2]);
  else
    return 0.0;
}

double cube2_mls_t::integrate_over_intersection(double *f, int num0, int num1)
{
  if (loc == FCE && num_non_trivial > 1)
    return simplex[0].integrate_over_intersection(num0, num1, f[t0p0], f[t0p1], f[t0p2])
         + simplex[1].integrate_over_intersection(num0, num1, f[t1p0], f[t1p1], f[t1p2]);
  else
    return 0.0;
}

double cube2_mls_t::integrate_in_dir(double *f, int dir)
{
  return simplex[0].integrate_in_dir(dir, f[t0p0], f[t0p1], f[t0p2])
       + simplex[1].integrate_in_dir(dir, f[t1p0], f[t1p1], f[t1p2]);
}

double cube2_mls_t::measure_of_domain()
{
  switch (loc){
  case INS: return (x1-x0)*(y1-y0);                break;
  case OUT: return 0.0;                            break;
  case FCE: return simplex[0].measure_of_domain()
                 + simplex[1].measure_of_domain(); break;
  }
}

double cube2_mls_t::measure_of_interface(int num)
{
  if (loc == FCE)
    return simplex[0].measure_of_interface(num)
         + simplex[1].measure_of_interface(num);
  else
    return 0.0;
}

double cube2_mls_t::measure_of_colored_interface(int num0, int num1)
{
  if (loc == FCE)
    return simplex[0].measure_of_colored_interface(num0, num1)
         + simplex[1].measure_of_colored_interface(num0, num1);
  else
    return 0.0;
}

double cube2_mls_t::measure_in_dir(int dir)
{
  switch (loc){
  case INS:
    switch (dir){
    case 0: case 1: return y1-y0; break;
    case 2: case 3: return x1-x0; break;
    }
    break;
  case OUT:
    return 0; break;
  case FCE:
    return simplex[0].measure_in_dir(dir)
         + simplex[1].measure_in_dir(dir); break;
  }
}
