#include "cube3_mls.h"

void cube3_mls_t::construct_domain(double *phi, std::vector<action_t> &action, std::vector<int> &color)
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

    for (int j = 0; j < 8; j++)
    {
      all_negative = (all_negative && (phi[i*8+j] < 0.0));
      all_positive = (all_positive && (phi[i*8+j] > 0.0));
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
        for (int j = 0; j < color.size(); j++)
          non_trivial_color[j] = color[i];
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

    /* Apply non trivial actions to every simplices */
    for (int j = 0; j < num_non_trivial; j++)
    {
      int s = non_trivial[j]*8;
      simplex[0].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t0p0], phi[s+t0p1], phi[s+t0p2], phi[s+t0p3]);
      simplex[1].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t1p0], phi[s+t1p1], phi[s+t1p2], phi[s+t1p3]);
      simplex[2].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t2p0], phi[s+t2p1], phi[s+t2p2], phi[s+t2p3]);
      simplex[3].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t3p0], phi[s+t3p1], phi[s+t3p2], phi[s+t3p3]);
      simplex[4].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t4p0], phi[s+t4p1], phi[s+t4p2], phi[s+t4p3]);
#ifdef CUBE3_MLS_KUHN
      simplex[5].do_action(non_trivial_color[j], non_trivial_action[j], phi[s+t5p0], phi[s+t5p1], phi[s+t5p2], phi[s+t5p3]);
#endif
    }
  }
}

double cube3_mls_t::integrate_over_domain(double* f)
{
  switch (loc){
  case INS: return (x1-x0)*(y1-y0)*(z1-z0)*(f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7])/8.0;   break;
  case OUT: return 0.0;                                                                     break;
  case FCE: return simplex[0].integrate_over_domain(f[t0p0], f[t0p1], f[t0p2], f[t0p3])
                 + simplex[1].integrate_over_domain(f[t1p0], f[t1p1], f[t1p2], f[t1p3])
                 + simplex[2].integrate_over_domain(f[t2p0], f[t2p1], f[t2p2], f[t2p3])
                 + simplex[3].integrate_over_domain(f[t3p0], f[t3p1], f[t3p2], f[t3p3])
          #ifdef CUBE3_MLS_KUHN
                 + simplex[5].integrate_over_domain(f[t5p0], f[t5p1], f[t5p2], f[t5p3])
          #endif
                 + simplex[4].integrate_over_domain(f[t4p0], f[t4p1], f[t4p2], f[t4p3]);    break;
  }
}

double cube3_mls_t::integrate_over_interface(double *f, int num)
{
  if (loc == FCE)
    return simplex[0].integrate_over_interface(num, f[t0p0], f[t0p1], f[t0p2], f[t0p3]) +
           simplex[1].integrate_over_interface(num, f[t1p0], f[t1p1], f[t1p2], f[t1p3]) +
           simplex[2].integrate_over_interface(num, f[t2p0], f[t2p1], f[t2p2], f[t2p3]) +
           simplex[3].integrate_over_interface(num, f[t3p0], f[t3p1], f[t3p2], f[t3p3]) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_interface(num, f[t5p0], f[t5p1], f[t5p2], f[t5p3]) +
#endif
           simplex[4].integrate_over_interface(num, f[t4p0], f[t4p1], f[t4p2], f[t4p3]);
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_colored_interface(double *f, int num0, int num1)
{
  if (loc == FCE)
    return simplex[0].integrate_over_colored_interface(num0, num1, f[t0p0], f[t0p1], f[t0p2], f[t0p3]) +
           simplex[1].integrate_over_colored_interface(num0, num1, f[t1p0], f[t1p1], f[t1p2], f[t1p3]) +
           simplex[2].integrate_over_colored_interface(num0, num1, f[t2p0], f[t2p1], f[t2p2], f[t2p3]) +
           simplex[3].integrate_over_colored_interface(num0, num1, f[t3p0], f[t3p1], f[t3p2], f[t3p3]) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_colored_interface(num0, num1, f[t5p0], f[t5p1], f[t5p2], f[t5p3]) +
#endif
           simplex[4].integrate_over_colored_interface(num0, num1, f[t4p0], f[t4p1], f[t4p2], f[t4p3]);
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_intersection(double *f, int num0, int num1)
{
  if (loc == FCE && num_non_trivial > 1)
    return simplex[0].integrate_over_intersection(num0, num1, f[t0p0], f[t0p1], f[t0p2], f[t0p3]) +
           simplex[1].integrate_over_intersection(num0, num1, f[t1p0], f[t1p1], f[t1p2], f[t1p3]) +
           simplex[2].integrate_over_intersection(num0, num1, f[t2p0], f[t2p1], f[t2p2], f[t2p3]) +
           simplex[3].integrate_over_intersection(num0, num1, f[t3p0], f[t3p1], f[t3p2], f[t3p3]) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_intersection(num0, num1, f[t5p0], f[t5p1], f[t5p2], f[t5p3]) +
#endif
           simplex[4].integrate_over_intersection(num0, num1, f[t4p0], f[t4p1], f[t4p2], f[t4p3]);
  else
    return 0.0;
}

double cube3_mls_t::integrate_over_intersection(double *f, int num0, int num1, int num2)
{
  if (loc == FCE && num_non_trivial > 2)
    return simplex[0].integrate_over_intersection(num0, num1, num2, f[t0p0], f[t0p1], f[t0p2], f[t0p3]) +
           simplex[1].integrate_over_intersection(num0, num1, num2, f[t1p0], f[t1p1], f[t1p2], f[t1p3]) +
           simplex[2].integrate_over_intersection(num0, num1, num2, f[t2p0], f[t2p1], f[t2p2], f[t2p3]) +
           simplex[3].integrate_over_intersection(num0, num1, num2, f[t3p0], f[t3p1], f[t3p2], f[t3p3]) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].integrate_over_intersection(num0, num1, num2, f[t5p0], f[t5p1], f[t5p2], f[t5p3]) +
#endif
           simplex[4].integrate_over_intersection(num0, num1, num2, f[t4p0], f[t4p1], f[t4p2], f[t4p3]);
  else
    return 0.0;
}

double cube3_mls_t::integrate_in_dir(double *f, int dir)
{
  return simplex[0].integrate_in_dir(dir, f[t0p0], f[t0p1], f[t0p2], f[t0p3]) +
         simplex[1].integrate_in_dir(dir, f[t1p0], f[t1p1], f[t1p2], f[t1p3]) +
         simplex[2].integrate_in_dir(dir, f[t2p0], f[t2p1], f[t2p2], f[t2p3]) +
         simplex[3].integrate_in_dir(dir, f[t3p0], f[t3p1], f[t3p2], f[t3p3]) +
#ifdef CUBE3_MLS_KUHN
         simplex[5].integrate_in_dir(dir, f[t5p0], f[t5p1], f[t5p2], f[t5p3]) +
#endif
         simplex[4].integrate_in_dir(dir, f[t4p0], f[t4p1], f[t4p2], f[t4p3]);
}

double cube3_mls_t::measure_of_domain()
{
  switch (loc){
  case INS: return (x1-x0)*(y1-y0)*(z1-z0);         break;
  case OUT: return 0.0;                             break;
  case FCE: return simplex[0].measure_of_domain()
                 + simplex[1].measure_of_domain()
                 + simplex[2].measure_of_domain()
                 + simplex[3].measure_of_domain()
          #ifdef CUBE3_MLS_KUHN
                 + simplex[5].measure_of_domain()
          #endif
                 + simplex[4].measure_of_domain();  break;
  }
}

double cube3_mls_t::measure_of_interface(int num)
{
  if (loc == FCE)
    return simplex[0].measure_of_interface(num) +
           simplex[1].measure_of_interface(num) +
           simplex[2].measure_of_interface(num) +
           simplex[3].measure_of_interface(num) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].measure_of_interface(num) +
#endif
           simplex[4].measure_of_interface(num);
  else
    return 0.0;
}

double cube3_mls_t::measure_of_colored_interface(int num0, int num1)
{
  if (loc == FCE)
    return simplex[0].measure_of_colored_interface(num0, num1) +
           simplex[1].measure_of_colored_interface(num0, num1) +
           simplex[2].measure_of_colored_interface(num0, num1) +
           simplex[3].measure_of_colored_interface(num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].measure_of_colored_interface(num0, num1) +
#endif
           simplex[4].measure_of_colored_interface(num0, num1);
  else
    return 0.0;
}

double cube3_mls_t::measure_of_intersection(int num0, int num1)
{
  if (loc == FCE && num_non_trivial > 1)
    return simplex[0].measure_of_intersection(num0, num1) +
           simplex[1].measure_of_intersection(num0, num1) +
           simplex[2].measure_of_intersection(num0, num1) +
           simplex[3].measure_of_intersection(num0, num1) +
#ifdef CUBE3_MLS_KUHN
           simplex[5].measure_of_intersection(num0, num1) +
#endif
           simplex[4].measure_of_intersection(num0, num1);
  else
    return 0.0;
}

double cube3_mls_t::measure_in_dir(int dir)
{
  switch (loc){
  case OUT: return 0;
  case INS:
    switch (dir) {
    case 0: case 1: return (y1-y0)*(z1-z0);
    case 2: case 3: return (x1-x0)*(z1-z0);
    case 4: case 5: return (x1-x0)*(y1-y0);
    }
  case FCE:
  return simplex[0].measure_in_dir(dir) +
         simplex[1].measure_in_dir(dir) +
         simplex[2].measure_in_dir(dir) +
         simplex[3].measure_in_dir(dir) +
#ifdef CUBE3_MLS_KUHN
         simplex[5].measure_in_dir(dir) +
#endif
         simplex[4].measure_in_dir(dir);
  }
}
