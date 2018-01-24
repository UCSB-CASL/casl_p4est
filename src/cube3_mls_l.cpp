#include "cube3_mls_l.h"

void cube3_mls_l_t::construct_domain(std::vector<CF_3 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
   num_of_lsfs = phi.size();

   // 2-by-2-by-2 grid
   double x[2] = { x0, x1 };
   double y[2] = { y0, y1 };
   double z[2] = { z0, z1 };

 #ifdef CASL_THROWS
   if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
           throw std::domain_error("[CASL_ERROR]: (cube3_mls_l_quadratic_t::construct_domain) sizes of phi, acn and clr do not coincide.");
 #endif

   std::vector<double> phi_all(num_of_lsfs*n_nodes, -1.);

   /* Eliminate unnecessary splitting */

   loc = INS;

   std::vector<CF_3 *>   nt_phi;
   std::vector<action_t> nt_acn;
   std::vector<int>      nt_clr;
   std::vector<int>      nt_idx;

   double band = lip*diag;

   for (unsigned int p = 0; p < num_of_lsfs; p++)
   {
     bool all_negative = true;
     bool all_positive = true;

     // check how far cube is from interface
     double value = (*phi[p]) ( .5*(x0+x1), .5*(y0+y1), .5*(z0+z1) );

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
       for (short k = 0; k < n_nodes_dir; ++k)
         for (short j = 0; j < n_nodes_dir; ++j)
           for (short i = 0; i < n_nodes_dir; ++i)
           {
             double value = (*phi[p]) ( x[i], y[j], z[k] );
//             value = -1;
             phi_all[n_nodes*p + n_nodes_dir*n_nodes_dir*k + n_nodes_dir*j + i] = value;
             all_negative = (all_negative && (value < 0.));
             all_positive = (all_positive && (value > 0.));
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

//   if (nt_idx.size() > 1) loc = OUT;

   if (loc == FCE)
   {
     if (nt_acn[0] == ADDITION) nt_acn[0] = INTERSECTION;

     double x[8] = {x0,x1,x0,x1,x0,x1,x0,x1};
     double y[8] = {y0,y0,y1,y1,y0,y0,y1,y1};
     double z[8] = {z0,z0,z0,z0,z1,z1,z1,z1};

     /* Split a cube into 6 simplices */
     simplex.clear();
     simplex.reserve(NTETS);
     simplex.push_back(simplex3_mls_l_t(x[t0p0],y[t0p0],z[t0p0], x[t0p1],y[t0p1],z[t0p1], x[t0p2],y[t0p2],z[t0p2], x[t0p3],y[t0p3],z[t0p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t1p0],y[t1p0],z[t1p0], x[t1p1],y[t1p1],z[t1p1], x[t1p2],y[t1p2],z[t1p2], x[t1p3],y[t1p3],z[t1p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t2p0],y[t2p0],z[t2p0], x[t2p1],y[t2p1],z[t2p1], x[t2p2],y[t2p2],z[t2p2], x[t2p3],y[t2p3],z[t2p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t3p0],y[t3p0],z[t3p0], x[t3p1],y[t3p1],z[t3p1], x[t3p2],y[t3p2],z[t3p2], x[t3p3],y[t3p3],z[t3p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t4p0],y[t4p0],z[t4p0], x[t4p1],y[t4p1],z[t4p1], x[t4p2],y[t4p2],z[t4p2], x[t4p3],y[t4p3],z[t4p3])); //simplex.back().set_use_linear(use_linear);
#ifdef cube3_mls_l_KUHN
     simplex.push_back(simplex3_mls_l_t(x[t5p0],y[t5p0],z[t5p0], x[t5p1],y[t5p1],z[t5p1], x[t5p2],y[t5p2],z[t5p2], x[t5p3],y[t5p3],z[t5p3])); //simplex.back().set_use_linear(use_linear);
#endif

     // TODO: mark appropriate edges for integrate_in_dir
#ifdef cube3_mls_l_KUHN
     simplex[0].tris[0].dir = 1; simplex[0].tris[3].dir = 4;
     simplex[1].tris[0].dir = 3; simplex[1].tris[3].dir = 4;
     simplex[2].tris[0].dir = 1; simplex[2].tris[3].dir = 2;
     simplex[3].tris[0].dir = 3; simplex[3].tris[3].dir = 0;
     simplex[4].tris[0].dir = 5; simplex[4].tris[3].dir = 2;
     simplex[5].tris[0].dir = 5; simplex[5].tris[3].dir = 0;
#endif
     // it doesn't make sense to do it for the MIDDLE_CUT triangulation

     num_of_lsfs = nt_idx.size();

     std::vector<double> phi_s(n_nodes_simplex*num_of_lsfs, -1);

     for (int s = 0; s < NTETS; ++s)
     {
       for (int i = 0; i < num_of_lsfs; ++i)
       {
         int phi_idx = nt_idx[i];
         for (int n = 0; n < n_nodes_simplex; ++n)
           phi_s[n_nodes_simplex*i + n] = phi_all[n_nodes*phi_idx + tp_l[s][n]];
       }
       simplex[s].construct_domain(phi_s, nt_acn, nt_clr);

 //      simplex[s].construct_domain(nt_phi, nt_acn, nt_clr);
     }

//     simplex[0].construct_domain(nt_phi, nt_acn, nt_clr);
//     simplex[1].construct_domain(nt_phi, nt_acn, nt_clr);
//     simplex[2].construct_domain(nt_phi, nt_acn, nt_clr);
//     simplex[3].construct_domain(nt_phi, nt_acn, nt_clr);
//     simplex[4].construct_domain(nt_phi, nt_acn, nt_clr);
//     simplex[5].construct_domain(nt_phi, nt_acn, nt_clr);
   }
}

void cube3_mls_l_t::construct_domain(std::vector<double> &phi_all, std::vector<action_t> &acn, std::vector<int> &clr)
{
   num_of_lsfs = acn.size();

   // 2-by-2-by-2 grid
   double x[2] = { x0, x1 };
   double y[2] = { y0, y1 };
   double z[2] = { z0, z1 };

 #ifdef CASL_THROWS
   if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
           throw std::domain_error("[CASL_ERROR]: (cube3_mls_l_quadratic_t::construct_domain) sizes of phi, acn and clr do not coincide.");
 #endif

   /* Eliminate unnecessary splitting */

   loc = INS;

   std::vector<CF_3 *>   nt_phi;
   std::vector<action_t> nt_acn;
   std::vector<int>      nt_clr;
   std::vector<int>      nt_idx;

   double band = lip*diag;

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
       for (short k = 0; k < n_nodes_dir; ++k)
         for (short j = 0; j < n_nodes_dir; ++j)
           for (short i = 0; i < n_nodes_dir; ++i)
           {
             double value = phi_all[n_nodes*p + n_nodes_dir*n_nodes_dir*k + n_nodes_dir*j + i];
             all_negative = (all_negative && (value < 0.));
             all_positive = (all_positive && (value > 0.));
           }
     }

     if (all_positive)
     {
       if (acn[p] == INTERSECTION)
       {
         loc = OUT;
//         nt_phi.clear();
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
//         nt_phi.clear();
         nt_acn.clear();
         nt_clr.clear();
         nt_idx.clear();
       }
     }
     else if (loc == FCE || (loc == INS && acn[p] == INTERSECTION) || (loc == OUT && acn[p] == ADDITION))
     {
       loc = FCE;
//       nt_phi.push_back(phi[p]);
       nt_acn.push_back(acn[p]);
       nt_clr.push_back(clr[p]);
       nt_idx.push_back(p);
     }
   }

//   if (nt_idx.size() > 1) loc = OUT;

   if (loc == FCE)
   {
     if (nt_acn[0] == ADDITION) nt_acn[0] = INTERSECTION;

     double x[8] = {x0,x1,x0,x1,x0,x1,x0,x1};
     double y[8] = {y0,y0,y1,y1,y0,y0,y1,y1};
     double z[8] = {z0,z0,z0,z0,z1,z1,z1,z1};

     /* Split a cube into 6 simplices */
     simplex.clear();
     simplex.reserve(NTETS);
     simplex.push_back(simplex3_mls_l_t(x[t0p0],y[t0p0],z[t0p0], x[t0p1],y[t0p1],z[t0p1], x[t0p2],y[t0p2],z[t0p2], x[t0p3],y[t0p3],z[t0p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t1p0],y[t1p0],z[t1p0], x[t1p1],y[t1p1],z[t1p1], x[t1p2],y[t1p2],z[t1p2], x[t1p3],y[t1p3],z[t1p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t2p0],y[t2p0],z[t2p0], x[t2p1],y[t2p1],z[t2p1], x[t2p2],y[t2p2],z[t2p2], x[t2p3],y[t2p3],z[t2p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t3p0],y[t3p0],z[t3p0], x[t3p1],y[t3p1],z[t3p1], x[t3p2],y[t3p2],z[t3p2], x[t3p3],y[t3p3],z[t3p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[t4p0],y[t4p0],z[t4p0], x[t4p1],y[t4p1],z[t4p1], x[t4p2],y[t4p2],z[t4p2], x[t4p3],y[t4p3],z[t4p3])); //simplex.back().set_use_linear(use_linear);
#ifdef cube3_mls_l_KUHN
     simplex.push_back(simplex3_mls_l_t(x[t5p0],y[t5p0],z[t5p0], x[t5p1],y[t5p1],z[t5p1], x[t5p2],y[t5p2],z[t5p2], x[t5p3],y[t5p3],z[t5p3])); //simplex.back().set_use_linear(use_linear);
#endif

     // TODO: mark appropriate edges for integrate_in_dir
#ifdef cube3_mls_l_KUHN
     simplex[0].tris[0].dir = 1; simplex[0].tris[3].dir = 4;
     simplex[1].tris[0].dir = 3; simplex[1].tris[3].dir = 4;
     simplex[2].tris[0].dir = 1; simplex[2].tris[3].dir = 2;
     simplex[3].tris[0].dir = 3; simplex[3].tris[3].dir = 0;
     simplex[4].tris[0].dir = 5; simplex[4].tris[3].dir = 2;
     simplex[5].tris[0].dir = 5; simplex[5].tris[3].dir = 0;
#endif
     // it doesn't make sense to do it for the MIDDLE_CUT triangulation

     num_of_lsfs = nt_idx.size();

     std::vector<double> phi_s(n_nodes_simplex*num_of_lsfs, -1);

     for (int s = 0; s < NTETS; ++s)
     {
       for (int i = 0; i < num_of_lsfs; ++i)
       {
         int phi_idx = nt_idx[i];
         for (int n = 0; n < n_nodes_simplex; ++n)
           phi_s[n_nodes_simplex*i + n] = phi_all[n_nodes*phi_idx + tp_l[s][n]];
       }
       simplex[s].construct_domain(phi_s, nt_acn, nt_clr);
     }
   }
}

double cube3_mls_l_t::integrate_over_domain(CF_3 &f)
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
    #ifdef cube3_mls_l_KUHN
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

double cube3_mls_l_t::integrate_over_interface(CF_3& f, int num)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_interface(f, num) +
           simplex[1].integrate_over_interface(f, num) +
           simplex[2].integrate_over_interface(f, num) +
           simplex[3].integrate_over_interface(f, num) +
#ifdef cube3_mls_l_KUHN
           simplex[5].integrate_over_interface(f, num) +
#endif
           simplex[4].integrate_over_interface(f, num);
  }
  else
    return 0.0;
}

double cube3_mls_l_t::integrate_over_colored_interface(CF_3& f, int num0, int num1)
{
  if (loc == FCE)
  {
    return simplex[0].integrate_over_colored_interface(f, num0, num1) +
           simplex[1].integrate_over_colored_interface(f, num0, num1) +
           simplex[2].integrate_over_colored_interface(f, num0, num1) +
           simplex[3].integrate_over_colored_interface(f, num0, num1) +
#ifdef cube3_mls_l_KUHN
           simplex[5].integrate_over_colored_interface(f, num0, num1) +
#endif
           simplex[4].integrate_over_colored_interface(f, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_l_t::integrate_over_intersection(CF_3& f, int num0, int num1)
{
  if (loc == FCE && num_of_lsfs > 1)
  {
    return simplex[0].integrate_over_intersection(f, num0, num1) +
           simplex[1].integrate_over_intersection(f, num0, num1) +
           simplex[2].integrate_over_intersection(f, num0, num1) +
           simplex[3].integrate_over_intersection(f, num0, num1) +
#ifdef cube3_mls_l_KUHN
           simplex[5].integrate_over_intersection(f, num0, num1) +
#endif
           simplex[4].integrate_over_intersection(f, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_l_t::integrate_over_intersection(CF_3& f, int num0, int num1, int num2)
{
  if (loc == FCE && num_of_lsfs > 2)
  {
    return simplex[0].integrate_over_intersection(f, num0, num1, num2) +
           simplex[1].integrate_over_intersection(f, num0, num1, num2) +
           simplex[2].integrate_over_intersection(f, num0, num1, num2) +
           simplex[3].integrate_over_intersection(f, num0, num1, num2) +
#ifdef cube3_mls_l_KUHN
           simplex[5].integrate_over_intersection(f, num0, num1, num2) +
#endif
           simplex[4].integrate_over_intersection(f, num0, num1, num2);
  }
  else
    return 0.0;
}

double cube3_mls_l_t::integrate_in_dir(CF_3& f, int dir)
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
#ifdef cube3_mls_l_KUHN
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






/* Quadrature points */

void cube3_mls_l_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
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

void cube3_mls_l_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
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

void cube3_mls_l_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
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

void cube3_mls_l_t::quadrature_over_intersection(int num0, int num1, int num2, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
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

void cube3_mls_l_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
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
      }
    case FCE:
      {
        simplex[0].quadrature_in_dir(dir, weights, X, Y, Z);
        simplex[1].quadrature_in_dir(dir, weights, X, Y, Z);
        simplex[2].quadrature_in_dir(dir, weights, X, Y, Z);
        simplex[3].quadrature_in_dir(dir, weights, X, Y, Z);
        simplex[4].quadrature_in_dir(dir, weights, X, Y, Z);
        simplex[5].quadrature_in_dir(dir, weights, X, Y, Z);
      }
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
  }
}

//void cube3_mls_l_t::interpolate_to_cube(double *in, double *out)
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


//double cube3_mls_l_t::measure_of_domain()
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
//    #ifdef cube3_mls_l_KUHN
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

//double cube3_mls_l_t::measure_of_interface(int num)
//{
//  if (loc == FCE)
//  {
//    return simplex[0].integrate_over_interface(1.,1.,1.,1., num) +
//           simplex[1].integrate_over_interface(1.,1.,1.,1., num) +
//           simplex[2].integrate_over_interface(1.,1.,1.,1., num) +
//           simplex[3].integrate_over_interface(1.,1.,1.,1., num) +
//#ifdef cube3_mls_l_KUHN
//           simplex[5].integrate_over_interface(1.,1.,1.,1., num) +
//#endif
//           simplex[4].integrate_over_interface(1.,1.,1.,1., num);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_l_t::measure_of_colored_interface(int num0, int num1)
//{
//  if (loc == FCE)
//  {
//    return simplex[0].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//           simplex[1].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//           simplex[2].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//           simplex[3].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//#ifdef cube3_mls_l_KUHN
//           simplex[5].integrate_over_colored_interface(1.,1.,1.,1., num0, num1) +
//#endif
//           simplex[4].integrate_over_colored_interface(1.,1.,1.,1., num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_l_t::measure_of_intersection(int num0, int num1)
//{
//  if (loc == FCE && num_non_trivial > 1)
//  {
//    return simplex[0].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//           simplex[1].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//           simplex[2].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//           simplex[3].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//#ifdef cube3_mls_l_KUHN
//           simplex[5].integrate_over_intersection(1.,1.,1.,1., num0, num1) +
//#endif
//           simplex[4].integrate_over_intersection(1.,1.,1.,1., num0, num1);
//  }
//  else
//    return 0.0;
//}

//double cube3_mls_l_t::measure_in_dir(int dir)
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
//#ifdef cube3_mls_l_KUHN
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

//double cube3_mls_l_t::interpolate_linear(double *f, double x, double y, double z)
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

//double cube3_mls_l_t::interpolate_quadratic(double *f, double *fxx, double *fyy, double *fzz, double x, double y, double z)
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
