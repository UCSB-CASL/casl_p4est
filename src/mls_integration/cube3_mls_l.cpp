#include "cube3_mls_l.h"

void cube3_mls_l_t::construct_domain(std::vector<double> &phi_all, std::vector<action_t> &acn, std::vector<int> &clr)
{
   num_of_lsfs = acn.size();

 #ifdef CASL_THROWS
   if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
           throw std::domain_error("[CASL_ERROR]: (cube3_mls_l_quadratic_t::construct_domain) sizes of phi, acn and clr do not coincide.");
 #endif

   /* Eliminate unnecessary splitting */

   loc = INS;

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
       for (unsigned short k = 0; k < n_nodes_dir; ++k)
         for (unsigned short j = 0; j < n_nodes_dir; ++j)
           for (unsigned short i = 0; i < n_nodes_dir; ++i)
           {
             double value = phi_all[n_nodes*p + n_nodes_dir*n_nodes_dir*k + n_nodes_dir*j + i];
             all_negative = (all_negative && (value < 0.));
             all_positive = (all_positive && (value > 0.));
           }
     }

     if (all_positive)
     {
       if (acn[p] == CUBE_MLS_INTERSECTION)
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
       if (acn[p] == CUBE_MLS_ADDITION)
       {
         loc = INS;
//         nt_phi.clear();
         nt_acn.clear();
         nt_clr.clear();
         nt_idx.clear();
       }
     }
     else if (loc == FCE || (loc == INS && acn[p] == CUBE_MLS_INTERSECTION) || (loc == OUT && acn[p] == CUBE_MLS_ADDITION))
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
     if (nt_acn[0] == CUBE_MLS_ADDITION) nt_acn[0] = CUBE_MLS_INTERSECTION;

     double x[8] = {x0,x1,x0,x1,x0,x1,x0,x1};
     double y[8] = {y0,y0,y1,y1,y0,y0,y1,y1};
     double z[8] = {z0,z0,z0,z0,z1,z1,z1,z1};

     /* Split a cube into 6 simplices */
     simplex.clear();
     simplex.reserve(NTETS);
     simplex.push_back(simplex3_mls_l_t(x[l3_t0p0],y[l3_t0p0],z[l3_t0p0], x[l3_t0p1],y[l3_t0p1],z[l3_t0p1], x[l3_t0p2],y[l3_t0p2],z[l3_t0p2], x[l3_t0p3],y[l3_t0p3],z[l3_t0p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[l3_t1p0],y[l3_t1p0],z[l3_t1p0], x[l3_t1p1],y[l3_t1p1],z[l3_t1p1], x[l3_t1p2],y[l3_t1p2],z[l3_t1p2], x[l3_t1p3],y[l3_t1p3],z[l3_t1p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[l3_t2p0],y[l3_t2p0],z[l3_t2p0], x[l3_t2p1],y[l3_t2p1],z[l3_t2p1], x[l3_t2p2],y[l3_t2p2],z[l3_t2p2], x[l3_t2p3],y[l3_t2p3],z[l3_t2p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[l3_t3p0],y[l3_t3p0],z[l3_t3p0], x[l3_t3p1],y[l3_t3p1],z[l3_t3p1], x[l3_t3p2],y[l3_t3p2],z[l3_t3p2], x[l3_t3p3],y[l3_t3p3],z[l3_t3p3])); //simplex.back().set_use_linear(use_linear);
     simplex.push_back(simplex3_mls_l_t(x[l3_t4p0],y[l3_t4p0],z[l3_t4p0], x[l3_t4p1],y[l3_t4p1],z[l3_t4p1], x[l3_t4p2],y[l3_t4p2],z[l3_t4p2], x[l3_t4p3],y[l3_t4p3],z[l3_t4p3])); //simplex.back().set_use_linear(use_linear);
#ifdef cube3_mls_l_KUHN
     simplex.push_back(simplex3_mls_l_t(x[l3_t5p0],y[l3_t5p0],z[l3_t5p0], x[l3_t5p1],y[l3_t5p1],z[l3_t5p1], x[l3_t5p2],y[l3_t5p2],z[l3_t5p2], x[l3_t5p3],y[l3_t5p3],z[l3_t5p3])); //simplex.back().set_use_linear(use_linear);
#endif

     // TODO: mark appropriate edges for integrate_in_dir
//#ifdef cube3_mls_l_KUHN
//     simplex[0].tris_[0].dir = 1; simplex[0].tris_[3].dir = 4;
//     simplex[1].tris_[0].dir = 3; simplex[1].tris_[3].dir = 4;
//     simplex[2].tris_[0].dir = 1; simplex[2].tris_[3].dir = 2;
//     simplex[3].tris_[0].dir = 3; simplex[3].tris_[3].dir = 0;
//     simplex[4].tris_[0].dir = 5; simplex[4].tris_[3].dir = 2;
//     simplex[5].tris_[0].dir = 5; simplex[5].tris_[3].dir = 0;

     /* integration in dir
      * 0: s3->f3 + s5->f3
      * 1: s0->f0 + s2->f0
      * 2: s2->f3 + s4->f3
      * 3: s1->f0 + s3->f0
      * 4: s0->f3 + s1->f3
      * 5: s4->f0 + s5->f0
      */
//#endif
     // it doesn't make sense to do it for the MIDDLE_CUT triangulation

     num_of_lsfs = nt_idx.size();

     std::vector<double> phi_s(n_nodes_simplex*num_of_lsfs, -1);

     for (unsigned int s = 0; s < NTETS; ++s)
     {
       for (unsigned int i = 0; i < num_of_lsfs; ++i)
       {
         unsigned int phi_idx = nt_idx[i];
         for (unsigned int n = 0; n < n_nodes_simplex; ++n)
           phi_s[n_nodes_simplex*i + n] = phi_all[n_nodes*phi_idx + tp_l[s][n]];
       }
       simplex[s].construct_domain(phi_s, nt_acn, nt_clr);
     }
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
      } break;
    case FCE:
      {
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
