#include "cube3_mls_quadratic.h"

void cube3_mls_quadratic_t::construct_domain(std::vector< std::vector<double> > &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  bool all_positive, all_negative;

  num_of_lsfs = phi.size();

#ifdef CASL_THROWS
  if (num_of_lsfs != acn.size() || num_of_lsfs != clr.size())
          throw std::domain_error("[CASL_ERROR]: (cube3_mls_quadratic_t::construct_domain) sizes of phi, acn and clr do not coincide.");
#endif

  /* Eliminate unnecessary splitting */
  loc = INS;
  double *F;
  for (unsigned int i = 0; i < num_of_lsfs; i++)
  {
    all_negative = true;
    all_positive = true;

    F = phi[i].data();

    for (int j = 0; j < n_nodes; j++)
    {
      all_negative = (all_negative && (F[j] < 0.));
      all_positive = (all_positive && (F[j] > 0.));
    }

    if (all_positive)
    {
      if (acn[i] == INTERSECTION) loc = OUT;
    }
    else if (all_negative)
    {
      if (acn[i] == ADDITION) loc = INS;
    }
    else if ((loc == INS && acn[i] == INTERSECTION) || (loc == OUT && acn[i] == ADDITION))
    {
      loc = FCE;
    }

  }

  if (loc == FCE)
  {
    double xc = .5*(x0+x1);
    double yc = .5*(y0+y1);
    double zc = .5*(z0+z1);

    double x[27] = { x0, xc, x1, x0, xc, x1, x0, xc, x1,
                     x0, xc, x1, x0, xc, x1, x0, xc, x1,
                     x0, xc, x1, x0, xc, x1, x0, xc, x1 };

    double y[27] = { y0, y0, y0, yc, yc, yc, y1, y1, y1,
                     y0, y0, y0, yc, yc, yc, y1, y1, y1,
                     y0, y0, y0, yc, yc, yc, y1, y1, y1 };

    double z[27] = { z0, z0, z0, z0, z0, z0, z0, z0, z0,
                     zc, zc, zc, zc, zc, zc, zc, zc, zc,
                     z1, z1, z1, z1, z1, z1, z1, z1, z1 };

    /* Split a cube into 5 simplices */
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


    std::vector< std::vector<double> > phi_s0(num_of_lsfs, std::vector<double> (n_nodes_simplex, -1));
    std::vector< std::vector<double> > phi_s1(num_of_lsfs, std::vector<double> (n_nodes_simplex, -1));
    std::vector< std::vector<double> > phi_s2(num_of_lsfs, std::vector<double> (n_nodes_simplex, -1));
    std::vector< std::vector<double> > phi_s3(num_of_lsfs, std::vector<double> (n_nodes_simplex, -1));
    std::vector< std::vector<double> > phi_s4(num_of_lsfs, std::vector<double> (n_nodes_simplex, -1));
    std::vector< std::vector<double> > phi_s5(num_of_lsfs, std::vector<double> (n_nodes_simplex, -1));

    for (short i = 0; i < num_of_lsfs; ++i)
    {
      phi_s0[i][0] = phi[i][t0p0];     phi_s1[i][0] = phi[i][t1p0];     phi_s2[i][0] = phi[i][t2p0];     phi_s3[i][0] = phi[i][t3p0];     phi_s4[i][0] = phi[i][t4p0];     phi_s5[i][0] = phi[i][t5p0];
      phi_s0[i][1] = phi[i][t0p1];     phi_s1[i][1] = phi[i][t1p1];     phi_s2[i][1] = phi[i][t2p1];     phi_s3[i][1] = phi[i][t3p1];     phi_s4[i][1] = phi[i][t4p1];     phi_s5[i][1] = phi[i][t5p1];
      phi_s0[i][2] = phi[i][t0p2];     phi_s1[i][2] = phi[i][t1p2];     phi_s2[i][2] = phi[i][t2p2];     phi_s3[i][2] = phi[i][t3p2];     phi_s4[i][2] = phi[i][t4p2];     phi_s5[i][2] = phi[i][t5p2];
      phi_s0[i][3] = phi[i][t0p3];     phi_s1[i][3] = phi[i][t1p3];     phi_s2[i][3] = phi[i][t2p3];     phi_s3[i][3] = phi[i][t3p3];     phi_s4[i][3] = phi[i][t4p3];     phi_s5[i][3] = phi[i][t5p3];
      phi_s0[i][4] = phi[i][t0p4];     phi_s1[i][4] = phi[i][t1p4];     phi_s2[i][4] = phi[i][t2p4];     phi_s3[i][4] = phi[i][t3p4];     phi_s4[i][4] = phi[i][t4p4];     phi_s5[i][4] = phi[i][t5p4];
      phi_s0[i][5] = phi[i][t0p5];     phi_s1[i][5] = phi[i][t1p5];     phi_s2[i][5] = phi[i][t2p5];     phi_s3[i][5] = phi[i][t3p5];     phi_s4[i][5] = phi[i][t4p5];     phi_s5[i][5] = phi[i][t5p5];
      phi_s0[i][6] = phi[i][t0p6];     phi_s1[i][6] = phi[i][t1p6];     phi_s2[i][6] = phi[i][t2p6];     phi_s3[i][6] = phi[i][t3p6];     phi_s4[i][6] = phi[i][t4p6];     phi_s5[i][6] = phi[i][t5p6];
      phi_s0[i][7] = phi[i][t0p7];     phi_s1[i][7] = phi[i][t1p7];     phi_s2[i][7] = phi[i][t2p7];     phi_s3[i][7] = phi[i][t3p7];     phi_s4[i][7] = phi[i][t4p7];     phi_s5[i][7] = phi[i][t5p7];
      phi_s0[i][8] = phi[i][t0p8];     phi_s1[i][8] = phi[i][t1p8];     phi_s2[i][8] = phi[i][t2p8];     phi_s3[i][8] = phi[i][t3p8];     phi_s4[i][8] = phi[i][t4p8];     phi_s5[i][8] = phi[i][t5p8];
      phi_s0[i][9] = phi[i][t0p9];     phi_s1[i][9] = phi[i][t1p9];     phi_s2[i][9] = phi[i][t2p9];     phi_s3[i][9] = phi[i][t3p9];     phi_s4[i][9] = phi[i][t4p9];     phi_s5[i][9] = phi[i][t5p9];
    }

    simplex[0].construct_domain(phi_s0, acn, clr);
    simplex[1].construct_domain(phi_s1, acn, clr);
    simplex[2].construct_domain(phi_s2, acn, clr);
    simplex[3].construct_domain(phi_s3, acn, clr);
    simplex[4].construct_domain(phi_s4, acn, clr);
    simplex[5].construct_domain(phi_s5, acn, clr);
  }
}

double cube3_mls_quadratic_t::integrate_over_domain(std::vector<double> &f)
{
  switch (loc){
    case INS:
      {
        double alpha = 1./sqrt(3.);

        double F0 = interpolate_quad(f, -alpha, -alpha, -alpha);
        double F1 = interpolate_quad(f,  alpha, -alpha, -alpha);
        double F2 = interpolate_quad(f, -alpha,  alpha, -alpha);
        double F3 = interpolate_quad(f,  alpha,  alpha, -alpha);
        double F4 = interpolate_quad(f, -alpha, -alpha,  alpha);
        double F5 = interpolate_quad(f,  alpha, -alpha,  alpha);
        double F6 = interpolate_quad(f, -alpha,  alpha,  alpha);
        double F7 = interpolate_quad(f,  alpha,  alpha,  alpha);

        return (x1-x0)*(y1-y0)*(z1-z0)*(F0+F1+F2+F3+F4+F5+F6+F7)/8.0;
      } break;
    case OUT: return 0.0; break;
    case FCE:
      {
        std::vector<double> F_s0(n_nodes_simplex, 1);
        std::vector<double> F_s1(n_nodes_simplex, 1);
        std::vector<double> F_s2(n_nodes_simplex, 1);
        std::vector<double> F_s3(n_nodes_simplex, 1);
        std::vector<double> F_s4(n_nodes_simplex, 1);
        std::vector<double> F_s5(n_nodes_simplex, 1);

        F_s0[0] = f[t0p0];    F_s1[0] = f[t1p0];    F_s2[0] = f[t2p0];    F_s3[0] = f[t3p0];    F_s4[0] = f[t4p0];    F_s5[0] = f[t5p0];
        F_s0[1] = f[t0p1];    F_s1[1] = f[t1p1];    F_s2[1] = f[t2p1];    F_s3[1] = f[t3p1];    F_s4[1] = f[t4p1];    F_s5[1] = f[t5p1];
        F_s0[2] = f[t0p2];    F_s1[2] = f[t1p2];    F_s2[2] = f[t2p2];    F_s3[2] = f[t3p2];    F_s4[2] = f[t4p2];    F_s5[2] = f[t5p2];
        F_s0[3] = f[t0p3];    F_s1[3] = f[t1p3];    F_s2[3] = f[t2p3];    F_s3[3] = f[t3p3];    F_s4[3] = f[t4p3];    F_s5[3] = f[t5p3];
        F_s0[4] = f[t0p4];    F_s1[4] = f[t1p4];    F_s2[4] = f[t2p4];    F_s3[4] = f[t3p4];    F_s4[4] = f[t4p4];    F_s5[4] = f[t5p4];
        F_s0[5] = f[t0p5];    F_s1[5] = f[t1p5];    F_s2[5] = f[t2p5];    F_s3[5] = f[t3p5];    F_s4[5] = f[t4p5];    F_s5[5] = f[t5p5];
        F_s0[6] = f[t0p6];    F_s1[6] = f[t1p6];    F_s2[6] = f[t2p6];    F_s3[6] = f[t3p6];    F_s4[6] = f[t4p6];    F_s5[6] = f[t5p6];
        F_s0[7] = f[t0p7];    F_s1[7] = f[t1p7];    F_s2[7] = f[t2p7];    F_s3[7] = f[t3p7];    F_s4[7] = f[t4p7];    F_s5[7] = f[t5p7];
        F_s0[8] = f[t0p8];    F_s1[8] = f[t1p8];    F_s2[8] = f[t2p8];    F_s3[8] = f[t3p8];    F_s4[8] = f[t4p8];    F_s5[8] = f[t5p8];
        F_s0[9] = f[t0p9];    F_s1[9] = f[t1p9];    F_s2[9] = f[t2p9];    F_s3[9] = f[t3p9];    F_s4[9] = f[t4p9];    F_s5[9] = f[t5p9];

//        return 0.;
        return simplex[0].integrate_over_domain(F_s0)
             + simplex[1].integrate_over_domain(F_s1)
             + simplex[2].integrate_over_domain(F_s2)
             + simplex[3].integrate_over_domain(F_s3)
             + simplex[4].integrate_over_domain(F_s4)
             + simplex[5].integrate_over_domain(F_s5);
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

double cube3_mls_quadratic_t::integrate_over_interface(std::vector<double> &f, int num)
{
  if (loc == FCE)
  {
    std::vector<double> F_s0(n_nodes_simplex, 1);
    std::vector<double> F_s1(n_nodes_simplex, 1);
    std::vector<double> F_s2(n_nodes_simplex, 1);
    std::vector<double> F_s3(n_nodes_simplex, 1);
    std::vector<double> F_s4(n_nodes_simplex, 1);
    std::vector<double> F_s5(n_nodes_simplex, 1);

    F_s0[0] = f[t0p0];    F_s1[0] = f[t1p0];    F_s2[0] = f[t2p0];    F_s3[0] = f[t3p0];    F_s4[0] = f[t4p0];    F_s5[0] = f[t5p0];
    F_s0[1] = f[t0p1];    F_s1[1] = f[t1p1];    F_s2[1] = f[t2p1];    F_s3[1] = f[t3p1];    F_s4[1] = f[t4p1];    F_s5[1] = f[t5p1];
    F_s0[2] = f[t0p2];    F_s1[2] = f[t1p2];    F_s2[2] = f[t2p2];    F_s3[2] = f[t3p2];    F_s4[2] = f[t4p2];    F_s5[2] = f[t5p2];
    F_s0[3] = f[t0p3];    F_s1[3] = f[t1p3];    F_s2[3] = f[t2p3];    F_s3[3] = f[t3p3];    F_s4[3] = f[t4p3];    F_s5[3] = f[t5p3];
    F_s0[4] = f[t0p4];    F_s1[4] = f[t1p4];    F_s2[4] = f[t2p4];    F_s3[4] = f[t3p4];    F_s4[4] = f[t4p4];    F_s5[4] = f[t5p4];
    F_s0[5] = f[t0p5];    F_s1[5] = f[t1p5];    F_s2[5] = f[t2p5];    F_s3[5] = f[t3p5];    F_s4[5] = f[t4p5];    F_s5[5] = f[t5p5];
    F_s0[6] = f[t0p6];    F_s1[6] = f[t1p6];    F_s2[6] = f[t2p6];    F_s3[6] = f[t3p6];    F_s4[6] = f[t4p6];    F_s5[6] = f[t5p6];
    F_s0[7] = f[t0p7];    F_s1[7] = f[t1p7];    F_s2[7] = f[t2p7];    F_s3[7] = f[t3p7];    F_s4[7] = f[t4p7];    F_s5[7] = f[t5p7];
    F_s0[8] = f[t0p8];    F_s1[8] = f[t1p8];    F_s2[8] = f[t2p8];    F_s3[8] = f[t3p8];    F_s4[8] = f[t4p8];    F_s5[8] = f[t5p8];
    F_s0[9] = f[t0p9];    F_s1[9] = f[t1p9];    F_s2[9] = f[t2p9];    F_s3[9] = f[t3p9];    F_s4[9] = f[t4p9];    F_s5[9] = f[t5p9];

    return simplex[0].integrate_over_interface(F_s0, num)
         + simplex[1].integrate_over_interface(F_s1, num)
         + simplex[2].integrate_over_interface(F_s2, num)
         + simplex[3].integrate_over_interface(F_s3, num)
         + simplex[4].integrate_over_interface(F_s4, num)
         + simplex[5].integrate_over_interface(F_s5, num);
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_over_colored_interface(std::vector<double> &f, int num0, int num1)
{
  if (loc == FCE)
  {
    std::vector<double> F_s0(n_nodes_simplex, 1);
    std::vector<double> F_s1(n_nodes_simplex, 1);
    std::vector<double> F_s2(n_nodes_simplex, 1);
    std::vector<double> F_s3(n_nodes_simplex, 1);
    std::vector<double> F_s4(n_nodes_simplex, 1);
    std::vector<double> F_s5(n_nodes_simplex, 1);

    F_s0[0] = f[t0p0];    F_s1[0] = f[t1p0];    F_s2[0] = f[t2p0];    F_s3[0] = f[t3p0];    F_s4[0] = f[t4p0];    F_s5[0] = f[t5p0];
    F_s0[1] = f[t0p1];    F_s1[1] = f[t1p1];    F_s2[1] = f[t2p1];    F_s3[1] = f[t3p1];    F_s4[1] = f[t4p1];    F_s5[1] = f[t5p1];
    F_s0[2] = f[t0p2];    F_s1[2] = f[t1p2];    F_s2[2] = f[t2p2];    F_s3[2] = f[t3p2];    F_s4[2] = f[t4p2];    F_s5[2] = f[t5p2];
    F_s0[3] = f[t0p3];    F_s1[3] = f[t1p3];    F_s2[3] = f[t2p3];    F_s3[3] = f[t3p3];    F_s4[3] = f[t4p3];    F_s5[3] = f[t5p3];
    F_s0[4] = f[t0p4];    F_s1[4] = f[t1p4];    F_s2[4] = f[t2p4];    F_s3[4] = f[t3p4];    F_s4[4] = f[t4p4];    F_s5[4] = f[t5p4];
    F_s0[5] = f[t0p5];    F_s1[5] = f[t1p5];    F_s2[5] = f[t2p5];    F_s3[5] = f[t3p5];    F_s4[5] = f[t4p5];    F_s5[5] = f[t5p5];
    F_s0[6] = f[t0p6];    F_s1[6] = f[t1p6];    F_s2[6] = f[t2p6];    F_s3[6] = f[t3p6];    F_s4[6] = f[t4p6];    F_s5[6] = f[t5p6];
    F_s0[7] = f[t0p7];    F_s1[7] = f[t1p7];    F_s2[7] = f[t2p7];    F_s3[7] = f[t3p7];    F_s4[7] = f[t4p7];    F_s5[7] = f[t5p7];
    F_s0[8] = f[t0p8];    F_s1[8] = f[t1p8];    F_s2[8] = f[t2p8];    F_s3[8] = f[t3p8];    F_s4[8] = f[t4p8];    F_s5[8] = f[t5p8];
    F_s0[9] = f[t0p9];    F_s1[9] = f[t1p9];    F_s2[9] = f[t2p9];    F_s3[9] = f[t3p9];    F_s4[9] = f[t4p9];    F_s5[9] = f[t5p9];

    return simplex[0].integrate_over_colored_interface(F_s0, num0, num1)
         + simplex[1].integrate_over_colored_interface(F_s1, num0, num1)
         + simplex[2].integrate_over_colored_interface(F_s2, num0, num1)
         + simplex[3].integrate_over_colored_interface(F_s3, num0, num1)
         + simplex[4].integrate_over_colored_interface(F_s4, num0, num1)
         + simplex[5].integrate_over_colored_interface(F_s5, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_over_intersection(std::vector<double> &f, int num0, int num1)
{
  if (loc == FCE && num_of_lsfs > 1)
  {
    std::vector<double> F_s0(n_nodes_simplex, 1);
    std::vector<double> F_s1(n_nodes_simplex, 1);
    std::vector<double> F_s2(n_nodes_simplex, 1);
    std::vector<double> F_s3(n_nodes_simplex, 1);
    std::vector<double> F_s4(n_nodes_simplex, 1);
    std::vector<double> F_s5(n_nodes_simplex, 1);

    F_s0[0] = f[t0p0];    F_s1[0] = f[t1p0];    F_s2[0] = f[t2p0];    F_s3[0] = f[t3p0];    F_s4[0] = f[t4p0];    F_s5[0] = f[t5p0];
    F_s0[1] = f[t0p1];    F_s1[1] = f[t1p1];    F_s2[1] = f[t2p1];    F_s3[1] = f[t3p1];    F_s4[1] = f[t4p1];    F_s5[1] = f[t5p1];
    F_s0[2] = f[t0p2];    F_s1[2] = f[t1p2];    F_s2[2] = f[t2p2];    F_s3[2] = f[t3p2];    F_s4[2] = f[t4p2];    F_s5[2] = f[t5p2];
    F_s0[3] = f[t0p3];    F_s1[3] = f[t1p3];    F_s2[3] = f[t2p3];    F_s3[3] = f[t3p3];    F_s4[3] = f[t4p3];    F_s5[3] = f[t5p3];
    F_s0[4] = f[t0p4];    F_s1[4] = f[t1p4];    F_s2[4] = f[t2p4];    F_s3[4] = f[t3p4];    F_s4[4] = f[t4p4];    F_s5[4] = f[t5p4];
    F_s0[5] = f[t0p5];    F_s1[5] = f[t1p5];    F_s2[5] = f[t2p5];    F_s3[5] = f[t3p5];    F_s4[5] = f[t4p5];    F_s5[5] = f[t5p5];
    F_s0[6] = f[t0p6];    F_s1[6] = f[t1p6];    F_s2[6] = f[t2p6];    F_s3[6] = f[t3p6];    F_s4[6] = f[t4p6];    F_s5[6] = f[t5p6];
    F_s0[7] = f[t0p7];    F_s1[7] = f[t1p7];    F_s2[7] = f[t2p7];    F_s3[7] = f[t3p7];    F_s4[7] = f[t4p7];    F_s5[7] = f[t5p7];
    F_s0[8] = f[t0p8];    F_s1[8] = f[t1p8];    F_s2[8] = f[t2p8];    F_s3[8] = f[t3p8];    F_s4[8] = f[t4p8];    F_s5[8] = f[t5p8];
    F_s0[9] = f[t0p9];    F_s1[9] = f[t1p9];    F_s2[9] = f[t2p9];    F_s3[9] = f[t3p9];    F_s4[9] = f[t4p9];    F_s5[9] = f[t5p9];

    return simplex[0].integrate_over_intersection(F_s0, num0, num1)
         + simplex[1].integrate_over_intersection(F_s1, num0, num1)
         + simplex[2].integrate_over_intersection(F_s2, num0, num1)
         + simplex[3].integrate_over_intersection(F_s3, num0, num1)
         + simplex[4].integrate_over_intersection(F_s4, num0, num1)
         + simplex[5].integrate_over_intersection(F_s5, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_over_intersection(std::vector<double> &f, int num0, int num1, int num2)
{
  if (loc == FCE && num_of_lsfs > 2)
  {
    std::vector<double> F_s0(n_nodes_simplex, 1);
    std::vector<double> F_s1(n_nodes_simplex, 1);
    std::vector<double> F_s2(n_nodes_simplex, 1);
    std::vector<double> F_s3(n_nodes_simplex, 1);
    std::vector<double> F_s4(n_nodes_simplex, 1);
    std::vector<double> F_s5(n_nodes_simplex, 1);

    F_s0[0] = f[t0p0];    F_s1[0] = f[t1p0];    F_s2[0] = f[t2p0];    F_s3[0] = f[t3p0];    F_s4[0] = f[t4p0];    F_s5[0] = f[t5p0];
    F_s0[1] = f[t0p1];    F_s1[1] = f[t1p1];    F_s2[1] = f[t2p1];    F_s3[1] = f[t3p1];    F_s4[1] = f[t4p1];    F_s5[1] = f[t5p1];
    F_s0[2] = f[t0p2];    F_s1[2] = f[t1p2];    F_s2[2] = f[t2p2];    F_s3[2] = f[t3p2];    F_s4[2] = f[t4p2];    F_s5[2] = f[t5p2];
    F_s0[3] = f[t0p3];    F_s1[3] = f[t1p3];    F_s2[3] = f[t2p3];    F_s3[3] = f[t3p3];    F_s4[3] = f[t4p3];    F_s5[3] = f[t5p3];
    F_s0[4] = f[t0p4];    F_s1[4] = f[t1p4];    F_s2[4] = f[t2p4];    F_s3[4] = f[t3p4];    F_s4[4] = f[t4p4];    F_s5[4] = f[t5p4];
    F_s0[5] = f[t0p5];    F_s1[5] = f[t1p5];    F_s2[5] = f[t2p5];    F_s3[5] = f[t3p5];    F_s4[5] = f[t4p5];    F_s5[5] = f[t5p5];
    F_s0[6] = f[t0p6];    F_s1[6] = f[t1p6];    F_s2[6] = f[t2p6];    F_s3[6] = f[t3p6];    F_s4[6] = f[t4p6];    F_s5[6] = f[t5p6];
    F_s0[7] = f[t0p7];    F_s1[7] = f[t1p7];    F_s2[7] = f[t2p7];    F_s3[7] = f[t3p7];    F_s4[7] = f[t4p7];    F_s5[7] = f[t5p7];
    F_s0[8] = f[t0p8];    F_s1[8] = f[t1p8];    F_s2[8] = f[t2p8];    F_s3[8] = f[t3p8];    F_s4[8] = f[t4p8];    F_s5[8] = f[t5p8];
    F_s0[9] = f[t0p9];    F_s1[9] = f[t1p9];    F_s2[9] = f[t2p9];    F_s3[9] = f[t3p9];    F_s4[9] = f[t4p9];    F_s5[9] = f[t5p9];

    return simplex[0].integrate_over_intersection(F_s0, num0, num1, num2)
         + simplex[1].integrate_over_intersection(F_s1, num0, num1, num2)
         + simplex[2].integrate_over_intersection(F_s2, num0, num1, num2)
         + simplex[3].integrate_over_intersection(F_s3, num0, num1, num2)
         + simplex[4].integrate_over_intersection(F_s4, num0, num1, num2)
         + simplex[5].integrate_over_intersection(F_s5, num0, num1, num2);
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::integrate_in_dir(std::vector<double> &f, int dir)
{
  switch (loc){
    case OUT: return 0;
    case INS:
      {
        double alpha = 1./sqrt(3.);
        double F0 = 1, F1 = 1, F2 = 1, F3 = 1, s = 0;

        switch (dir) {
          case 0: s = (y1-y0)*(z1-z0);
            F0 = interpolate_quad(f, -1., -alpha, -alpha);
            F1 = interpolate_quad(f, -1.,  alpha, -alpha);
            F2 = interpolate_quad(f, -1., -alpha,  alpha);
            F3 = interpolate_quad(f, -1.,  alpha,  alpha); break;
          case 1: s = (y1-y0)*(z1-z0);
            F0 = interpolate_quad(f,  1., -alpha, -alpha);
            F1 = interpolate_quad(f,  1.,  alpha, -alpha);
            F2 = interpolate_quad(f,  1., -alpha,  alpha);
            F3 = interpolate_quad(f,  1.,  alpha,  alpha); break;
          case 2: s = (x1-x0)*(z1-z0);
            F0 = interpolate_quad(f, -alpha, -1., -alpha);
            F1 = interpolate_quad(f,  alpha, -1., -alpha);
            F2 = interpolate_quad(f, -alpha, -1.,  alpha);
            F3 = interpolate_quad(f,  alpha, -1.,  alpha); break;
          case 3: s = (x1-x0)*(z1-z0);
            F0 = interpolate_quad(f, -alpha,  1., -alpha);
            F1 = interpolate_quad(f,  alpha,  1., -alpha);
            F2 = interpolate_quad(f, -alpha,  1.,  alpha);
            F3 = interpolate_quad(f,  alpha,  1.,  alpha); break;
          case 4: s = (x1-x0)*(y1-y0);
            F0 = interpolate_quad(f, -alpha, -alpha, -1.);
            F1 = interpolate_quad(f,  alpha, -alpha, -1.);
            F2 = interpolate_quad(f, -alpha,  alpha, -1.);
            F3 = interpolate_quad(f,  alpha,  alpha, -1.); break;
          case 5: s = (x1-x0)*(y1-y0);
            F0 = interpolate_quad(f, -alpha, -alpha,  1.);
            F1 = interpolate_quad(f,  alpha, -alpha,  1.);
            F2 = interpolate_quad(f, -alpha,  alpha,  1.);
            F3 = interpolate_quad(f,  alpha,  alpha,  1.); break;
        }

        return 0.25*s*(F0+F1+F2+F3);
      }
    case FCE:
      {
        std::vector<double> F_s0(n_nodes_simplex, 1);
        std::vector<double> F_s1(n_nodes_simplex, 1);
        std::vector<double> F_s2(n_nodes_simplex, 1);
        std::vector<double> F_s3(n_nodes_simplex, 1);
        std::vector<double> F_s4(n_nodes_simplex, 1);
        std::vector<double> F_s5(n_nodes_simplex, 1);

        F_s0[0] = f[t0p0];    F_s1[0] = f[t1p0];    F_s2[0] = f[t2p0];    F_s3[0] = f[t3p0];    F_s4[0] = f[t4p0];    F_s5[0] = f[t5p0];
        F_s0[1] = f[t0p1];    F_s1[1] = f[t1p1];    F_s2[1] = f[t2p1];    F_s3[1] = f[t3p1];    F_s4[1] = f[t4p1];    F_s5[1] = f[t5p1];
        F_s0[2] = f[t0p2];    F_s1[2] = f[t1p2];    F_s2[2] = f[t2p2];    F_s3[2] = f[t3p2];    F_s4[2] = f[t4p2];    F_s5[2] = f[t5p2];
        F_s0[3] = f[t0p3];    F_s1[3] = f[t1p3];    F_s2[3] = f[t2p3];    F_s3[3] = f[t3p3];    F_s4[3] = f[t4p3];    F_s5[3] = f[t5p3];
        F_s0[4] = f[t0p4];    F_s1[4] = f[t1p4];    F_s2[4] = f[t2p4];    F_s3[4] = f[t3p4];    F_s4[4] = f[t4p4];    F_s5[4] = f[t5p4];
        F_s0[5] = f[t0p5];    F_s1[5] = f[t1p5];    F_s2[5] = f[t2p5];    F_s3[5] = f[t3p5];    F_s4[5] = f[t4p5];    F_s5[5] = f[t5p5];
        F_s0[6] = f[t0p6];    F_s1[6] = f[t1p6];    F_s2[6] = f[t2p6];    F_s3[6] = f[t3p6];    F_s4[6] = f[t4p6];    F_s5[6] = f[t5p6];
        F_s0[7] = f[t0p7];    F_s1[7] = f[t1p7];    F_s2[7] = f[t2p7];    F_s3[7] = f[t3p7];    F_s4[7] = f[t4p7];    F_s5[7] = f[t5p7];
        F_s0[8] = f[t0p8];    F_s1[8] = f[t1p8];    F_s2[8] = f[t2p8];    F_s3[8] = f[t3p8];    F_s4[8] = f[t4p8];    F_s5[8] = f[t5p8];
        F_s0[9] = f[t0p9];    F_s1[9] = f[t1p9];    F_s2[9] = f[t2p9];    F_s3[9] = f[t3p9];    F_s4[9] = f[t4p9];    F_s5[9] = f[t5p9];

        return simplex[0].integrate_in_dir(F_s0, dir)
             + simplex[1].integrate_in_dir(F_s1, dir)
             + simplex[2].integrate_in_dir(F_s2, dir)
             + simplex[3].integrate_in_dir(F_s3, dir)
             + simplex[4].integrate_in_dir(F_s4, dir)
             + simplex[5].integrate_in_dir(F_s5, dir);
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


double cube3_mls_quadratic_t::measure_of_domain()
{
  switch (loc){
    case INS:
      {
        return (x1-x0)*(y1-y0)*(z1-z0);
      } break;
    case OUT: return 0.0; break;
    case FCE:
      {
        std::vector<double> F(n_nodes_simplex, 1);

        return simplex[0].integrate_over_domain(F)
            +  simplex[1].integrate_over_domain(F)
            +  simplex[2].integrate_over_domain(F)
            +  simplex[3].integrate_over_domain(F)
            +  simplex[4].integrate_over_domain(F)
            +  simplex[5].integrate_over_domain(F);
      } break;
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0;
  }
}

double cube3_mls_quadratic_t::measure_of_interface(int num)
{
  if (loc == FCE)
  {
    std::vector<double> F(n_nodes_simplex, 1);

    return simplex[0].integrate_over_interface(F, num) +
           simplex[1].integrate_over_interface(F, num) +
           simplex[2].integrate_over_interface(F, num) +
           simplex[3].integrate_over_interface(F, num) +
           simplex[4].integrate_over_interface(F, num) +
           simplex[5].integrate_over_interface(F, num);
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::measure_of_colored_interface(int num0, int num1)
{
  if (loc == FCE)
  {
    std::vector<double> F(n_nodes_simplex, 1);

    return simplex[0].integrate_over_colored_interface(F, num0, num1) +
           simplex[1].integrate_over_colored_interface(F, num0, num1) +
           simplex[2].integrate_over_colored_interface(F, num0, num1) +
           simplex[3].integrate_over_colored_interface(F, num0, num1) +
           simplex[5].integrate_over_colored_interface(F, num0, num1) +
           simplex[4].integrate_over_colored_interface(F, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::measure_of_intersection(int num0, int num1)
{
  if (loc == FCE && num_of_lsfs > 1)
  {
    std::vector<double> F(n_nodes_simplex, 1);

    return simplex[0].integrate_over_intersection(F, num0, num1) +
           simplex[1].integrate_over_intersection(F, num0, num1) +
           simplex[2].integrate_over_intersection(F, num0, num1) +
           simplex[3].integrate_over_intersection(F, num0, num1) +
           simplex[4].integrate_over_intersection(F, num0, num1) +
           simplex[5].integrate_over_intersection(F, num0, num1);
  }
  else
    return 0.0;
}

double cube3_mls_quadratic_t::measure_in_dir(int dir)
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
        std::vector<double> F(n_nodes_simplex, 1);

        return simplex[0].integrate_in_dir(F, dir) +
               simplex[1].integrate_in_dir(F, dir) +
               simplex[2].integrate_in_dir(F, dir) +
               simplex[3].integrate_in_dir(F, dir) +
               simplex[4].integrate_in_dir(F, dir) +
               simplex[5].integrate_in_dir(F, dir);
      }
    default:
#ifdef CASL_THROWS
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif
      return 0;
  }
}

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
