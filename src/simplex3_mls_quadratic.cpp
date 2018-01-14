#include "simplex3_mls_quadratic.h"


//--------------------------------------------------
// Constructors
//--------------------------------------------------
simplex3_mls_quadratic_t::simplex3_mls_quadratic_t()
{
  vtxs.reserve(8);
  edgs.reserve(27);
  tris.reserve(20);
  tets.reserve(6);

  eps = 1.0e-13;
}

simplex3_mls_quadratic_t::simplex3_mls_quadratic_t(double x0, double y0, double z0,
                                                   double x1, double y1, double z1,
                                                   double x2, double y2, double z2,
                                                   double x3, double y3, double z3,
                                                   double x4, double y4, double z4,
                                                   double x5, double y5, double z5,
                                                   double x6, double y6, double z6,
                                                   double x7, double y7, double z7,
                                                   double x8, double y8, double z8,
                                                   double x9, double y9, double z9)
{
  if (1) // usually there will be only one cut
  {
    vtxs.reserve(20);
    edgs.reserve(27);
    tris.reserve(20);
    tets.reserve(6);
  }

  /* fill the vectors with the initial structure */
  /*
   *             3
   *             | o
   *            o|   o
   *             |     o
   *             |       o
   *           o |         o
   *             7-----------9
   *            /|           | o
   *          o/ |           |   o
   *          /  |           |     o
   *         8   |           |       o
   *         |   |           |         o
   *        o|   0-----------6-----------2
   *         |  /           /        o
   *         | /           /     o
   *       o |/           /  o
   *         4-----------5
   *        /        o
   *      o/     o
   *      /  o
   *     1
   *
   */
  vtxs.push_back(vtx3_t(x0,y0,z0));
  vtxs.push_back(vtx3_t(x1,y1,z1));
  vtxs.push_back(vtx3_t(x2,y2,z2));
  vtxs.push_back(vtx3_t(x3,y3,z3));
  vtxs.push_back(vtx3_t(x4,y4,z4));
  vtxs.push_back(vtx3_t(x5,y5,z5));
  vtxs.push_back(vtx3_t(x6,y6,z6));
  vtxs.push_back(vtx3_t(x7,y7,z7));
  vtxs.push_back(vtx3_t(x8,y8,z8));
  vtxs.push_back(vtx3_t(x9,y9,z9));

  edgs.push_back(edg3_t(0,4,1));
  edgs.push_back(edg3_t(0,6,2));
  edgs.push_back(edg3_t(0,7,3));
  edgs.push_back(edg3_t(1,5,2));
  edgs.push_back(edg3_t(1,8,3));
  edgs.push_back(edg3_t(2,9,3));

  tris.push_back(tri3_t(1,2,3,5,4,3));
  tris.push_back(tri3_t(0,2,3,5,2,1));
  tris.push_back(tri3_t(0,1,3,4,2,0));
  tris.push_back(tri3_t(0,1,2,3,1,0));

  tets.push_back(tet3_t(0,1,2,3,0,1,2,3));

//  use_linear = true;
  eps = 1.0e-13;

  diag = MAX(fabs(x0-x1), fabs(y0-y1), fabs(z0-z1));
}




//--------------------------------------------------
// Constructing domain
//--------------------------------------------------
void simplex3_mls_quadratic_t::construct_domain(std::vector<CF_3 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{

  bool needs_refinement = true;
  int last_vtxs_size = 0;

  int initial_refinement = 0;
  int n;

  std::vector<vtx3_t> vtxs_initial = vtxs;
  std::vector<edg3_t> edgs_initial = edgs;
  std::vector<tri3_t> tris_initial = tris;
  std::vector<tet3_t> tets_initial = tets;


  while(1)
  {
    for (int i = 0; i < initial_refinement; ++i)
    {
      n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
      n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
      n = tets.size(); for (int i = 0; i < n; i++) refine_tet(i);
    }

    int refine_level = 0;

    // loop over LSFs
    for (short phi_idx = 0; phi_idx < phi.size(); ++phi_idx)
    {
      last_vtxs_size = 0;

      invalid_reconstruction = true;

      while (invalid_reconstruction)
      {
        needs_refinement = true;

        while (needs_refinement)
        {
          // interpolate to all vertices
          for (int i = last_vtxs_size; i < vtxs.size(); ++i)
            if (!vtxs[i].is_recycled)
            {
              vtxs[i].value = (*phi[phi_idx]) ( vtxs[i].x, vtxs[i].y, vtxs[i].z );
              perturb(vtxs[i].value, eps);
            }

          // check validity of data on each edge
          needs_refinement = false;
          for (int i = 0; i < edgs.size(); ++i)
            if (!edgs[i].is_split)
            {
              edg3_t *e = &edgs[i];

              double phi0 = vtxs[e->vtx0].value;
              double phi1 = vtxs[e->vtx1].value;
              double phi2 = vtxs[e->vtx2].value;

              if (phi1*phi0 < 0 && phi0*phi2 > 0)
              {
                needs_refinement = true;
                break;
              }

//              if (phi0*phi2 > 0)
//              {
//                double c0 =     phi0;
//                double c1 = -3.*phi0 + 4.*phi1 -    phi2;
//                double c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

//                if (c2 > EPS)
//                {
//                  double d = c1*c1 - 4.*c0*c2;
//                  if (d > 0)
//                  {
//                    double a1 = .5*(-c1-sqrt(d))/c2;
//                    double a2 = .5*(-c1+sqrt(d))/c2;
//                    if (a1 > 0. && a1 < 1. ||
//                        a2 > 0. && a2 < 1.)
//                    {
//                      std::cout << a1 << " " << a2 << "\n";
//                      needs_refinement = true;
//                      break;
//                    }
//                  }
//                }
//              }

              if (phi0*phi2 > 0)
              {
                double c0 = phi0;
                double c1 = -3.*phi0 + 4.*phi1 -    phi2;
                double c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

                if (fabs(c2) > EPS)
                {
                  double a_ext = -.5*c1/c2;
                  if (a_ext > 0. && a_ext < 1.)
                  {
                    double phi_ext = c0 + c1*a_ext + c2*a_ext*a_ext;

                    if (phi0*phi_ext < 0)
                    {
//                      std::cout << a_ext << " " << phi_ext << "\n";
                      needs_refinement = true;
                      break;
                    }
                  }
                }
              }
            }

          // check validity of data on each face
          if (!needs_refinement)
            for (int i = 0; i < tris.size(); ++i)
              if (!tris[i].is_split)
              {
                tri3_t *f = &tris[i];
                edg3_t *e0 = &edgs[f->edg0];
                edg3_t *e1 = &edgs[f->edg1];
                edg3_t *e2 = &edgs[f->edg2];

                double phi0 = vtxs[f->vtx0].value;
                double phi1 = vtxs[f->vtx1].value;
                double phi2 = vtxs[f->vtx2].value;
                double phi3 = vtxs[e2->vtx1].value;
                double phi4 = vtxs[e0->vtx1].value;
                double phi5 = vtxs[e1->vtx1].value;

                if (phi0*phi1 > 0 && phi1*phi2 > 0 && phi2*phi3 > 0 && phi3*phi4 > 0 && phi4*phi5 > 0)
                {
                  double paa = 2.*phi0 + 2.*phi1 - 4.*phi3;
                  double pab = 4.*phi0 - 4.*phi3 + 4.*phi4 - 4.*phi5;
                  double pbb = 2.*phi0 + 2.*phi2 - 4.*phi5;
                  double pa = -3.*phi0 -    phi1 + 4.*phi3;
                  double pb = -3.*phi0 -    phi2 + 4.*phi5;

                  double det = 4.*paa*pbb - pab*pab;

                  if (fabs(det) > EPS)
                  {
                    double a = (pb*pab - 2.*pa*pbb)/det;
                    double b = (pa*pab - 2.*pb*paa)/det;

                    if (a > 0 && b > 0 && a+b < 1.)
                    {
                      double phi_extremum = paa*a*a + pab*a*b + pbb*b*b + pa*a + pb*b + phi0;

                      if (phi_extremum*phi0 < 0)
                      {
//                        std::cout << "face " << a << " " << b << "\n";
                        needs_refinement = true;
                        break;
                      }
                    }
                  }

                }
              }

          last_vtxs_size = vtxs.size();

          // refine if necessary
          if (needs_refinement && refine_level < max_refinement_ - initial_refinement)
          {
            int n;
            n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
            n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
            n = tets.size(); for (int i = 0; i < n; i++) refine_tet(i);
            refine_level++;
          } else if (needs_refinement) {
            std::cout << "Cannot resolve invalid geometry (bad)\n";
            needs_refinement = false;
          }
        }

        invalid_reconstruction = false;

        std::vector<vtx3_t> vtx_tmp = vtxs;
        std::vector<edg3_t> edg_tmp = edgs;
        std::vector<tri3_t> tri_tmp = tris;
        std::vector<tet3_t> tet_tmp = tets;

        int n;
        n = vtxs.size(); for (int i = 0; i < n; i++) {                                    do_action_vtx(i, clr[phi_idx], acn[phi_idx]); }
        n = edgs.size(); for (int i = 0; i < n; i++) {                                    do_action_edg(i, clr[phi_idx], acn[phi_idx]); }
        n = tris.size(); for (int i = 0; i < n; i++) { if (invalid_reconstruction) break; do_action_tri(i, clr[phi_idx], acn[phi_idx]); }
        n = tets.size(); for (int i = 0; i < n; i++) { if (invalid_reconstruction) break; do_action_tet(i, clr[phi_idx], acn[phi_idx]); }

        if (invalid_reconstruction && refine_level < max_refinement_ - initial_refinement)
        {
          vtxs = vtx_tmp;
          edgs = edg_tmp;
          tris = tri_tmp;
          tets = tet_tmp;

          n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
          n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
          n = tets.size(); for (int i = 0; i < n; i++) refine_tet(i);

          refine_level++;
        } else {
//          if (invalid_reconstruction) std::cout << "Cannot resolve invalid geometry\n";
          invalid_reconstruction = false;
        }
      }

      eps *= 0.5;
    }

    // sort everything before integration
    for (int i = 0; i < edgs.size(); i++)
    {
      edg3_t *edg = &edgs[i];
      if (need_swap(edg->vtx0, edg->vtx2)) { swap(edg->vtx0, edg->vtx2); }
    }

    for (int i = 0; i < tris.size(); i++)
    {
      tri3_t *tri = &tris[i];
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); swap(tri->c_vtx12, tri->c_vtx02); swap(tri->ab12, tri->ab02); }
      if (need_swap(tri->vtx1, tri->vtx2)) { swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2); swap(tri->c_vtx02, tri->c_vtx01); swap(tri->ab02, tri->ab01); }
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); swap(tri->c_vtx12, tri->c_vtx02); swap(tri->ab12, tri->ab02); }
    }

    for (int i = 0; i < tets.size(); i++)
    {
      tet3_t *tet = &tets[i];
      if (need_swap(tet->vtx0, tet->vtx1)) { swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1); }
      if (need_swap(tet->vtx1, tet->vtx2)) { swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2); }
      if (need_swap(tet->vtx2, tet->vtx3)) { swap(tet->vtx2, tet->vtx3); swap(tet->tri2, tet->tri3); }
      if (need_swap(tet->vtx0, tet->vtx1)) { swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1); }
      if (need_swap(tet->vtx1, tet->vtx2)) { swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2); }
      if (need_swap(tet->vtx0, tet->vtx1)) { swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1); }
    }

    if (check_for_overlapping_)
    {
      // check for overlapping volumes
      double v_before = volume(tets[0].vtx0, tets[0].vtx1, tets[0].vtx2, tets[0].vtx3);
      double v_after  = 0;

      // compute volume after using linear representation
      for (int i = 0; i < tets.size(); ++i)
        if (!tets[i].is_split)
          v_after += volume(tets[i].vtx0, tets[i].vtx1, tets[i].vtx2, tets[i].vtx3);

      if (fabs(v_before-v_after) > EPS)
      {
        if (initial_refinement == max_refinement_)
        {
          std::cout << "Can't resolve overlapping " << fabs(v_before-v_after) << "\n";
          break;
        } else {
          ++initial_refinement;
          //std::cout << "Overlapping " << fabs(s_before-s_after) << "\n";
          vtxs = vtxs_initial;
          edgs = edgs_initial;
          tris = tris_initial;
          tets = tets_initial;
        }
      } else {
        break;
      }
    } else {
      break;
    }
    break;
  }
}


void simplex3_mls_quadratic_t::construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{

  bool needs_refinement = true;
  int last_vtxs_size = 0;

  int initial_refinement = 0;
  int n;

  std::vector<double> phi_current(nodes_per_tet, -1);

  std::vector<vtx3_t> vtxs_initial = vtxs;
  std::vector<edg3_t> edgs_initial = edgs;
  std::vector<tri3_t> tris_initial = tris;
  std::vector<tet3_t> tets_initial = tets;

  while(1)
  {
    for (int i = 0; i < initial_refinement; ++i)
    {
      n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
      n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
      n = tets.size(); for (int i = 0; i < n; i++) refine_tet(i);
    }

    int refine_level = 0;

    // loop over LSFs
    for (short phi_idx = 0; phi_idx < acn.size(); ++phi_idx)
    {
      for (int i = 0; i < nodes_per_tet; ++i)
      {
        vtxs[i].value  = phi[nodes_per_tet*phi_idx + i];
        phi_current[i] = phi[nodes_per_tet*phi_idx + i];
      }

      last_vtxs_size = nodes_per_tri;

      invalid_reconstruction = true;

      while (invalid_reconstruction)
      {
        needs_refinement = true;

        while (needs_refinement)
        {
          // interpolate to all vertices
          for (int i = last_vtxs_size; i < vtxs.size(); ++i)
            if (!vtxs[i].is_recycled)
            {
              double xyz[3] = { vtxs[i].x, vtxs[i].y, vtxs[i].z };
              vtxs[i].value = interpolate_from_parent (phi_current, xyz );
              perturb(vtxs[i].value, eps);
            }

          // check validity of data on each edge
          needs_refinement = false;
          for (int i = 0; i < edgs.size(); ++i)
            if (!edgs[i].is_split)
            {
              edg3_t *e = &edgs[i];

              double phi0 = vtxs[e->vtx0].value;
              double phi1 = vtxs[e->vtx1].value;
              double phi2 = vtxs[e->vtx2].value;

              if (phi1*phi0 < 0 && phi0*phi2 > 0)
              {
                needs_refinement = true;
                break;
              }

              if (phi0*phi2 > 0)
              {
                double c0 = phi0;
                double c1 = -3.*phi0 + 4.*phi1 -    phi2;
                double c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

                if (fabs(c2) > EPS)
                {
                  double a_ext = -.5*c1/c2;
                  if (a_ext > 0. && a_ext < 1.)
                  {
                    double phi_ext = c0 + c1*a_ext + c2*a_ext*a_ext;

                    if (phi0*phi_ext < 0)
                    {
//                      std::cout << a_ext << " " << phi_ext << "\n";
                      needs_refinement = true;
                      break;
                    }
                  }
                }
              }
            }

          // check validity of data on each face
          if (!needs_refinement)
            for (int i = 0; i < tris.size(); ++i)
              if (!tris[i].is_split)
              {
                tri3_t *f = &tris[i];
                edg3_t *e0 = &edgs[f->edg0];
                edg3_t *e1 = &edgs[f->edg1];
                edg3_t *e2 = &edgs[f->edg2];

                double phi0 = vtxs[f->vtx0].value;
                double phi1 = vtxs[f->vtx1].value;
                double phi2 = vtxs[f->vtx2].value;
                double phi3 = vtxs[e2->vtx1].value;
                double phi4 = vtxs[e0->vtx1].value;
                double phi5 = vtxs[e1->vtx1].value;

                if (phi0*phi1 > 0 && phi1*phi2 > 0 && phi2*phi3 > 0 && phi3*phi4 > 0 && phi4*phi5 > 0)
                {
                  double paa = 2.*phi0 + 2.*phi1 - 4.*phi3;
                  double pab = 4.*phi0 - 4.*phi3 + 4.*phi4 - 4.*phi5;
                  double pbb = 2.*phi0 + 2.*phi2 - 4.*phi5;
                  double pa = -3.*phi0 -    phi1 + 4.*phi3;
                  double pb = -3.*phi0 -    phi2 + 4.*phi5;

                  double det = 4.*paa*pbb - pab*pab;

                  if (fabs(det) > EPS)
                  {
                    double a = (pb*pab - 2.*pa*pbb)/det;
                    double b = (pa*pab - 2.*pb*paa)/det;

                    if (a > 0 && b > 0 && a+b < 1.)
                    {
                      double phi_extremum = paa*a*a + pab*a*b + pbb*b*b + pa*a + pb*b + phi0;

                      if (phi_extremum*phi0 < 0)
                      {
//                        std::cout << "face " << a << " " << b << "\n";
                        needs_refinement = true;
                        break;
                      }
                    }
                  }

                }
              }

          last_vtxs_size = vtxs.size();

          // refine if necessary
          if (needs_refinement && refine_level < max_refinement_ - initial_refinement)
          {
            int n;
            n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
            n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
            n = tets.size(); for (int i = 0; i < n; i++) refine_tet(i);
            refine_level++;
          } else if (needs_refinement) {
            std::cout << "Cannot resolve invalid geometry (bad)\n";
            needs_refinement = false;
          }
        }

        invalid_reconstruction = false;

        std::vector<vtx3_t> vtx_tmp = vtxs;
        std::vector<edg3_t> edg_tmp = edgs;
        std::vector<tri3_t> tri_tmp = tris;
        std::vector<tet3_t> tet_tmp = tets;

        int n;
        n = vtxs.size(); for (int i = 0; i < n; i++) {                                    do_action_vtx(i, clr[phi_idx], acn[phi_idx]); }
        n = edgs.size(); for (int i = 0; i < n; i++) {                                    do_action_edg(i, clr[phi_idx], acn[phi_idx]); }
        n = tris.size(); for (int i = 0; i < n; i++) { if (invalid_reconstruction && refine_level < max_refinement_ - initial_refinement) break; do_action_tri(i, clr[phi_idx], acn[phi_idx]); }
        n = tets.size(); for (int i = 0; i < n; i++) { if (invalid_reconstruction && refine_level < max_refinement_ - initial_refinement) break; do_action_tet(i, clr[phi_idx], acn[phi_idx]); }

        if (invalid_reconstruction && refine_level < max_refinement_ - initial_refinement)
        {
          vtxs = vtx_tmp;
          edgs = edg_tmp;
          tris = tri_tmp;
          tets = tet_tmp;

          n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
          n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
          n = tets.size(); for (int i = 0; i < n; i++) refine_tet(i);

          refine_level++;
        } else {
//          if (invalid_reconstruction) std::cout << "Cannot resolve invalid geometry\n";
          invalid_reconstruction = false;
        }
      }

      eps *= 0.5;
    }

    // sort everything before integration
    for (int i = 0; i < edgs.size(); i++)
    {
      edg3_t *edg = &edgs[i];
      if (need_swap(edg->vtx0, edg->vtx2)) { swap(edg->vtx0, edg->vtx2); }
    }

    for (int i = 0; i < tris.size(); i++)
    {
      tri3_t *tri = &tris[i];
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); swap(tri->c_vtx12, tri->c_vtx02); swap(tri->ab12, tri->ab02); }
      if (need_swap(tri->vtx1, tri->vtx2)) { swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2); swap(tri->c_vtx02, tri->c_vtx01); swap(tri->ab02, tri->ab01); }
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); swap(tri->c_vtx12, tri->c_vtx02); swap(tri->ab12, tri->ab02); }
    }

    for (int i = 0; i < tets.size(); i++)
    {
      tet3_t *tet = &tets[i];
      if (need_swap(tet->vtx0, tet->vtx1)) { swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1); }
      if (need_swap(tet->vtx1, tet->vtx2)) { swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2); }
      if (need_swap(tet->vtx2, tet->vtx3)) { swap(tet->vtx2, tet->vtx3); swap(tet->tri2, tet->tri3); }
      if (need_swap(tet->vtx0, tet->vtx1)) { swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1); }
      if (need_swap(tet->vtx1, tet->vtx2)) { swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2); }
      if (need_swap(tet->vtx0, tet->vtx1)) { swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1); }
    }

    if (check_for_overlapping_)
    {
      // check for overlapping volumes
      double v_before = volume(tets[0].vtx0, tets[0].vtx1, tets[0].vtx2, tets[0].vtx3);
      double v_after  = 0;

      // compute volume after using linear representation
      for (int i = 0; i < tets.size(); ++i)
        if (!tets[i].is_split)
          v_after += volume(tets[i].vtx0, tets[i].vtx1, tets[i].vtx2, tets[i].vtx3);

      if (fabs(v_before-v_after) > EPS)
      {
        if (initial_refinement == max_refinement_)
        {
          std::cout << "Can't resolve overlapping " << fabs(v_before-v_after) << "\n";
          break;
        } else {
          ++initial_refinement;
          //std::cout << "Overlapping " << fabs(s_before-s_after) << "\n";
          vtxs = vtxs_initial;
          edgs = edgs_initial;
          tris = tris_initial;
          tets = tets_initial;
        }
      } else {
        break;
      }
    } else {
      break;
    }
  }

//  n = edgs.size(); for (int i = 0; i < n; i++) if (!edgs[i].is_split && edgs[i].loc == FCE) deform_edge_in_normal_dir(i);
}

void simplex3_mls_quadratic_t::deform_middle_node(double &x_out, double &y_out,
                                                  double x, double y,
                                                  double x0, double y0,
                                                  double x1, double y1,
                                                  double x2, double y2,
                                                  double x3, double y3,
                                                  double x01, double y01)
{

  double Xa  = -x0+x1;
  double Xb  = -x0+x3;
  double Xab = x0-x1+x2-x3;

  double Ya  = -y0+y1;
  double Yb  = -y0+y3;
  double Yab = y0-y1+y2-y3;

  // solve quadratic equation
  double c2 = Xb*Yab - Yb*Xab;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = Ya*Xb - Xa*Yb + Xab*(y-y0) - Yab*(x-x0);   // the expansion of f at the center of (0,1)
  double c0 = Xa*(y-y0)-Ya*(x-x0);

  double b;

  if (fabs(c2) < eps) b = -c0/c1;
  else
  {
//    if (Fn<0) alpha = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
//    else      alpha = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    double b1 = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    double b2 = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    if      (b1 >= 0. && b1 <= 1.) b = b1;
    else if (b2 >= 0. && b2 <= 1.) b = b2;
    else
    {
      std::cout << "Warning: inverse mapping is incorrect! (" << c0 << " " << c1 << " " << c2 << " " << b1 << " " << b2 << ")\n";
      b = b1;
    }
  }

  double a = (x-x0-b*Xb)/(Xa+b*Xab);

  x_out = x0 + a*Xa + b*Xb + a*b*Xab + (1.-b)*((1.-a)*x0 + a*x1)*2.*(1.-a)*a*(-x0+2.*x01-x1);
  y_out = y0 + a*Ya + b*Yb + a*b*Yab + (1.-b)*((1.-a)*y0 + a*y1)*2.*(1.-a)*a*(-y0+2.*y01-y1);

}






//--------------------------------------------------
// Splitting
//--------------------------------------------------
void simplex3_mls_quadratic_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx3_t *vtx = &vtxs[n_vtx];

  if (vtx->is_recycled) return;

  switch (action){
  case INTERSECTION:  if (vtx->value > 0)                                                       vtx->set(OUT, -1, -1, -1);  break;
  case ADDITION:      if (vtx->value < 0)                                                       vtx->set(INS, -1, -1, -1);  break;
  case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==LNE || vtx->loc==PNT)  vtx->set(FCE, cn, -1, -1);  break;
  }
}

void simplex3_mls_quadratic_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg3_t *edg = &edgs[n_edg];

  int c0 = edg->c0;
  int c1 = edg->c1;

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  int num_negatives = 0;
  if (vtxs[edg->vtx0].value < 0) num_negatives++;
  if (vtxs[edg->vtx2].value < 0) num_negatives++;

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  edg->type = num_negatives;
#endif

  // auxiliary variables
  vtx3_t *c_vtx_x, *c_vtx_0x, *c_vtx_x2;
  edg3_t *c_edg0, *c_edg1;
  double r;

  switch (num_negatives){
  case 0: // ++
    /* split an edge */
    // no need to split

    /* apply rules */
    switch (action){
    case INTERSECTION:  edg->set(OUT,-1,-1);  break;
    case ADDITION:      /* do nothig */       break;
    case COLORATION:    /* do nothig */       break;
    }
    break;
  case 1: // -+
      {
//    edgs.reserve(edgs.size()+2);
//    edg = &edgs[n_edg];
    /* split an edge */
    edg->is_split = true;

    // new vertex

    // find intersection
    double a = find_intersection_quadratic(n_edg);
    edg->a = a;

    // map intersection point and new middle points to real space
    double xyz[3];
    double xyz_m[3];
    double xyz_p[3];
    mapping_edg(xyz,  n_edg, a);
    mapping_edg(xyz_m, n_edg, .5*a);
    mapping_edg(xyz_p, n_edg, a + .5*(1.-a));

//    double x_0 = vtxs[edg->vtx0].x, y_0 = vtxs[edg->vtx0].y;
//    double x_1 = vtxs[edg->vtx1].x, y_1 = vtxs[edg->vtx1].y;
//    double x_2 = vtxs[edg->vtx2].x, y_2 = vtxs[edg->vtx2].y;

    // create new vertices
    vtxs.push_back(vtx3_t(xyz_m[0],xyz_m[1],xyz_m[2])); int n_vtx_0x = vtxs.size()-1;
    vtxs.push_back(vtx3_t(xyz_p[0],xyz_p[1],xyz_p[2])); int n_vtx_x2 = vtxs.size()-1;
    vtxs.push_back(vtx3_t(xyz[0],xyz[1],xyz[2]));

    edg->c_vtx_x = vtxs.size()-1;

    // new edges
    edgs.push_back(edg3_t(edg->vtx0,    n_vtx_0x, edg->c_vtx_x)); edg = &edgs[n_edg]; // edges might have changed their addresses
    edgs.push_back(edg3_t(edg->c_vtx_x, n_vtx_x2, edg->vtx2   )); edg = &edgs[n_edg];

    edg->c_edg0 = edgs.size()-2;
    edg->c_edg1 = edgs.size()-1;

    /* apply rules */
    c_vtx_x  = &vtxs[edg->c_vtx_x];
    c_vtx_0x = &vtxs[n_vtx_0x];
    c_vtx_x2 = &vtxs[n_vtx_x2];

    c_edg0  = &edgs[edg->c_edg0];
    c_edg1  = &edgs[edg->c_edg1];

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    c_vtx_x->p_edg = n_edg;
    c_edg0->p_edg  = n_edg;
    c_edg1->p_edg  = n_edg;
#endif

    switch (action){
    case INTERSECTION:
      switch (edg->loc){
      case OUT: c_vtx_x->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_vtx_0x->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
      case INS: c_vtx_x->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
      case FCE: c_vtx_x->set(LNE, c0, cn, -1); c_edg0->set(FCE, c0, -1); c_vtx_0x->set(FCE, c0, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1); if (c0==cn)            c_vtx_x->set(FCE, c0, -1, -1);  break;
      case LNE: c_vtx_x->set(PNT, c0, c1, cn); c_edg0->set(LNE, c0, c1); c_vtx_0x->set(LNE, c0, c1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1); if (c0==cn || c1==cn)  c_vtx_x->set(LNE, c0, c1, -1);  break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    case ADDITION:
      switch (edg->loc){
      case OUT: c_vtx_x->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
      case INS: c_vtx_x->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); c_vtx_x2->set(INS, -1, -1, -1);                                                        break;
      case FCE: c_vtx_x->set(LNE, c0, cn, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(FCE, c0, -1); c_vtx_x2->set(FCE, c0, -1, -1); if (c0==cn)            c_vtx_x->set(FCE, c0, -1, -1);  break;
      case LNE: c_vtx_x->set(PNT, c0, c1, cn); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(LNE, c0, c1); c_vtx_x2->set(LNE, c0, c1, -1); if (c0==cn || c1==cn)  c_vtx_x->set(LNE, c0, c1, -1);  break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    case COLORATION:
      switch (edg->loc){
      case OUT: c_vtx_x->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_vtx_0x->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
      case INS: c_vtx_x->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); c_vtx_x2->set(INS, -1, -1, -1);                                                        break;
      case FCE: c_vtx_x->set(LNE, c0, cn, -1); c_edg0->set(FCE, cn, -1); c_vtx_0x->set(FCE, cn, -1, -1); c_edg1->set(FCE, c0, -1); c_vtx_x2->set(FCE, c0, -1, -1); if (c0==cn)            c_vtx_x->set(FCE, c0, -1, -1);  break;
      case LNE: c_vtx_x->set(PNT, c0, c1, cn); c_edg0->set(FCE, cn, -1); c_vtx_0x->set(FCE, cn, -1, -1); c_edg1->set(LNE, c0, c1); c_vtx_x2->set(LNE, c0, c1, -1); if (c0==cn || c1==cn)  c_vtx_x->set(LNE, c0, c1, -1);  break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    }
    break;
      }
  case 2: // --
    /* split an edge */
    // no need to split

    /* apply rules */
    switch (action) {
    case INTERSECTION:  /* do nothing */                                            break;
    case ADDITION:                                          edg->set(INS, -1, -1);  break;
    case COLORATION:    if (edg->loc==FCE || edg->loc==LNE) edg->set(FCE, cn, -1);  break;
    }
    break;
  }
}

void simplex3_mls_quadratic_t::do_action_tri(int n_tri, int cn, action_t action)
{
  tri3_t *tri = &tris[n_tri];

  if (tri->is_split) return;

  int cc = tri->c;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Determine type */
  int num_negatives = 0;
  if (vtxs[tri->vtx0].value < 0) num_negatives++;
  if (vtxs[tri->vtx1].value < 0) num_negatives++;
  if (vtxs[tri->vtx2].value < 0) num_negatives++;

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  tri->type = num_negatives;

  /* check whether vertices of a triangle and edges coincide */
  if (tri->vtx0 != edgs[tri->edg1].vtx0 || tri->vtx0 != edgs[tri->edg2].vtx0 ||
      tri->vtx1 != edgs[tri->edg0].vtx0 || tri->vtx1 != edgs[tri->edg2].vtx2 ||
      tri->vtx2 != edgs[tri->edg0].vtx2 || tri->vtx2 != edgs[tri->edg1].vtx2)
  {
    std::cout << vtxs[tri->vtx0].value << " " << vtxs[tri->vtx1].value << " " << vtxs[tri->vtx2].value << std::endl;
    throw std::domain_error("[CASL_ERROR]: Vertices of a triangle and edges do not coincide after sorting.");
  }

  /* check whether appropriate edges have been splitted */
  int e0_type_expect, e1_type_expect, e2_type_expect;

  switch (num_negatives){
  case 0: e0_type_expect = 0; e1_type_expect = 0; e2_type_expect = 0; break;
  case 1: e0_type_expect = 0; e1_type_expect = 1; e2_type_expect = 1; break;
  case 2: e0_type_expect = 1; e1_type_expect = 1; e2_type_expect = 2; break;
  case 3: e0_type_expect = 2; e1_type_expect = 2; e2_type_expect = 2; break;
  }

  if (edgs[tri->edg0].type != e0_type_expect || edgs[tri->edg1].type != e1_type_expect || edgs[tri->edg2].type != e2_type_expect)
    throw std::domain_error("[CASL_ERROR]: While splitting a triangle one of edges has an unexpected type.");
#endif

  // auxiliary variables
  edg3_t *c_edg0, *c_edg1;
  edg3_t *edg0, *edg1, *edg2;
  tri3_t *c_tri0, *c_tri1, *c_tri2;
  vtx3_t *vtx_u0, *vtx_u1;

  switch (num_negatives)
  {
  case 0: // (+++)
    /* split a triangle */
    // no need to split

    /* apply rules */
    switch (action){
    case INTERSECTION:  tri->set(OUT, -1);  break;
    case ADDITION:      /* do nothing */    break;
    case COLORATION:    /* do nothing */    break;
    }
    break;

  case 1: // (-++)
      {
//    edgs.reserve(edgs.size()+2);
//    tris.reserve(tris.size()+3);
//    tri = &tris[n_tri];
    /* split a triangle */
    tri->is_split = true;

    // new vertices
    tri->c_vtx01 = edgs[tri->edg2].c_vtx_x;
    tri->c_vtx02 = edgs[tri->edg1].c_vtx_x;

    // vertex along interface
    double ab_u0[2];
    find_middle_node(ab_u0[0], ab_u0[1], 0, edgs[tri->edg1].a, edgs[tri->edg2].a, 0, n_tri);

    if (ab_u0[1] > 1. - ab_u0[0]/edgs[tri->edg2].a)
    {
      invalid_reconstruction = true;
//      std::cout << "Intersecting edges: " << .5*edgs[tri->edg2].a << " " << .5*edgs[tri->edg1].a << " " << ab_u0[0] << " " << ab_u0[1] << "\n";

      double zeta0 = 1. - .5 - .5*edgs[tri->edg1].a;
      double zeta1 = 1. - ab_u0[0]/edgs[tri->edg2].a - ab_u0[1];
      double ratio = (zeta0-eps)/(zeta0-zeta1);

      ab_u0[0] = .5*edgs[tri->edg2].a + ratio*(ab_u0[0] - .5*edgs[tri->edg2].a);
      ab_u0[1] = .5*edgs[tri->edg1].a + ratio*(ab_u0[1] - .5*edgs[tri->edg1].a);
    }

    if (ab_u0[1] > 1. - ab_u0[0]/edgs[tri->edg2].a)
    {
      std::cout << "Intersecting edges: " << .5*edgs[tri->edg2].a << " " << .5*edgs[tri->edg1].a << " " << ab_u0[0] << " " << ab_u0[1] << "\n";
    }

//     DEBUGGING: check if linear reconstruction works
//    a_u0 = 0.5*edgs[tri->edg2].a;
//    b_u0 = 0.5*edgs[tri->edg1].a;

    double xyz_u0[3];
    mapping_tri(xyz_u0, n_tri, ab_u0);

//    // check if not to curvy
    double ab_u0_orig[2] = { .5*edgs[tri->edg2].a, .5*edgs[tri->edg1].a };
    double xyz_u0_orig[3];
    mapping_tri(xyz_u0_orig, n_tri, ab_u0_orig);

    double length = sqrt( SQR(vtxs[tri->c_vtx01].x - vtxs[tri->c_vtx02].x) +
                          SQR(vtxs[tri->c_vtx01].y - vtxs[tri->c_vtx02].y) +
                          SQR(vtxs[tri->c_vtx01].z - vtxs[tri->c_vtx02].z) );

    double deform = sqrt( SQR(xyz_u0[0]-xyz_u0_orig[0]) +
                          SQR(xyz_u0[1]-xyz_u0_orig[1]) +
                          SQR(xyz_u0[2]-xyz_u0_orig[2]) );

//    double length = sqrt( SQR(edgs[tri->edg1].a) + SQR(edgs[tri->edg2].a) );
//    double deform = sqrt( SQR(ab_u0[0] - ab_u0_orig[0]) +
//                          SQR(ab_u0[1] - ab_u0_orig[1]) );

    if (deform > length*curvature_limit_)
    {
//      std::cout << "High curvature: " << length << " " << deform << "\n";
      invalid_reconstruction = true;
    }

    double ab_u1[2] = { 0.5*edgs[tri->edg2].a, 0.5 };

    double a_new, b_new;
    deform_middle_node(a_new, b_new, ab_u1[0], ab_u1[1], 0, edgs[tri->edg1].a, edgs[tri->edg2].a, 0, 1, 0, 0, 1, ab_u0[0], ab_u0[1]);
    ab_u1[0] = a_new;
    ab_u1[1] = b_new;

    double xyz_u1[3];
    mapping_tri(xyz_u1, n_tri, ab_u1);

    //     DEBUGGING: check if linear reconstruction works
//    xyz_u0[0] = 0.5*(vtxs[tri->c_vtx01].x+vtxs[tri->c_vtx02].x);
//    xyz_u0[1] = 0.5*(vtxs[tri->c_vtx01].y+vtxs[tri->c_vtx02].y);
//    xyz_u0[2] = 0.5*(vtxs[tri->c_vtx01].z+vtxs[tri->c_vtx02].z);

//    xyz_u1[0] = 0.5*(vtxs[tri->c_vtx01].x+vtxs[tri->vtx2].x);
//    xyz_u1[1] = 0.5*(vtxs[tri->c_vtx01].y+vtxs[tri->vtx2].y);
//    xyz_u1[2] = 0.5*(vtxs[tri->c_vtx01].z+vtxs[tri->vtx2].z);

    vtxs.push_back(vtx3_t(xyz_u0[0], xyz_u0[1], xyz_u0[2]));
    vtxs.push_back(vtx3_t(xyz_u1[0], xyz_u1[1], xyz_u1[2]));

    int u0 = vtxs.size()-2;
    int u1 = vtxs.size()-1;

    // new edges
    edgs.push_back(edg3_t(tri->c_vtx01, u0, tri->c_vtx02));
    edgs.push_back(edg3_t(tri->c_vtx01, u1, tri->vtx2   ));

    // edges might have changed their addresses
    edg0 = &edgs[tri->edg0];
    edg1 = &edgs[tri->edg1];
    edg2 = &edgs[tri->edg2];

    tri->c_edg0 = edgs.size()-2;
    tri->c_edg1 = edgs.size()-1;

    // new triangles
    tris.push_back(tri3_t(tri->vtx0,    tri->c_vtx01, tri->c_vtx02, tri->c_edg0,  edg1->c_edg0, edg2->c_edg0)); tri = &tris[n_tri];
    tris.push_back(tri3_t(tri->c_vtx01, tri->c_vtx02, tri->vtx2,    edg1->c_edg1, tri->c_edg1,  tri->c_edg0));  tri = &tris[n_tri];
    tris.push_back(tri3_t(tri->c_vtx01, tri->vtx1,    tri->vtx2,    tri->edg0,    tri->c_edg1,  edg2->c_edg1)); tri = &tris[n_tri];

    tri->c_tri0 = tris.size()-3;
    tri->c_tri1 = tris.size()-2;
    tri->c_tri2 = tris.size()-1;

    /* apply rules */
    c_edg0 = &edgs[tri->c_edg0];
    c_edg1 = &edgs[tri->c_edg1];

    c_tri0 = &tris[tri->c_tri0];  c_tri0->dir = tri->dir; c_tri0->p_lsf = tri->p_lsf;
    c_tri1 = &tris[tri->c_tri1];  c_tri1->dir = tri->dir; c_tri1->p_lsf = tri->p_lsf;
    c_tri2 = &tris[tri->c_tri2];  c_tri2->dir = tri->dir; c_tri2->p_lsf = tri->p_lsf;

    vtx_u0 = &vtxs[u0];
    vtx_u1 = &vtxs[u1];

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
      throw std::domain_error("[CASL_ERROR]: While splitting a triangle one of child triangles is not consistent.");

    // track the parent
    c_edg0->p_tri = n_tri;
    c_edg1->p_tri = n_tri;
    c_tri0->p_tri = n_tri;
    c_tri1->p_tri = n_tri;
    c_tri2->p_tri = n_tri;
#endif

    switch (action){
    case INTERSECTION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(FCE, cn, -1); vtx_u0->set(FCE, cn, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case FCE: c_edg0->set(LNE, cc, cn); vtx_u0->set(LNE, cc, cn, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(FCE, cc); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tri->loc){
      case OUT: c_edg0->set(FCE, cn, -1); vtx_u0->set(FCE, cn, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(LNE, cc, cn); vtx_u0->set(LNE, cc, cn, -1); c_edg1->set(FCE, cc, -1); vtx_u1->set(FCE, cc, -1, -1); c_tri0->set(INS, -1); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(LNE, cc, cn); vtx_u0->set(LNE, cc, cn, -1); c_edg1->set(FCE, cc, -1); vtx_u1->set(FCE, cc, -1, -1); c_tri0->set(FCE, cn); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    }
    break;
      }
  case 2: // (--+)
      {
//    edgs.reserve(edgs.size()+2);
//    tris.reserve(tris.size()+3);
//    tri = &tris[n_tri];
    /* split a triangle */
    tri->is_split = true;

    tri->c_vtx02 = edgs[tri->edg1].c_vtx_x;
    tri->c_vtx12 = edgs[tri->edg0].c_vtx_x;

    // vertex along interface
    double ab_u1[2];
    find_middle_node(ab_u1[0], ab_u1[1], 1.-edgs[tri->edg0].a, edgs[tri->edg0].a, 0, edgs[tri->edg1].a, n_tri);

    if (ab_u1[1] < ab_u1[0]*edgs[tri->edg0].a/(1.-edgs[tri->edg0].a))
    {
      invalid_reconstruction = true;
      double ab_u1_orig[2] = {.5*(1.-edgs[tri->edg0].a), .5*(edgs[tri->edg0].a + edgs[tri->edg1].a)};

      double zeta0 = ab_u1_orig[1] - ab_u1_orig[0]*edgs[tri->edg0].a/(1.-edgs[tri->edg0].a);
      double zeta1 = ab_u1[1] - ab_u1[0]*edgs[tri->edg0].a/(1.-edgs[tri->edg0].a);
      double ratio = (zeta0-eps)/(zeta0-zeta1);

      ab_u1[0] = .5*(1.-edgs[tri->edg0].a) + ratio*(ab_u1[0] - .5*(1.-edgs[tri->edg0].a));
      ab_u1[1] = .5*(edgs[tri->edg0].a+edgs[tri->edg1].a) + ratio*(ab_u1[1] - .5*(edgs[tri->edg0].a+edgs[tri->edg1].a));
    }

    if (ab_u1[1] < ab_u1[0]*edgs[tri->edg0].a/(1.-edgs[tri->edg0].a))
    {
      std::cout << "Intersecting edges: " << .5*(1.-edgs[tri->edg0].a) << " " << .5*(edgs[tri->edg0].a+edgs[tri->edg1].a) << " " << ab_u1[0] << " " << ab_u1[1] << "\n";
    }

    // DEBUGGING: check if linear reconstruction works
//    a_u1 = 0.5*(1.-edgs[tri->edg0].a);
//    b_u1 = 0.5*(edgs[tri->edg1].a+edgs[tri->edg0].a);

    double xyz_u1[3];
    mapping_tri(xyz_u1, n_tri, ab_u1);


    // check if not to curvy
    double ab_u1_orig[2] = { .5*(1.-edgs[tri->edg0].a), .5*(edgs[tri->edg0].a + edgs[tri->edg1].a) };
    double xyz_u1_orig[3];
    mapping_tri(xyz_u1_orig, n_tri, ab_u1_orig);

    double length = sqrt( SQR(vtxs[tri->c_vtx02].x - vtxs[tri->c_vtx12].x) +
                          SQR(vtxs[tri->c_vtx02].y - vtxs[tri->c_vtx12].y) +
                          SQR(vtxs[tri->c_vtx02].z - vtxs[tri->c_vtx12].z) );

    double deform = sqrt( SQR(xyz_u1[0]-xyz_u1_orig[0]) +
                          SQR(xyz_u1[1]-xyz_u1_orig[1]) +
                          SQR(xyz_u1[2]-xyz_u1_orig[2]) );

//    double length = sqrt( SQR(1.-edgs[tri->edg0].a) + SQR(edgs[tri->edg0].a-edgs[tri->edg1].a) );
//    double deform = sqrt( SQR(ab_u1[0] - ab_u1_orig[0]) +
//                          SQR(ab_u1[1] - ab_u1_orig[1]) );

    if (deform > length*curvature_limit_)
    {
//      std::cout << "High curvature: " << length << " " << deform << "\n";
      invalid_reconstruction = true;
    }

    double ab_u0[2] = { 0.5*(1.-edgs[tri->edg0].a), 0.5*edgs[tri->edg0].a };

    double a_new, b_new;
    deform_middle_node(a_new, b_new, ab_u0[0], ab_u0[1],  (1.-edgs[tri->edg0].a),  edgs[tri->edg0].a, 0, edgs[tri->edg1].a, 0, 0, 1, 0, ab_u1[0], ab_u1[1]);
    ab_u0[0] = a_new;
    ab_u0[1] = b_new;

    double xyz_u0[3];
    mapping_tri(xyz_u0, n_tri, ab_u0);

    // DEBUGGING: check if linear reconstruction works
//    xyz_u0[0] = 0.5*(vtxs[tri->vtx0].x+vtxs[tri->c_vtx12].x);
//    xyz_u0[1] = 0.5*(vtxs[tri->vtx0].y+vtxs[tri->c_vtx12].y);
//    xyz_u0[2] = 0.5*(vtxs[tri->vtx0].z+vtxs[tri->c_vtx12].z);

//    xyz_u1[0] = 0.5*(vtxs[tri->c_vtx02].x+vtxs[tri->c_vtx12].x);
//    xyz_u1[1] = 0.5*(vtxs[tri->c_vtx02].y+vtxs[tri->c_vtx12].y);
//    xyz_u1[2] = 0.5*(vtxs[tri->c_vtx02].z+vtxs[tri->c_vtx12].z);

    vtxs.push_back(vtx3_t(xyz_u0[0], xyz_u0[1], xyz_u0[2]));
    vtxs.push_back(vtx3_t(xyz_u1[0], xyz_u1[1], xyz_u1[2]));

    int u0 = vtxs.size()-2;
    int u1 = vtxs.size()-1;

    // create new edges
    edgs.push_back(edg3_t(tri->vtx0,    u0, tri->c_vtx12));
    edgs.push_back(edg3_t(tri->c_vtx02, u1, tri->c_vtx12));

    // edges might have changed their addresses
    edg0 = &edgs[tri->edg0];
    edg1 = &edgs[tri->edg1];
    edg2 = &edgs[tri->edg2];

    tri->c_edg0 = edgs.size()-2;
    tri->c_edg1 = edgs.size()-1;

    tris.push_back(tri3_t(tri->vtx0,    tri->vtx1,    tri->c_vtx12, edg0->c_edg0, tri->c_edg0,  tri->edg2   )); tri = &tris[n_tri];
    tris.push_back(tri3_t(tri->vtx0,    tri->c_vtx02, tri->c_vtx12, tri->c_edg1,  tri->c_edg0,  edg1->c_edg0)); tri = &tris[n_tri];
    tris.push_back(tri3_t(tri->c_vtx02, tri->c_vtx12, tri->vtx2,    edg0->c_edg1, edg1->c_edg1, tri->c_edg1 )); tri = &tris[n_tri];

    tri->c_tri0 = tris.size()-3;
    tri->c_tri1 = tris.size()-2;
    tri->c_tri2 = tris.size()-1;

    /* apply rules */
    c_edg0 = &edgs[tri->c_edg0];
    c_edg1 = &edgs[tri->c_edg1];

    c_tri0 = &tris[tri->c_tri0];  c_tri0->dir = tri->dir; c_tri0->p_lsf = tri->p_lsf;
    c_tri1 = &tris[tri->c_tri1];  c_tri1->dir = tri->dir; c_tri1->p_lsf = tri->p_lsf;
    c_tri2 = &tris[tri->c_tri2];  c_tri2->dir = tri->dir; c_tri2->p_lsf = tri->p_lsf;

    vtx_u0 = &vtxs[u0];
    vtx_u1 = &vtxs[u1];

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
      throw std::domain_error("[CASL_ERROR]: While splitting a triangle one of child triangles is not consistent.");

    // track the parent
    c_edg0->p_tri = n_tri;
    c_edg1->p_tri = n_tri;
    c_tri0->p_tri = n_tri;
    c_tri1->p_tri = n_tri;
    c_tri2->p_tri = n_tri;
#endif

    switch (action){
    case INTERSECTION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(FCE, cn, -1); vtx_u1->set(FCE, cn, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
      case FCE: c_edg0->set(FCE, cc, -1); vtx_u0->set(FCE, cc, -1, -1); c_edg1->set(LNE, cc, cn); vtx_u1->set(LNE, cc, cn, -1); c_tri0->set(FCE, cc); c_tri1->set(FCE, cc); c_tri2->set(OUT, -1); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tri->loc){
      case OUT: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(FCE, cn, -1); vtx_u1->set(FCE, cn, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(LNE, cc, cn); vtx_u1->set(LNE, cc, cn, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(FCE, cn, -1); vtx_u0->set(FCE, cn, -1, -1); c_edg1->set(LNE, cc, cn); vtx_u1->set(LNE, cc, cn, -1); c_tri0->set(FCE, cn); c_tri1->set(FCE, cn); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    }
    break;
      }

  case 3: // (---)
    /* split a triangle */
    // no need to split

    /* apply rules */
    switch (action){
    case INTERSECTION:  /* do nothing */                          break;
    case ADDITION:                            tri->set(INS, -1);  break;
    case COLORATION:    if (tri->loc == FCE)  tri->set(FCE, cn);  break;
    }
    break;
  }
}
void simplex3_mls_quadratic_t::do_action_tet(int n_tet, int cn, action_t action)
{
  tet3_t *tet = &tets[n_tet];

  if (tet->is_split) return;

  /* Sort vertices */
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx2, tet->vtx3)) {swap(tet->vtx2, tet->vtx3); swap(tet->tri2, tet->tri3);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}

  /* Determine type */
  int num_negatives = 0;
  if (vtxs[tet->vtx0].value < 0) num_negatives++;
  if (vtxs[tet->vtx1].value < 0) num_negatives++;
  if (vtxs[tet->vtx2].value < 0) num_negatives++;
  if (vtxs[tet->vtx3].value < 0) num_negatives++;

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  tet->type = num_negatives;

  /* check whether vertices coincide */
  if (tet->vtx0 != tris[tet->tri1].vtx0 || tet->vtx0 != tris[tet->tri2].vtx0 || tet->vtx0 != tris[tet->tri3].vtx0 ||
      tet->vtx1 != tris[tet->tri0].vtx0 || tet->vtx1 != tris[tet->tri2].vtx1 || tet->vtx1 != tris[tet->tri3].vtx1 ||
      tet->vtx2 != tris[tet->tri1].vtx1 || tet->vtx2 != tris[tet->tri0].vtx1 || tet->vtx2 != tris[tet->tri3].vtx2 ||
      tet->vtx3 != tris[tet->tri1].vtx2 || tet->vtx3 != tris[tet->tri2].vtx2 || tet->vtx3 != tris[tet->tri0].vtx2)
    throw std::domain_error("[CASL_ERROR]: Vertices of a tetrahedron do not coincide with vertices of triangles after sorting.");

  /* check whether edges coincide */
  if (tris[tet->tri0].edg0 != tris[tet->tri1].edg0 || tris[tet->tri0].edg1 != tris[tet->tri2].edg0 || tris[tet->tri0].edg2 != tris[tet->tri3].edg0 ||
      tris[tet->tri1].edg1 != tris[tet->tri2].edg1 || tris[tet->tri1].edg2 != tris[tet->tri3].edg1 ||
      tris[tet->tri2].edg2 != tris[tet->tri3].edg2)
    throw std::domain_error("[CASL_ERROR]: Edges of different triangles in a tetrahedron do not coincide.");

  /* check if appropriate triangles have been splitted */
  int t0_type_expect, t1_type_expect, t2_type_expect, t3_type_expect;

  switch (num_negatives){
  case 0: t0_type_expect = 0; t1_type_expect = 0; t2_type_expect = 0; t3_type_expect = 0; break;
  case 1: t0_type_expect = 0; t1_type_expect = 1; t2_type_expect = 1; t3_type_expect = 1; break;
  case 2: t0_type_expect = 1; t1_type_expect = 1; t2_type_expect = 2; t3_type_expect = 2; break;
  case 3: t0_type_expect = 2; t1_type_expect = 2; t2_type_expect = 2; t3_type_expect = 3; break;
  case 4: t0_type_expect = 3; t1_type_expect = 3; t2_type_expect = 3; t3_type_expect = 3; break;
  }

  if (tris[tet->tri0].type != t0_type_expect ||
      tris[tet->tri1].type != t1_type_expect ||
      tris[tet->tri2].type != t2_type_expect ||
      tris[tet->tri3].type != t3_type_expect)
    throw std::domain_error("CASL_ERROR]: While splitting a tetrahedron one of triangles has an unexpected type.");
#endif

  edg3_t *c_edg;
  tri3_t *c_tri0, *c_tri1, *c_tri2, *c_tri3, *c_tri4, *c_tri5;
  tet3_t *c_tet0, *c_tet1, *c_tet2, *c_tet3, *c_tet4, *c_tet5;
  int n_tris, n_tets;
  tri3_t *tri0, *tri1, *tri2, *tri3;

  switch (num_negatives)
  {
  case 0: /* (++++) */
    /* split a tetrahedron */
    // no need to split

    /* apply rules */
    switch (action){
    case INTERSECTION:  tet->set(OUT);    break;
    case ADDITION:      /* do nothing */  break;
    case COLORATION:    /* do nothing */  break;
    }
    break;

  case 1: /* (-+++) */
//    tris.reserve(tris.size()+3);
//    tets.reserve(tets.size()+4);
//    tet = &tets[n_tet];
    /* split a tetrahedron */
    tet->is_split = true;

    // new vertices
    tet->c_vtx01 = tris[tet->tri2].c_vtx01;
    tet->c_vtx02 = tris[tet->tri1].c_vtx01;
    tet->c_vtx03 = tris[tet->tri1].c_vtx02;

    // new triangles
    tris.push_back(tri3_t(tet->c_vtx01, tet->c_vtx02, tet->c_vtx03, tris[tet->tri1].c_edg0, tris[tet->tri2].c_edg0, tris[tet->tri3].c_edg0));
    tris.push_back(tri3_t(tet->c_vtx01, tet->c_vtx02, tet->vtx3,    tris[tet->tri1].c_edg1, tris[tet->tri2].c_edg1, tris[tet->tri3].c_edg0));
    tris.push_back(tri3_t(tet->c_vtx01, tet->vtx2,    tet->vtx3,    tris[tet->tri1].edg0,   tris[tet->tri2].c_edg1, tris[tet->tri3].c_edg1));

    tet->c_tri0 = tris.size()-3;
    tet->c_tri1 = tris.size()-2;
    tet->c_tri2 = tris.size()-1;

//    construct_proper_mapping(tet->c_tri0, -1);

    tri0 = &tris[tet->tri0];
    tri1 = &tris[tet->tri1];
    tri2 = &tris[tet->tri2];
    tri3 = &tris[tet->tri3];

    // new tetrahedra
    tets.push_back(tet3_t(tet->vtx0,    tet->c_vtx01, tet->c_vtx02, tet->c_vtx03, tet->c_tri0,  tri1->c_tri0, tri2->c_tri0, tri3->c_tri0)); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->c_vtx01, tet->c_vtx02, tet->c_vtx03, tet->vtx3,    tri1->c_tri1, tri2->c_tri1, tet->c_tri1,  tet->c_tri0 )); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->c_vtx01, tet->c_vtx02, tet->vtx2,    tet->vtx3,    tri1->c_tri2, tet->c_tri2,  tet->c_tri1,  tri3->c_tri1)); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->c_vtx01, tet->vtx1,    tet->vtx2,    tet->vtx3,    tet->tri0,    tet->c_tri2,  tri2->c_tri2, tri3->c_tri2)); tet = &tets[n_tet];

    tet->c_tet0 = tets.size()-4;
    tet->c_tet1 = tets.size()-3;
    tet->c_tet2 = tets.size()-2;
    tet->c_tet3 = tets.size()-1;

    /* apply rules */
    c_tri0 = &tris[tet->c_tri0];
    c_tri1 = &tris[tet->c_tri1];
    c_tri2 = &tris[tet->c_tri2];

    c_tet0 = &tets[tet->c_tet0];
    c_tet1 = &tets[tet->c_tet1];
    c_tet2 = &tets[tet->c_tet2];
    c_tet3 = &tets[tet->c_tet3];

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2))
      throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child triangles is not consistent.");

    if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) || !tet_is_ok(tet->c_tet3))
      throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child tetrahedra is not consistent.");

    c_tri0->p_tet = n_tet;
    c_tri1->p_tet = n_tet;
    c_tri2->p_tet = n_tet;

    c_tet0->p_tet = n_tet;
    c_tet1->p_tet = n_tet;
    c_tet2->p_tet = n_tet;
    c_tet3->p_tet = n_tet;
#endif

    if (action == INTERSECTION || action == ADDITION) c_tri0->p_lsf = cn;

    switch (action)
    {
    case INTERSECTION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(FCE, cn); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(INS); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tet->loc){
      case OUT: c_tri0->set(FCE, cn); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(INS); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    }
    break;

  case 2: // --++
  {
//    tris.reserve(tris.size()+6);
//    tets.reserve(tets.size()+6);
//    tet = &tets[n_tet];
    /* split a tetrahedron */
    tet->is_split = true;

    // vertices
    tet->c_vtx02 = tris[tet->tri1].c_vtx01;
    tet->c_vtx03 = tris[tet->tri1].c_vtx02;
    tet->c_vtx12 = tris[tet->tri0].c_vtx01;
    tet->c_vtx13 = tris[tet->tri0].c_vtx02;

    double abc_u[3];
    double xyz_u[3];

    find_middle_node_tet(abc_u, n_tet);

    mapping_tet(xyz_u, n_tet, abc_u);

    double xyz_u_lin[3];
    xyz_u_lin[0] = 0.5*(vtxs[tet->c_vtx03].x+vtxs[tet->c_vtx12].x);
    xyz_u_lin[1] = 0.5*(vtxs[tet->c_vtx03].y+vtxs[tet->c_vtx12].y);
    xyz_u_lin[2] = 0.5*(vtxs[tet->c_vtx03].z+vtxs[tet->c_vtx12].z);

    double length = sqrt( SQR(vtxs[tet->c_vtx03].x - vtxs[tet->c_vtx12].x) +
                          SQR(vtxs[tet->c_vtx03].y - vtxs[tet->c_vtx12].y) +
                          SQR(vtxs[tet->c_vtx03].z - vtxs[tet->c_vtx12].z) );

    double deform = sqrt( SQR(xyz_u[0]-xyz_u_lin[0]) +
                          SQR(xyz_u[1]-xyz_u_lin[1]) +
                          SQR(xyz_u[2]-xyz_u_lin[2]) );

    if (deform > length*curvature_limit_)
    {
//      std::cout << "High curvature: " << length << " " << deform << "\n";
      invalid_reconstruction = true;
    }

//    xyz_u_lin[0] = 0.5*(vtxs[tet->c_vtx13].x+vtxs[tet->c_vtx02].x);
//    xyz_u_lin[1] = 0.5*(vtxs[tet->c_vtx13].y+vtxs[tet->c_vtx02].y);
//    xyz_u_lin[2] = 0.5*(vtxs[tet->c_vtx13].z+vtxs[tet->c_vtx02].z);
//    xyz_u[0] = 0.5*(vtxs[tet->c_vtx03].x+vtxs[tet->c_vtx12].x);
//    xyz_u[1] = 0.5*(vtxs[tet->c_vtx03].y+vtxs[tet->c_vtx12].y);
//    xyz_u[2] = 0.5*(vtxs[tet->c_vtx03].z+vtxs[tet->c_vtx12].z);

    vtxs.push_back(vtx3_t(xyz_u[0], xyz_u[1], xyz_u[2]));
//    vtxs.push_back(vtx3_t(xyz_u_lin[0], xyz_u_lin[1], xyz_u_lin[2]));

    int vn = vtxs.size()-1;

    // new edge
    edgs.push_back(edg3_t(tet->c_vtx03, vn, tet->c_vtx12));
    tet->c_edg = edgs.size()-1;

    // new triangles
    tris.push_back(tri3_t(tet->vtx0,    tet->c_vtx12, tet->c_vtx13, tris[tet->tri0].c_edg0, tris[tet->tri2].c_edg0,             tris[tet->tri3].c_edg0            ));
    tris.push_back(tri3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx12, tet->c_edg,             tris[tet->tri3].c_edg0,             edgs[tris[tet->tri1].edg1].c_edg0 ));
    tris.push_back(tri3_t(tet->c_vtx02, tet->c_vtx03, tet->c_vtx12, tet->c_edg,             tris[tet->tri3].c_edg1,             tris[tet->tri1].c_edg0            ));
    tris.push_back(tri3_t(tet->c_vtx03, tet->c_vtx12, tet->c_vtx13, tris[tet->tri0].c_edg0, tris[tet->tri2].c_edg1,             tet->c_edg                        ));
    tris.push_back(tri3_t(tet->c_vtx03, tet->c_vtx12, tet->vtx3,    tris[tet->tri0].c_edg1, edgs[tris[tet->tri1].edg1].c_edg1,  tet->c_edg                        ));
    tris.push_back(tri3_t(tet->c_vtx02, tet->c_vtx12, tet->vtx3,    tris[tet->tri0].c_edg1, tris[tet->tri1].c_edg1,             tris[tet->tri3].c_edg1            ));

    n_tris = tris.size();
    tet->c_tri0 = n_tris-6;
    tet->c_tri1 = n_tris-5;
    tet->c_tri2 = n_tris-4;
    tet->c_tri3 = n_tris-3;
    tet->c_tri4 = n_tris-2;
    tet->c_tri5 = n_tris-1;

    tri0 = &tris[tet->tri0];
    tri1 = &tris[tet->tri1];
    tri2 = &tris[tet->tri2];
    tri3 = &tris[tet->tri3];

    // new tetrahedra
    tets.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->c_vtx12, tet->c_vtx13, tri0->c_tri0, tet->c_tri0,  tri2->c_tri0, tri3->c_tri0)); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx12, tet->c_vtx13, tet->c_tri3,  tet->c_tri0,  tri2->c_tri1, tet->c_tri1 )); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->vtx0,    tet->c_vtx02, tet->c_vtx03, tet->c_vtx12, tet->c_tri2,  tet->c_tri1,  tri3->c_tri1, tri1->c_tri0)); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->c_vtx03, tet->c_vtx12, tet->c_vtx13, tet->vtx3,    tri0->c_tri1, tri2->c_tri2, tet->c_tri4,  tet->c_tri3 )); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->c_vtx02, tet->c_vtx03, tet->c_vtx12, tet->vtx3,    tet->c_tri4,  tet->c_tri5,  tri1->c_tri1, tet->c_tri2 )); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->c_vtx02, tet->c_vtx12, tet->vtx2,    tet->vtx3,    tri0->c_tri2, tri1->c_tri2, tet->c_tri5,  tri3->c_tri2)); tet = &tets[n_tet];

    n_tets = tets.size();
    tet->c_tet0 = n_tets-6;
    tet->c_tet1 = n_tets-5;
    tet->c_tet2 = n_tets-4;
    tet->c_tet3 = n_tets-3;
    tet->c_tet4 = n_tets-2;
    tet->c_tet5 = n_tets-1;

//    construct_proper_mapping(tet->c_tri2, -1);
//    construct_proper_mapping(tet->c_tri3, -1);

    /* apply rules */
    vtx3_t *c_vtx = &vtxs[vn];

    c_edg = &edgs[tet->c_edg];

    c_tri0 = &tris[tet->c_tri0];
    c_tri1 = &tris[tet->c_tri1];
    c_tri2 = &tris[tet->c_tri2];
    c_tri3 = &tris[tet->c_tri3];
    c_tri4 = &tris[tet->c_tri4];
    c_tri5 = &tris[tet->c_tri5];

    c_tet0 = &tets[tet->c_tet0];
    c_tet1 = &tets[tet->c_tet1];
    c_tet2 = &tets[tet->c_tet2];
    c_tet3 = &tets[tet->c_tet3];
    c_tet4 = &tets[tet->c_tet4];
    c_tet5 = &tets[tet->c_tet5];


#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2) ||
        !tri_is_ok(tet->c_tri3) || !tri_is_ok(tet->c_tri4) || !tri_is_ok(tet->c_tri5))
      throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child triangles is not consistent.");

    if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) ||
        !tet_is_ok(tet->c_tet3) || !tet_is_ok(tet->c_tet4) || !tet_is_ok(tet->c_tet5))
      throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child tetrahedra is not consistent.");

    c_edg->p_tet = n_tet;

    c_tri0->p_tet = n_tet;
    c_tri1->p_tet = n_tet;
    c_tri2->p_tet = n_tet;
    c_tri3->p_tet = n_tet;
    c_tri4->p_tet = n_tet;
    c_tri5->p_tet = n_tet;

    c_tet0->p_tet = n_tet;
    c_tet1->p_tet = n_tet;
    c_tet2->p_tet = n_tet;
    c_tet3->p_tet = n_tet;
    c_tet4->p_tet = n_tet;
    c_tet5->p_tet = n_tet;
#endif

    if (action == INTERSECTION || action == ADDITION){
      c_tri2->p_lsf = cn;
      c_tri3->p_lsf = cn;
    }

    switch (action)
    {
    case INTERSECTION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT,-1);  c_tet0->set(OUT); c_edg->set(OUT,-1,-1); c_vtx->set(OUT,-1,-1,-1);
                c_tri1->set(OUT,-1);  c_tet1->set(OUT);
                c_tri2->set(OUT,-1);  c_tet2->set(OUT);
                c_tri3->set(OUT,-1);  c_tet3->set(OUT);
                c_tri4->set(OUT,-1);  c_tet4->set(OUT);
                c_tri5->set(OUT,-1);  c_tet5->set(OUT);
                break;
      case INS: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(FCE,cn,-1); c_vtx->set(FCE,cn,-1,-1);
                c_tri1->set(INS,-1);  c_tet1->set(INS);
                c_tri2->set(FCE,cn);  c_tet2->set(INS);
                c_tri3->set(FCE,cn);  c_tet3->set(OUT);
                c_tri4->set(OUT,-1);  c_tet4->set(OUT);
                c_tri5->set(OUT,-1);  c_tet5->set(OUT);
                break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tet->loc){
      case OUT: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(FCE,cn,-1); c_vtx->set(FCE,cn,-1,-1);
                c_tri1->set(INS,-1);  c_tet1->set(INS);
                c_tri2->set(FCE,cn);  c_tet2->set(INS);
                c_tri3->set(FCE,cn);  c_tet3->set(OUT);
                c_tri4->set(OUT,-1);  c_tet4->set(OUT);
                c_tri5->set(OUT,-1);  c_tet5->set(OUT);
                break;
      case INS: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(INS,-1,-1); c_vtx->set(INS,-1,-1,-1);
                c_tri1->set(INS,-1);  c_tet1->set(INS);
                c_tri2->set(INS,-1);  c_tet2->set(INS);
                c_tri3->set(INS,-1);  c_tet3->set(INS);
                c_tri4->set(INS,-1);  c_tet4->set(INS);
                c_tri5->set(INS,-1);  c_tet5->set(INS);
                break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT,-1);  c_tet0->set(OUT); c_edg->set(OUT,-1,-1); c_vtx->set(OUT,-1,-1,-1);
                c_tri1->set(OUT,-1);  c_tet1->set(OUT);
                c_tri2->set(OUT,-1);  c_tet2->set(OUT);
                c_tri3->set(OUT,-1);  c_tet3->set(OUT);
                c_tri4->set(OUT,-1);  c_tet4->set(OUT);
                c_tri5->set(OUT,-1);  c_tet5->set(OUT);
                break;
      case INS: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(INS,-1,-1); c_vtx->set(INS,-1,-1,-1);
                c_tri1->set(INS,-1);  c_tet1->set(INS);
                c_tri2->set(INS,-1);  c_tet2->set(INS);
                c_tri3->set(INS,-1);  c_tet3->set(INS);
                c_tri4->set(INS,-1);  c_tet4->set(INS);
                c_tri5->set(INS,-1);  c_tet5->set(INS);
                break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    }
    break;
  }
  case 3: // ---+
//    tris.reserve(tris.size()+3);
//    tets.reserve(tets.size()+4);
//    tet = &tets[n_tet];
    /* split a tetrahedron */
    tet->is_split = true;

    // vertices
    tet->c_vtx03 = tris[tet->tri1].c_vtx02;
    tet->c_vtx13 = tris[tet->tri0].c_vtx02;
    tet->c_vtx23 = tris[tet->tri0].c_vtx12;

    // new triangles
    tris.push_back(tri3_t(tet->vtx0,    tet->vtx1,    tet->c_vtx23, tris[tet->tri0].c_edg0, tris[tet->tri1].c_edg0, tris[tet->tri2].edg2  ));
    tris.push_back(tri3_t(tet->vtx0,    tet->c_vtx13, tet->c_vtx23, tris[tet->tri0].c_edg1, tris[tet->tri1].c_edg0, tris[tet->tri2].c_edg0));
    tris.push_back(tri3_t(tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tris[tet->tri0].c_edg1, tris[tet->tri1].c_edg1, tris[tet->tri2].c_edg1));

    n_tris = tris.size();
    tet->c_tri0 = n_tris - 3;
    tet->c_tri1 = n_tris - 2;
    tet->c_tri2 = n_tris - 1;

    tri0 = &tris[tet->tri0];
    tri1 = &tris[tet->tri1];
    tri2 = &tris[tet->tri2];
    tri3 = &tris[tet->tri3];

    // new tetrahedra
    tets.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->vtx2,    tet->c_vtx23, tri0->c_tri0, tri1->c_tri0, tet->c_tri0,  tet->tri3   )); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->c_vtx13, tet->c_vtx23, tri0->c_tri1, tet->c_tri1,  tet->c_tri0,  tri2->c_tri0)); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tet->c_tri2,  tet->c_tri1,  tri1->c_tri1, tri2->c_tri1)); tet = &tets[n_tet];
    tets.push_back(tet3_t(tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tet->vtx3,    tri0->c_tri2, tri1->c_tri2, tri2->c_tri2, tet->c_tri2 )); tet = &tets[n_tet];

    n_tets = tets.size();
    tet->c_tet0 = n_tets-4;
    tet->c_tet1 = n_tets-3;
    tet->c_tet2 = n_tets-2;
    tet->c_tet3 = n_tets-1;

//    construct_proper_mapping(tet->c_tri2, -1);

    /* apply rules */
    c_tri0 = &tris[tet->c_tri0];
    c_tri1 = &tris[tet->c_tri1];
    c_tri2 = &tris[tet->c_tri2];

    c_tet0 = &tets[tet->c_tet0];
    c_tet1 = &tets[tet->c_tet1];
    c_tet2 = &tets[tet->c_tet2];
    c_tet3 = &tets[tet->c_tet3];

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2))
      throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child triangles is not consistent.");

    if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) || !tet_is_ok(tet->c_tet3))
      throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child tetrahedra is not consistent.");

    c_tri0->p_tet = n_tet;
    c_tri1->p_tet = n_tet;
    c_tri2->p_tet = n_tet;

    c_tet0->p_tet = n_tet;
    c_tet1->p_tet = n_tet;
    c_tet2->p_tet = n_tet;
    c_tet3->p_tet = n_tet;
#endif

    if (action == INTERSECTION || action == ADDITION) c_tri2->p_lsf = cn;

    switch (action)
    {
    case INTERSECTION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT,-1); c_tri1->set(OUT,-1); c_tri2->set(OUT,-1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(FCE,cn); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(OUT); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tet->loc){
      case OUT: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(FCE,cn); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT,-1); c_tri1->set(OUT,-1); c_tri2->set(OUT,-1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default: ;
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    }
    break;

  case 4: // ----
    // no need to split
    switch (action){
    case INTERSECTION:  /* do nothig */   break;
    case ADDITION:      tet->set(INS);    break;
    case COLORATION:    /* do nothig */   break;
    }
    break;
  }
}










//--------------------------------------------------
// Auxiliary tools for splitting
//--------------------------------------------------
double simplex3_mls_quadratic_t::find_intersection_quadratic(int e)
{
  double f0 = vtxs[edgs[e].vtx0].value;
  double f1 = vtxs[edgs[e].vtx1].value;
  double f2 = vtxs[edgs[e].vtx2].value;

  if (fabs(f0) < .8*eps) return .8*eps;
  if (fabs(f1) < .8*eps) return 0.5;
  if (fabs(f2) < .8*eps) return 1.-.8*eps;

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  if(f0*f2 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
#endif

  double fdd = (f2+f0-2.*f1)/0.25;

  double c2 = 0.5*fdd;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = (f2-f0)/1.;   // the expansion of f at the center of (0,1)
  double c0 = f1;

  double x;

  if (fabs(c2) < eps) { c0 = .5*(f0+f2); x = -c0/c1; }
  else
  {
    if (f2<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    else      x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
  }
#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  if (x < -0.5 || x > 0.5)
  {
    std::cout << f0 << " " << f1 << " " << f2 << " " << x << std::endl;
    throw std::domain_error("[CASL_ERROR]: ");
  }
#endif

  if (x <-0.5) return .8*eps;
  if (x > 0.5) return 1.-.8*eps;

  return .5+x;
}

void simplex3_mls_quadratic_t::find_middle_node(double &x_out, double &y_out, double x0, double y0, double x1, double y1, int n_tri)
{
  tri3_t *tri = &tris[n_tri];

  // compute normal
  double tx = x1-x0;
  double ty = y1-y0;
  double norm = sqrt(tx*tx+ty*ty);
  tx /= norm;
  ty /= norm;
  double nx =-ty;
  double ny = tx;

  // fetch values of LSF
  std::vector<int> nv(nodes_per_tri, -1);

  nv[0] = tri->vtx0;
  nv[1] = tri->vtx1;
  nv[2] = tri->vtx2;
  nv[3] = edgs[tri->edg2].vtx1;
  nv[4] = edgs[tri->edg0].vtx1;
  nv[5] = edgs[tri->edg1].vtx1;

//  std::vector<double> f(nodes_per_tri, 0);

//  for (short i = 0; i < nodes_per_tri; ++i)
//    f[i] = vtxs[nv[i]].value;

  double a = 0.5*(x0+x1);
  double b = 0.5*(y0+y1);

  double N[nodes_per_tri]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};
  double Na[nodes_per_tri] = {-3.+4.*a+4.*b,            4.*a-1.,      0,            4.-8.*a-4.*b,   4.*b,  -4.*b};
  double Nb[nodes_per_tri] = {-3.+4.*a+4.*b,            0,            4.*b-1.,     -4.*a,           4.*a,   4.-4.*a-8.*b};
  double Naa[nodes_per_tri] = {4, 4, 0,-8, 0, 0};
  double Nab[nodes_per_tri] = {4, 0, 0,-4, 4,-4};
  double Nbb[nodes_per_tri] = {4, 0, 4, 0, 0,-8};

  double F = 0, Fn = 0, Fnn = 0;
  double f;
  for (short i = 0; i < nodes_per_tri; ++i)
  {
    f = vtxs[nv[i]].value;
    F   += f*N[i];
    Fn  += f*(Na[i]*nx+Nb[i]*ny);
    Fnn += f*(Naa[i]*nx*nx + 2.*Nab[i]*nx*ny + Nbb[i]*ny*ny);
  }

  // solve quadratic equation
  double c2 = 0.5*Fnn;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = Fn;   // the expansion of f at the center of (0,1)
  double c0 = F;

  double alpha;

  if (fabs(c2) < EPS) alpha = -c0/c1;
  else
  {
//    if (Fn<0) alpha = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
//    else      alpha = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    double alpha1 = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    double alpha2 = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    if (fabs(alpha1)>fabs(alpha2)) alpha = alpha2;
    else alpha = alpha1;
  }

//  alpha = 0;

  x_out = 0.5*(x0+x1) + alpha*nx;
  y_out = 0.5*(y0+y1) + alpha*ny;

  while (x_out + y_out > 1. || x_out < 0. || y_out < 0.)
  {
    std::cout << "Warning: point is outside of a triangle! (" << .5*(x0+x1) << " " << .5*(y0+y1) << " " << x_out << " " << y_out << ")\n";
    invalid_reconstruction = true;
    if (x_out < 0.) alpha = (.5*(x0+x1)-eps)/(.5*(x0+x1)-x_out)*alpha;
    else if (y_out < 0.) alpha = (.5*(y0+y1)-eps)/(.5*(y0+y1)-y_out)*alpha;
    else if (1.-x_out-y_out < 0.) alpha = (1.-.5*(x0+x1)-.5*(y0+y1)-eps)/(1. -.5*(x0+x1)-.5*(y0+y1)-(1.-x_out-y_out))*alpha;

    x_out = 0.5*(x0+x1) + alpha*nx;
    y_out = 0.5*(y0+y1) + alpha*ny;
  }

  if (y_out > 1.-x_out || x_out < 0. || y_out < 0.) {
    std::cout << "Warning: point is outside of a triangle! (" << .5*(x0+x1) << " " << .5*(y0+y1) << " " << x_out << " " << y_out << ")\n";
  }

}


void simplex3_mls_quadratic_t::find_middle_node_tet(double abc_out[3], int n_tet)
{
  tet3_t *tet = &tets[n_tet];

  // get coordinates of intersection points
  int e01 = tris[tet->tri3].edg2;
  int e02 = tris[tet->tri3].edg1;
  int e03 = tris[tet->tri1].edg1;
  int e12 = tris[tet->tri0].edg2;
  int e23 = tris[tet->tri0].edg0;
  int e13 = tris[tet->tri0].edg1;

  double r13 = edgs[e13].a;
  double r12 = edgs[e12].a;
  double r03 = edgs[e03].a;
  double r02 = edgs[e02].a;

  double abc03[3] = { 0.,     0.,   r03 };
  double abc02[3] = { 0.,     r02,  0. };
  double abc12[3] = { 1.-r12, r12,  0. };
  double abc13[3] = { 1.-r13, 0.,   r13 };

  //  double n[3] = { t0[1]*t1[2] - t0[2]*t1[1],
  //                  t0[2]*t1[0] - t0[0]*t1[2],
  //                  t0[0]*t1[1] - t0[1]*t1[0] };

  //  double norm = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);

  //  n[0] /= norm;
  //  n[1] /= norm;
  //  n[2] /= norm;

//  double t0[3] = { abc03[0]-abc12[0], abc03[1]-abc12[1], abc03[2]-abc12[2] };
//  double t1[3] = { abc02[0]-abc13[0], abc02[1]-abc13[1], abc02[2]-abc13[2] };

//  double nx = t0[1]*t1[2] - t0[2]*t1[1];
//  double ny = t0[2]*t1[0] - t0[0]*t1[2];
//  double nz = t0[0]*t1[1] - t0[1]*t1[0];

////  double t1[3] = { abc12[1]*abc03[2] - abc12[2]*abc03[1],
////                   abc12[2]*abc03[0] - abc12[0]*abc03[2],
////                   abc12[0]*abc03[1] - abc12[1]*abc03[0] };

  double t0[3] = { abc03[0]-abc12[0], abc03[1]-abc12[1], abc03[2]-abc12[2] };
  double t1[3] = { abc12[1],
                  -abc12[0],
                   0.};

  double nx = t0[1]*t1[2] - t0[2]*t1[1];
  double ny = t0[2]*t1[0] - t0[0]*t1[2];
  double nz = t0[0]*t1[1] - t0[1]*t1[0];

  double norm = sqrt(nx*nx + ny*ny + nz*nz);

  nx /= norm;
  ny /= norm;
  nz /= norm;

  // fetch values of LSF
  std::vector<int> nv(nodes_per_tet, -1);

  nv[0] = tet->vtx0;
  nv[1] = tet->vtx1;
  nv[2] = tet->vtx2;
  nv[3] = tet->vtx3;
  nv[4] = edgs[e01].vtx1;
  nv[5] = edgs[e12].vtx1;
  nv[6] = edgs[e02].vtx1;
  nv[7] = edgs[e03].vtx1;
  nv[8] = edgs[e13].vtx1;
  nv[9] = edgs[e23].vtx1;

  double a = 0.5*(abc12[0]+abc03[0]);
  double b = 0.5*(abc12[1]+abc03[1]);
  double c = 0.5*(abc12[2]+abc03[2]);

//  double a = 0.25*(abc12[0]+abc03[0]+abc02[0]+abc13[0]);
//  double b = 0.25*(abc12[1]+abc03[1]+abc02[1]+abc13[1]);
//  double c = 0.25*(abc12[2]+abc03[2]+abc02[2]+abc13[2]);

//  double a = 0.5*(abc02[0]+abc13[0]);
//  double b = 0.5*(abc02[1]+abc13[1]);
//  double c = 0.5*(abc02[2]+abc13[2]);

  double d = 1.-a-b-c;
  double N[nodes_per_tet]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  double Na[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
  double Nb[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
  double Nc[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

  double Naa[nodes_per_tet] = { 4, 4, 0, 0,-8, 0, 0, 0, 0, 0 };
  double Nbb[nodes_per_tet] = { 4, 0, 4, 0, 0, 0,-8, 0, 0, 0 };
  double Ncc[nodes_per_tet] = { 4, 0, 0, 4, 0, 0, 0,-8, 0, 0 };
  double Nab[nodes_per_tet] = { 4, 0, 0, 0,-4, 4,-4, 0, 0, 0 };
  double Nbc[nodes_per_tet] = { 4, 0, 0, 0, 0, 0,-4,-4, 0, 4 };
  double Nca[nodes_per_tet] = { 4, 0, 0, 0,-4, 0, 0,-4, 4, 0 };


  double F = 0, Fx = 0, Fy = 0, Fz = 0, Fxx = 0, Fyy = 0, Fzz = 0, Fxy = 0, Fyz = 0, Fzx = 0;
  double f;
  for (short i = 0; i < nodes_per_tet; ++i)
  {
    f = vtxs[nv[i]].value;

    F   += f*N[i];

    Fx  += f*Na[i];
    Fy  += f*Nb[i];
    Fz  += f*Nc[i];

    Fxx += f*Naa[i];
    Fyy += f*Nbb[i];
    Fzz += f*Ncc[i];

    Fxy += f*Nab[i];
    Fyz += f*Nbc[i];
    Fzx += f*Nca[i];
  }

//  nx = Fx; ny = Fy; nz = Fz;

//  norm = sqrt(nx*nx + ny*ny + nz*nz);

//  nx /= norm;
//  ny /= norm;
//  nz /= norm;

  double Fn  = Fx*nx + Fy*ny + Fz*nz;
  double Fnn = Fxx*nx*nx + Fyy*ny*ny + Fzz*nz*nz + 2.*Fxy*nx*ny + 2.*Fyz*ny*nz + 2.*Fzx*nz*nx;

  // solve quadratic equation
  double c2 = 0.5*Fnn;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = Fn;   // the expansion of f at the center of (0,1)
  double c0 = F;

  double alpha;

  if (fabs(c2) < eps) alpha = -c0/c1;
  else
  {
//    if (Fn<0) alpha = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
//    else      alpha = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    double alpha1 = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    double alpha2 = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    if (fabs(alpha1)>fabs(alpha2)) alpha = alpha2;
    else alpha = alpha1;
  }

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  if (alpha != alpha) throw std::domain_error("[CASL_ERROR]: ");
#endif

//  alpha = 0;

  abc_out[0] = a + alpha*nx;
  abc_out[1] = b + alpha*ny;
  abc_out[2] = c + alpha*nz;

  while (abc_out[0] + abc_out[1] + abc_out[2] > 1. || abc_out[0] < 0. || abc_out[1] < 0. || abc_out[2] < 0.)
  {
    std::cout << "Warning: point is outside of a tetrahedron! (" << alpha << " " << a << " " << b << " " << c << " " << abc_out[0] << " " << abc_out[1] << " " << abc_out[2] << ")\n";
    invalid_reconstruction = true;
    if (abc_out[0] < 0.) alpha = (a-eps)/(a-abc_out[0])*alpha;
    else if (abc_out[1] < 0.) alpha = (b-eps)/(b-abc_out[1])*alpha;
    else if (abc_out[2] < 0.) alpha = (c-eps)/(c-abc_out[2])*alpha;
    else if (abc_out[0] + abc_out[1] + abc_out[2] > 1.) alpha = (1.-a-b-c-eps)/(1.-a-b-c - (1.-abc_out[0]-abc_out[1]-abc_out[2]))*alpha;

    abc_out[0] = a + alpha*nx;
    abc_out[1] = b + alpha*ny;
    abc_out[2] = c + alpha*nz;
  }

  if (abc_out[0] + abc_out[1] + abc_out[2] > 1. || abc_out[0] < 0. || abc_out[1] < 0. || abc_out[2] < 0.) {
    std::cout << "Warning: point is outside of a tetrahedron! (" << alpha << " " << a << " " << b << " " << c << " " << abc_out[0] << " " << abc_out[1] << " " << abc_out[2] << ")\n";
  }

//  if (a + b + c > 1. || a < 0. || b < 0. || c < 0.) {
//    std::cout << "Warning: point is outside of a triangle! (" << a << " " << b << " " << c << " " << abc_out[0] << " " << abc_out[1] << " " << abc_out[2] << ")\n";
//  }

}


//void simplex3_mls_quadratic_t::find_middle_node_tet(double abc_out[3], int n_tet)
//{
//  tet3_t *tet = &tets[n_tet];

//  // get coordinates of intersection points
//  int e01 = tris[tet->tri3].edg2;
//  int e02 = tris[tet->tri3].edg1;
//  int e03 = tris[tet->tri1].edg1;
//  int e12 = tris[tet->tri0].edg2;
//  int e23 = tris[tet->tri0].edg0;
//  int e13 = tris[tet->tri0].edg1;

//  double r13 = edgs[e13].a;
//  double r12 = edgs[e12].a;
//  double r03 = edgs[e03].a;
//  double r02 = edgs[e02].a;

//  double abc03[3] = { 0.,     0.,   r03 };
//  double abc02[3] = { 0.,     r02,  0. };
//  double abc12[3] = { 1.-r12, r12,  0. };
//  double abc13[3] = { 1.-r13, 0.,   r13 };

////  double t0[3] = { abc03[0]-abc12[0], abc03[1]-abc12[1], abc03[2]-abc12[2] };
////  double t1[3] = { abc02[0]-abc13[0], abc02[1]-abc13[1], abc02[2]-abc13[2] };

//////  double n[3] = { t0[1]*t1[2] - t0[2]*t1[1],
//////                  t0[2]*t1[0] - t0[0]*t1[2],
//////                  t0[0]*t1[1] - t0[1]*t1[0] };

//////  double norm = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);

//////  n[0] /= norm;
//////  n[1] /= norm;
//////  n[2] /= norm;

////  double nx = t0[1]*t1[2] - t0[2]*t1[1];
////  double ny = t0[2]*t1[0] - t0[0]*t1[2];
////  double nz = t0[0]*t1[1] - t0[1]*t1[0];

//  double t0[3] = { abc03[0]-abc12[0], abc03[1]-abc12[1], abc03[2]-abc12[2] };

//  double t1[3] = { abc12[1]*abc03[2] - abc12[2]*abc03[1],
//                   abc12[2]*abc03[0] - abc12[0]*abc03[2],
//                   abc12[0]*abc03[1] - abc12[1]*abc03[0] };

//  double nx = t0[1]*t1[2] - t0[2]*t1[1];
//  double ny = t0[2]*t1[0] - t0[0]*t1[2];
//  double nz = t0[0]*t1[1] - t0[1]*t1[0];

//  double norm = sqrt(nx*nx + ny*ny + nz*nz);

//  nx /= norm;
//  ny /= norm;
//  nz /= norm;

//  // fetch values of LSF
//  std::vector<int> nv(nodes_per_tet, -1);

//  nv[0] = tet->vtx0;
//  nv[1] = tet->vtx1;
//  nv[2] = tet->vtx2;
//  nv[3] = tet->vtx3;
//  nv[4] = edgs[e01].vtx1;
//  nv[5] = edgs[e12].vtx1;
//  nv[6] = edgs[e02].vtx1;
//  nv[7] = edgs[e03].vtx1;
//  nv[8] = edgs[e13].vtx1;
//  nv[9] = edgs[e23].vtx1;

//  double a = 0.5*(abc12[0]+abc03[0]);
//  double b = 0.5*(abc12[1]+abc03[1]);
//  double c = 0.5*(abc12[2]+abc03[2]);

////  double a = 0.25*(abc12[0]+abc03[0]+abc02[0]+abc13[0]);
////  double b = 0.25*(abc12[1]+abc03[1]+abc02[1]+abc13[1]);
////  double c = 0.25*(abc12[2]+abc03[2]+abc02[2]+abc13[2]);

////  double a = 0.5*(abc02[0]+abc13[0]);
////  double b = 0.5*(abc02[1]+abc13[1]);
////  double c = 0.5*(abc02[2]+abc13[2]);

//  double tolerance = 1.e-16;
//  double F = 1;

//  while (fabs(F) > tolerance)
//  {
//    double d = 1.-a-b-c;
//    double N[nodes_per_tet]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

//    double Na[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
//    double Nb[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
//    double Nc[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

//    F = 0;
//    double Fx = 0, Fy = 0, Fz = 0;
//    double f;
//    for (short i = 0; i < nodes_per_tet; ++i)
//    {
//      f = vtxs[nv[i]].value;

//      F   += f*N[i];

//      Fx  += f*Na[i];
//      Fy  += f*Nb[i];
//      Fz  += f*Nc[i];
//    }

////    nx = Fx; ny = Fy; nz = Fz;

////    double norm = sqrt(nx*nx + ny*ny + nz*nz);

////    nx /= norm;
////    ny /= norm;
////    nz /= norm;

//    double Fn  = Fx*nx + Fy*ny + Fz*nz;
//    double change_a = F*nx/Fn;
//    double change_b = F*ny/Fn;
//    double change_c = F*nz/Fn;

//    a -= change_a;
//    b -= change_b;
//    c -= change_c;
//  }

//  abc_out[0] = a;
//  abc_out[1] = b;
//  abc_out[2] = c;

//}


bool simplex3_mls_quadratic_t::need_swap(int v0, int v1)
{
//  double dif = vtxs[v0].value - vtxs[v1].value;
//  if (dif > 0.)
//  if (fabs(dif) < .8*eps){ // if values are too close, sort vertices by their numbers
//    if (v0 > v1) return true;
//    else         return false;
//  } else if (dif > 0.0){ // otherwise sort by values
//    return true;
//  } else {
//    return false;
//  }

  if (vtxs[v0].value > vtxs[v1].value)
    return true;
  else if (vtxs[v0].value < vtxs[v1].value)
    return false;
  else
  {
    if (v0 > v1) return true;
    else         return false;
  }
}








//--------------------------------------------------
// Refinement
//--------------------------------------------------

void simplex3_mls_quadratic_t::refine_all()
{
  int n;
  n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
  n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
  n = tets.size(); for (int i = 0; i < n; i++) refine_tet(i);

  std::cout << "Refined!\n";
}

void simplex3_mls_quadratic_t::refine_edg(int n_edg)
{
  edg3_t *edg = &edgs[n_edg];

  if (edg->is_split) return;
  else edg->is_split = true;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  /* Create two new vertices */
  double xyz_v01[3];
  double xyz_v12[3];

  mapping_edg(xyz_v01, n_edg, 0.25);
  mapping_edg(xyz_v12, n_edg, 0.75);

  vtxs.push_back(vtx3_t(xyz_v01[0], xyz_v01[1], xyz_v01[2]));
  vtxs.push_back(vtx3_t(xyz_v12[0], xyz_v12[1], xyz_v12[2]));

  int n_vtx01 = vtxs.size()-2;
  int n_vtx12 = vtxs.size()-1;

  /* Create two new edges */
  edgs.push_back(edg3_t(edg->vtx0, n_vtx01, edg->vtx1)); edg = &edgs[n_edg];
  edgs.push_back(edg3_t(edg->vtx1, n_vtx12, edg->vtx2)); edg = &edgs[n_edg];

  edg->c_edg0 = edgs.size()-2;
  edg->c_edg1 = edgs.size()-1;

  /* Transfer properties to new objects */
  loc_t loc = edg->loc;
  int c0 = edg->c0;
  int c1 = edg->c1;

  vtxs[n_vtx01].set(loc, c0, c1, -1);
  vtxs[n_vtx12].set(loc, c0, c1, -1);

  edgs[edg->c_edg0].set(loc, c0, c1);
  edgs[edg->c_edg1].set(loc, c0, c1);
}

void simplex3_mls_quadratic_t::refine_tri(int n_tri)
{
  tri3_t *tri = &tris[n_tri];

  if (tri->is_split) return;
  else tri->is_split = true;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Create 3 new vertices */
  double xyz[3];
  double ab[2];
  ab[0] = .25; ab[1] = .25; mapping_tri(xyz, n_tri, ab); vtxs.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  ab[0] = .50; ab[1] = .25; mapping_tri(xyz, n_tri, ab); vtxs.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  ab[0] = .25; ab[1] = .50; mapping_tri(xyz, n_tri, ab); vtxs.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));

  int n_u0 = vtxs.size()-3;
  int n_u1 = vtxs.size()-2;
  int n_u2 = vtxs.size()-1;

  /* Create 3 new edges */
  int n_v0 = tri->vtx0;
  int n_v1 = tri->vtx1;
  int n_v2 = tri->vtx2;
  int n_v01 = edgs[tri->edg2].vtx1;
  int n_v12 = edgs[tri->edg0].vtx1;
  int n_v02 = edgs[tri->edg1].vtx1;

  edgs.push_back(edg3_t(n_v02, n_u0, n_v01));
  edgs.push_back(edg3_t(n_v01, n_u1, n_v12));
  edgs.push_back(edg3_t(n_v02, n_u2, n_v12));

  /* Create 4 new triangles */
  int n_edg0 = edgs.size()-3;
  int n_edg1 = edgs.size()-2;
  int n_edg2 = edgs.size()-1;

  tris.push_back(tri3_t(n_v0,  n_v01, n_v02, n_edg0, edgs[tri->edg1].c_edg0, edgs[tri->edg2].c_edg0)); tri = &tris[n_tri];
  tris.push_back(tri3_t(n_v1,  n_v01, n_v12, n_edg1, edgs[tri->edg0].c_edg0, edgs[tri->edg2].c_edg1)); tri = &tris[n_tri];
  tris.push_back(tri3_t(n_v2,  n_v02, n_v12, n_edg2, edgs[tri->edg0].c_edg1, edgs[tri->edg1].c_edg1)); tri = &tris[n_tri];
  tris.push_back(tri3_t(n_v01, n_v02, n_v12, n_edg2, n_edg1,                 n_edg0));                 tri = &tris[n_tri];

  int n_tri0 = tris.size()-4;
  int n_tri1 = tris.size()-3;
  int n_tri2 = tris.size()-2;
  int n_tri3 = tris.size()-1;

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  tri_is_ok(n_tri0);
  tri_is_ok(n_tri1);
  tri_is_ok(n_tri2);
  tri_is_ok(n_tri3);
#endif

  tri->c_edg0 = n_edg0;
  tri->c_edg1 = n_edg1;
  tri->c_edg2 = n_edg2;

  tri->c_tri0 = n_tri0;
  tri->c_tri1 = n_tri1;
  tri->c_tri2 = n_tri2;
  tri->c_tri3 = n_tri3;

  /* Transfer properties */
  loc_t loc = tri->loc;
  int c = tri->c;
  int dir = tri->dir;

  vtxs[n_u0].set(loc, c, -1, -1);
  vtxs[n_u1].set(loc, c, -1, -1);
  vtxs[n_u2].set(loc, c, -1, -1);

  edgs[n_edg0].set(loc, c, -1);
  edgs[n_edg1].set(loc, c, -1);
  edgs[n_edg2].set(loc, c, -1);

  tris[n_tri0].set(loc, c); tris[n_tri0].dir = dir;
  tris[n_tri1].set(loc, c); tris[n_tri1].dir = dir;
  tris[n_tri2].set(loc, c); tris[n_tri2].dir = dir;
  tris[n_tri3].set(loc, c); tris[n_tri3].dir = dir;
}

void simplex3_mls_quadratic_t::refine_tet(int n_tet)
{
  tet3_t *tet = &tets[n_tet];

  if (tet->is_split) return;
  else tet->is_split = true;

  /* Sort vertices */
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx2, tet->vtx3)) {swap(tet->vtx2, tet->vtx3); swap(tet->tri2, tet->tri3);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}

  // fetch vertices
  tri3_t *t0 = &tris[tet->tri0];
  tri3_t *t1 = &tris[tet->tri1];
  tri3_t *t2 = &tris[tet->tri2];
  tri3_t *t3 = &tris[tet->tri3];

  int e01 = t3->edg2;
  int e02 = t3->edg1;
  int e03 = t1->edg1;
  int e12 = t0->edg2;
  int e23 = t0->edg0;
  int e13 = t0->edg1;

  int nv0 = tet->vtx0;
  int nv1 = tet->vtx1;
  int nv2 = tet->vtx2;
  int nv3 = tet->vtx3;
  int nv01 = edgs[e01].vtx1;
  int nv12 = edgs[e12].vtx1;
  int nv02 = edgs[e02].vtx1;
  int nv03 = edgs[e03].vtx1;
  int nv13 = edgs[e13].vtx1;
  int nv23 = edgs[e23].vtx1;

  // create one more vertex
  double abc[3] = { .25, .25, .25 };
  double xyz[3];

  mapping_tet(xyz, n_tet, abc);

  vtxs.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  int nv = vtxs.size() - 1;

  // create one more edge
  edgs.push_back(edg3_t(nv12, nv, nv03));
  int ne = edgs.size() - 1;

  // create 8 more triagnles
  tris.push_back(tri3_t(nv01, nv02, nv03, t1->c_edg0, t2->c_edg0, t3->c_edg0));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];
  tris.push_back(tri3_t(nv01, nv12, nv13, t0->c_edg0, t2->c_edg1, t3->c_edg1));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];
  tris.push_back(tri3_t(nv02, nv12, nv23, t0->c_edg1, t1->c_edg1, t3->c_edg2));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];
  tris.push_back(tri3_t(nv03, nv13, nv23, t0->c_edg2, t1->c_edg2, t2->c_edg2));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];
  tris.push_back(tri3_t(nv01, nv03, nv12, ne,         t3->c_edg1, t2->c_edg0));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];
  tris.push_back(tri3_t(nv13, nv03, nv12, ne,         t0->c_edg0, t2->c_edg2));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];
  tris.push_back(tri3_t(nv23, nv03, nv12, ne,         t0->c_edg1, t1->c_edg2));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];
  tris.push_back(tri3_t(nv02, nv03, nv12, ne,         t3->c_edg2, t1->c_edg0));  t0 = &tris[tet->tri0]; t1 = &tris[tet->tri1]; t2 = &tris[tet->tri2]; t3 = &tris[tet->tri3];

  int cf0 = tris.size() - 8;
  int cf1 = tris.size() - 7;
  int cf2 = tris.size() - 6;
  int cf3 = tris.size() - 5;
  int cf4 = tris.size() - 4;
  int cf5 = tris.size() - 3;
  int cf6 = tris.size() - 2;
  int cf7 = tris.size() - 1;

  // create 8 more tetrahedra
  tets.push_back(tet3_t(nv0,  nv01, nv02, nv03, cf0, t1->c_tri0, t2->c_tri0, t3->c_tri0));
  tets.push_back(tet3_t(nv1,  nv01, nv12, nv13, cf1, t0->c_tri0, t2->c_tri1, t3->c_tri1));
  tets.push_back(tet3_t(nv2,  nv02, nv12, nv23, cf2, t0->c_tri1, t1->c_tri1, t3->c_tri2));
  tets.push_back(tet3_t(nv3,  nv03, nv13, nv23, cf3, t0->c_tri2, t1->c_tri2, t2->c_tri2));
  tets.push_back(tet3_t(nv01, nv02, nv03, nv12, cf7, cf4,        t3->c_tri3, cf0));
  tets.push_back(tet3_t(nv03, nv12, nv13, nv23, t0->c_tri3, cf3, cf6,        cf5));
  tets.push_back(tet3_t(nv01, nv03, nv12, nv13, cf5, cf1,        t2->c_tri3, cf4));
  tets.push_back(tet3_t(nv02, nv03, nv12, nv23, cf6, cf2,        t1->c_tri3, cf7));

  int n_tet0 = tets.size() - 8;
  int n_tet1 = tets.size() - 7;
  int n_tet2 = tets.size() - 6;
  int n_tet3 = tets.size() - 5;
  int n_tet4 = tets.size() - 4;
  int n_tet5 = tets.size() - 3;
  int n_tet6 = tets.size() - 2;
  int n_tet7 = tets.size() - 1;

  // transfer properties
  loc_t loc = tets[n_tet].loc;

  vtxs[nv].set(loc, -1, -1, -1);

  edgs[ne].set(loc, -1, -1);

  tris[cf0].set(loc, -1);
  tris[cf1].set(loc, -1);
  tris[cf2].set(loc, -1);
  tris[cf3].set(loc, -1);
  tris[cf4].set(loc, -1);
  tris[cf5].set(loc, -1);
  tris[cf6].set(loc, -1);
  tris[cf7].set(loc, -1);

  tets[n_tet0].set(loc);
  tets[n_tet1].set(loc);
  tets[n_tet2].set(loc);
  tets[n_tet3].set(loc);
  tets[n_tet4].set(loc);
  tets[n_tet5].set(loc);
  tets[n_tet6].set(loc);
  tets[n_tet7].set(loc);


#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
  tri_is_ok(cf0);
  tri_is_ok(cf1);
  tri_is_ok(cf2);
  tri_is_ok(cf3);
  tri_is_ok(cf4);
  tri_is_ok(cf5);
  tri_is_ok(cf6);
  tri_is_ok(cf7);

  tet_is_ok(n_tet0);
  tet_is_ok(n_tet1);
  tet_is_ok(n_tet2);
  tet_is_ok(n_tet3);
  tet_is_ok(n_tet4);
  tet_is_ok(n_tet5);
  tet_is_ok(n_tet6);
  tet_is_ok(n_tet7);
#endif

}





//--------------------------------------------------
// Integration
//--------------------------------------------------
double simplex3_mls_quadratic_t::integrate_over_domain(CF_3 &f)
{
//  return 0;
  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
  double xyz[3];

  // quadrature points
  static double alph = (5.+3.*sqrt(5.))/20.;
  static double beta = (5.-   sqrt(5.))/20.;

  static double abc0[3] = { beta, beta, beta };
  static double abc1[3] = { alph, beta, beta };
  static double abc2[3] = { beta, alph, beta };
  static double abc3[3] = { beta, beta, alph };

  /* integrate over tetrahedra */
  for (unsigned int i = 0; i < tets.size(); i++)
    if (!tets[i].is_split && tets[i].loc == INS)
    {
//      tet3_t *tet = &tets[i];

      mapping_tet(xyz, i, abc0); f0 = f( xyz[0], xyz[1], xyz[2] );
      mapping_tet(xyz, i, abc1); f1 = f( xyz[0], xyz[1], xyz[2] );
      mapping_tet(xyz, i, abc2); f2 = f( xyz[0], xyz[1], xyz[2] );
      mapping_tet(xyz, i, abc3); f3 = f( xyz[0], xyz[1], xyz[2] );

      w0 = jacobian_tet(i, abc0);
      w1 = jacobian_tet(i, abc1);
      w2 = jacobian_tet(i, abc2);
      w3 = jacobian_tet(i, abc3);

      result += w0*f0 + w1*f1 + w2*f2 + w3*f3;
//      result += volume(s->vtx0, s->vtx1, s->vtx2, s->vtx3);
    }

  return result*0.125/3.;
}

//double simplex3_mls_quadratic_t::integrate_over_domain(CF_3 &f)
//{
//  double result = 0.0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
//  double xyz[3];

//  double abc0[3] = { 0., 0., 0. };
//  double abc1[3] = { 1., 0., 0. };
//  double abc2[3] = { 0., 1., 0. };
//  double abc3[3] = { 0., 0., 1. };

//  /* integrate over tetrahedra */
//  for (unsigned int i = 0; i < tets.size(); i++)
//    if (!tets[i].is_split && tets[i].loc == INS)
//    {
//      tet3_t *s = &tets[i];

//      mapping_tet(xyz, i, abc0); f0 = f( xyz[0], xyz[1], xyz[2] );
//      mapping_tet(xyz, i, abc1); f1 = f( xyz[0], xyz[1], xyz[2] );
//      mapping_tet(xyz, i, abc2); f2 = f( xyz[0], xyz[1], xyz[2] );
//      mapping_tet(xyz, i, abc3); f3 = f( xyz[0], xyz[1], xyz[2] );

////      result += (w0*f0 + w1*f1 + w2*f2 + w3*f3 + w4*f4);
//      result += (f0+f1+f2+f3)/4.0*volume(s->vtx0, s->vtx1, s->vtx2, s->vtx3);
//    }

//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if (result != result)
//    throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif

//  return result;
//}

//double simplex3_mls_quadratic_t::integrate_over_domain(CF_3 &f)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0;
//  double xyz[3];

//  // quadrature points, degree 3
//  double alph = (5.+3.*sqrt(5.))/20.;
//  double beta = (5.-   sqrt(5.))/20.;

//  double abc0[3] = { 0.25, 0.25, 0.25 };
//  double abc1[3] = { 0.5, 1./6., 1./6. };
//  double abc2[3] = { 1./6., 0.5, 1./6. };
//  double abc3[3] = { 1./6., 1./6., 0.5 };
//  double abc4[3] = { 1./6., 1./6., 1./6. };

//  double V = 0.5/3.;

//  /* integrate over tetrahedra */
//  for (unsigned int i = 0; i < tets.size(); i++)
//    if (!tets[i].is_split && tets[i].loc == INS)
//    {
//      tet3_t *s = &tets[i];

//      mapping_tet(xyz, i, abc0); f0 = f( xyz[0], xyz[1], xyz[2] );
//      mapping_tet(xyz, i, abc1); f1 = f( xyz[0], xyz[1], xyz[2] );
//      mapping_tet(xyz, i, abc2); f2 = f( xyz[0], xyz[1], xyz[2] );
//      mapping_tet(xyz, i, abc3); f3 = f( xyz[0], xyz[1], xyz[2] );
//      mapping_tet(xyz, i, abc4); f4 = f( xyz[0], xyz[1], xyz[2] );

//      w0 = -.8*jacobian_tet(i, abc0);
//      w1 = .45*jacobian_tet(i, abc1);
//      w2 = .45*jacobian_tet(i, abc2);
//      w3 = .45*jacobian_tet(i, abc3);
//      w4 = .45*jacobian_tet(i, abc4);

//      result += (w0*f0 + w1*f1 + w2*f2 + w3*f3 + w4*f4);
////      result += (f0+f1+f2+f3+f4)/5.0*volume(s->vtx0, s->vtx1, s->vtx2, s->vtx3);
//    }

//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if (result != result)
//    throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
//#endif

////  return result;
//  return result*V;
//}


double simplex3_mls_quadratic_t::integrate_over_interface(CF_3 &f, int num)
{
//  return 0;
  bool integrate_specific = (num != -1);

  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0;
  double f0 = 0, f1 = 0, f2 = 0;
  double xyz[3];

  // quadrature points
  static double ab0[2] = { .0, .5 };
  static double ab1[2] = { .5, .0 };
  static double ab2[2] = { .5, .5 };

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris.size(); i++)
  {
    tri3_t *t = &tris[i];
    if (!t->is_split && t->loc == FCE)
      if (!integrate_specific
          || (integrate_specific && t->c == num))
      {
//        construct_proper_mapping(i, -1);
        // map quadrature points into real space and interpolate integrand
        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );

        // scale weights by Jacobian
        w0 = jacobian_tri(i, ab0);
        w1 = jacobian_tri(i, ab1);
        w2 = jacobian_tri(i, ab2);

        result += w0*f0 + w1*f1 + w2*f2;
//        result += fabs(f0)+fabs(f1)+fabs(f2);
      }
  }

  return result/6.;
}


//double simplex3_mls_quadratic_t::integrate_over_interface(std::vector<double> &f, int num)
//{
//  bool integrate_specific = (num != -1);

//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0;
//  double f0 = 0, f1 = 0, f2 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 1./6., 1./6. };
//  double ab1[2] = { 2./3., 1./6. };
//  double ab2[2] = { 1./6., 2./3. };

//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == FCE)
//      if (!integrate_specific
//          || (integrate_specific && t->c == num))
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = interpolate_from_parent(f, xyz);
//        mapping_tri(xyz, i, ab1); f1 = interpolate_from_parent(f, xyz);
//        mapping_tri(xyz, i, ab2); f2 = interpolate_from_parent(f, xyz);

////        double r0 = 0.814;
////        double d = 0.133;

////        double theta = 0.379;
////        double phy = 0.312;

////        double cosT = cos(theta);
////        double sinT = sin(theta);
////        double cosP = cos(phy);
////        double sinP = sin(phy);
////        double xc_0 = round((-d*sinT*cosP-0.02)*100.)/100.; double yc_0 = round(( d*cosT*cosP-0.07)*100.)/100.; double zc_0 = round(( d*sinP-0.03)*100.)/100.;

////        mapping_tri(xyz, i, ab0);  f0 = fabs(r0 - sqrt(pow(xyz[0]-xc_0, 2.0) + pow(xyz[1]-yc_0, 2.0) + pow(xyz[2]-zc_0, 2.0)));
////        mapping_tri(xyz, i, ab1);  f1 = fabs(r0 - sqrt(pow(xyz[0]-xc_0, 2.0) + pow(xyz[1]-yc_0, 2.0) + pow(xyz[2]-zc_0, 2.0)));
////        mapping_tri(xyz, i, ab2);  f2 = fabs(r0 - sqrt(pow(xyz[0]-xc_0, 2.0) + pow(xyz[1]-yc_0, 2.0) + pow(xyz[2]-zc_0, 2.0)));

//        // scale weights by Jacobian
//        w0 = jacobian_tri(i, ab0);
//        w1 = jacobian_tri(i, ab1);
//        w2 = jacobian_tri(i, ab2);

//        result += w0*f0 + w1*f1 + w2*f2;
////        result += f0 + f1 + f2;
//      }
//  }

//  return result/6.;
//}

//double simplex3_mls_quadratic_t::integrate_over_interface(CF_3 &f, int num)
//{
//  bool integrate_specific = (num != -1);

//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 0., 0. };
//  double ab1[2] = { 1., 0. };
//  double ab2[2] = { 0., 1. };

//  max_dist_error_ = 0;
//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == FCE)
//      if (!integrate_specific
//          || (integrate_specific && t->c == num))
//      {
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );

//        result += (f0+f1+f2)/3.*area(t->vtx0, t->vtx1, t->vtx2);
//      }
//  }

//  return result;
//}

//double simplex3_mls_quadratic_t::integrate_over_interface(CF_3 &f, int num)
//{
//  bool integrate_specific = (num != -1);

//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 1./3., 1./3. };
//  double ab1[2] = { .2, .6 };
//  double ab2[2] = { .2, .2 };
//  double ab3[2] = { .6, .2 };

//  max_dist_error_ = 0;
//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == FCE)
//      if (!integrate_specific
//          || (integrate_specific && t->c == num))
//      {
////        construct_proper_mapping(i, -1);
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab3); f3 = f( xyz[0], xyz[1], xyz[2] );

//        // scale weights by Jacobian
//        w0 =-27.*jacobian_tri(i, ab0);
//        w1 = 25.*jacobian_tri(i, ab1);
//        w2 = 25.*jacobian_tri(i, ab2);
//        w3 = 25.*jacobian_tri(i, ab3);

//        result += w0*f0 + w1*f1 + w2*f2 + w3*f3;
//      }
//  }

//  return result/96.;
//}

//double simplex3_mls_quadratic_t::integrate_over_interface(CF_3 &f, int num)
//{
//  bool integrate_specific = (num != -1);

//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0;
//  double xyz[3];

//  // quadrature points
////  double ab0[2] = { .0, .0 };
////  double ab1[2] = { .5, .0 };
////  double ab2[2] = { 1., .0 };
////  double ab3[2] = { .5, .5 };
////  double ab4[2] = { .0, 1. };
////  double ab5[2] = { .0, .5 };
////  double ab6[2] = { 1./3., 1./3. };

//  double ab0[2] = { 0.33333333333333333, 0.33333333333333333 };
//  double ab1[2] = { 0.79742698535308720, 0.10128650732345633 };
//  double ab2[2] = { 0.10128650732345633, 0.79742698535308720 };
//  double ab3[2] = { 0.10128650732345633, 0.10128650732345633 };
//  double ab4[2] = { 0.05971587178976981, 0.47014206410511505 };
//  double ab5[2] = { 0.47014206410511505, 0.05971587178976981 };
//  double ab6[2] = { 0.47014206410511505, 0.47014206410511505 };


//  max_dist_error_ = 0;
//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == FCE)
//      if (!integrate_specific
//          || (integrate_specific && t->c == num))
//      {
//        construct_proper_mapping(i, -1);
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab3); f3 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab4); f4 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab5); f5 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab6); f6 = f( xyz[0], xyz[1], xyz[2] );

//        // scale weights by Jacobian
//        w0 = 0.22500000000000000;
//        w1 = 0.12593918054482717;
//        w2 = 0.12593918054482717;
//        w3 = 0.12593918054482717;
//        w4 = 0.13239415278850616;
//        w5 = 0.13239415278850616;
//        w6 = 0.13239415278850616;

//        w0 *= jacobian_tri(i, ab0);
//        w1 *= jacobian_tri(i, ab1);
//        w2 *= jacobian_tri(i, ab2);
//        w3 *= jacobian_tri(i, ab3);
//        w4 *= jacobian_tri(i, ab4);
//        w5 *= jacobian_tri(i, ab5);
//        w6 *= jacobian_tri(i, ab6);

//        result += w0*f0 + w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5 + w6*f6;
////        result += fabs(f0)+fabs(f1)+fabs(f2)+fabs(f3);
////        result = MAX(MAX(result, fabs(f0), fabs(f1)), fabs(f2), fabs(f3));

////        mapping_tri(xyz, i, t->ab01); f0 = f( xyz[0], xyz[1], xyz[2] );
////        mapping_tri(xyz, i, t->ab12); f1 = f( xyz[0], xyz[1], xyz[2] );
////        mapping_tri(xyz, i, t->ab02); f2 = f( xyz[0], xyz[1], xyz[2] );
////        result += fabs(f0)+fabs(f1)+fabs(f2)+fabs(f3);

//      }
//  }

////  std::cout << max_dist_error_ << std::endl;
////  return max_dist_error_;

////#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
////  if (result != result)
////  {
////    std::cout << w0 << " " << w1 << " " << w2 << " " << w3 << " " << w4 << " " << w5 << " " << w6 << std::endl;
////    std::cout << f0 << " " << f1 << " " << f2 << " " << f3 << " " << f4 << " " << f5 << " " << f6 << std::endl;

////    std::cout << &f <<std::endl;
////    throw std::domain_error("[CASL_ERROR]: Something went wrong during surface integration.");
////  }
////#endif

//  return 0.5*result;
//}

//double simplex3_mls_quadratic_t::integrate_over_interface(CF_3 &f, int num)
//{
//  bool integrate_specific = (num != -1);

//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0;
//  double w7 = 0, w8 = 0, w9 = 0, w10 = 0, w11 = 0, w12 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0;
//  double f7 = 0, f8 = 0, f9 = 0, f10 = 0, f11 = 0, f12 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 0.333333333333333,  0.333333333333333 };
//  double ab1[2] = { 0.479308067841923,  0.260345966079038 };
//  double ab2[2] = { 0.260345966079038,  0.479308067841923 };
//  double ab3[2] = { 0.260345966079038,  0.260345966079038 };
//  double ab4[2] = { 0.869739794195568,  0.065130102902216 };
//  double ab5[2] = { 0.065130102902216,  0.869739794195568 };
//  double ab6[2] = { 0.065130102902216,  0.065130102902216 };
//  double ab7[2] = { 0.638444188569809,  0.312865496004875 };
//  double ab8[2] = { 0.638444188569809,  0.048690315425316 };
//  double ab9[2] = { 0.312865496004875,  0.638444188569809 };
//  double ab10[2] = { 0.312865496004875,  0.048690315425316 };
//  double ab11[2] = { 0.048690315425316,  0.638444188569809 };
//  double ab12[2] = { 0.048690315425316,  0.312865496004875 };


//  max_dist_error_ = 0;
//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == FCE)
//      if (!integrate_specific
//          || (integrate_specific && t->c == num))
//      {
////        construct_proper_mapping(i, -1);
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab3); f3 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab4); f4 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab5); f5 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab6); f6 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab7); f7 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab8); f8 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab9); f9 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab10); f10 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab11); f11 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab12); f12 = f( xyz[0], xyz[1], xyz[2] );

//        // scale weights by Jacobian
//        w0 = -0.149570044467670;
//        w1 =  0.175615257433204;
//        w2 =  0.175615257433204;
//        w3 =  0.175615257433204;
//        w4 =  0.053347235608839;
//        w5 =  0.053347235608839;
//        w6 =  0.053347235608839;
//        w7 =  0.077113760890257;
//        w8 =  0.077113760890257;
//        w9 =  0.077113760890257;
//        w10 =  0.077113760890257;
//        w11 =  0.077113760890257;
//        w12 =  0.077113760890257;

//        w0 *= jacobian_tri(i, ab0);
//        w1 *= jacobian_tri(i, ab1);
//        w2 *= jacobian_tri(i, ab2);
//        w3 *= jacobian_tri(i, ab3);
//        w4 *= jacobian_tri(i, ab4);
//        w5 *= jacobian_tri(i, ab5);
//        w6 *= jacobian_tri(i, ab6);
//        w7 *= jacobian_tri(i, ab7);
//        w8 *= jacobian_tri(i, ab8);
//        w9 *= jacobian_tri(i, ab9);
//        w10 *= jacobian_tri(i, ab10);
//        w11 *= jacobian_tri(i, ab11);
//        w12 *= jacobian_tri(i, ab12);

//        result += w0*f0 + w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5 + w6*f6
//            + w7*f7 + w8*f8 + w9*f9 + w10*f10 + w11*f11 + w12*f12;
////        result += fabs(f0)+fabs(f1)+fabs(f2)+fabs(f3);
////        result = MAX(MAX(result, fabs(f0), fabs(f1)), fabs(f2), fabs(f3));

////        mapping_tri(xyz, i, t->ab01); f0 = f( xyz[0], xyz[1], xyz[2] );
////        mapping_tri(xyz, i, t->ab12); f1 = f( xyz[0], xyz[1], xyz[2] );
////        mapping_tri(xyz, i, t->ab02); f2 = f( xyz[0], xyz[1], xyz[2] );
////        result += fabs(f0)+fabs(f1)+fabs(f2)+fabs(f3);

//      }
//  }

////  std::cout << max_dist_error_ << std::endl;
////  return max_dist_error_;
//  return 0.5*result;
//}

//double simplex3_mls_quadratic_t::integrate_over_interface(std::vector<double> &f, int num)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0;
//  double f0 = 0, f1 = 0, f2 = 0;
//  double xyz[3];

//  // quadrature points
//  double a0 = .5*(1.-sqrt(.6));
//  double a1 = .5;
//  double a2 = .5*(1.+sqrt(.6));

//  /* integrate over edges */
//  for (unsigned int i = 0; i < edgs.size(); i++)
//  {
//    edg3_t *e = &edgs[i];
//    if (!e->is_split && e->loc == FCE)
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_edg(xyz, i, a0); f0 = interpolate_from_parent(f, xyz);
//        mapping_edg(xyz, i, a1); f1 = interpolate_from_parent(f, xyz);
//        mapping_edg(xyz, i, a2); f2 = interpolate_from_parent(f, xyz);

//        // scale weights by Jacobian
//        w0 = 5.*jacobian_edg(i, a0);
//        w1 = 8.*jacobian_edg(i, a1);
//        w2 = 5.*jacobian_edg(i, a2);

//        result += w0*f0 + w1*f1+w2*f2;
//      }
//  }

//  return result/18.;
//}

//double simplex3_mls_quadratic_t::integrate_over_colored_interface(CF_3 &f, int num0, int num1)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0;
//  double f0 = 0, f1 = 0, f2 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 1./6., 1./6. };
//  double ab1[2] = { 2./3., 1./6. };
//  double ab2[2] = { 1./6., 2./3. };

//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == FCE)
//      if (t->p_lsf == num0 && t->c == num1)
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );

//        // scale weights by Jacobian
//        w0 = jacobian_tri(i, ab0)/6.;
//        w1 = jacobian_tri(i, ab1)/6.;
//        w2 = jacobian_tri(i, ab2)/6.;

//        result += w0*f0 + w1*f1 + w2*f2;
//      }
//  }

//  return result;
//}


double simplex3_mls_quadratic_t::integrate_over_colored_interface(CF_3 &f, int num0, int num1)
{
  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0;
  double f0 = 0, f1 = 0, f2 = 0;
  double xyz[3];

  // quadrature points
  static double ab0[2] = { .0, .5 };
  static double ab1[2] = { .5, .0 };
  static double ab2[2] = { .5, .5 };

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris.size(); i++)
  {
    tri3_t *t = &tris[i];
    if (!t->is_split && t->loc == FCE)
      if (t->p_lsf == num0 && t->c == num1)
      {
        // map quadrature points into real space and interpolate integrand
        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );

        // scale weights by Jacobian
        w0 = jacobian_tri(i, ab0);
        w1 = jacobian_tri(i, ab1);
        w2 = jacobian_tri(i, ab2);

        result += w0*f0 + w1*f1 + w2*f2;
      }
  }

  return result/6.;
}

//double simplex3_mls_quadratic_t::integrate_over_intersection(CF_3 &f, int num0, int num1)
//{
//  bool integrate_specific = (num0 != -1 && num1 != -1);

//  double result = 0.0;
//  double w0 = 0, w1 = 0;
//  double f0 = 0, f1 = 0;
//  double xyz[3];

//  // quadrature points
//  double a0 = .5*(1.-1./sqrt(3.));
//  double a1 = .5*(1.+1./sqrt(3.));

//  /* integrate over edges */
//  for (unsigned int i = 0; i < edgs.size(); i++)
//  {
//    edg3_t *e = &edgs[i];
//    if (!e->is_split && e->loc == LNE)
//      if ( !integrate_specific
//           || (integrate_specific
//               && (e->c0 == num0 || e->c1 == num0)
//               && (e->c0 == num1 || e->c1 == num1)) )
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_edg(xyz, i, a0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_edg(xyz, i, a1); f1 = f( xyz[0], xyz[1], xyz[2] );

//        // scale weights by Jacobian
//        w0 = jacobian_edg(i, a0)/2.;
//        w1 = jacobian_edg(i, a1)/2.;

//        result += w0*f0 + w1*f1;
//      }
//  }

//  return result;
//}

double simplex3_mls_quadratic_t::integrate_over_intersection(CF_3 &f, int num0, int num1)
{
  bool integrate_specific = (num0 != -1 && num1 != -1);

  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0;
  double f0 = 0, f1 = 0, f2 = 0;
  double xyz[3];

  // quadrature points
  static double a0 = 0.;
  static double a1 = .5;
  static double a2 = 1.;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs.size(); i++)
  {
    edg3_t *e = &edgs[i];
    if (!e->is_split && e->loc == LNE)
      if ( !integrate_specific
           || (integrate_specific
               && (e->c0 == num0 || e->c1 == num0)
               && (e->c0 == num1 || e->c1 == num1)) )
      {
        // map quadrature points into real space and interpolate integrand
        mapping_edg(xyz, i, a0); f0 = f.value(xyz);
        mapping_edg(xyz, i, a1); f1 = f.value(xyz);
        mapping_edg(xyz, i, a2); f2 = f.value(xyz);

        // scale weights by Jacobian
        w0 = jacobian_edg(i, a0);
        w1 = jacobian_edg(i, a1)*4.;
        w2 = jacobian_edg(i, a2);

        result += w0*f0 + w1*f1 + w2*f2;
      }
  }

  return result/6.;
}


//double simplex3_mls_quadratic_t::integrate_over_intersection(CF_3 &f, int num0, int num1)
//{
//  bool integrate_specific = (num0 != -1 && num1 != -1);
////  double result = 0.0;
////  double w0 = 0, w1 = 0, w2 = 0;
////  double f0 = 0, f1 = 0, f2 = 0;
////  double x,y;

////  // quadrature points
////  double a0 = .5*(1.-sqrt(.6));
////  double a1 = .5;
////  double a2 = .5*(1.+sqrt(.6));

////  /* integrate over edges */
////  for (unsigned int i = 0; i < edgs.size(); i++)
////  {
////    edg2_t *e = &edgs[i];
////    if (!e->is_split && e->loc == FCE)
////      if ((!integrate_specific && e->c0 >= 0) || (integrate_specific && e->c0 == num))
////      {
////        // map quadrature points into real space and interpolate integrand
////        mapping_edg(x, y, i, a0); f0 = interpolate_from_parent(f, x, y);
////        mapping_edg(x, y, i, a1); f1 = interpolate_from_parent(f, x, y);
////        mapping_edg(x, y, i, a2); f2 = interpolate_from_parent(f, x, y);

////        // scale weights by Jacobian
////        w0 = 5./9.*jacobian_edg(i, a0)/2.;
////        w1 = 8./9.*jacobian_edg(i, a1)/2.;
////        w2 = 5./9.*jacobian_edg(i, a2)/2.;

////        result += w0*f0 + w1*f1+w2*f2;
////      }
////  }

////  return result;

//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0;
//  double f0 = 0, f1 = 0, f2 = 0;
//  double xyz[3];

//  // quadrature points
//  double a0 = .5*(1.-sqrt(.6));
//  double a1 = .5;
//  double a2 = .5*(1.+sqrt(.6));

//  /* integrate over edges */
//  for (unsigned int i = 0; i < edgs.size(); i++)
//  {
//    edg3_t *e = &edgs[i];
//    if (!e->is_split && e->loc == LNE)
//      if ( !integrate_specific
//           || (integrate_specific
//               && (e->c0 == num0 || e->c1 == num0)
//               && (e->c0 == num1 || e->c1 == num1)) )
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_edg(xyz, i, a0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_edg(xyz, i, a1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_edg(xyz, i, a2); f2 = f( xyz[0], xyz[1], xyz[2] );

//        // scale weights by Jacobian
//        w0 = 5.*jacobian_edg(i, a0);
//        w1 = 8.*jacobian_edg(i, a1);
//        w2 = 5.*jacobian_edg(i, a2);

//        result += w0*f0 + w1*f1+w2*f2;
//      }
//  }

//  return result/18.;
//}

double simplex3_mls_quadratic_t::integrate_over_intersection(CF_3 &f, int num0, int num1, int num2)
{
  double result = 0.0;
  bool integrate_specific = (num0 != -1 && num1 != -1 && num2 != -1);

  for (unsigned int i = 0; i < vtxs.size(); i++)
  {
    vtx3_t *v = &vtxs[i];
    if (v->loc == PNT)
      if ( !integrate_specific
           || (integrate_specific
               && (v->c0 == num0 || v->c1 == num0 || v->c2 == num0)
               && (v->c0 == num1 || v->c1 == num1 || v->c2 == num1)
               && (v->c0 == num2 || v->c1 == num2 || v->c2 == num2)) )
      {
        result += f( v->x, v->y, v->z );
      }
  }

  return result;
}

//double simplex3_mls_quadratic_t::integrate_in_dir(CF_3 &f, int dir)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0;
//  double f0 = 0, f1 = 0, f2 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 1./6., 1./6. };
//  double ab1[2] = { 2./3., 1./6. };
//  double ab2[2] = { 1./6., 2./3. };

//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == INS)
//      if (t->dir == dir)
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );

//        // scale weights by Jacobian
//        w0 = jacobian_tri(i, ab0)/6.;
//        w1 = jacobian_tri(i, ab1)/6.;
//        w2 = jacobian_tri(i, ab2)/6.;

//        result += w0*f0 + w1*f1 + w2*f2;
//      }
//  }

//  return result;
//}

// Linear integration
//double simplex3_mls_quadratic_t::integrate_in_dir(CF_3 &f, int dir)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 0., 0. };
//  double ab1[2] = { 1., 0. };
//  double ab2[2] = { 0., 1. };

//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == INS)
//      if (t->dir == dir)
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );

//        result += (f0+f1+f2)/3.*area(t->vtx0, t->vtx1, t->vtx2);
//      }
//  }

////  return result/96.;
//  return result;
//}

//double simplex3_mls_quadratic_t::integrate_in_dir(CF_3 &f, int dir)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
//  double xyz[3];

//  // quadrature points
//  double ab0[2] = { 1./3., 1./3. };
//  double ab1[2] = { .2, .6 };
//  double ab2[2] = { .2, .2 };
//  double ab3[2] = { .6, .2 };

//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri3_t *t = &tris[i];
//    if (!t->is_split && t->loc == INS)
//      if (t->dir == dir)
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );
//        mapping_tri(xyz, i, ab3); f3 = f( xyz[0], xyz[1], xyz[2] );


//        // scale weights by Jacobian
//        w0 =-27.*jacobian_tri(i, ab0);
//        w1 = 25.*jacobian_tri(i, ab1);
//        w2 = 25.*jacobian_tri(i, ab2);
//        w3 = 25.*jacobian_tri(i, ab3);

//        result += w0*f0 + w1*f1 + w2*f2 + w3*f3;
//      }
//  }

//  return result/96.;
//}

double simplex3_mls_quadratic_t::integrate_in_dir(CF_3 &f, int dir)
{
  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0;
  double f0 = 0, f1 = 0, f2 = 0;
  double xyz[3];

  // quadrature points
  static double ab0[2] = { .0, .5 };
  static double ab1[2] = { .5, .0 };
  static double ab2[2] = { .5, .5 };

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris.size(); i++)
  {
    tri3_t *t = &tris[i];
    if (!t->is_split && t->loc == INS)
      if (t->dir == dir)
      {
        // map quadrature points into real space and interpolate integrand
        mapping_tri(xyz, i, ab0); f0 = f( xyz[0], xyz[1], xyz[2] );
        mapping_tri(xyz, i, ab1); f1 = f( xyz[0], xyz[1], xyz[2] );
        mapping_tri(xyz, i, ab2); f2 = f( xyz[0], xyz[1], xyz[2] );

        // scale weights by Jacobian
        w0 = jacobian_tri(i, ab0);
        w1 = jacobian_tri(i, ab1);
        w2 = jacobian_tri(i, ab2);

        result += w0*f0 + w1*f1 + w2*f2;
      }
  }

  return result/6.;
}




//--------------------------------------------------
// Jacobians
//--------------------------------------------------
double simplex3_mls_quadratic_t::jacobian_edg(int n_edg, double a)
{
  edg3_t *edg = &edgs[n_edg];

  double Na[3] = {-3.+4.*a, 4.-8.*a, -1.+4.*a};

  double X = vtxs[edg->vtx0].x * Na[0] + vtxs[edg->vtx1].x * Na[1] + vtxs[edg->vtx2].x * Na[2];
  double Y = vtxs[edg->vtx0].y * Na[0] + vtxs[edg->vtx1].y * Na[1] + vtxs[edg->vtx2].y * Na[2];
  double Z = vtxs[edg->vtx0].z * Na[0] + vtxs[edg->vtx1].z * Na[1] + vtxs[edg->vtx2].z * Na[2];

  return sqrt(X*X+Y*Y+Z*Z);
}

double simplex3_mls_quadratic_t::jacobian_tri(int n_tri, double *ab)
{
  tri3_t *tri = &tris[n_tri];

  if (tri->is_curved) { // if triangle is curved, then use the two-stage mapping
    // first, map a reference element into a 2D triangle on the surface
    double a = ab[0];
    double b = ab[1];

    double N_2d[nodes_per_tri]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};
    double Na_2d[nodes_per_tri] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
    double Nb_2d[nodes_per_tri] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

    double A[nodes_per_tri] = { 0, 1, 0, tri->ab01[0], tri->ab12[0], tri->ab02[0] };
    double B[nodes_per_tri] = { 0, 0, 1, tri->ab01[1], tri->ab12[1], tri->ab02[1] };

    double Aa = 0, Ba = 0;
    double Ab = 0, Bb = 0;

    a = 0;
    b = 0;

    for (int i = 0; i < nodes_per_tri; ++i)
    {
      a += A[i]*N_2d[i];   Aa += A[i]*Na_2d[i];   Ba += B[i]*Na_2d[i];
      b += B[i]*N_2d[i];   Ab += A[i]*Nb_2d[i];   Bb += B[i]*Nb_2d[i];
    }

    double jacobian_2d = fabs(Aa*Bb-Ab*Ba);

    // second, map the 2d surface triangle into 3D
//    a = ab[0];
//    b = ab[1];

    double Na[nodes_per_tri] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
    double Nb[nodes_per_tri] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

    int nv0 = tri->vtx0;
    int nv1 = tri->vtx1;
    int nv2 = tri->vtx2;

    double X[nodes_per_tri] = { vtxs[nv0].x, vtxs[nv1].x, vtxs[nv2].x, tri->g_vtx01[0], tri->g_vtx12[0], tri->g_vtx02[0] };
    double Y[nodes_per_tri] = { vtxs[nv0].y, vtxs[nv1].y, vtxs[nv2].y, tri->g_vtx01[1], tri->g_vtx12[1], tri->g_vtx02[1] };
    double Z[nodes_per_tri] = { vtxs[nv0].z, vtxs[nv1].z, vtxs[nv2].z, tri->g_vtx01[2], tri->g_vtx12[2], tri->g_vtx02[2] };

    double Xa = 0, Ya = 0, Za = 0;
    double Xb = 0, Yb = 0, Zb = 0;

    for (int i = 0; i < nodes_per_tri; ++i)
    {
      Xa += X[i]*Na[i];   Ya += Y[i]*Na[i];   Za += Z[i]*Na[i];
      Xb += X[i]*Nb[i];   Yb += Y[i]*Nb[i];   Zb += Z[i]*Nb[i];
    }

//    return 1.*sqrt((Xa*Xa+Ya*Ya+Za*Za)*(Xb*Xb+Yb*Yb+Zb*Zb) - pow(Xa*Xb+Ya*Yb+Za*Zb, 2.));
    double result = jacobian_2d*sqrt((Xa*Xa+Ya*Ya+Za*Za)*(Xb*Xb+Yb*Yb+Zb*Zb) - pow(Xa*Xb+Ya*Yb+Za*Zb, 2.));

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    if (result != result)
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif

    return result;

  } else { // if triangle is not curved, then a one-stage mapping suffies

    double a = ab[0];
    double b = ab[1];

    double Na[6] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
    double Nb[6] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

    double X1 = vtxs[tri->vtx0].x*Na[0] + vtxs[tri->vtx1].x*Na[1] + vtxs[tri->vtx2].x*Na[2] + vtxs[edgs[tri->edg2].vtx1].x*Na[3] + vtxs[edgs[tri->edg0].vtx1].x*Na[4] + vtxs[edgs[tri->edg1].vtx1].x*Na[5];
    double Y1 = vtxs[tri->vtx0].y*Na[0] + vtxs[tri->vtx1].y*Na[1] + vtxs[tri->vtx2].y*Na[2] + vtxs[edgs[tri->edg2].vtx1].y*Na[3] + vtxs[edgs[tri->edg0].vtx1].y*Na[4] + vtxs[edgs[tri->edg1].vtx1].y*Na[5];
    double Z1 = vtxs[tri->vtx0].z*Na[0] + vtxs[tri->vtx1].z*Na[1] + vtxs[tri->vtx2].z*Na[2] + vtxs[edgs[tri->edg2].vtx1].z*Na[3] + vtxs[edgs[tri->edg0].vtx1].z*Na[4] + vtxs[edgs[tri->edg1].vtx1].z*Na[5];

    double X2 = vtxs[tri->vtx0].x*Nb[0] + vtxs[tri->vtx1].x*Nb[1] + vtxs[tri->vtx2].x*Nb[2] + vtxs[edgs[tri->edg2].vtx1].x*Nb[3] + vtxs[edgs[tri->edg0].vtx1].x*Nb[4] + vtxs[edgs[tri->edg1].vtx1].x*Nb[5];
    double Y2 = vtxs[tri->vtx0].y*Nb[0] + vtxs[tri->vtx1].y*Nb[1] + vtxs[tri->vtx2].y*Nb[2] + vtxs[edgs[tri->edg2].vtx1].y*Nb[3] + vtxs[edgs[tri->edg0].vtx1].y*Nb[4] + vtxs[edgs[tri->edg1].vtx1].y*Nb[5];
    double Z2 = vtxs[tri->vtx0].z*Nb[0] + vtxs[tri->vtx1].z*Nb[1] + vtxs[tri->vtx2].z*Nb[2] + vtxs[edgs[tri->edg2].vtx1].z*Nb[3] + vtxs[edgs[tri->edg0].vtx1].z*Nb[4] + vtxs[edgs[tri->edg1].vtx1].z*Nb[5];

//    double xyz0[3] = { vtxs[tri->vtx0].x, vtxs[tri->vtx0].y, vtxs[tri->vtx0].z };
//    double xyz1[3] = { vtxs[tri->vtx1].x, vtxs[tri->vtx1].y, vtxs[tri->vtx1].z };
//    double xyz2[3] = { vtxs[tri->vtx2].x, vtxs[tri->vtx2].y, vtxs[tri->vtx2].z };

//    double xyz3[3] = { vtxs[edgs[tri->edg2].vtx1].x, vtxs[edgs[tri->edg2].vtx1].y, vtxs[edgs[tri->edg2].vtx1].z };
//    double xyz4[3] = { vtxs[edgs[tri->edg0].vtx1].x, vtxs[edgs[tri->edg0].vtx1].y, vtxs[edgs[tri->edg0].vtx1].z };
//    double xyz5[3] = { vtxs[edgs[tri->edg1].vtx1].x, vtxs[edgs[tri->edg1].vtx1].y, vtxs[edgs[tri->edg1].vtx1].z };

//    xyz3[0] -= .5*(xyz0[0]+xyz1[0]);
//    xyz3[1] -= .5*(xyz0[1]+xyz1[1]);
//    xyz3[2] -= .5*(xyz0[2]+xyz1[2]);

//    xyz4[0] -= .5*(xyz1[0]+xyz2[0]);
//    xyz4[1] -= .5*(xyz1[1]+xyz2[1]);
//    xyz4[2] -= .5*(xyz1[2]+xyz2[2]);

//    xyz5[0] -= .5*(xyz0[0]+xyz2[0]);
//    xyz5[1] -= .5*(xyz0[1]+xyz2[1]);
//    xyz5[2] -= .5*(xyz0[2]+xyz2[2]);

//    double jac = ((X1*X1+Y1*Y1+Z1*Z1)*(X2*X2+Y2*Y2+Z2*Z2) - pow(X1*X2+Y1*Y2+Z1*Z2,2.));

    //
    double result = sqrt(fabs((X1*X1+Y1*Y1+Z1*Z1)*(X2*X2+Y2*Y2+Z2*Z2) - pow(X1*X2+Y1*Y2+Z1*Z2,2.)));

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
    if (result != result)
      throw std::domain_error("[CASL_ERROR]: Something went wrong during integration.");
#endif

    return result;
  }
}

double simplex3_mls_quadratic_t::jacobian_tet(int n_tet, double *abc)
{
  tet3_t *tet = &tets[n_tet];

  double a = abc[0];
  double b = abc[1];
  double c = abc[2];

  double Na[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
  double Nb[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
  double Nc[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

  int e03 = tris[tet->tri1].edg1;
  int e01 = tris[tet->tri3].edg2;
  int e02 = tris[tet->tri3].edg1;
  int e12 = tris[tet->tri0].edg2;
  int e23 = tris[tet->tri0].edg0;
  int e13 = tris[tet->tri0].edg1;

  std::vector<int> nv(nodes_per_tet, -1);

  nv[0] = tet->vtx0;
  nv[1] = tet->vtx1;
  nv[2] = tet->vtx2;
  nv[3] = tet->vtx3;
  nv[4] = edgs[e01].vtx1;
  nv[5] = edgs[e12].vtx1;
  nv[6] = edgs[e02].vtx1;
  nv[7] = edgs[e03].vtx1;
  nv[8] = edgs[e13].vtx1;
  nv[9] = edgs[e23].vtx1;

  double jac[3][3];

  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      jac[i][j] = 0;

  for (int i = 0; i < nodes_per_tet; ++i)
  {
    jac[0][0] += vtxs[nv[i]].x*Na[i];    jac[0][1] += vtxs[nv[i]].x*Nb[i];    jac[0][2] += vtxs[nv[i]].x*Nc[i];
    jac[1][0] += vtxs[nv[i]].y*Na[i];    jac[1][1] += vtxs[nv[i]].y*Nb[i];    jac[1][2] += vtxs[nv[i]].y*Nc[i];
    jac[2][0] += vtxs[nv[i]].z*Na[i];    jac[2][1] += vtxs[nv[i]].z*Nb[i];    jac[2][2] += vtxs[nv[i]].z*Nc[i];
  }

  return fabs( jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1]) - jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0]) + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]) );
}





////--------------------------------------------------
//// Interpolation
////--------------------------------------------------
//double simplex3_mls_quadratic_t::interpolate_from_parent(std::vector<double> &f, double* xyz)
//{
//  // map real point to reference element
//  vtx3_t *v0 = &vtxs[0];
//  vtx3_t *v1 = &vtxs[1];
//  vtx3_t *v2 = &vtxs[2];
//  vtx3_t *v3 = &vtxs[3];

//  double A[9], A_inv[9], D[3];
//  A[3*0+0] = v1->x - v0->x; A[3*0+1] = v2->x - v0->x; A[3*0+2] = v3->x - v0->x; D[0] = xyz[0] - v0->x;
//  A[3*1+0] = v1->y - v0->y; A[3*1+1] = v2->y - v0->y; A[3*1+2] = v3->y - v0->y; D[1] = xyz[1] - v0->y;
//  A[3*2+0] = v1->z - v0->z; A[3*2+1] = v2->z - v0->z; A[3*2+2] = v3->z - v0->z; D[2] = xyz[2] - v0->z;

//  inv_mat3(A, A_inv);

//  double a = A_inv[3*0+0]*D[0] + A_inv[3*0+1]*D[1] + A_inv[3*0+2]*D[2];
//  double b = A_inv[3*1+0]*D[0] + A_inv[3*1+1]*D[1] + A_inv[3*1+2]*D[2];
//  double c = A_inv[3*2+0]*D[0] + A_inv[3*2+1]*D[1] + A_inv[3*2+2]*D[2];

//  // compute nodal functions
//  double d = 1.-a-b-c;
//  double N[nodes_per_tet]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

//  double result = 0;

//  for (short i = 0; i < nodes_per_tet; ++i)
//  {
//    result += N[i]*f[i];
//  }

//  return result;
//}

//void simplex3_mls_quadratic_t::inv_mat3(double *in, double *out)
//{
//  double det = in[3*0+0]*(in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2]) -
//               in[3*0+1]*(in[3*1+0]*in[3*2+2] - in[3*1+2]*in[3*2+0]) +
//               in[3*0+2]*(in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1]);

//  out[3*0+0] = (in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2])/det;
//  out[3*0+1] = (in[3*0+2]*in[3*2+1] - in[3*2+2]*in[3*0+1])/det;
//  out[3*0+2] = (in[3*0+1]*in[3*1+2] - in[3*1+1]*in[3*0+2])/det;

//  out[3*1+0] = (in[3*1+2]*in[3*2+0] - in[3*2+2]*in[3*1+0])/det;
//  out[3*1+1] = (in[3*0+0]*in[3*2+2] - in[3*2+0]*in[3*0+2])/det;
//  out[3*1+2] = (in[3*0+2]*in[3*1+0] - in[3*1+2]*in[3*0+0])/det;

//  out[3*2+0] = (in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1])/det;
//  out[3*2+1] = (in[3*0+1]*in[3*2+0] - in[3*2+1]*in[3*0+0])/det;
//  out[3*2+2] = (in[3*0+0]*in[3*1+1] - in[3*1+0]*in[3*0+1])/det;
//}









//--------------------------------------------------
// Mapping
//--------------------------------------------------
void simplex3_mls_quadratic_t::mapping_edg(double* xyz, int n_edg, double a)
{
  edg3_t *edg = &edgs[n_edg];

  double N0 = 1.-3.*a+2.*a*a;
  double N1 = 4.*a-4.*a*a;
  double N2 = -a+2.*a*a;

  xyz[0] = vtxs[edg->vtx0].x * N0 + vtxs[edg->vtx1].x * N1 + vtxs[edg->vtx2].x * N2;
  xyz[1] = vtxs[edg->vtx0].y * N0 + vtxs[edg->vtx1].y * N1 + vtxs[edg->vtx2].y * N2;
  xyz[2] = vtxs[edg->vtx0].z * N0 + vtxs[edg->vtx1].z * N1 + vtxs[edg->vtx2].z * N2;
}

void simplex3_mls_quadratic_t::mapping_tri(double* xyz, int n_tri, double* ab)
{
  tri3_t *tri = &tris[n_tri];

  if (tri->is_curved) { // if a triangle is curved, then use the two-stage mapping
    // first, map a reference element into a 2D triangle on the surface
    double a = ab[0];
    double b = ab[1];

    double N_2d[nodes_per_tri]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

    double A[nodes_per_tri] = { 0, 1, 0, tri->ab01[0], tri->ab12[0], tri->ab02[0] };
    double B[nodes_per_tri] = { 0, 0, 1, tri->ab01[1], tri->ab12[1], tri->ab02[1] };

    a = 0;
    b = 0;

    for (int i = 0; i < nodes_per_tri; ++i)
    {
      a += A[i]*N_2d[i];
      b += B[i]*N_2d[i];
    }

//    a = ab[0];
//    b = ab[1];

    // second, map the 2d surface triangle into 3D
    double N[nodes_per_tri]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

    int nv0 = tri->vtx0;
    int nv1 = tri->vtx1;
    int nv2 = tri->vtx2;

    double X[nodes_per_tri] = { vtxs[nv0].x, vtxs[nv1].x, vtxs[nv2].x, tri->g_vtx01[0], tri->g_vtx12[0], tri->g_vtx02[0] };
    double Y[nodes_per_tri] = { vtxs[nv0].y, vtxs[nv1].y, vtxs[nv2].y, tri->g_vtx01[1], tri->g_vtx12[1], tri->g_vtx02[1] };
    double Z[nodes_per_tri] = { vtxs[nv0].z, vtxs[nv1].z, vtxs[nv2].z, tri->g_vtx01[2], tri->g_vtx12[2], tri->g_vtx02[2] };

    xyz[0] = 0;
    xyz[1] = 0;
    xyz[2] = 0;

    for (int i = 0; i < nodes_per_tri; ++i)
    {
      xyz[0] += X[i]*N[i];
      xyz[1] += Y[i]*N[i];
      xyz[2] += Z[i]*N[i];
    }

  } else { // if a triangle is not curved, then a one-stage mapping suffies
    double a = ab[0];
    double b = ab[1];

    double N[nodes_per_tri]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

    int nv0 = tri->vtx0;
    int nv1 = tri->vtx1;
    int nv2 = tri->vtx2;
    int nv3 = edgs[tri->edg2].vtx1;
    int nv4 = edgs[tri->edg0].vtx1;
    int nv5 = edgs[tri->edg1].vtx1;

    xyz[0] = vtxs[nv0].x * N[0] + vtxs[nv1].x * N[1] + vtxs[nv2].x * N[2] + vtxs[nv3].x * N[3] + vtxs[nv4].x * N[4] + vtxs[nv5].x * N[5];
    xyz[1] = vtxs[nv0].y * N[0] + vtxs[nv1].y * N[1] + vtxs[nv2].y * N[2] + vtxs[nv3].y * N[3] + vtxs[nv4].y * N[4] + vtxs[nv5].y * N[5];
    xyz[2] = vtxs[nv0].z * N[0] + vtxs[nv1].z * N[1] + vtxs[nv2].z * N[2] + vtxs[nv3].z * N[3] + vtxs[nv4].z * N[4] + vtxs[nv5].z * N[5];
  }
}

void simplex3_mls_quadratic_t::mapping_tet(double *xyz, int n_tet, double* abc)
{
  tet3_t *tet = &tets[n_tet];

  int e01 = tris[tet->tri3].edg2;
  int e02 = tris[tet->tri3].edg1;
  int e03 = tris[tet->tri1].edg1;
  int e12 = tris[tet->tri0].edg2;
  int e13 = tris[tet->tri0].edg1;
  int e23 = tris[tet->tri0].edg0;

  std::vector<int> nv(nodes_per_tet, -1);

  nv[0] = tet->vtx0;
  nv[1] = tet->vtx1;
  nv[2] = tet->vtx2;
  nv[3] = tet->vtx3;
  nv[4] = edgs[e01].vtx1;
  nv[5] = edgs[e12].vtx1;
  nv[6] = edgs[e02].vtx1;
  nv[7] = edgs[e03].vtx1;
  nv[8] = edgs[e13].vtx1;
  nv[9] = edgs[e23].vtx1;

  double a = abc[0];
  double b = abc[1];
  double c = abc[2];
  double d = 1.-a-b-c;
  double N[nodes_per_tet]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  xyz[0] = 0.;
  xyz[1] = 0.;
  xyz[2] = 0.;

  for (short i = 0; i < nodes_per_tet; ++i)
  {
    xyz[0] += N[i]*vtxs[nv[i]].x;
    xyz[1] += N[i]*vtxs[nv[i]].y;
    xyz[2] += N[i]*vtxs[nv[i]].z;
  }
}

double simplex3_mls_quadratic_t::find_root(double phi, double phi_n, double phi_nn)
{
  double c2 = 0.5*phi_nn;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = phi_n;   // the expansion of f at the center of (0,1)
  double c0 = phi;

  double x;

  if (fabs(c2) < eps) { x = -c0/c1; }
  else
  {
//    if (c1<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
//    else      x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    double alpha1 = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    double alpha2 = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    if (fabs(alpha1)>fabs(alpha2)) x = alpha2;
    else x = alpha1;
  }
//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if (x < -0.5 || x > 0.5)
//  {
//    throw std::domain_error("[CASL_ERROR]: ");
//  }
//#endif

//  if (x <-0.5) return eps;
//  if (x > 0.5) return 1.-eps;

  return x;
}


void simplex3_mls_quadratic_t::construct_proper_mapping(int tri_idx, int phi_idx)
{
  tri3_t *tri = &tris[tri_idx];

  tri->is_curved = true;
//  tri->is_curved = false;

  int v0 = tri->vtx0;
  int v1 = tri->vtx1;
  int v2 = tri->vtx2;

  /* find normal to triangle */
  double t01[3] = { vtxs[v1].x - vtxs[v0].x, vtxs[v1].y - vtxs[v0].y, vtxs[v1].z - vtxs[v0].z };
  double t02[3] = { vtxs[v2].x - vtxs[v0].x, vtxs[v2].y - vtxs[v0].y, vtxs[v2].z - vtxs[v0].z };

  double normal[3] = { t01[1]*t02[2] - t01[2]*t02[1],
                       t01[2]*t02[0] - t01[0]*t02[2],
                       t01[0]*t02[1] - t01[1]*t02[0] };

  double norm = sqrt(pow(normal[0], 2.) + pow(normal[1], 2.) + pow(normal[2], 2.));

  normal[0] /= norm;
  normal[1] /= norm;
  normal[2] /= norm;

  /* deform edges in normal direction */
  double xyz_start[3];
//  double xyz_f[3];
  double phi_value, phi_n_value, phi_nn_value;
  double phi_fwd;
  double phi_bwd;
  int u0, u1;
  double d;

  double shift = 0.5;

  // point 01
  u0 = v0;
  u1 = v1;

  xyz_start[0] = .5*(vtxs[u0].x+vtxs[u1].x);
  xyz_start[1] = .5*(vtxs[u0].y+vtxs[u1].y);
  xyz_start[2] = .5*(vtxs[u0].z+vtxs[u1].z);

  interpolate_from_parent_with_derivatives(xyz_start, normal, phi_value, phi_n_value, phi_nn_value);
//  interpolate_from_parent_with_derivatives(xyz_start, phi_value, phi_n_value, phi_nn_value, normal);

//  phi_value = (*phi_->at(0))(xyz_start[0], xyz_start[1], xyz_start[2]);
//  phi_fwd = (*phi_->at(0))(xyz_start[0] + shift*diag*normal[0], xyz_start[1] + shift*diag*normal[1], xyz_start[2] + shift*diag*normal[2]);
//  phi_bwd = (*phi_->at(0))(xyz_start[0] - shift*diag*normal[0], xyz_start[1] - shift*diag*normal[1], xyz_start[2] - shift*diag*normal[2]);
//  phi_n_value = (phi_fwd-phi_bwd)/diag;
//  phi_nn_value = (phi_fwd - 2.*phi_value + phi_bwd)/diag/diag;
//  phi_nn_value = 0;

  d = find_root(phi_value, phi_n_value, phi_nn_value);

//  d = 0;

  tri->g_vtx01[0] = xyz_start[0] + d*normal[0];
  tri->g_vtx01[1] = xyz_start[1] + d*normal[1];
  tri->g_vtx01[2] = xyz_start[2] + d*normal[2];

//  tri->g_vtx01[0] = vtxs[edgs[tri->edg2].vtx1].x;
//  tri->g_vtx01[1] = vtxs[edgs[tri->edg2].vtx1].y;
//  tri->g_vtx01[2] = vtxs[edgs[tri->edg2].vtx1].z;

  // point 12
  u0 = v1;
  u1 = v2;

  xyz_start[0] = .5*(vtxs[u0].x+vtxs[u1].x);
  xyz_start[1] = .5*(vtxs[u0].y+vtxs[u1].y);
  xyz_start[2] = .5*(vtxs[u0].z+vtxs[u1].z);

  interpolate_from_parent_with_derivatives(xyz_start, normal, phi_value, phi_n_value, phi_nn_value);
//  interpolate_from_parent_with_derivatives(xyz_start, phi_value, phi_n_value, phi_nn_value, normal);

//  phi_value = (*phi_->at(0))(xyz_start[0], xyz_start[1], xyz_start[2]);
//  phi_fwd = (*phi_->at(0))(xyz_start[0] + shift*diag*normal[0], xyz_start[1] + shift*diag*normal[1], xyz_start[2] + shift*diag*normal[2]);
//  phi_bwd = (*phi_->at(0))(xyz_start[0] - shift*diag*normal[0], xyz_start[1] - shift*diag*normal[1], xyz_start[2] - shift*diag*normal[2]);
//  phi_n_value = (phi_fwd-phi_bwd)/diag;
//  phi_nn_value = (phi_fwd - 2.*phi_value + phi_bwd)/diag/diag;
//  phi_nn_value = 0;

  d = find_root(phi_value, phi_n_value, phi_nn_value);

//  d = 0;

  tri->g_vtx12[0] = xyz_start[0] + d*normal[0];
  tri->g_vtx12[1] = xyz_start[1] + d*normal[1];
  tri->g_vtx12[2] = xyz_start[2] + d*normal[2];

//  tri->g_vtx12[0] = vtxs[edgs[tri->edg0].vtx1].x;
//  tri->g_vtx12[1] = vtxs[edgs[tri->edg0].vtx1].y;
//  tri->g_vtx12[2] = vtxs[edgs[tri->edg0].vtx1].z;

  // point 20
  u0 = v2;
  u1 = v0;

  xyz_start[0] = .5*(vtxs[u0].x+vtxs[u1].x);
  xyz_start[1] = .5*(vtxs[u0].y+vtxs[u1].y);
  xyz_start[2] = .5*(vtxs[u0].z+vtxs[u1].z);

  interpolate_from_parent_with_derivatives(xyz_start, normal, phi_value, phi_n_value, phi_nn_value);
//  interpolate_from_parent_with_derivatives(xyz_start, phi_value, phi_n_value, phi_nn_value, normal);

//  phi_value = (*phi_->at(0))(xyz_start[0], xyz_start[1], xyz_start[2]);
//  phi_fwd = (*phi_->at(0))(xyz_start[0] + shift*diag*normal[0], xyz_start[1] + shift*diag*normal[1], xyz_start[2] + shift*diag*normal[2]);
//  phi_bwd = (*phi_->at(0))(xyz_start[0] - shift*diag*normal[0], xyz_start[1] - shift*diag*normal[1], xyz_start[2] - shift*diag*normal[2]);
//  phi_n_value = (phi_fwd-phi_bwd)/diag;
//  phi_nn_value = (phi_fwd - 2.*phi_value + phi_bwd)/diag/diag;
//  phi_nn_value = 0;

  d = find_root(phi_value, phi_n_value, phi_nn_value);

//  d = 0;

  tri->g_vtx02[0] = xyz_start[0] + d*normal[0];
  tri->g_vtx02[1] = xyz_start[1] + d*normal[1];
  tri->g_vtx02[2] = xyz_start[2] + d*normal[2];

//  tri->g_vtx02[0] = vtxs[edgs[tri->edg1].vtx1].x;
//  tri->g_vtx02[1] = vtxs[edgs[tri->edg1].vtx1].y;
//  tri->g_vtx02[2] = vtxs[edgs[tri->edg1].vtx1].z;


//  normal[0] = t01[1]*t02[2] - t01[2]*t02[1];
//  normal[1] = t01[2]*t02[0] - t01[0]*t02[2];
//  normal[2] = t01[0]*t02[1] - t01[1]*t02[0];

//  norm = sqrt(pow(normal[0], 2.) + pow(normal[1], 2.) + pow(normal[2], 2.));

//  normal[0] /= norm;
//  normal[1] /= norm;
//  normal[2] /= norm;

  /* find projections of old midpoints onto triangle's plane */
  vtx3_t *vtx0 = &vtxs[v0];
  vtx3_t *vtx1 = &vtxs[v1];
  vtx3_t *vtx2 = &vtxs[v2];

  double A[9], A_inv[9], D[3];
  A[3*0+0] = vtx1->x - vtx0->x; A[3*0+1] = vtx2->x - vtx0->x; A[3*0+2] = normal[0];
  A[3*1+0] = vtx1->y - vtx0->y; A[3*1+1] = vtx2->y - vtx0->y; A[3*1+2] = normal[1];
  A[3*2+0] = vtx1->z - vtx0->z; A[3*2+1] = vtx2->z - vtx0->z; A[3*2+2] = normal[2];

  inv_mat3(A, A_inv);
  double xyz_test[3];
  double xyz_target[3];

  int u = edgs[tri->edg0].vtx1;

  D[0] = vtxs[u].x - vtx0->x;
  D[1] = vtxs[u].y - vtx0->y;
  D[2] = vtxs[u].z - vtx0->z;

  tri->ab12[0] = A_inv[3*0+0]*D[0] + A_inv[3*0+1]*D[1] + A_inv[3*0+2]*D[2];
  tri->ab12[1] = A_inv[3*1+0]*D[0] + A_inv[3*1+1]*D[1] + A_inv[3*1+2]*D[2];
//  tri->ab12[0] = 0.5;
//  tri->ab12[1] = 0.5;
//  tri->ab12[2] = A_inv[3*2+0]*D[0] + A_inv[3*2+1]*D[1] + A_inv[3*2+2]*D[2];

//  xyz_target[0] = vtxs[u].x;
//  xyz_target[1] = vtxs[u].y;
//  xyz_target[2] = vtxs[u].z;

//  invert_mapping_tri(tri_idx, xyz_target, tri->ab12);

  u = edgs[tri->edg1].vtx1;

  D[0] = vtxs[u].x - vtx0->x;
  D[1] = vtxs[u].y - vtx0->y;
  D[2] = vtxs[u].z - vtx0->z;

  tri->ab02[0] = A_inv[3*0+0]*D[0] + A_inv[3*0+1]*D[1] + A_inv[3*0+2]*D[2];
  tri->ab02[1] = A_inv[3*1+0]*D[0] + A_inv[3*1+1]*D[1] + A_inv[3*1+2]*D[2];
//  tri->ab02[0] = 0.0;
//  tri->ab02[1] = 0.5;
//  tri->ab02[2] = A_inv[3*2+0]*D[0] + A_inv[3*2+1]*D[1] + A_inv[3*2+2]*D[2];

//  xyz_target[0] = vtxs[u].x;
//  xyz_target[1] = vtxs[u].y;
//  xyz_target[2] = vtxs[u].z;

//  invert_mapping_tri(tri_idx, xyz_target, tri->ab02);

  u = edgs[tri->edg2].vtx1;

  D[0] = vtxs[u].x - vtx0->x;
  D[1] = vtxs[u].y - vtx0->y;
  D[2] = vtxs[u].z - vtx0->z;

  tri->ab01[0] = A_inv[3*0+0]*D[0] + A_inv[3*0+1]*D[1] + A_inv[3*0+2]*D[2];
  tri->ab01[1] = A_inv[3*1+0]*D[0] + A_inv[3*1+1]*D[1] + A_inv[3*1+2]*D[2];
//  tri->ab01[0] = 0.5;
//  tri->ab01[1] = 0.0;
//  tri->ab01[2] = A_inv[3*2+0]*D[0] + A_inv[3*2+1]*D[1] + A_inv[3*2+2]*D[2];

//  xyz_target[0] = vtxs[u].x;
//  xyz_target[1] = vtxs[u].y;
//  xyz_target[2] = vtxs[u].z;

//  invert_mapping_tri(tri_idx, xyz_target, tri->ab01);

//  double dist_error = 0;

//  u = edgs[tri->edg0].vtx1;
//  mapping_tri(xyz_test, tri_idx, tri->ab12);
//  dist_error = sqrt(pow(xyz_test[0] - vtxs[u].x, 2.) + pow(xyz_test[1] - vtxs[u].y, 2.) + pow(xyz_test[2] - vtxs[u].z, 2.));
//  max_dist_error_ = MAX(dist_error, max_dist_error_);

//  u = edgs[tri->edg1].vtx1;
//  mapping_tri(xyz_test, tri_idx, tri->ab02);
//  dist_error = sqrt(pow(xyz_test[0] - vtxs[u].x, 2.) + pow(xyz_test[1] - vtxs[u].y, 2.) + pow(xyz_test[2] - vtxs[u].z, 2.));
//  max_dist_error_ = MAX(dist_error, max_dist_error_);

//  u = edgs[tri->edg2].vtx1;
//  mapping_tri(xyz_test, tri_idx, tri->ab01);
//  dist_error = sqrt(pow(xyz_test[0] - vtxs[u].x, 2.) + pow(xyz_test[1] - vtxs[u].y, 2.) + pow(xyz_test[2] - vtxs[u].z, 2.));
//  max_dist_error_ = MAX(dist_error, max_dist_error_);

}

#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
bool simplex3_mls_quadratic_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = (edgs[e0].vtx0 == v1 || edgs[e0].vtx2 == v1) && (edgs[e0].vtx0 == v2 || edgs[e0].vtx2 == v2);
  result = result && (edgs[e1].vtx0 == v0 || edgs[e1].vtx2 == v0) && (edgs[e1].vtx0 == v2 || edgs[e1].vtx2 == v2);
  result = result && (edgs[e2].vtx0 == v0 || edgs[e2].vtx2 == v0) && (edgs[e2].vtx0 == v1 || edgs[e2].vtx2 == v1);
  return result;
}

bool simplex3_mls_quadratic_t::tri_is_ok(int t)
{
  tri3_t *tri = &tris[t];
  bool result = tri_is_ok(tri->vtx0, tri->vtx1, tri->vtx2, tri->edg0, tri->edg1, tri->edg2);
  if (!result) std::cout << "Inconsistent triangle!\n";
  return result;
}

bool simplex3_mls_quadratic_t::tet_is_ok(int s)
{
  bool result = true;
  tet3_t *tet = &tets[s];

  tri3_t *tri;

  tri = &tris[tet->tri0];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);
  if (!result)
    throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child tetrahedra is not consistent.");

  tri = &tris[tet->tri1];
  result = result && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);
  if (!result)
    throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child tetrahedra is not consistent.");

  tri = &tris[tet->tri2];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);
  if (!result)
    throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child tetrahedra is not consistent.");

  tri = &tris[tet->tri3];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0);

  if (!result)
    throw std::domain_error("[CASL_ERROR]: While splitting a tetrahedron one of child tetrahedra is not consistent.");
  return result;
}
#endif



//void simplex3_mls_quadratic_t::interpolate_from_neighbors(int v)
//{
//  vtx3_t *vtx = &vtxs[v];
//  vtx->value = vtx->ratio*vtxs[vtx->n_vtx0].value + (1.0-vtx->ratio)*vtxs[vtx->n_vtx1].value;
//}

//void simplex3_mls_quadratic_t::interpolate_from_parent(int v)
//{
//  double vol0 = volume(v, 1, 2, 3);
//  double vol1 = volume(0, v, 2, 3);
//  double vol2 = volume(0, 1, v, 3);
//  double vol3 = volume(0, 1, 2, v);
//  double vol  = volume(0, 1, 2, 3);

//  #ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//    if (vol < eps)
//      throw std::domain_error("[CASL_ERROR]: Division by zero.");
//  #endif

//  vtxs[v].value = (vol0*vtxs[0].value + vol1*vtxs[1].value + vol2*vtxs[2].value + vol3*vtxs[3].value)/vol;
//}

//void simplex3_mls_quadratic_t::interpolate_from_parent(vtx3_t &vertex)
//{
//  double vol0 = volume(vertex, vtxs[1], vtxs[2], vtxs[3]);
//  double vol1 = volume(vtxs[0], vertex, vtxs[2], vtxs[3]);
//  double vol2 = volume(vtxs[0], vtxs[1], vertex, vtxs[3]);
//  double vol3 = volume(vtxs[0], vtxs[1], vtxs[2], vertex);
//  double vol  = volume(vtxs[0], vtxs[1], vtxs[2], vtxs[3]);

//  #ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//    if (vol < eps)
//      throw std::domain_error("[CASL_ERROR]: Division by zero.");
//  #endif

//  vertex.value = (vol0*vtxs[0].value + vol1*vtxs[1].value + vol2*vtxs[2].value + vol3*vtxs[3].value)/vol;
//}

//void simplex3_mls_quadratic_t::interpolate_all(double &p0, double &p1, double &p2, double &p3)
//{
//  vtxs[0].value = p0;
//  vtxs[1].value = p1;
//  vtxs[2].value = p2;
//  vtxs[3].value = p3;

//  for (unsigned int i = 4; i < vtxs.size(); i++) interpolate_from_neighbors(i);
////  for (int i = 4; i < vtxs.size(); i++) interpolate_from_parent(i);
//}

//double simplex3_mls_quadratic_t::find_intersection_linear(int v0, int v1)
//{
//  vtx3_t *vtx0 = &vtxs[v0];
//  vtx3_t *vtx1 = &vtxs[v1];
//  double nx = vtx1->x - vtx0->x;
//  double ny = vtx1->y - vtx0->y;
//  double nz = vtx1->z - vtx0->z;
//  double l = sqrt(nx*nx+ny*ny+nz*nz);
//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if(l < 0.8*eps) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
//#endif
//  nx /= l;
//  ny /= l;
//  nz /= l;
//  double f0 = vtx0->value;
//  double f1 = vtx1->value;

//  if(fabs(f0)<0.8*eps) return 0.+0.8*eps;
//  if(fabs(f1)<0.8*eps) return l-0.8*eps;

//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
//#endif

//  double c1 =     (f1-f0)/l;          //  the expansion of f at the center of (a,b)
//  double c0 = 0.5*(f1+f0);

//  double x = -c0/c1;

//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
//#endif

//  return 1.-(x+0.5*l)/l;
//}

//double simplex3_mls_quadratic_t::find_intersection_quadratic(int e)
//{
//  vtx3_t *vtx0 = &vtxs[edgs[e].vtx0];
//  vtx3_t *vtx1 = &vtxs[edgs[e].vtx1];
//  double nx = vtx1->x - vtx0->x;
//  double ny = vtx1->y - vtx0->y;
//  double nz = vtx1->z - vtx0->z;
//  double l = sqrt(nx*nx+ny*ny+nz*nz);
//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if(l < 0.8*eps) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
//#endif
//  nx /= l;
//  ny /= l;
//  nz /= l;
//  double f0 = vtx0->value;
//  double f01 = edgs[e].value;
//  double f1 = vtx1->value;

//  if (fabs(f0)  < 0.8*eps) return (l-0.8*eps)/l;
//  if (fabs(f01) < 0.8*eps) return 0.5;
//  if (fabs(f1)  < 0.8*eps) return (0.+0.8*eps)/l;

//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
//#endif

//  double fdd = (f1+f0-2.*f01)/(0.25*l*l);

//  double c2 = 0.5*fdd;   // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
//  double c1 = (f1-f0)/l; //  the expansion of f at the center of (a,b)
//  double c0 = f01;

//  double x;

//  if(fabs(c2)<eps) x = -c0/c1;
//  else
//  {
//    if(f1<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
//    else     x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
//  }
//#ifdef SIMPLEX3_MLS_QUADRATIC_DEBUG
//  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
//#endif

////  if (x < -0.5*l) return (l-eps)/l;
////  if (x > 0.5*l) return (0.+eps)/l;

//  return 1.-(x+0.5*l)/l;
//}

//void simplex3_mls_quadratic_t::get_edge_coords(int e, double xyz[])
//{
//  vtx3_t *vtx0 = &vtxs[edgs[e].vtx0];
//  vtx3_t *vtx1 = &vtxs[edgs[e].vtx1];

//  xyz[0] = 0.5*(vtx0->x+vtx1->x);
//  xyz[1] = 0.5*(vtx0->y+vtx1->y);
//  xyz[2] = 0.5*(vtx0->z+vtx1->z);
//}

//double simplex3_mls_quadratic_t::length(int vtx0, int vtx1)
//{
//  return sqrt(pow(vtxs[vtx0].x - vtxs[vtx1].x, 2.0)
//            + pow(vtxs[vtx0].y - vtxs[vtx1].y, 2.0)
//            + pow(vtxs[vtx0].z - vtxs[vtx1].z, 2.0));
//}
//double simplex3_mls_quadratic_t::area(int vtx0, int vtx1, int vtx2)
//{
//  double x01 = vtxs[vtx1].x - vtxs[vtx0].x; double x02 = vtxs[vtx2].x - vtxs[vtx0].x;
//  double y01 = vtxs[vtx1].y - vtxs[vtx0].y; double y02 = vtxs[vtx2].y - vtxs[vtx0].y;
//  double z01 = vtxs[vtx1].z - vtxs[vtx0].z; double z02 = vtxs[vtx2].z - vtxs[vtx0].z;

//  return 0.5*sqrt(pow(y01*z02-z01*y02,2.0) + pow(z01*x02-x01*z02,2.0) + pow(x01*y02-y01*x02,2.0));
//}

double simplex3_mls_quadratic_t::volume(int vtx0, int vtx1, int vtx2, int vtx3)
{
  double a11 = vtxs[vtx1].x-vtxs[vtx0].x; double a12 = vtxs[vtx1].y-vtxs[vtx0].y; double a13 = vtxs[vtx1].z-vtxs[vtx0].z;
  double a21 = vtxs[vtx2].x-vtxs[vtx0].x; double a22 = vtxs[vtx2].y-vtxs[vtx0].y; double a23 = vtxs[vtx2].z-vtxs[vtx0].z;
  double a31 = vtxs[vtx3].x-vtxs[vtx0].x; double a32 = vtxs[vtx3].y-vtxs[vtx0].y; double a33 = vtxs[vtx3].z-vtxs[vtx0].z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}

double simplex3_mls_quadratic_t::volume(vtx3_t &vtx0, vtx3_t &vtx1, vtx3_t &vtx2, vtx3_t &vtx3)
{
  double a11 = vtx1.x-vtx0.x; double a12 = vtx1.y-vtx0.y; double a13 = vtx1.z-vtx0.z;
  double a21 = vtx2.x-vtx0.x; double a22 = vtx2.y-vtx0.y; double a23 = vtx2.z-vtx0.z;
  double a31 = vtx3.x-vtx0.x; double a32 = vtx3.y-vtx0.y; double a33 = vtx3.z-vtx0.z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}

double simplex3_mls_quadratic_t::area(int vtx0, int vtx1, int vtx2)
{
  double x01 = vtxs[vtx1].x - vtxs[vtx0].x; double x02 = vtxs[vtx2].x - vtxs[vtx0].x;
  double y01 = vtxs[vtx1].y - vtxs[vtx0].y; double y02 = vtxs[vtx2].y - vtxs[vtx0].y;
  double z01 = vtxs[vtx1].z - vtxs[vtx0].z; double z02 = vtxs[vtx2].z - vtxs[vtx0].z;

  return 0.5*sqrt(pow(y01*z02-z01*y02,2.0) + pow(z01*x02-x01*z02,2.0) + pow(x01*y02-y01*x02,2.0));
}






//--------------------------------------------------
// Interpolation
//--------------------------------------------------
double simplex3_mls_quadratic_t::interpolate_from_parent(std::vector<double> &f, double* xyz)
{
  // map real point to reference element
  vtx3_t *v0 = &vtxs[0];
  vtx3_t *v1 = &vtxs[1];
  vtx3_t *v2 = &vtxs[2];
  vtx3_t *v3 = &vtxs[3];

  double A[9], A_inv[9], D[3];
  A[3*0+0] = v1->x - v0->x; A[3*0+1] = v2->x - v0->x; A[3*0+2] = v3->x - v0->x; D[0] = xyz[0] - v0->x;
  A[3*1+0] = v1->y - v0->y; A[3*1+1] = v2->y - v0->y; A[3*1+2] = v3->y - v0->y; D[1] = xyz[1] - v0->y;
  A[3*2+0] = v1->z - v0->z; A[3*2+1] = v2->z - v0->z; A[3*2+2] = v3->z - v0->z; D[2] = xyz[2] - v0->z;

  inv_mat3(A, A_inv);

  double a = A_inv[3*0+0]*D[0] + A_inv[3*0+1]*D[1] + A_inv[3*0+2]*D[2];
  double b = A_inv[3*1+0]*D[0] + A_inv[3*1+1]*D[1] + A_inv[3*1+2]*D[2];
  double c = A_inv[3*2+0]*D[0] + A_inv[3*2+1]*D[1] + A_inv[3*2+2]*D[2];

  // compute nodal functions
  double d = 1.-a-b-c;
  double N[nodes_per_tet]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  double result = 0;

  for (short i = 0; i < nodes_per_tet; ++i)
  {
    result += N[i]*f[i];
  }

  return result;
}

void simplex3_mls_quadratic_t::inv_mat3(double *in, double *out)
{
  double det = in[3*0+0]*(in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2]) -
               in[3*0+1]*(in[3*1+0]*in[3*2+2] - in[3*1+2]*in[3*2+0]) +
               in[3*0+2]*(in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1]);

  out[3*0+0] = (in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2])/det;
  out[3*0+1] = (in[3*0+2]*in[3*2+1] - in[3*2+2]*in[3*0+1])/det;
  out[3*0+2] = (in[3*0+1]*in[3*1+2] - in[3*1+1]*in[3*0+2])/det;

  out[3*1+0] = (in[3*1+2]*in[3*2+0] - in[3*2+2]*in[3*1+0])/det;
  out[3*1+1] = (in[3*0+0]*in[3*2+2] - in[3*2+0]*in[3*0+2])/det;
  out[3*1+2] = (in[3*0+2]*in[3*1+0] - in[3*1+2]*in[3*0+0])/det;

  out[3*2+0] = (in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1])/det;
  out[3*2+1] = (in[3*0+1]*in[3*2+0] - in[3*2+1]*in[3*0+0])/det;
  out[3*2+2] = (in[3*0+0]*in[3*1+1] - in[3*1+0]*in[3*0+1])/det;
}


double simplex3_mls_quadratic_t::interpolate_from_parent_with_derivatives(double* xyz, double normal[3], double &F, double &Fn, double &Fnn)
{
  tet3_t *tet = &tets[0];

  // get coordinates of intersection points
  int e01 = tris[tet->tri3].edg2;
  int e02 = tris[tet->tri3].edg1;
  int e03 = tris[tet->tri1].edg1;
  int e12 = tris[tet->tri0].edg2;
  int e23 = tris[tet->tri0].edg0;
  int e13 = tris[tet->tri0].edg1;

  std::vector<int> nv(nodes_per_tet, -1);

  nv[0] = tet->vtx0;
  nv[1] = tet->vtx1;
  nv[2] = tet->vtx2;
  nv[3] = tet->vtx3;
  nv[4] = edgs[e01].vtx1;
  nv[5] = edgs[e12].vtx1;
  nv[6] = edgs[e02].vtx1;
  nv[7] = edgs[e03].vtx1;
  nv[8] = edgs[e13].vtx1;
  nv[9] = edgs[e23].vtx1;

  // map real point to reference element
  vtx3_t *v0 = &vtxs[nv[0]];
  vtx3_t *v1 = &vtxs[nv[1]];
  vtx3_t *v2 = &vtxs[nv[2]];
  vtx3_t *v3 = &vtxs[nv[3]];

  double A[9], A_inv[9], D[3];
  A[3*0+0] = v1->x - v0->x; A[3*0+1] = v2->x - v0->x; A[3*0+2] = v3->x - v0->x; D[0] = xyz[0] - v0->x;
  A[3*1+0] = v1->y - v0->y; A[3*1+1] = v2->y - v0->y; A[3*1+2] = v3->y - v0->y; D[1] = xyz[1] - v0->y;
  A[3*2+0] = v1->z - v0->z; A[3*2+1] = v2->z - v0->z; A[3*2+2] = v3->z - v0->z; D[2] = xyz[2] - v0->z;

  inv_mat3(A, A_inv);

  double a = A_inv[3*0+0]*D[0] + A_inv[3*0+1]*D[1] + A_inv[3*0+2]*D[2];
  double b = A_inv[3*1+0]*D[0] + A_inv[3*1+1]*D[1] + A_inv[3*1+2]*D[2];
  double c = A_inv[3*2+0]*D[0] + A_inv[3*2+1]*D[1] + A_inv[3*2+2]*D[2];

  // compute nodal functions
  double d = 1.-a-b-c;
  double N[nodes_per_tet]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  double Na[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
  double Nb[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
  double Nc[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

  double Naa[nodes_per_tet] = { 4, 4, 0, 0,-8, 0, 0, 0, 0, 0 };
  double Nbb[nodes_per_tet] = { 4, 0, 4, 0, 0, 0,-8, 0, 0, 0 };
  double Ncc[nodes_per_tet] = { 4, 0, 0, 4, 0, 0, 0,-8, 0, 0 };
  double Nab[nodes_per_tet] = { 4, 0, 0, 0,-4, 4,-4, 0, 0, 0 };
  double Nbc[nodes_per_tet] = { 4, 0, 0, 0, 0, 0,-4,-4, 0, 4 };
  double Nca[nodes_per_tet] = { 4, 0, 0, 0,-4, 0, 0,-4, 4, 0 };

  double Fa = 0, Fb = 0, Fc = 0, Faa = 0, Fbb = 0, Fcc = 0, Fab = 0, Fbc = 0, Fca = 0;
  F = 0;
  double f;
  for (short i = 0; i < nodes_per_tet; ++i)
  {
    f = vtxs[nv[i]].value;

    F   += f*N[i];

    Fa  += f*Na[i];
    Fb  += f*Nb[i];
    Fc  += f*Nc[i];

    Faa += f*Naa[i];
    Fbb += f*Nbb[i];
    Fcc += f*Ncc[i];

    Fab += f*Nab[i];
    Fbc += f*Nbc[i];
    Fca += f*Nca[i];
  }

  double Fx = Fa*A_inv[3*0+0] + Fb*A_inv[3*1+0] + Fc*A_inv[3*2+0];
  double Fy = Fa*A_inv[3*0+1] + Fb*A_inv[3*1+1] + Fc*A_inv[3*2+1];
  double Fz = Fa*A_inv[3*0+2] + Fb*A_inv[3*1+2] + Fc*A_inv[3*2+2];

  int alph, beta;

  alph = 0; beta = 0;
  double Fxx = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
             + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
             + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
             + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
             + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
             + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 1; beta = 1;
  double Fyy = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 2; beta = 2;
  double Fzz = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 0; beta = 1;
  double Fxy = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 1; beta = 2;
  double Fyz = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 2; beta = 0;
  double Fzx = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  double nx = normal[0];
  double ny = normal[1];
  double nz = normal[2];

  Fn  = Fx*nx + Fy*ny + Fz*nz;
  Fnn = Fxx*nx*nx + Fyy*ny*ny + Fzz*nz*nz + 2.*Fxy*nx*ny + 2.*Fyz*ny*nz + 2.*Fzx*nz*nx;
}


double simplex3_mls_quadratic_t::interpolate_from_parent_with_derivatives(double* xyz, double &F, double &Fn, double &Fnn, double *normal)
{
  tet3_t *tet = &tets[0];

  // get coordinates of intersection points
  int e01 = tris[tet->tri3].edg2;
  int e02 = tris[tet->tri3].edg1;
  int e03 = tris[tet->tri1].edg1;
  int e12 = tris[tet->tri0].edg2;
  int e23 = tris[tet->tri0].edg0;
  int e13 = tris[tet->tri0].edg1;

  std::vector<int> nv(nodes_per_tet, -1);

  nv[0] = tet->vtx0;
  nv[1] = tet->vtx1;
  nv[2] = tet->vtx2;
  nv[3] = tet->vtx3;
  nv[4] = edgs[e01].vtx1;
  nv[5] = edgs[e12].vtx1;
  nv[6] = edgs[e02].vtx1;
  nv[7] = edgs[e03].vtx1;
  nv[8] = edgs[e13].vtx1;
  nv[9] = edgs[e23].vtx1;

  // map real point to reference element
  vtx3_t *v0 = &vtxs[nv[0]];
  vtx3_t *v1 = &vtxs[nv[1]];
  vtx3_t *v2 = &vtxs[nv[2]];
  vtx3_t *v3 = &vtxs[nv[3]];

  double A[9], A_inv[9], D[3];
  A[3*0+0] = v1->x - v0->x; A[3*0+1] = v2->x - v0->x; A[3*0+2] = v3->x - v0->x; D[0] = xyz[0] - v0->x;
  A[3*1+0] = v1->y - v0->y; A[3*1+1] = v2->y - v0->y; A[3*1+2] = v3->y - v0->y; D[1] = xyz[1] - v0->y;
  A[3*2+0] = v1->z - v0->z; A[3*2+1] = v2->z - v0->z; A[3*2+2] = v3->z - v0->z; D[2] = xyz[2] - v0->z;

  inv_mat3(A, A_inv);

  double a = A_inv[3*0+0]*D[0] + A_inv[3*0+1]*D[1] + A_inv[3*0+2]*D[2];
  double b = A_inv[3*1+0]*D[0] + A_inv[3*1+1]*D[1] + A_inv[3*1+2]*D[2];
  double c = A_inv[3*2+0]*D[0] + A_inv[3*2+1]*D[1] + A_inv[3*2+2]*D[2];

  // compute nodal functions
  double d = 1.-a-b-c;
  double N[nodes_per_tet]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  double Na[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
  double Nb[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
  double Nc[nodes_per_tet] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

  double Naa[nodes_per_tet] = { 4, 4, 0, 0,-8, 0, 0, 0, 0, 0 };
  double Nbb[nodes_per_tet] = { 4, 0, 4, 0, 0, 0,-8, 0, 0, 0 };
  double Ncc[nodes_per_tet] = { 4, 0, 0, 4, 0, 0, 0,-8, 0, 0 };
  double Nab[nodes_per_tet] = { 4, 0, 0, 0,-4, 4,-4, 0, 0, 0 };
  double Nbc[nodes_per_tet] = { 4, 0, 0, 0, 0, 0,-4,-4, 0, 4 };
  double Nca[nodes_per_tet] = { 4, 0, 0, 0,-4, 0, 0,-4, 4, 0 };

  double Fa = 0, Fb = 0, Fc = 0, Faa = 0, Fbb = 0, Fcc = 0, Fab = 0, Fbc = 0, Fca = 0;
  F = 0;
  double f;
  for (short i = 0; i < nodes_per_tet; ++i)
  {
    f = vtxs[nv[i]].value;

    F   += f*N[i];

    Fa  += f*Na[i];
    Fb  += f*Nb[i];
    Fc  += f*Nc[i];

    Faa += f*Naa[i];
    Fbb += f*Nbb[i];
    Fcc += f*Ncc[i];

    Fab += f*Nab[i];
    Fbc += f*Nbc[i];
    Fca += f*Nca[i];
  }

  double Fx = Fa*A_inv[3*0+0] + Fb*A_inv[3*1+0] + Fc*A_inv[3*2+0];
  double Fy = Fa*A_inv[3*0+1] + Fb*A_inv[3*1+1] + Fc*A_inv[3*2+1];
  double Fz = Fa*A_inv[3*0+2] + Fb*A_inv[3*1+2] + Fc*A_inv[3*2+2];

  double norm = sqrt(Fx*Fx + Fy*Fy + Fz*Fz);

  normal[0] = Fx/norm;
  normal[1] = Fy/norm;
  normal[2] = Fz/norm;

  int alph, beta;

  alph = 0; beta = 0;
  double Fxx = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      +        Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      +        Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      +        Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      +        Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      +        Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 1; beta = 1;
  double Fyy = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 2; beta = 2;
  double Fzz = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 0; beta = 1;
  double Fxy = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 1; beta = 2;
  double Fyz = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  alph = 2; beta = 0;
  double Fzx = Faa*A_inv[3*0+alph]*A_inv[3*0+beta]
      + Fbb*A_inv[3*1+alph]*A_inv[3*1+beta]
      + Fcc*A_inv[3*2+alph]*A_inv[3*2+beta]
      + Fab*(A_inv[3*0+alph]*A_inv[3*1+beta] + A_inv[3*1+alph]*A_inv[3*0+beta])
      + Fbc*(A_inv[3*1+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*1+beta])
      + Fca*(A_inv[3*0+alph]*A_inv[3*2+beta] + A_inv[3*2+alph]*A_inv[3*0+beta]);

  double nx = normal[0];
  double ny = normal[1];
  double nz = normal[2];

  Fn  = Fx*nx + Fy*ny + Fz*nz;
  Fnn = Fxx*nx*nx + Fyy*ny*ny + Fzz*nz*nz + 2.*Fxy*nx*ny + 2.*Fyz*ny*nz + 2.*Fzx*nz*nx;
}


void simplex3_mls_quadratic_t::deform_edge_in_normal_dir(int n_edg)
{
  edg3_t *edg = &edgs[n_edg];

  double xyz_start[3] = { .5*(vtxs[edg->vtx0].x + vtxs[edg->vtx2].x),
                          .5*(vtxs[edg->vtx0].y + vtxs[edg->vtx2].y),
                          .5*(vtxs[edg->vtx0].z + vtxs[edg->vtx2].z) };

  double phi_value;
  double phi_n_value;
  double phi_nn_value;
  double normal[3];

  interpolate_from_parent_with_derivatives(xyz_start, phi_value, phi_n_value, phi_nn_value, normal);

  double d = find_root(phi_value, phi_n_value, phi_nn_value);

  vtxs[edg->vtx1].x = xyz_start[0] + d*normal[0];
  vtxs[edg->vtx1].y = xyz_start[1] + d*normal[1];
  vtxs[edg->vtx1].z = xyz_start[2] + d*normal[2];
}

void simplex3_mls_quadratic_t::invert_mapping_tri(int tri_idx, double xyz[3], double ab[2])
{
  tri3_t *tri = &tris[tri_idx];

  double a = ab[0];
  double b = ab[1];

  int nv0 = tri->vtx0;
  int nv1 = tri->vtx1;
  int nv2 = tri->vtx2;

  double X[nodes_per_tri] = { vtxs[nv0].x, vtxs[nv1].x, vtxs[nv2].x, tri->g_vtx01[0], tri->g_vtx12[0], tri->g_vtx02[0] };
  double Y[nodes_per_tri] = { vtxs[nv0].y, vtxs[nv1].y, vtxs[nv2].y, tri->g_vtx01[1], tri->g_vtx12[1], tri->g_vtx02[1] };
  double Z[nodes_per_tri] = { vtxs[nv0].z, vtxs[nv1].z, vtxs[nv2].z, tri->g_vtx01[2], tri->g_vtx12[2], tri->g_vtx02[2] };

  double tolerance = 1.e-15;
  double error = 1.;
  int max_iterations = 1000;
  int iteration = 0;
  double error_nm1 = 2.;

  double xyz_g[3];
  double Xa, Ya, Za;
  double Xb, Yb, Zb;

  double Xaa, Yaa, Zaa;
  double Xab, Yab, Zab;
  double Xbb, Ybb, Zbb;

  while (error > tolerance && iteration < max_iterations)
  {
    double N[nodes_per_tri]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};
    double Na[nodes_per_tri] = {-3.+4.*a+4.*b,            4.*a-1.,      0,            4.-8.*a-4.*b,   4.*b,  -4.*b};
    double Nb[nodes_per_tri] = {-3.+4.*a+4.*b,            0,            4.*b-1.,     -4.*a,           4.*a,   4.-4.*a-8.*b};
    double Naa[nodes_per_tri] = {4, 4, 0,-8, 0, 0};
    double Nab[nodes_per_tri] = {4, 0, 0,-4, 4,-4};
    double Nbb[nodes_per_tri] = {4, 0, 4, 0, 0,-8};

    xyz_g[0] = 0;
    xyz_g[1] = 0;
    xyz_g[2] = 0;

    Xa = 0; Ya = 0; Za = 0;
    Xb = 0; Yb = 0; Zb = 0;

    Xaa = 0; Yaa = 0; Zaa = 0;
    Xab = 0; Yab = 0; Zab = 0;
    Xbb = 0; Ybb = 0; Zbb = 0;

    for (int i = 0; i < nodes_per_tri; ++i)
    {
      xyz_g[0] += X[i]*N[i];
      xyz_g[1] += Y[i]*N[i];
      xyz_g[2] += Z[i]*N[i];

      Xa += X[i]*Na[i];   Ya += Y[i]*Na[i];   Za += Z[i]*Na[i];
      Xb += X[i]*Nb[i];   Yb += Y[i]*Nb[i];   Zb += Z[i]*Nb[i];

      Xaa += X[i]*Naa[i];   Yaa += Y[i]*Naa[i];   Zaa += Z[i]*Naa[i];
      Xab += X[i]*Nab[i];   Yab += Y[i]*Nab[i];   Zab += Z[i]*Nab[i];
      Xbb += X[i]*Nbb[i];   Ybb += Y[i]*Nbb[i];   Zbb += Z[i]*Nbb[i];
    }
    error_nm1 = error;
    error = sqrt(pow(xyz_g[0]-xyz[0], 2.) + pow(xyz_g[1]-xyz[1], 2.) + pow(xyz_g[2]-xyz[2],2.));

//    if (error_nm1 < error)
//      std::cout << error_nm1 << " " << error << std::endl;

    if (error > tolerance)
    {
      double Fa = 2.*( (xyz_g[0]-xyz[0])*Xa + (xyz_g[1]-xyz[1])*Ya + (xyz_g[2]-xyz[2])*Za );
      double Fb = 2.*( (xyz_g[0]-xyz[0])*Xb + (xyz_g[1]-xyz[1])*Yb + (xyz_g[2]-xyz[2])*Zb );

      double Faa = 2.*( Xa*Xa + Ya*Ya + Za*Za + (xyz_g[0]-xyz[0])*Xaa + (xyz_g[1]-xyz[1])*Yaa + (xyz_g[2]-xyz[2])*Zaa );
      double Fab = 2.*( Xa*Xb + Ya*Yb + Za*Zb + (xyz_g[0]-xyz[0])*Xab + (xyz_g[1]-xyz[1])*Yab + (xyz_g[2]-xyz[2])*Zab );
      double Fbb = 2.*( Xb*Xb + Yb*Yb + Zb*Zb + (xyz_g[0]-xyz[0])*Xbb + (xyz_g[1]-xyz[1])*Ybb + (xyz_g[2]-xyz[2])*Zbb );

      double na = Fa;
      double nb = Fb;

      double norm = sqrt(na*na + nb*nb);

      na /= norm;
      nb /= norm;

      double Fn  = Fa*na + Fb*nb;
      double Fnn = Faa*na*na + 2.*Fab*na*nb + Fbb*nb*nb;

      double alpha = - Fn/Fnn;

      a += alpha*na;
      b += alpha*nb;

//      std::cout << "Iteration " << iteration << ": " << alpha*na << " " << alpha*nb << " " << error << std::endl;
    }

    iteration++;

  }


  ab[0] = a;
  ab[1] = b;
}
