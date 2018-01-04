#include "simplex2_mls_quadratic.h"
#include "casl_math.h"


//--------------------------------------------------
// Constructors
//--------------------------------------------------
//simplex2_mls_quadratic_t::simplex2_mls_quadratic_t()
//{
//  vtxs.reserve(8);
//  edgs.reserve(27);
//  tris.reserve(20);

//  d_max_ = 1.;

//  eps = 1.e-10;
//  eps_abc_ = eps;
//  eps_xyz_ = eps*d_max_;
//}

simplex2_mls_quadratic_t::simplex2_mls_quadratic_t(double x0, double y0,
                                                   double x1, double y1,
                                                   double x2, double y2,
                                                   double x3, double y3,
                                                   double x4, double y4,
                                                   double x5, double y5)
{
  x0_ = x0; y0_ = y0;
  x1_ = x1; y1_ = y1;
  x2_ = x2; y2_ = y2;
  x3_ = x3; y3_ = y3;
  x4_ = x4; y4_ = y4;
  x5_ = x5; y5_ = y5;

  d_max_ = MAX( sqrt(SQR(x0_-x1_) + SQR(y0_-y1_)),
                sqrt(SQR(x1_-x2_) + SQR(y1_-y2_)),
                sqrt(SQR(x2_-x0_) + SQR(y2_-y0_)) );

  eps_ = 1.e-10;
  eps_abc_ = eps_;
  eps_xyz_ = eps_*d_max_;

  // usually there will be only one cut
  vtxs.reserve(15);
  edgs.reserve(10);
  tris.reserve(4);

  /* fill the vectors with the initial structure */
  /* 2
   * |\
   * 5 4
   * |  \
   * 0-3-1
   */
  vtxs.push_back(vtx2_t(x0_,y0_));
  vtxs.push_back(vtx2_t(x1_,y1_));
  vtxs.push_back(vtx2_t(x2_,y2_));
  vtxs.push_back(vtx2_t(x3_,y3_));
  vtxs.push_back(vtx2_t(x4_,y4_));
  vtxs.push_back(vtx2_t(x5_,y5_));

  edgs.push_back(edg2_t(1,4,2));
  edgs.push_back(edg2_t(0,5,2));
  edgs.push_back(edg2_t(0,3,1));

  tris.push_back(tri2_t(0,1,2,0,1,2));
}




//--------------------------------------------------
// Constructing domain
//--------------------------------------------------
void simplex2_mls_quadratic_t::construct_domain(std::vector<CF_2 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{

  double phi0, phi1, phi2;
  double c0, c1, c2;
  double a_ext, phi_ext;

  int initial_refinement = 0;
  int n;

  std::vector<vtx2_t> vtxs_initial = vtxs;
  std::vector<edg2_t> edgs_initial = edgs;
  std::vector<tri2_t> tris_initial = tris;

  while(1)
  {
    for (int i = 0; i < initial_refinement; ++i)
    {
      n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
      n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
    }

    // loop over LSFs
    int refine_level = 0;
    for (short phi_idx = 0; phi_idx < phi.size(); ++phi_idx)
    {
      int last_vtxs_size = 0;

      invalid_reconstruction_ = true;

      while (invalid_reconstruction_)
      {
        bool needs_refinement = true;

        while (needs_refinement)
        {
          // interpolate to all vertices
          for (int i = last_vtxs_size; i < vtxs.size(); ++i)
            if (!vtxs[i].is_recycled)
            {
              vtxs[i].value = (*phi[phi_idx]) ( vtxs[i].x, vtxs[i].y );
              perturb(vtxs[i].value, eps_xyz_);
            }

          // check validity of data on each edge
          needs_refinement = false;
          for (int i = 0; i < edgs.size(); ++i)
            if (!edgs[i].is_split)
            {
              edg2_t *e = &edgs[i];

              phi0 = vtxs[e->vtx0].value;
              phi1 = vtxs[e->vtx1].value;
              phi2 = vtxs[e->vtx2].value;

              if (phi0*phi2 > 0 && phi0*phi1 < 0)
              {
                needs_refinement = true;
                break;
              }

//              if (phi0 > 0 && phi2 > 0 ||
//                  phi0 < 0 && phi2 < 0)
//              {
//                c0 = phi0;
//                c1 = -3.*phi0+4.*phi1-phi2;
//                c2 = 2.*phi0 - 4.*phi1 + 2.*phi2;

//                if (c2 > EPS)
//                {
//                  d = c1*c1 - 4.*c0*c2;
//                  if (d > 0)
//                  {
//                    a1 = .5*(-c1-sqrt(d))/c2;
//                    a2 = .5*(-c1+sqrt(d))/c2;
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
                c0 = phi0;
                c1 = -3.*phi0 + 4.*phi1 -    phi2;
                c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

                if (fabs(c2) > EPS)
                {
                  a_ext = -.5*c1/c2;
                  if (a_ext > 0. && a_ext < 1.)
                  {
                    phi_ext = c0 + c1*a_ext + c2*a_ext*a_ext;

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

          last_vtxs_size = vtxs.size();

          // refine if necessary
          if (needs_refinement && refine_level < max_refinement_ - initial_refinement)
          {
            int n;
            n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
            n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
            refine_level++;
          } else if (needs_refinement) {
            std::cout << "Cannot resolve invalid geometry (bad)\n";
            needs_refinement = false;
          }

        }

        invalid_reconstruction_ = false;

        std::vector<vtx2_t> vtx_tmp = vtxs;
        std::vector<edg2_t> edg_tmp = edgs;
        std::vector<tri2_t> tri_tmp = tris;

        int n;
        n = vtxs.size(); for (int i = 0; i < n; i++) do_action_vtx(i, clr[phi_idx], acn[phi_idx]);
        n = edgs.size(); for (int i = 0; i < n; i++) do_action_edg(i, clr[phi_idx], acn[phi_idx]);
        n = tris.size(); for (int i = 0; i < n; i++) do_action_tri(i, clr[phi_idx], acn[phi_idx]);

        if (invalid_reconstruction_ && refine_level < max_refinement_ - initial_refinement)
        {
          vtxs = vtx_tmp;
          edgs = edg_tmp;
          tris = tri_tmp;

          n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
          n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
          refine_level++;
        } else {
//          if (invalid_reconstruction_) std::cout << "Cannot resolve invalid geometry\n";
          invalid_reconstruction_ = false;
        }
      }

      eps_xyz_ *= 0.5;
      eps_abc_ *= 0.5;
    }

    // sort everything before integration
    for (int i = 0; i < edgs.size(); i++)
    {
      edg2_t *edg = &edgs[i];
      if (need_swap(edg->vtx0, edg->vtx2)) { swap(edg->vtx0, edg->vtx2); }
    }

    for (int i = 0; i < tris.size(); i++)
    {
      tri2_t *tri = &tris[i];
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); }
      if (need_swap(tri->vtx1, tri->vtx2)) { swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2); }
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); }
    }

    // check for overlapping volumes
    double s_before = area(tris[0].vtx0, tris[0].vtx1, tris[0].vtx2);
    double s_after  = 0;

    // compute volume after using linear representation
    for (int i = 0; i < tris.size(); ++i)
      if (!tris[i].is_split)
        s_after += area(tris[i].vtx0, tris[i].vtx1, tris[i].vtx2);

    if (fabs(s_before-s_after) > EPS)
    {
      if (initial_refinement == max_refinement_)
        std::cout << "Can't resolve overlapping " << fabs(s_before-s_after) << "\n";
      else {
        ++initial_refinement;
//        std::cout << "Overlapping " << fabs(s_before-s_after) << "\n";
        vtxs = vtxs_initial;
        edgs = edgs_initial;
        tris = tris_initial;
      }
    } else {
      break;
    }
  }
}


void simplex2_mls_quadratic_t::construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{

  double phi0, phi1, phi2;
  double c0, c1, c2;
  double a_ext, phi_ext;

  int initial_refinement = 0;
  int n;

  std::vector<double> phi_current(nodes_per_tri_, -1);

  std::vector<vtx2_t> vtxs_initial = vtxs;
  std::vector<edg2_t> edgs_initial = edgs;
  std::vector<tri2_t> tris_initial = tris;

  while(1)
  {
    for (int i = 0; i < initial_refinement; ++i)
    {
      n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
      n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
    }

    // loop over LSFs
    int refine_level = 0;
    for (short phi_idx = 0; phi_idx < acn.size(); ++phi_idx)
    {

      for (int i = 0; i < nodes_per_tri_; ++i)
      {
        vtxs[i].value  = phi[nodes_per_tri_*phi_idx + i];
        phi_current[i] = phi[nodes_per_tri_*phi_idx + i];
      }

      int last_vtxs_size = nodes_per_tri_;

      invalid_reconstruction_ = true;

      while (invalid_reconstruction_)
      {
        bool needs_refinement = true;

        while (needs_refinement)
        {
          // interpolate to all vertices
          for (int i = last_vtxs_size; i < vtxs.size(); ++i)
            if (!vtxs[i].is_recycled)
            {
              vtxs[i].value = interpolate_from_parent (phi_current, vtxs[i].x, vtxs[i].y );
              perturb(vtxs[i].value, eps_xyz_);
            }

          // check validity of data on each edge
          needs_refinement = false;
          for (int i = 0; i < edgs.size(); ++i)
            if (!edgs[i].is_split)
            {
              edg2_t *e = &edgs[i];

              phi0 = vtxs[e->vtx0].value;
              phi1 = vtxs[e->vtx1].value;
              phi2 = vtxs[e->vtx2].value;

              if (phi0*phi2 > 0 && phi0*phi1 < 0)
              {
                needs_refinement = true;
                break;
              }

//              if (phi0 > 0 && phi2 > 0 ||
//                  phi0 < 0 && phi2 < 0)
//              {
//                c0 = phi0;
//                c1 = -3.*phi0 + 4.*phi1 -    phi2;
//                c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

//                if (c2 > EPS)
//                {
//                  d = c1*c1 - 4.*c0*c2;
//                  if (d > 0)
//                  {
//                    a1 = .5*(-c1-sqrt(d))/c2;
//                    a2 = .5*(-c1+sqrt(d))/c2;
//                    if ( (a1 > 0. && a1 < 1.) ||
//                         (a2 > 0. && a2 < 1.) )
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
                c0 = phi0;
                c1 = -3.*phi0 + 4.*phi1 -    phi2;
                c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

                if (fabs(c2) > EPS)
                {
                  a_ext = -.5*c1/c2;
                  if (a_ext > 0. && a_ext < 1.)
                  {
                    phi_ext = c0 + c1*a_ext + c2*a_ext*a_ext;

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

          last_vtxs_size = vtxs.size();

          // refine if necessary
          if (needs_refinement && refine_level < max_refinement_ - initial_refinement)
          {
            int n;
            n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
            n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
            refine_level++;
          } else if (needs_refinement) {
            std::cout << "Cannot resolve invalid geometry (bad)\n";
            needs_refinement = false;
          }

        }

        invalid_reconstruction_ = false;

        std::vector<vtx2_t> vtx_tmp = vtxs;
        std::vector<edg2_t> edg_tmp = edgs;
        std::vector<tri2_t> tri_tmp = tris;

        int n;
        n = vtxs.size(); for (int i = 0; i < n; i++) do_action_vtx(i, clr[phi_idx], acn[phi_idx]);
        n = edgs.size(); for (int i = 0; i < n; i++) do_action_edg(i, clr[phi_idx], acn[phi_idx]);
        n = tris.size(); for (int i = 0; i < n; i++) do_action_tri(i, clr[phi_idx], acn[phi_idx]);

        if (invalid_reconstruction_ && refine_level < max_refinement_ - initial_refinement)
        {
          vtxs = vtx_tmp;
          edgs = edg_tmp;
          tris = tri_tmp;

          n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
          n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);
          refine_level++;
        } else {
//          if (invalid_reconstruction_) std::cout << "Cannot resolve invalid geometry\n";
          invalid_reconstruction_ = false;
        }
      }

      eps_abc_ *= 0.5;
      eps_xyz_ *= 0.5;
    }

    // sort everything before integration
    for (int i = 0; i < edgs.size(); i++)
    {
      edg2_t *edg = &edgs[i];
      if (need_swap(edg->vtx0, edg->vtx2)) { swap(edg->vtx0, edg->vtx2); }
    }

    for (int i = 0; i < tris.size(); i++)
    {
      tri2_t *tri = &tris[i];
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); }
      if (need_swap(tri->vtx1, tri->vtx2)) { swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2); }
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); }
    }

    // check for overlapping volumes
    double s_before = area(tris[0].vtx0, tris[0].vtx1, tris[0].vtx2);
    double s_after  = 0;

    // compute volume after using linear representation
    for (int i = 0; i < tris.size(); ++i)
      if (!tris[i].is_split)
        s_after += area(tris[i].vtx0, tris[i].vtx1, tris[i].vtx2);

    if (fabs(s_before-s_after) > EPS)
    {
      if (initial_refinement == max_refinement_)
      {
        std::cout << "Can't resolve overlapping " << fabs(s_before-s_after) << "\n";
      } else {
        ++initial_refinement;
//        std::cout << "Overlapping " << fabs(s_before-s_after) << "\n";
        vtxs = vtxs_initial;
        edgs = edgs_initial;
        tris = tris_initial;
      }
    } else {
      break;
    }
  }
}







//--------------------------------------------------
// Splitting
//--------------------------------------------------
void simplex2_mls_quadratic_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx2_t *vtx = &vtxs[n_vtx];

  if (vtx->is_recycled) return;

  switch (action){
  case INTERSECTION:  if (vtx->value > 0)                                      vtx->set(OUT, -1, -1);  break;
  case ADDITION:      if (vtx->value < 0)                                      vtx->set(INS, -1, -1);  break;
  case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==PNT)  vtx->set(FCE, cn, -1);  break;
  }
}

void simplex2_mls_quadratic_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg2_t *edg = &edgs[n_edg];

  int c0 = edg->c0;

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  int num_negatives = 0;
  if (vtxs[edg->vtx0].value < 0) num_negatives++;
  if (vtxs[edg->vtx2].value < 0) num_negatives++;

#ifdef CASL_THROWS
  edg->type = num_negatives;
#endif

  // auxiliary variables
  vtx2_t *c_vtx_x, *c_vtx_0x, *c_vtx_x2;
  edg2_t *c_edg0, *c_edg1;
  double r;

  switch (num_negatives){
  case 0: // ++
    /* split an edge */
    // no need to split

    /* apply rules */
    switch (action){
    case INTERSECTION:  edg->set(OUT,-1); break;
    case ADDITION:      /* do nothig */   break;
    case COLORATION:    /* do nothig */   break;
    }
    break;
  case 1: // -+
      {
    /* split an edge */
    edg->is_split = true;

    // new vertex

    // find intersection
    double a = find_intersection_quadratic(n_edg);
    edg->a = a;

    // map intersection point and new middle points to real space
    double x,y,xm,ym,xp,yp;
    mapping_edg(x,  y,  n_edg, a);
    mapping_edg(xm, ym, n_edg, .5*a);
    mapping_edg(xp, yp, n_edg, a + .5*(1.-a));

//    double x_0 = vtxs[edg->vtx0].x, y_0 = vtxs[edg->vtx0].y;
//    double x_1 = vtxs[edg->vtx1].x, y_1 = vtxs[edg->vtx1].y;
//    double x_2 = vtxs[edg->vtx2].x, y_2 = vtxs[edg->vtx2].y;

    // create new vertices
    vtxs.push_back(vtx2_t(xm,ym)); int n_vtx_0x = vtxs.size()-1;
    vtxs.push_back(vtx2_t(xp,yp)); int n_vtx_x2 = vtxs.size()-1;
    vtxs.push_back(vtx2_t(x,y));

    edg->c_vtx_x = vtxs.size()-1;

//    vtxs.back().n_vtx0 = edg->vtx0;
//    vtxs.back().n_vtx1 = edg->vtx1;
//    vtxs.back().ratio  = r;

    // new edges
    edgs.push_back(edg2_t(edg->vtx0,    n_vtx_0x, edg->c_vtx_x)); edg = &edgs[n_edg]; // edges might have changed their addresses
    edgs.push_back(edg2_t(edg->c_vtx_x, n_vtx_x2, edg->vtx2   )); edg = &edgs[n_edg];

    edg->c_edg0 = edgs.size()-2;
    edg->c_edg1 = edgs.size()-1;

    /* apply rules */
    vtxs[edg->vtx1].is_recycled = true;

    c_vtx_x  = &vtxs[edg->c_vtx_x];
    c_vtx_0x = &vtxs[n_vtx_0x];
    c_vtx_x2 = &vtxs[n_vtx_x2];
    c_edg0  = &edgs[edg->c_edg0];
    c_edg1  = &edgs[edg->c_edg1];

    c_edg0->dir = edg->dir;
    c_edg1->dir = edg->dir;

    c_edg0->p_lsf = edg->p_lsf;
    c_edg1->p_lsf = edg->p_lsf;

#ifdef CASL_THROWS
    c_vtx_x->p_edg = n_edg;
    c_edg0->p_edg  = n_edg;
    c_edg1->p_edg  = n_edg;
#endif
    switch (action){
    case INTERSECTION:
      switch (edg->loc){
      case OUT: c_vtx_x->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_vtx_0x->set(OUT, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
      case INS: c_vtx_x->set(FCE, cn, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
      case FCE: c_vtx_x->set(PNT, c0, cn); c_edg0->set(FCE, c0); c_vtx_0x->set(FCE, c0, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); if (c0==cn)  c_vtx_x->set(FCE, c0, -1);  break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    case ADDITION:
      switch (edg->loc){
      case OUT: c_vtx_x->set(FCE, cn, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
      case INS: c_vtx_x->set(INS, -1, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(INS, -1); c_vtx_x2->set(INS, -1, -1); break;
      case FCE: c_vtx_x->set(PNT, c0, cn); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(FCE, c0); c_vtx_x2->set(FCE, c0, -1); if (c0==cn)  c_vtx_x->set(FCE, c0, -1);  break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    case COLORATION:
      switch (edg->loc){
      case OUT: c_vtx_x->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_vtx_0x->set(OUT, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
      case INS: c_vtx_x->set(INS, -1, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(INS, -1); c_vtx_x2->set(INS, -1, -1); break;
      case FCE: c_vtx_x->set(PNT, c0, cn); c_edg0->set(FCE, cn); c_vtx_0x->set(FCE, cn, -1); c_edg1->set(FCE, c0); c_vtx_x2->set(FCE, c0, -1); if (c0==cn)  c_vtx_x->set(FCE, c0, -1);  break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    }
      }
    break;
  case 2: // --
    /* split an edge */
    // no need to split

    /* apply rules */
    switch (action) {
    case INTERSECTION:  /* do nothing */                        break;
    case ADDITION:                          edg->set(INS, -1);  break;
    case COLORATION:    if (edg->loc==FCE)  edg->set(FCE, cn);  break;
    }
    break;
  }
}

void simplex2_mls_quadratic_t::do_action_tri(int n_tri, int cn, action_t action)
{
  tri2_t *tri = &tris[n_tri];

  if (tri->is_split) return;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Determine type */
  int num_negatives = 0;
  if (vtxs[tri->vtx0].value < 0) num_negatives++;
  if (vtxs[tri->vtx1].value < 0) num_negatives++;
  if (vtxs[tri->vtx2].value < 0) num_negatives++;

#ifdef CASL_THROWS
  tri->type = num_negatives;

  /* check whether vertices of a triangle and edges coincide */
  if (tri->vtx0 != edgs[tri->edg1].vtx0 || tri->vtx0 != edgs[tri->edg2].vtx0 ||
      tri->vtx1 != edgs[tri->edg0].vtx0 || tri->vtx1 != edgs[tri->edg2].vtx2 ||
      tri->vtx2 != edgs[tri->edg0].vtx2 || tri->vtx2 != edgs[tri->edg1].vtx2)
  {
    std::cout << vtxs[tri->vtx0].value << " : " << vtxs[tri->vtx1].value << " : " << vtxs[tri->vtx2].value << std::endl;
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
  edg2_t *c_edg0, *c_edg1;
  edg2_t *edg0, *edg1, *edg2;
  tri2_t *c_tri0, *c_tri1, *c_tri2;
  vtx2_t *vtx_u0, *vtx_u1;

  switch (num_negatives)
  {
    case 0: // (+++)
      /* split a triangle */
      // no need to split

      /* apply rules */
      switch (action){
        case INTERSECTION:  tri->set(OUT);    break;
        case ADDITION:      /* do nothing */  break;
        case COLORATION:    /* do nothing */  break;
      }
      break;

    case 1: // (-++)
      {
        /* split a triangle */
        tri->is_split = true;

        /* new vertices */
        tri->c_vtx01 = edgs[tri->edg2].c_vtx_x;
        tri->c_vtx02 = edgs[tri->edg1].c_vtx_x;

        // coordinates of new vertices in reference element
        double abc_v01[] = { edgs[tri->edg2].a, 0. };
        double abc_v02[] = { 0., edgs[tri->edg1].a };

        /* vertex along interface */
        double abc_u0_lin[2] = { .5*(abc_v01[0] + abc_v02[0]), .5*(abc_v01[1] + abc_v02[1]) };
        double abc_u0[2];
        find_middle_node(abc_u0, abc_v02, abc_v01, n_tri);

        /* check for an intersection with an auxiliary straight edge */
        if (check_for_edge_intersections_)
        {
          // two points on the edge
          double *edge_vtx0   = abc_v01;
          double  edge_vtx1[] = { 0., 1. };

          // tangent and normal vectors
          double t[] = { edge_vtx1[0] - edge_vtx0[0], edge_vtx1[1] - edge_vtx0[1] };
          double n[] = { -t[1], t[0] };
          double norm = sqrt(SQR(n[0]) + SQR(n[1]));
          n[0] /= norm; n[1] /= norm;

          // level-set function for the edge: phi = na*(a-a0) + nb*(b-b0)
          double dist_to_edge = n[0]*(abc_u0[0]-edge_vtx0[0]) + n[1]*(abc_u0[1]-edge_vtx0[1]);

          if (dist_to_edge < 0)
          {
            // set the whole reconstruction for refinement
            invalid_reconstruction_ = true;

            // bring the midpoint below the edge in case the max level of refinement is reached
            double dist_to_edge_lin = n[0]*(abc_u0_lin[0]-edge_vtx0[0]) + n[1]*(abc_u0_lin[1]-edge_vtx0[1]);
            double ratio = (dist_to_edge_lin-eps_abc_)/(dist_to_edge_lin-dist_to_edge);

            abc_u0[0] = abc_u0_lin[0] + ratio*(abc_u0[0] - abc_u0_lin[0]);
            abc_u0[1] = abc_u0_lin[1] + ratio*(abc_u0[1] - abc_u0_lin[1]);
          }
        }

        /* midpoint of the auxiliary edge */
        double abc_u1[2] = { 0.5*edgs[tri->edg2].a, 0.5 };

        if (adjust_auxiliary_midpoint_)
        {
          double *quad_node0   = abc_v01;
          double *quad_node1   = abc_v02;
          double  quad_node2[] = { 1., 0. };
          double  quad_node3[] = { 0., 1. };
          double *quad_node4   = abc_u0;

          deform_middle_node(abc_u1, abc_u1, quad_node0, quad_node1, quad_node2, quad_node3, quad_node4);
        }

        /* map midpoints to physical space */
        double xyz_u0[2]; mapping_tri(xyz_u0, n_tri, abc_u0);
        double xyz_u1[2]; mapping_tri(xyz_u1, n_tri, abc_u1);

        /* check if deformation is not too high */
        if (check_for_curvature_)
        {
          double xyz_u0_lin[2]; mapping_tri(xyz_u0_lin, n_tri, abc_u0_lin);

          double length = sqrt( SQR(vtxs[tri->c_vtx01].x - vtxs[tri->c_vtx02].x) +
                                SQR(vtxs[tri->c_vtx01].y - vtxs[tri->c_vtx02].y) );

          double deform = sqrt( SQR(xyz_u0[0]-xyz_u0_lin[0]) +
                                SQR(xyz_u0[1]-xyz_u0_lin[1]) );

          if (deform > length*curvature_limit_)
          {
            //std::cout << "High curvature: " << length << " " << deform << "\n";
            invalid_reconstruction_ = true;
          }
        }

        vtxs.push_back(vtx2_t(xyz_u0[0], xyz_u0[1]));
        vtxs.push_back(vtx2_t(xyz_u1[0], xyz_u1[1]));

        int u0 = vtxs.size()-2;
        int u1 = vtxs.size()-1;

        // new edges
        edgs.push_back(edg2_t(tri->c_vtx01, u0, tri->c_vtx02));
        edgs.push_back(edg2_t(tri->c_vtx01, u1, tri->vtx2   ));

        // edges might have changed their addresses
        edg0 = &edgs[tri->edg0];
        edg1 = &edgs[tri->edg1];
        edg2 = &edgs[tri->edg2];

        tri->c_edg0 = edgs.size()-2;
        tri->c_edg1 = edgs.size()-1;

        // new triangles
        tris.push_back(tri2_t(tri->vtx0,    tri->c_vtx01, tri->c_vtx02, tri->c_edg0,  edg1->c_edg0, edg2->c_edg0)); tri = &tris[n_tri];
        tris.push_back(tri2_t(tri->c_vtx01, tri->c_vtx02, tri->vtx2,    edg1->c_edg1, tri->c_edg1,  tri->c_edg0));  tri = &tris[n_tri];
        tris.push_back(tri2_t(tri->c_vtx01, tri->vtx1,    tri->vtx2,    tri->edg0,    tri->c_edg1,  edg2->c_edg1)); tri = &tris[n_tri];

        tri->c_tri0 = tris.size()-3;
        tri->c_tri1 = tris.size()-2;
        tri->c_tri2 = tris.size()-1;

        /* apply rules */
        c_edg0 = &edgs[tri->c_edg0];
        c_edg1 = &edgs[tri->c_edg1];

        c_tri0 = &tris[tri->c_tri0];
        c_tri1 = &tris[tri->c_tri1];
        c_tri2 = &tris[tri->c_tri2];

        vtx_u0 = &vtxs[u0];
        vtx_u1 = &vtxs[u1];

        if (action == INTERSECTION || action == ADDITION) c_edg0->p_lsf = cn;

#ifdef CASL_THROWS
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
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(FCE, cn); vtx_u0->set(FCE, cn, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
              default:
#ifdef CASL_THROWS
                throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
            } break;
          case ADDITION:
            switch (tri->loc){
              case OUT: c_edg0->set(FCE, cn); vtx_u0->set(FCE, cn, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef CASL_THROWS
                throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
            } break;
          case COLORATION:
            switch (tri->loc){
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef CASL_THROWS
                throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
            } break;
        }
        break;
      }

    case 2: // (--+)
      {
        /* split a triangle */
        tri->is_split = true;

        /* new vertices */
        tri->c_vtx02 = edgs[tri->edg1].c_vtx_x;
        tri->c_vtx12 = edgs[tri->edg0].c_vtx_x;

        // coordinates of new vertices in reference element
        double abc_v02[] = { 0.,                   edgs[tri->edg1].a };
        double abc_v12[] = { 1.-edgs[tri->edg0].a, edgs[tri->edg0].a };

        /* vertex along interface */
        double abc_u1_lin[2] = { .5*(abc_v02[0] + abc_v12[0]), .5*(abc_v02[1] + abc_v12[1]) };
        double abc_u1[2];
        find_middle_node(abc_u1, abc_v12, abc_v02, n_tri);

        /* check for an intersection with an auxiliary straight edge */
        if (check_for_edge_intersections_)
        {
          // two points on the edge
          double edge_vtx0[2] = { 0., 0.};
          double edge_vtx1[2] = { abc_v12[0], abc_v12[1] };

          // tangent and normal vectors
          double t[] = { edge_vtx1[0] - edge_vtx0[0], edge_vtx1[1] - edge_vtx0[1] };
          double n[] = { -t[1], t[0] };
          double norm = sqrt(SQR(n[0]) + SQR(n[1]));
          n[0] /= norm; n[1] /= norm;

          // level-set function for the edge: phi = na*(a-a0) + nb*(b-b0)
          double dist_to_edge = n[0]*(abc_u1[0]-edge_vtx0[0]) + n[1]*(abc_u1[1]-edge_vtx0[1]);

          if (dist_to_edge < 0)
          {
            // set the whole reconstruction for refinement
            invalid_reconstruction_ = true;

            // bring the midpoint below the edge in case the max level of refinement is reached
            double dist_to_edge_lin = n[0]*(abc_u1_lin[0]-edge_vtx0[0]) + n[1]*(abc_u1_lin[1]-edge_vtx0[1]);
            double ratio = (dist_to_edge_lin-eps_abc_)/(dist_to_edge_lin-dist_to_edge);

            abc_u1[0] = abc_u1_lin[0] + ratio*(abc_u1[0] - abc_u1_lin[0]);
            abc_u1[1] = abc_u1_lin[1] + ratio*(abc_u1[1] - abc_u1_lin[1]);
          }
        }

        /* midpoint of the auxiliary edge */
        double abc_u0[] = { 0.5*(1.-edgs[tri->edg0].a), 0.5*edgs[tri->edg0].a };

        if (adjust_auxiliary_midpoint_)
        {
          double quad_node0[] = { abc_v12[0], abc_v12[1] };
          double quad_node1[] = { abc_v02[0], abc_v02[1] };
          double quad_node2[] = { 0., 0. };
          double quad_node3[] = { 1., 0. };
          double quad_node4[] = { abc_u1[0], abc_u1[1] };

          deform_middle_node(abc_u0, abc_u0, quad_node0, quad_node1, quad_node2, quad_node3, quad_node4);
        }

        /* map midpoints to physical space */
        double xyz_u0[2]; mapping_tri(xyz_u0, n_tri, abc_u0);
        double xyz_u1[2]; mapping_tri(xyz_u1, n_tri, abc_u1);

        /* check if deformation is not too high */
        if (check_for_curvature_)
        {
          double xyz_u1_lin[2]; mapping_tri(xyz_u1_lin, n_tri, abc_u1_lin);

          double length = sqrt( SQR(vtxs[tri->c_vtx02].x - vtxs[tri->c_vtx12].x) +
                                SQR(vtxs[tri->c_vtx02].y - vtxs[tri->c_vtx12].y) );

          double deform = sqrt( SQR(xyz_u1[0]-xyz_u1_lin[0]) +
                                SQR(xyz_u1[1]-xyz_u1_lin[1]) );

          if (deform > length*curvature_limit_)
          {
            //std::cout << "High curvature: " << length << " " << deform << "\n";
            invalid_reconstruction_ = true;
          }
        }

        vtxs.push_back(vtx2_t(xyz_u0[0], xyz_u0[1]));
        vtxs.push_back(vtx2_t(xyz_u1[0], xyz_u1[1]));

        int u0 = vtxs.size()-2;
        int u1 = vtxs.size()-1;

        // create new edges
        edgs.push_back(edg2_t(tri->vtx0,    u0, tri->c_vtx12));
        edgs.push_back(edg2_t(tri->c_vtx02, u1, tri->c_vtx12));

        // edges might have changed their addresses
        edg0 = &edgs[tri->edg0];
        edg1 = &edgs[tri->edg1];
        edg2 = &edgs[tri->edg2];

        tri->c_edg0 = edgs.size()-2;
        tri->c_edg1 = edgs.size()-1;

        tris.push_back(tri2_t(tri->vtx0,    tri->vtx1,    tri->c_vtx12, edg0->c_edg0, tri->c_edg0,  tri->edg2   )); tri = &tris[n_tri];
        tris.push_back(tri2_t(tri->vtx0,    tri->c_vtx02, tri->c_vtx12, tri->c_edg1,  tri->c_edg0,  edg1->c_edg0)); tri = &tris[n_tri];
        tris.push_back(tri2_t(tri->c_vtx02, tri->c_vtx12, tri->vtx2,    edg0->c_edg1, edg1->c_edg1, tri->c_edg1 )); tri = &tris[n_tri];

        tri->c_tri0 = tris.size()-3;
        tri->c_tri1 = tris.size()-2;
        tri->c_tri2 = tris.size()-1;

        /* apply rules */
        c_edg0 = &edgs[tri->c_edg0];
        c_edg1 = &edgs[tri->c_edg1];

        c_tri0 = &tris[tri->c_tri0];
        c_tri1 = &tris[tri->c_tri1];
        c_tri2 = &tris[tri->c_tri2];

        vtx_u0 = &vtxs[u0];
        vtx_u1 = &vtxs[u1];

        if (action == INTERSECTION || action == ADDITION) c_edg1->p_lsf = cn;

#ifdef CASL_THROWS
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
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(FCE, cn); vtx_u1->set(FCE, cn, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
              default:
#ifdef CASL_THROWS
                throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
            } break;
          case ADDITION:
            switch (tri->loc){
              case OUT: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(FCE, cn); vtx_u1->set(FCE, cn, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef CASL_THROWS
                throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
            } break;
          case COLORATION:
            switch (tri->loc){
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef CASL_THROWS
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
        case INTERSECTION:  /* do nothing */  break;
        case ADDITION:      tri->set(INS);    break;
        case COLORATION:                      break;
      }
      break;
  }
}







//--------------------------------------------------
// Auxiliary tools for splitting
//--------------------------------------------------
double simplex2_mls_quadratic_t::find_intersection_quadratic(int e)
{
  double f0 = vtxs[edgs[e].vtx0].value;
  double f1 = vtxs[edgs[e].vtx1].value;
  double f2 = vtxs[edgs[e].vtx2].value;

  if (fabs(f0) < .8*eps_xyz_) return .8*eps_abc_;
  if (fabs(f1) < .8*eps_xyz_) return 0.5;
  if (fabs(f2) < .8*eps_xyz_) return 1.-.8*eps_abc_;

#ifdef CASL_THROWS
  if(f0*f2 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
#endif

  double fdd = (f2+f0-2.*f1)/0.25;

  double c2 = 0.5*fdd;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = (f2-f0)/1.;   // the expansion of f at the center of (0,1)
  double c0 = f1;

  double x;

  if (fabs(c2) < EPS) { c0 = .5*(f0+f2); x = -c0/c1; }
  else
  {
    if (f2<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    else      x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
  }
#ifdef CASL_THROWS
  if (x < -0.5 || x > 0.5)
  {
    std::cout << f0 << " " << f1 << " " << f2 << " " << x << std::endl;
    throw std::domain_error("[CASL_ERROR]: ");
  }
#endif

  if (x <-0.5) return    .8*eps_abc_;
  if (x > 0.5) return 1.-.8*eps_abc_;

  return .5+x;
}

//void simplex2_mls_quadratic_t::find_middle_node(double &x_out, double &y_out, double x0, double y0, double x1, double y1, int n_tri)
void simplex2_mls_quadratic_t::find_middle_node(double *xyz_out, double *xyz0, double *xyz1, int n_tri)
{
  tri2_t *tri = &tris[n_tri];

  // compute normal
  double tx = xyz1[0]-xyz0[0];
  double ty = xyz1[1]-xyz0[1];
  double norm = sqrt(tx*tx+ty*ty);
  tx /= norm;
  ty /= norm;
  double nx =-ty;
  double ny = tx;

  // fetch values of LSF
  std::vector<int> nv(nodes_per_tri_, -1);

  nv[0] = tri->vtx0;
  nv[1] = tri->vtx1;
  nv[2] = tri->vtx2;
  nv[3] = edgs[tri->edg2].vtx1;
  nv[4] = edgs[tri->edg0].vtx1;
  nv[5] = edgs[tri->edg1].vtx1;

//  std::vector<double> f(nodes_per_tri_, 0);

//  for (short i = 0; i < nodes_per_tri_; ++i)
//    f[i] = vtxs[nv[i]].value;

  double a = 0.5*(xyz0[0]+xyz1[0]);
  double b = 0.5*(xyz0[1]+xyz1[1]);

  double N[nodes_per_tri_]   = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};
  double Na[nodes_per_tri_]  = {-3.+4.*a+4.*b,            4.*a-1.,      0,            4.-8.*a-4.*b,   4.*b,   -4.*b};
  double Nb[nodes_per_tri_]  = {-3.+4.*a+4.*b,            0,            4.*b-1.,       -4.*a,          4.*a,   4.-4.*a-8.*b};
  double Naa[nodes_per_tri_] = {4,4,0,-8,0,0};
  double Nab[nodes_per_tri_] = {4,0,0,-4,4,-4};
  double Nbb[nodes_per_tri_] = {4,0,4,0,0,-8};

  double F = 0, Fn = 0, Fnn = 0;
  double f;
  for (short i = 0; i < nodes_per_tri_; ++i)
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

  xyz_out[0] = a + alpha*nx;
  xyz_out[1] = b + alpha*ny;

  while (xyz_out[0] + xyz_out[1] > 1. || xyz_out[0] < 0. || xyz_out[1] < 0.)
  {
    invalid_reconstruction_ = true;
    if      (xyz_out[0] < 0.)               alpha = (a-eps_abc_)/(a-xyz_out[0])*alpha;
    else if (xyz_out[1] < 0.)               alpha = (b-eps_abc_)/(b-xyz_out[1])*alpha;
    else if (1.-xyz_out[0]-xyz_out[1] < 0.) alpha = (1.-a-b-eps_abc_)/(1. -a-b-(1.-xyz_out[0]-xyz_out[1]))*alpha;

    xyz_out[0] = a + alpha*nx;
    xyz_out[1] = b + alpha*ny;

//    std::cout << "Here!\n";
//    std::cout << "Warning: point is outside of a triangle! (" << .5*(x0+x1) << " " << .5*(y0+y1) << " " << x_out << " " << y_out << ")\n";

  }
}

//void simplex2_mls_quadratic_t::deform_middle_node(double &x_out, double &y_out,
//                                                  double x, double y,
//                                                  double x0, double y0,
//                                                  double x1, double y1,
//                                                  double x2, double y2,
//                                                  double x3, double y3,
//                                                  double x01, double y01)
void simplex2_mls_quadratic_t::deform_middle_node(double *xyz_out,
                                                  double *xyz_in,
                                                  double *xyz0,
                                                  double *xyz1,
                                                  double *xyz2,
                                                  double *xyz3,
                                                  double *xyz01)
{

  double Xa  = -xyz0[0] + xyz1[0];
  double Xb  = -xyz0[0] + xyz3[0];
  double Xab =  xyz0[0] - xyz1[0] + xyz2[0] - xyz3[0];

  double Ya  = -xyz0[1] + xyz1[1];
  double Yb  = -xyz0[1] + xyz3[1];
  double Yab =  xyz0[1] - xyz1[1] + xyz2[1] - xyz3[1];

  // solve quadratic equation
  // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0
  double c2 = Xb*Yab - Yb*Xab;
  double c1 = Ya*Xb - Xa*Yb + Xab*(xyz_in[1]-xyz0[1]) - Yab*(xyz_in[0]-xyz0[0]);
  double c0 = Xa*(xyz_in[1]-xyz0[1])-Ya*(xyz_in[0]-xyz0[0]);

  double b;

  if (fabs(c2) < EPS) b = -c0/c1;
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

  double a = (xyz_in[0]-xyz0[0]-b*Xb)/(Xa+b*Xab);

  xyz_out[0] = xyz0[0] + a*Xa + b*Xb + a*b*Xab + (1.-b)*((1.-a)*xyz0[0] + a*xyz1[0])*2.*(1.-a)*a*(-xyz0[0]+2.*xyz01[0]-xyz1[0]);
  xyz_out[1] = xyz0[1] + a*Ya + b*Yb + a*b*Yab + (1.-b)*((1.-a)*xyz0[1] + a*xyz1[1])*2.*(1.-a)*a*(-xyz0[1]+2.*xyz01[1]-xyz1[1]);
}

//void simplex2_mls_quadratic_t::find_middle_node(double &x_out, double &y_out, double x0, double y0, double x1, double y1, int n_tri)
//{
//  tri2_t *tri = &tris[n_tri];

//  // compute normal
//  double tx = x1-x0;
//  double ty = y1-y0;
//  double norm = sqrt(tx*tx+ty*ty);
//  tx /= norm;
//  ty /= norm;
//  double nx =-ty;
//  double ny = tx;

//  // fetch values of LSF
//  std::vector<int> nv(nodes_per_tri_, -1);

//  nv[0] = tri->vtx0;
//  nv[1] = tri->vtx1;
//  nv[2] = tri->vtx2;
//  nv[3] = edgs[tri->edg2].vtx1;
//  nv[4] = edgs[tri->edg0].vtx1;
//  nv[5] = edgs[tri->edg1].vtx1;

////  std::vector<double> f(nodes_per_tri_, 0);

////  for (short i = 0; i < nodes_per_tri_; ++i)
////    f[i] = vtxs[nv[i]].value;

//  double a = 0.5*(x0+x1);
//  double b = 0.5*(y0+y1);
//  double tolerance = 1.e-16;
//  double F = 1;

//  while (F > tolerance)
//  {

//    double N[nodes_per_tri_]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};
//    double Na[nodes_per_tri_] = {-3.+4.*a+4.*b,            4.*a-1.,      0,            4.-8.*a-4.*b,   4.*b,   -4.*b};
//    double Nb[nodes_per_tri_] = {-3.+4.*a+4.*b,            0,            4.*b-1.,       -4.*a,          4.*a,   4.-4.*a-8.*b};

//    F = 0;
//    double Fn = 0;
//    double f;
//    for (short i = 0; i < nodes_per_tri_; ++i)
//    {
//      f = vtxs[nv[i]].value;
//      F   += f*N[i];
//      Fn  += f*(Na[i]*nx+Nb[i]*ny);
//    }

//    double change_a = F*nx/Fn;
//    double change_b = F*ny/Fn;

//    a -= change_a;
//    b -= change_b;
//  }

//  x_out = a;
//  y_out = b;

//}

bool simplex2_mls_quadratic_t::need_swap(int v0, int v1)
{
//  double dif = vtxs[v0].value - vtxs[v1].value;
//  if (fabs(dif) < eps){ // if values are too close, sort vertices by their numbers
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

void simplex2_mls_quadratic_t::refine_all()
{
  int n;
  n = edgs.size(); for (int i = 0; i < n; i++) refine_edg(i);
  n = tris.size(); for (int i = 0; i < n; i++) refine_tri(i);

  std::cout << "Refined!\n";
}

void simplex2_mls_quadratic_t::refine_edg(int n_edg)
{
  edg2_t *edg = &edgs[n_edg];

  if (edg->is_split) return;
  else edg->is_split = true;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  /* Create two new vertices */
  double x_v01, y_v01, x_v12, y_v12;
  mapping_edg(x_v01, y_v01, n_edg, 0.25);
  mapping_edg(x_v12, y_v12, n_edg, 0.75);

  vtxs.push_back(vtx2_t(x_v01, y_v01));
  vtxs.push_back(vtx2_t(x_v12, y_v12));

  int n_vtx01 = vtxs.size()-2;
  int n_vtx12 = vtxs.size()-1;

  /* Create two new edges */
  edgs.push_back(edg2_t(edg->vtx0, n_vtx01, edg->vtx1)); edg = &edgs[n_edg];
  edgs.push_back(edg2_t(edg->vtx1, n_vtx12, edg->vtx2)); edg = &edgs[n_edg];

  edg->c_edg0 = edgs.size()-2;
  edg->c_edg1 = edgs.size()-1;

  /* Transfer properties to new objects */
  loc_t loc = edg->loc;
  int c = edg->c0;

  int dir = edg->dir;

  vtxs[n_vtx01].set(loc, c, -1);
  vtxs[n_vtx12].set(loc, c, -1);

  edgs[edg->c_edg0].set(loc, c);
  edgs[edg->c_edg1].set(loc, c);

  edgs[edg->c_edg0].dir = dir;
  edgs[edg->c_edg1].dir = dir;
}

void simplex2_mls_quadratic_t::refine_tri(int n_tri)
{
  tri2_t *tri = &tris[n_tri];

  if (tri->is_split) return;
  else tri->is_split = true;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Create 3 new vertices */
  double xyz[2];
  double abc[2];
  abc[0] = .25; abc[1] = .25; mapping_tri(xyz, n_tri, abc); vtxs.push_back(vtx2_t(xyz[0], xyz[1]));
  abc[0] = .50; abc[1] = .25; mapping_tri(xyz, n_tri, abc); vtxs.push_back(vtx2_t(xyz[0], xyz[1]));
  abc[0] = .25; abc[1] = .50; mapping_tri(xyz, n_tri, abc); vtxs.push_back(vtx2_t(xyz[0], xyz[1]));

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

  edgs.push_back(edg2_t(n_v02, n_u0, n_v01));
  edgs.push_back(edg2_t(n_v01, n_u1, n_v12));
  edgs.push_back(edg2_t(n_v02, n_u2, n_v12));

  /* Create 4 new triangles */
  int n_edg0 = edgs.size()-3;
  int n_edg1 = edgs.size()-2;
  int n_edg2 = edgs.size()-1;

  tris.push_back(tri2_t(n_v0,  n_v01, n_v02, n_edg0, edgs[tri->edg1].c_edg0, edgs[tri->edg2].c_edg0)); tri = &tris[n_tri];
  tris.push_back(tri2_t(n_v1,  n_v01, n_v12, n_edg1, edgs[tri->edg0].c_edg0, edgs[tri->edg2].c_edg1)); tri = &tris[n_tri];
  tris.push_back(tri2_t(n_v2,  n_v02, n_v12, n_edg2, edgs[tri->edg0].c_edg1, edgs[tri->edg1].c_edg1)); tri = &tris[n_tri];
  tris.push_back(tri2_t(n_v01, n_v02, n_v12, n_edg2, n_edg1,                 n_edg0));                 tri = &tris[n_tri];

  int n_tri0 = tris.size()-4;
  int n_tri1 = tris.size()-3;
  int n_tri2 = tris.size()-2;
  int n_tri3 = tris.size()-1;

  /* Transfer properties */
  loc_t loc = tri->loc;

  vtxs[n_u0].set(loc, -1, -1);
  vtxs[n_u1].set(loc, -1, -1);
  vtxs[n_u2].set(loc, -1, -1);

  edgs[n_edg0].set(loc, -1);
  edgs[n_edg1].set(loc, -1);
  edgs[n_edg2].set(loc, -1);

  tris[n_tri0].set(loc);
  tris[n_tri1].set(loc);
  tris[n_tri2].set(loc);
  tris[n_tri3].set(loc);
}





//--------------------------------------------------
// Integration
//--------------------------------------------------
//double simplex2_mls_quadratic_t::integrate_over_domain(std::vector<double> &f)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0;
//  double f0 = 0, f1 = 0, f2 = 0;
//  double x,y;

//  // quadrature points
//  double a0 = 1./6., b0 = 1./6.;
//  double a1 = 2./3., b1 = 1./6.;
//  double a2 = 1./6., b2 = 2./3.;

//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri2_t *t = &tris[i];
//    if (!t->is_split && t->loc == INS)
//    {
//      // map quadrature points into real space and interpolate integrand
//      mapping_tri(x, y, i, a0, b0); f0 = interpolate_from_parent(f, x, y);
//      mapping_tri(x, y, i, a1, b1); f1 = interpolate_from_parent(f, x, y);
//      mapping_tri(x, y, i, a2, b2); f2 = interpolate_from_parent(f, x, y);

//      // scale weights by Jacobian
//      w0 = jacobian_tri(i, a0, b0)/6.;
//      w1 = jacobian_tri(i, a1, b1)/6.;
//      w2 = jacobian_tri(i, a2, b2)/6.;

//      result += w0*f0 + w1*f1 + w2*f2;
////      result += area(t->vtx0, t->vtx1, t->vtx2);
//    }
//  }

//  return result;
//}

//double simplex2_mls_quadratic_t::integrate_over_domain(CF_2 &f)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
//  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
//  double x,y;

//  // quadrature points, order 3
//  double a0 = 1./3., b0 = 1./3.;
//  double a1 = 0.2, b1 = 0.6;
//  double a2 = 0.2, b2 = 0.2;
//  double a3 = 0.6, b3 = 0.2;

//  /* integrate over triangles */
//  for (unsigned int i = 0; i < tris.size(); i++)
//  {
//    tri2_t *t = &tris[i];
//    if (!t->is_split && t->loc == INS)
//    {
//      // map quadrature points into real space and interpolate integrand
//      mapping_tri(x, y, i, a0, b0); f0 = f( x, y );
//      mapping_tri(x, y, i, a1, b1); f1 = f( x, y );
//      mapping_tri(x, y, i, a2, b2); f2 = f( x, y );
//      mapping_tri(x, y, i, a3, b3); f3 = f( x, y );

//      // scale weights by Jacobian
//      w0 =-27.*jacobian_tri(i, a0, b0);
//      w1 = 25.*jacobian_tri(i, a1, b1);
//      w2 = 25.*jacobian_tri(i, a2, b2);
//      w3 = 25.*jacobian_tri(i, a3, b3);

//      result += w0*f0 + w1*f1 + w2*f2 + w3*f3;
////      result += area(t->vtx0, t->vtx1, t->vtx2);
//    }
//  }

//  return result/96.;
//}

double simplex2_mls_quadratic_t::integrate_over_domain(CF_2 &f)
{
  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
  double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
  double xyz[3];

  // quadrature points, degree 2
  double abc0[] = {.0, .5};
  double abc1[] = {.5, .0};
  double abc2[] = {.5, .5};

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris.size(); i++)
  {
    tri2_t *t = &tris[i];
    if (!t->is_split && t->loc == INS)
    {
      // map quadrature points into real space and interpolate integrand
      mapping_tri(xyz, i, abc0); f0 = f.value( xyz );
      mapping_tri(xyz, i, abc1); f1 = f.value( xyz );
      mapping_tri(xyz, i, abc2); f2 = f.value( xyz );

      // scale weights by Jacobian
      w0 = jacobian_tri(i, abc0[0], abc0[1]);
      w1 = jacobian_tri(i, abc1[0], abc1[1]);
      w2 = jacobian_tri(i, abc2[0], abc2[1]);

      result += w0*f0 + w1*f1 + w2*f2 + w3*f3;
    }
  }

  return result/6.;
}

double simplex2_mls_quadratic_t::integrate_over_interface(CF_2 &f, int num)
{
  bool integrate_specific = (num != -1);

  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0;
  double f0 = 0, f1 = 0, f2 = 0;
  double x,y;

  // quadrature points
  double a0 = 0.;
  double a1 = .5;
  double a2 = 1.;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs.size(); i++)
  {
    edg2_t *e = &edgs[i];
    if (!e->is_split && e->loc == FCE)
      if ((!integrate_specific && e->c0 >= 0) || (integrate_specific && e->c0 == num))
      {
        // map quadrature points into real space and interpolate integrand
        mapping_edg(x, y, i, a0); f0 = f( x, y );
        mapping_edg(x, y, i, a1); f1 = f( x, y );
        mapping_edg(x, y, i, a2); f2 = f( x, y );

        // scale weights by Jacobian
        w0 = jacobian_edg(i, a0);
        w1 = jacobian_edg(i, a1)*4.;
        w2 = jacobian_edg(i, a2);

        result += w0*f0 + w1*f1 + w2*f2;
      }
  }

  return result/6.;
}

//double simplex2_mls_quadratic_t::integrate_over_interface(CF_2 &f, int num)
//{
//  bool integrate_specific = (num != -1);

//  double result = 0.0;
//  double w0 = 0, w1 = 0;
//  double f0 = 0, f1 = 0;
//  double x,y;

//  // quadrature points, order 3
//  double a0 = .5*(1.-1./sqrt(3.));
//  double a1 = .5*(1.+1./sqrt(3.));

//  /* integrate over edges */
//  for (unsigned int i = 0; i < edgs.size(); i++)
//  {
//    edg2_t *e = &edgs[i];
//    if (!e->is_split && e->loc == FCE)
//      if ((!integrate_specific && e->c0 >= 0) || (integrate_specific && e->c0 == num))
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_edg(x, y, i, a0); f0 = f( x, y );
//        mapping_edg(x, y, i, a1); f1 = f( x, y );

//        // scale weights by Jacobian
//        w0 = jacobian_edg(i, a0)/2.;
//        w1 = jacobian_edg(i, a1)/2.;

//        result += w0*f0 + w1*f1;
////        result += (fabs(f0)+fabs(f1));
//      }
//  }

//  return result;

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
//}

//// integrate over colored interfaces (num0 - parental lsf, num1 - coloring lsf)
//double simplex2_mls_quadratic_t::integrate_over_colored_interface(CF_2 &f, int num0, int num1)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0;
//  double f0 = 0, f1 = 0;
//  double x,y;

//  // quadrature points
//  double a0 = .5*(1.-1./sqrt(3.));
//  double a1 = .5*(1.+1./sqrt(3.));

//  /* integrate over edges */
//  for (unsigned int i = 0; i < edgs.size(); i++)
//  {
//    edg2_t *e = &edgs[i];
//    if (!e->is_split && e->loc == FCE)
//      if (e->p_lsf == num0 && e->c0 == num1)
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_edg(x, y, i, a0); f0 = f( x, y );
//        mapping_edg(x, y, i, a1); f1 = f( x, y );

//        // scale weights by Jacobian
//        w0 = jacobian_edg(i, a0)/2.;
//        w1 = jacobian_edg(i, a1)/2.;

//        result += w0*f0 + w1*f1;
//      }
//  }

//  return result;
//}

// integrate over colored interfaces (num0 - parental lsf, num1 - coloring lsf)
double simplex2_mls_quadratic_t::integrate_over_colored_interface(CF_2 &f, int num0, int num1)
{
  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0;
  double f0 = 0, f1 = 0, f2 = 0;
  double x,y;

  // quadrature points
  double a0 = 0.;
  double a1 = .5;
  double a2 = 1.;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs.size(); i++)
  {
    edg2_t *e = &edgs[i];
    if (!e->is_split && e->loc == FCE)
      if (e->p_lsf == num0 && e->c0 == num1)
      {
        // map quadrature points into real space and interpolate integrand
        mapping_edg(x, y, i, a0); f0 = f( x, y );
        mapping_edg(x, y, i, a1); f1 = f( x, y );
        mapping_edg(x, y, i, a2); f2 = f( x, y );

        // scale weights by Jacobian
        w0 = jacobian_edg(i, a0);
        w1 = jacobian_edg(i, a1)*4.;
        w2 = jacobian_edg(i, a2);

        result += w0*f0 + w1*f1 + w2*f2;
      }
  }

  return result/6.;
}

double simplex2_mls_quadratic_t::integrate_over_intersection(CF_2 &f, int num0, int num1)
{
  double result = 0.0;
  bool integrate_specific = (num0 != -1 && num1 != -1);

  for (unsigned int i = 0; i < vtxs.size(); i++)
  {
    vtx2_t *v = &vtxs[i];
    if (v->loc == PNT)
      if ( !integrate_specific
           || (integrate_specific
               && (v->c0 == num0 || v->c1 == num0)
               && (v->c0 == num1 || v->c1 == num1)) )
      {
        result += f( v->x, v->y );
      }
  }

  return result;
}

//double simplex2_mls_quadratic_t::integrate_in_dir(CF_2 &f, int dir)
//{
//  double result = 0.0;
//  double w0 = 0, w1 = 0;
//  double f0 = 0, f1 = 0;
//  double x,y;

//  // quadrature points
//  double a0 = .5*(1.-1./sqrt(3.));
//  double a1 = .5*(1.+1./sqrt(3.));

//  /* integrate over edges */
//  for (unsigned int i = 0; i < edgs.size(); i++)
//  {
//    edg2_t *e = &edgs[i];
//    if (!e->is_split && e->loc == INS)
//      if (e->dir == dir)
//      {
//        // map quadrature points into real space and interpolate integrand
//        mapping_edg(x, y, i, a0); f0 = f( x, y );
//        mapping_edg(x, y, i, a1); f1 = f( x, y );

//        // scale weights by Jacobian
//        w0 = jacobian_edg(i, a0)/2.;
//        w1 = jacobian_edg(i, a1)/2.;

//        result += w0*f0 + w1*f1;
//      }
//  }

//  return result;
//}

double simplex2_mls_quadratic_t::integrate_in_dir(CF_2 &f, int dir)
{
  double result = 0.0;
  double w0 = 0, w1 = 0, w2 = 0;
  double f0 = 0, f1 = 0, f2 = 0;
  double x,y;

  // quadrature points
  double a0 = 0.;
  double a1 = .5;
  double a2 = 1.;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs.size(); i++)
  {
    edg2_t *e = &edgs[i];
    if (!e->is_split && e->loc == INS)
      if (e->dir == dir)
      {
        // map quadrature points into real space and interpolate integrand
        mapping_edg(x, y, i, a0); f0 = f( x, y );
        mapping_edg(x, y, i, a1); f1 = f( x, y );
        mapping_edg(x, y, i, a2); f2 = f( x, y );

        // scale weights by Jacobian
        w0 = jacobian_edg(i, a0);
        w1 = jacobian_edg(i, a1)*4.;
        w2 = jacobian_edg(i, a2);

        result += w0*f0 + w1*f1 + w2*f2;
      }
  }

  return result/6.;
}







//--------------------------------------------------
// Jacobians
//--------------------------------------------------
double simplex2_mls_quadratic_t::jacobian_edg(int n_edg, double a)
{
  edg2_t *edg = &edgs[n_edg];

  double Na[3] = {-3.+4.*a, 4.-8.*a, -1.+4.*a};

  double X = vtxs[edg->vtx0].x * Na[0] + vtxs[edg->vtx1].x * Na[1] + vtxs[edg->vtx2].x * Na[2];
  double Y = vtxs[edg->vtx0].y * Na[0] + vtxs[edg->vtx1].y * Na[1] + vtxs[edg->vtx2].y * Na[2];

  return sqrt(X*X+Y*Y);
}

double simplex2_mls_quadratic_t::jacobian_tri(int n_tri, double a, double b)
{
  tri2_t *tri = &tris[n_tri];

  double Na[6] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
  double Nb[6] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

  double j00 = vtxs[tri->vtx0].x*Na[0] + vtxs[tri->vtx1].x*Na[1] + vtxs[tri->vtx2].x*Na[2] + vtxs[edgs[tri->edg2].vtx1].x*Na[3] + vtxs[edgs[tri->edg0].vtx1].x*Na[4] + vtxs[edgs[tri->edg1].vtx1].x*Na[5];
  double j10 = vtxs[tri->vtx0].y*Na[0] + vtxs[tri->vtx1].y*Na[1] + vtxs[tri->vtx2].y*Na[2] + vtxs[edgs[tri->edg2].vtx1].y*Na[3] + vtxs[edgs[tri->edg0].vtx1].y*Na[4] + vtxs[edgs[tri->edg1].vtx1].y*Na[5];
  double j01 = vtxs[tri->vtx0].x*Nb[0] + vtxs[tri->vtx1].x*Nb[1] + vtxs[tri->vtx2].x*Nb[2] + vtxs[edgs[tri->edg2].vtx1].x*Nb[3] + vtxs[edgs[tri->edg0].vtx1].x*Nb[4] + vtxs[edgs[tri->edg1].vtx1].x*Nb[5];
  double j11 = vtxs[tri->vtx0].y*Nb[0] + vtxs[tri->vtx1].y*Nb[1] + vtxs[tri->vtx2].y*Nb[2] + vtxs[edgs[tri->edg2].vtx1].y*Nb[3] + vtxs[edgs[tri->edg0].vtx1].y*Nb[4] + vtxs[edgs[tri->edg1].vtx1].y*Nb[5];

  return fabs(j00*j11-j01*j10);
//  return j00*j11-j01*j10;
//  return sqrt(j00*j00*j11*j11 - 2.*j00*j01*j11*j10 + j01*j10*j01*j10);
}





//--------------------------------------------------
// Interpolation
//--------------------------------------------------
double simplex2_mls_quadratic_t::interpolate_from_parent(std::vector<double> &f, double x, double y)
{
  // map real point to reference element
  vtx2_t *v0 = &vtxs[0];
  vtx2_t *v1 = &vtxs[1];
  vtx2_t *v2 = &vtxs[2];

  double a = ( (x-v0->x)*(v2->y-v0->y) - (y-v0->y)*(v2->x-v0->x) ) / ( (v1->x-v0->x)*(v2->y-v0->y) - (v1->y-v0->y)*(v2->x-v0->x) );
  double b = ( (x-v0->x)*(v1->y-v0->y) - (y-v0->y)*(v1->x-v0->x) ) / ( (v2->x-v0->x)*(v1->y-v0->y) - (v2->y-v0->y)*(v1->x-v0->x) );

  // compute nodal functions
  double N[nodes_per_tri_]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

  double result = 0;

  for (short i = 0; i < nodes_per_tri_; ++i)
  {
    result += N[i]*f[i];
  }

  return result;
}









//--------------------------------------------------
// Mapping
//--------------------------------------------------
void simplex2_mls_quadratic_t::mapping_edg(double &x, double &y, int n_edg, double a)
{
  edg2_t *edg = &edgs[n_edg];

  double N0 = 1.-3.*a+2.*a*a;
  double N1 = 4.*a-4.*a*a;
  double N2 = -a+2.*a*a;

  x = vtxs[edg->vtx0].x * N0 + vtxs[edg->vtx1].x * N1 + vtxs[edg->vtx2].x * N2;
  y = vtxs[edg->vtx0].y * N0 + vtxs[edg->vtx1].y * N1 + vtxs[edg->vtx2].y * N2;
}

//void simplex2_mls_quadratic_t::mapping_tri(double &x, double &y, int n_tri, double a, double b)
void simplex2_mls_quadratic_t::mapping_tri(double *xyz, int n_tri, double *abc)
{
  tri2_t *tri = &tris[n_tri];

  double a = abc[0];
  double b = abc[1];
  double N[nodes_per_tri_]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

  int nv0 = tri->vtx0;
  int nv1 = tri->vtx1;
  int nv2 = tri->vtx2;
  int nv3 = edgs[tri->edg2].vtx1;
  int nv4 = edgs[tri->edg0].vtx1;
  int nv5 = edgs[tri->edg1].vtx1;

  xyz[0] = vtxs[nv0].x * N[0] + vtxs[nv1].x * N[1] + vtxs[nv2].x * N[2] + vtxs[nv3].x * N[3] + vtxs[nv4].x * N[4] + vtxs[nv5].x * N[5];
  xyz[1] = vtxs[nv0].y * N[0] + vtxs[nv1].y * N[1] + vtxs[nv2].y * N[2] + vtxs[nv3].y * N[3] + vtxs[nv4].y * N[4] + vtxs[nv5].y * N[5];
}



//--------------------------------------------------
// Debugging tools
//--------------------------------------------------
#ifdef CASL_THROWS
bool simplex2_mls_quadratic_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = (edgs[e0].vtx0 == v1 || edgs[e0].vtx2 == v1) && (edgs[e0].vtx0 == v2 || edgs[e0].vtx2 == v2);
  result = result && (edgs[e1].vtx0 == v0 || edgs[e1].vtx2 == v0) && (edgs[e1].vtx0 == v2 || edgs[e1].vtx2 == v2);
  result = result && (edgs[e2].vtx0 == v0 || edgs[e2].vtx2 == v0) && (edgs[e2].vtx0 == v1 || edgs[e2].vtx2 == v1);
  return result;
}

bool simplex2_mls_quadratic_t::tri_is_ok(int t)
{
  tri2_t *tri = &tris[t];
  bool result = tri_is_ok(tri->vtx0, tri->vtx1, tri->vtx2, tri->edg0, tri->edg1, tri->edg2);
  if (!result)
  {
    std::cout << "Inconsistent triangle!\n";
  }
  return result;
}
#endif



//double simplex2_mls_quadratic_t::length(int vtx0, int vtx1)
//{
//  return sqrt(pow(vtxs[vtx0].x - vtxs[vtx1].x, 2.0)
//            + pow(vtxs[vtx0].y - vtxs[vtx1].y, 2.0));
//}

double simplex2_mls_quadratic_t::area(int vtx0, int vtx1, int vtx2)
{
  double x01 = vtxs[vtx1].x - vtxs[vtx0].x; double x02 = vtxs[vtx2].x - vtxs[vtx0].x;
  double y01 = vtxs[vtx1].y - vtxs[vtx0].y; double y02 = vtxs[vtx2].y - vtxs[vtx0].y;

  return 0.5*fabs(x01*y02-y01*x02);
}

//void simplex2_mls_quadratic_t::accept_reconstruction()
//{
//  int n;
//  n = edgs.size(); for (int i = 0; i < n; i++) if (!edgs[i].is_split && edgs[i].to_split) edgs[i].is_split = true;
//  n = tris.size(); for (int i = 0; i < n; i++) if (!tris[i].is_split && tris[i].to_split) tris[i].is_split = true;
//}

//void simplex2_mls_quadratic_t::discard_reconstruction()
//{
//  int n;
//  n = edgs.size(); for (int i = 0; i < n; i++) if (!edgs[i].is_split && edgs[i].to_split) edgs[i].to_split = false;
//  n = tris.size(); for (int i = 0; i < n; i++) if (!tris[i].is_split && tris[i].to_split) tris[i].to_split = false;
//}




//void simplex2_mls_quadratic_t::interpolate_from_neighbors(int v)
//{
//  vtx2_t *vtx = &vtxs[v];
//  vtx->value = vtx->ratio*vtxs[vtx->n_vtx0].value + (1.0-vtx->ratio)*vtxs[vtx->n_vtx1].value;
//}

//void simplex2_mls_quadratic_t::interpolate_all(double &p0, double &p1, double &p2)
//{
//  vtxs[0].value = p0;
//  vtxs[1].value = p1;
//  vtxs[2].value = p2;

//  for (unsigned int i = 3; i < vtxs.size(); i++)
//  {
//    interpolate_from_neighbors(i);
//  }
//}

//double simplex2_mls_quadratic_t::find_intersection_linear(int v0, int v1)
//{
//  vtx2_t *vtx0 = &vtxs[v0];
//  vtx2_t *vtx1 = &vtxs[v1];
//  double nx = vtx1->x - vtx0->x;
//  double ny = vtx1->y - vtx0->y;
//  double l = sqrt(nx*nx+ny*ny);
//#ifdef CASL_THROWS
//  if(l < eps) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
//#endif
//  double f0 = vtx0->value;
//  double f1 = vtx1->value;

//  if(fabs(f0)<eps) return 0.+eps;
//  if(fabs(f1)<eps) return l-eps;

//#ifdef CASL_THROWS
//  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
//#endif

//  double c1 =     (f1-f0)/l;          //  the expansion of f at the center of (a,b)
//  double c0 = 0.5*(f1+f0);

//  double x = -c0/c1;

//#ifdef CASL_THROWS
//  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
//#endif

//  return 1.-(x+0.5*l)/l;
//}

//double simplex2_mls_quadratic_t::find_intersection_quadratic(int e)
//{
//  vtx2_t *vtx0 = &vtxs[edgs[e].vtx0];
//  vtx2_t *vtx1 = &vtxs[edgs[e].vtx1];
//  double nx = vtx1->x - vtx0->x;
//  double ny = vtx1->y - vtx0->y;
//  double l = sqrt(nx*nx+ny*ny);
//#ifdef CASL_THROWS
//  if(l < eps) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
//#endif
//  double f0 = vtx0->value;
//  double f01 = edgs[e].value;
//  double f1 = vtx1->value;

//  if (fabs(f0)  < eps) return (l-eps)/l;
//  if (fabs(f01) < eps) return 0.5;
//  if (fabs(f1)  < eps) return (0.+eps)/l;

//#ifdef CASL_THROWS
//  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
//#endif

//  double fdd = (f1+f0-2.*f01)/(0.25*l*l);

//  double c2 = 0.5*fdd;                // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
//  double c1 = (f1-f0)/l;          //  the expansion of f at the center of (a,b)
//  double c0 = f01;

//  double x;

//  if(fabs(c2)<eps) x = -c0/c1;
//  else
//  {
//    if(f1<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
//    else     x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
//  }
//#ifdef CASL_THROWS
//  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
//#endif

//  if (x < -0.5*l) return (l-eps)/l;
//  if (x > 0.5*l) return (0.+eps)/l;

//  return 1.-(x+0.5*l)/l;
//}

//void simplex2_mls_quadratic_t::get_edge_coords(int e, double xyz[])
//{
//  vtx2_t *vtx0 = &vtxs[edgs[e].vtx0];
//  vtx2_t *vtx1 = &vtxs[edgs[e].vtx1];

//  xyz[0] = 0.5*(vtx0->x+vtx1->x);
//  xyz[1] = 0.5*(vtx0->y+vtx1->y);
//}

//double simplex2_mls_t::find_intersection_brent(int v0, int v1)
//{
//#ifdef CASL_THROWS
//  if(phi_cf == NULL) throw std::invalid_argument("[CASL_ERROR]: LSF is not provided.");
//#endif

//  vtx2_t *vtx0 = &vtxs[v0];
//  vtx2_t *vtx1 = &vtxs[v1];

//  double x0 = vtx0->x; double y0 = vtx0->y;
//  double x1 = vtx1->x; double y1 = vtx1->y;

//  double l = sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0));

//  double r_a = 0;
//  double r_b = 1;
//  double r_c = 0.5;
//  double r_s = 0.5;
//  double r_d = 0.5;

//  double f_a = phi_cf(x0 + r_a*(x1-x0), y0 + r_a*(y1-y0));
//  double f_b = phi_cf(x0 + r_b*(x1-x0), y0 + r_b*(y1-y0));
//  double f_c = 1.0;
//  double f_s = 1.0;
//  double f_d = 1.0;

//  double tmp = 0;
//  double tol = d_tol/l;

//#ifdef CASL_THROWS
//  if(f_a*f_b >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
//#endif

//  if (fabs(f_a) < fabs(f_b))
//  {
//    tmp = r_b; r_b = r_a; r_a = tmp;
//    tmp = f_b; f_b = f_a; f_a = tmp;
//  }

//  r_c = r_a; f_c = f_a;

//  bool mflag = true;

//  while (fabs(f_b) > eps && fabs(r_a-r_b) > tol)
//  {
//    f_c = phi_cf(x0 + r_c*(x1-x0), y0 + r_c*(y1-y0));

//    if (fabs(f_a-f_c) > eps && fabs(f_b-f_c) > eps)
//    {
//      r_s = r_a*f_b*f_c/(f_a-f_b)/(f_a-f_c) +
//            r_b*f_c*f_a/(f_b-f_c)/(f_b-f_a) +
//            r_c*f_a*f_b/(f_c-f_a)/(f_c-f_b);
//    } else {
//      r_s = r_b - f_b*(r_b-r_a)/(f_b-f_a);
//    }

//    if ( (r_s > b && r_s > (3.*r_a+r_b)/4.) ||
//         (r_s < b && r_s < (3.*r_a+r_b)/4.) ||
//         ( mflag && fabs(r_s-r_b) >= 0.5*fabs(r_b-r_c)) ||
//         (!mflag && fabs(r_s-r_b) >= 0.5*fabs(r_c-r_d)) ||
//         ( mflag && fabs(r_b-r_c) < tol) ||
//         (!mflag && fabs(r_c-r_d) < tol) )
//    {
//      r_s = 0.5*(r_a+r_b); mflag = true;
//    } else {
//      mflag = false;
//    }

//    f_s = phi_cf(x0 + r_s*(x1-x0), y0 + r_s*(y1-y0));

//    r_d = r_c; f_d = f_c;
//    r_c = r_b; f_c = f_b;

//    if (f_a*f_s < 0.) {r_b = r_s; f_b = f_s;}
//    else              {r_a = r_s; f_a = f_s;}

//    if (fabs(f_a) < fabs(f_b))
//    {
//      tmp = r_b; r_b = r_a; r_a = tmp;
//      tmp = f_b; f_b = f_a; f_a = tmp;
//    }
//  }

//  return r_b;
//}


//void simplex2_mls_quadratic_t::do_action(int cn, action_t action)
//{
//  /* Process elements */
//  int n;
//  n = vtxs.size(); for (int i = 0; i < n; i++) do_action_vtx(i, cn, action);
//  n = edgs.size(); for (int i = 0; i < n; i++) do_action_edg(i, cn, action);
//  n = tris.size(); for (int i = 0; i < n; i++) do_action_tri(i, cn, action);
//}


//double simplex2_mls_quadratic_t::integrate_in_non_cart_dir(double f0, double f1, double f2, int num)
//{
//  /* interpolate function values to vertices */
//  interpolate_all(f0, f1, f2);

//  double result = 0.0;

//  /* integrate over edges */
//  for (unsigned int i = 0; i < edgs.size(); i++)
//  {
//    edg2_t *e = &edgs[i];
//    if (!e->is_split && (e->loc == INS || e->loc == FCE))
//      if (e->p_lsf == num)
//        result += length(e->vtx0, e->vtx1)*(vtxs[e->vtx0].value + vtxs[e->vtx1].value)/2.0;
//  }

//  return result;
//}

double simplex2_mls_quadratic_t::distance(double x0, double y0, double x1, double y1, double x2, double y2)
{
  double val = pow(x1-x0,2.) + pow(y1-y0,2.) - pow( ((x1-x0)*(x1-x2) + (y1-y0)*(y1-y2)), 2. )/( pow(x1-x2,2.) + pow(y1-y2,2.) );

  if (val <0) std::cout << val << " : " << x0 << ", " << y0 << " : " << x1 << ", " << y1 << " : " << x2 << ", " << y2 << " Problems!\n";

  if (val <0) val = EPS; //std::cout << val << " Problems!\n";
  return sqrt(val);

}



