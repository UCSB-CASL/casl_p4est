#include "simplex3_mls_q.h"

//--------------------------------------------------
// Class Constructors
//--------------------------------------------------
simplex3_mls_q_t::simplex3_mls_q_t(double x0, double y0, double z0,
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
  // usually there will be only one cut
  vtxs_.reserve(20);
  edgs_.reserve(27);
  tris_.reserve(20);
  tets_.reserve(6);

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
  vtxs_.push_back(vtx3_t(x0,y0,z0));
  vtxs_.push_back(vtx3_t(x1,y1,z1));
  vtxs_.push_back(vtx3_t(x2,y2,z2));
  vtxs_.push_back(vtx3_t(x3,y3,z3));
  vtxs_.push_back(vtx3_t(x4,y4,z4));
  vtxs_.push_back(vtx3_t(x5,y5,z5));
  vtxs_.push_back(vtx3_t(x6,y6,z6));
  vtxs_.push_back(vtx3_t(x7,y7,z7));
  vtxs_.push_back(vtx3_t(x8,y8,z8));
  vtxs_.push_back(vtx3_t(x9,y9,z9));

  edgs_.push_back(edg3_t(0,4,1));
  edgs_.push_back(edg3_t(0,6,2));
  edgs_.push_back(edg3_t(0,7,3));
  edgs_.push_back(edg3_t(1,5,2));
  edgs_.push_back(edg3_t(1,8,3));
  edgs_.push_back(edg3_t(2,9,3));

  tris_.push_back(tri3_t(1,2,3,5,4,3));
  tris_.push_back(tri3_t(0,2,3,5,2,1));
  tris_.push_back(tri3_t(0,1,3,4,2,0));
  tris_.push_back(tri3_t(0,1,2,3,1,0));

  tets_.push_back(tet3_t(0,1,2,3,0,1,2,3));

  tris_[0].dir = 0;
  tris_[1].dir = 1;
  tris_[2].dir = 2;
  tris_[3].dir = 3;

  // pre-compute inverse matrix for mapping of the original simplex onto the reference simplex
  vtx3_t *v0 = &vtxs_[0];
  vtx3_t *v1 = &vtxs_[1];
  vtx3_t *v2 = &vtxs_[2];
  vtx3_t *v3 = &vtxs_[3];

  double A[9];
  A[3*0+0] = v1->x - v0->x; A[3*0+1] = v2->x - v0->x; A[3*0+2] = v3->x - v0->x;
  A[3*1+0] = v1->y - v0->y; A[3*1+1] = v2->y - v0->y; A[3*1+2] = v3->y - v0->y;
  A[3*2+0] = v1->z - v0->z; A[3*2+1] = v2->z - v0->z; A[3*2+2] = v3->z - v0->z;

  inv_mat3(A, map_parent_to_ref_);

  // compute resolution limit (all edges with legnth < 2*eps_ will be split at the middle)
  double l01 = length(0, 1);
  double l02 = length(0, 2);
  double l03 = length(0, 3);
  double l12 = length(1, 2);
  double l13 = length(1, 3);
  double l23 = length(2, 3);

  lmin_ = l01;
  lmin_ = lmin_ < l02 ? lmin_ : l02;
  lmin_ = lmin_ < l03 ? lmin_ : l03;
  lmin_ = lmin_ < l12 ? lmin_ : l12;
  lmin_ = lmin_ < l13 ? lmin_ : l13;
  lmin_ = lmin_ < l23 ? lmin_ : l23;

  eps_ = eps_rel_*lmin_;
}




//--------------------------------------------------
// Constructing domain
//--------------------------------------------------
void simplex3_mls_q_t::construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  num_phi_ = acn.size();

  if (clr.size() != num_phi_) std::invalid_argument("[CASL_ERROR]: (simplex3_mls_q_t) Numbers of actions and colors are not equal.");
  if (phi.size() != num_phi_*nodes_per_tet_) std::invalid_argument("[CASL_ERROR]: (simplex3_mls_q_t) Numbers of actions and colors are not equal.");

  bool needs_refinement = true;
  int last_vtxs_size = 0;

  int initial_refinement = 0;
  int n;

  std::vector<double> phi_current(nodes_per_tet_, -1);

  std::vector<vtx3_t> vtxs_initial = vtxs_;
  std::vector<edg3_t> edgs_initial = edgs_;
  std::vector<tri3_t> tris_initial = tris_;
  std::vector<tet3_t> tets_initial = tets_;

  while(1)
  {
    for (int i = 0; i < initial_refinement; ++i)
    {
      n = edgs_.size(); for (int i = 0; i < n; i++) refine_edg(i);
      n = tris_.size(); for (int i = 0; i < n; i++) refine_tri(i);
      n = tets_.size(); for (int i = 0; i < n; i++) refine_tet(i);
    }

    int refine_level = 0;

    // loop over LSFs
    for (short phi_idx = 0; phi_idx < num_phi_; ++phi_idx)
    {
      phi_max_ = 0;
      for (int i = 0; i < nodes_per_tet_; ++i)
      {
        vtxs_[i].value  = phi[nodes_per_tet_*phi_idx + i];
        phi_current[i] = phi[nodes_per_tet_*phi_idx + i];
        phi_max_ = phi_max_ > fabs(phi_current[i]) ? phi_max_ : fabs(phi_current[i]);
      }

      phi_eps_ = eps_rel_*phi_max_;

      for (int i = 0; i < nodes_per_tet_; ++i)
        perturb(vtxs_[i].value, phi_eps_);

      // compute curvature
      compute_curvature();

      last_vtxs_size = nodes_per_tet_;

      invalid_reconstruction_ = true;

      while (invalid_reconstruction_)
      {
        needs_refinement = true;

        while (needs_refinement)
        {
          // interpolate to all vertices
          for (int i = last_vtxs_size; i < vtxs_.size(); ++i)
            if (!vtxs_[i].is_recycled)
            {
              double xyz[3] = { vtxs_[i].x, vtxs_[i].y, vtxs_[i].z };
              vtxs_[i].value = interpolate_from_parent (phi_current, xyz );
              perturb(vtxs_[i].value, phi_eps_);
            }

          last_vtxs_size = vtxs_.size();

          // check validity of data on each edge
          needs_refinement = false;
          if (check_for_valid_data_)
          {
          int n = edgs_.size();
          for (int i = 0; i < n; ++i)
            if (!edgs_[i].is_split)
            {
              edg3_t *e = &edgs_[i];

              sort_edg(i);

              double phi0 = vtxs_[e->vtx0].value;
              double phi1 = vtxs_[e->vtx1].value;
              double phi2 = vtxs_[e->vtx2].value;

              if (!same_sign(phi0, phi1) && same_sign(phi0, phi2))
              {
                needs_refinement = true;
                e->to_refine = true;
                e->a = .5;
//                smart_refine_edg(i);
              }

              if (!e->to_refine && same_sign(phi0, phi2))
              {
                double c1 = -3.*phi0 + 4.*phi1 - phi2;
                double c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

                if (fabs(c1) > DBL_MIN && fabs(c1) < 2.*fabs(c2))
                {
                  double a_ext = -.5*c1/c2;

                  if (a_ext > 0. && a_ext < 1.)
                  {
                    double phi_ext = phi0 + c1*a_ext + c2*a_ext*a_ext;

                    if (!same_sign(phi0, phi_ext))
                    {
                      needs_refinement = true;
                      e->to_refine = true;
                      e->a = need_swap(e->vtx0, e->vtx2) ? 1.-a_ext : a_ext;
//                      smart_refine_edg(i);
                    }
                  }
                }
              }
            }

          // check validity of data on each face
          n = tris_.size();
//          if (!needs_refinement)
            for (int i = 0; i < n; ++i)
              if (!tris_[i].is_split)
              {
                tri3_t *f = &tris_[i];

                if (need_swap(f->vtx0, f->vtx1)) {swap(f->vtx0, f->vtx1); swap(f->edg0, f->edg1);}
                if (need_swap(f->vtx1, f->vtx2)) {swap(f->vtx1, f->vtx2); swap(f->edg1, f->edg2);}
                if (need_swap(f->vtx0, f->vtx1)) {swap(f->vtx0, f->vtx1); swap(f->edg0, f->edg1);}

                edg3_t *e0 = &edgs_[f->edg0];
                edg3_t *e1 = &edgs_[f->edg1];
                edg3_t *e2 = &edgs_[f->edg2];

                if (!e0->to_refine &&
                    !e1->to_refine &&
                    !e2->to_refine )
                {

                  double phi0 = vtxs_[f->vtx0].value;
                  double phi1 = vtxs_[f->vtx1].value;
                  double phi2 = vtxs_[f->vtx2].value;
                  double phi3 = vtxs_[e2->vtx1].value;
                  double phi4 = vtxs_[e0->vtx1].value;
                  double phi5 = vtxs_[e1->vtx1].value;

                  if (same_sign(phi0, phi1) && same_sign(phi1, phi2))
                  {
                    double paa = 2.*phi0 + 2.*phi1 - 4.*phi3;
                    double pab = 4.*phi0 - 4.*phi3 + 4.*phi4 - 4.*phi5;
                    double pbb = 2.*phi0 + 2.*phi2 - 4.*phi5;

                    double pa = -3.*phi0 - phi1 + 4.*phi3;
                    double pb = -3.*phi0 - phi2 + 4.*phi5;

                    double det = 4.*paa*pbb - pab*pab;
                    double a = (pb*pab - 2.*pa*pbb);
                    double b = (pa*pab - 2.*pb*paa);

                    if (fabs(a) > DBL_MIN && fabs(b) > DBL_MIN && fabs(a+b) < fabs(det))
                    {
                      // calculate critical point
                      a /= det;
                      b /= det;

                      if (a > DBL_MIN && b > DBL_MIN && a+b < 1.)
                      {
                        // calculate value of a level-set function at the found critical point
                        double phi_extremum = paa*a*a + pab*a*b + pbb*b*b + pa*a + pb*b + phi0;

                        // check whether the level-set function changes sign
                        if (!same_sign(phi_extremum, phi0))
                        {
                          needs_refinement = true;
                          f->to_refine = true;
                          f->a = a;
                          f->b = b;
//                          smart_refine_tri(i, a, b);
                          std::cout << "Face refinement!\n";
                        }
                      }
                    }

                  }
                }
              }
          }

          // refine if necessary
          if (needs_refinement && refine_level < max_refinement_ - initial_refinement)
          {
            for (int i = 0; i < edgs_.size(); i++) smart_refine_edg(i);
            for (int i = 0; i < tris_.size(); i++) smart_refine_tri(i);
            for (int i = 0; i < tets_.size(); i++) smart_refine_tet(i);
            refine_level++;
          } else if (needs_refinement) {
            std::cout << "Cannot resolve invalid geometry (bad)\n";
            needs_refinement = false;
          }
        }

        invalid_reconstruction_ = false;

        vtxs_tmp_ = vtxs_;
        edgs_tmp_ = edgs_;
        tris_tmp_ = tris_;
        tets_tmp_ = tets_;

        int n;
        n = vtxs_.size(); for (int i = 0; i < n; i++) { do_action_vtx(i, clr[phi_idx], acn[phi_idx]); }
        n = edgs_.size(); for (int i = 0; i < n; i++) { do_action_edg(i, clr[phi_idx], acn[phi_idx]); }
        n = tris_.size(); for (int i = 0; i < n; i++) { do_action_tri(i, clr[phi_idx], acn[phi_idx]); }
//        std::cout << "[ERROR]: here "  << refine_level << "\n";
        n = tets_.size(); for (int i = 0; i < n; i++) { do_action_tet(i, clr[phi_idx], acn[phi_idx]); }

        if (invalid_reconstruction_ && refine_level < max_refinement_ - initial_refinement)
        {
          vtxs_ = vtxs_tmp_;
          edgs_ = edgs_tmp_;
          tris_ = tris_tmp_;
          tets_ = tets_tmp_;

          for (int i = 0; i < edgs_.size(); i++) smart_refine_edg(i);
          for (int i = 0; i < tris_.size(); i++) smart_refine_tri(i);
          for (int i = 0; i < tets_.size(); i++) smart_refine_tet(i);

          refine_level++;
        } else {
          if (invalid_reconstruction_)
          {
            std::cout << initial_refinement << "Cannot resolve invalid geometry\n";
            if (initial_refinement < 3)
            {
              ++initial_refinement;
              vtxs_ = vtxs_initial;
              edgs_ = edgs_initial;
              tris_ = tris_initial;
              tets_ = tets_initial;
              break;
            } else {
              invalid_reconstruction_ = false;
            }
          }
//          invalid_reconstruction_ = false;
        }
      }
      if (invalid_reconstruction_) break;
    }

    if (invalid_reconstruction_) continue;


    // sort everything before integration
    for (int i = 0; i < edgs_.size(); i++)
    {
      edg3_t *edg = &edgs_[i];
      if (need_swap(edg->vtx0, edg->vtx2)) { swap(edg->vtx0, edg->vtx2); }
    }

    for (int i = 0; i < tris_.size(); i++)
    {
      tri3_t *tri = &tris_[i];
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); swap(tri->c_vtx12, tri->c_vtx02); swap(tri->ab12, tri->ab02); }
      if (need_swap(tri->vtx1, tri->vtx2)) { swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2); swap(tri->c_vtx02, tri->c_vtx01); swap(tri->ab02, tri->ab01); }
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); swap(tri->c_vtx12, tri->c_vtx02); swap(tri->ab12, tri->ab02); }
    }

    for (int i = 0; i < tets_.size(); i++)
    {
      tet3_t *tet = &tets_[i];
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
      double v_before = volume(tets_[0].vtx0, tets_[0].vtx1, tets_[0].vtx2, tets_[0].vtx3);
      double v_after  = 0;

      // compute volume after using linear representation
      for (int i = 0; i < tets_.size(); ++i)
        if (!tets_[i].is_split)
          v_after += volume(tets_[i].vtx0, tets_[i].vtx1, tets_[i].vtx2, tets_[i].vtx3);

      if (fabs(v_before-v_after) > 1.e-15)
      {
        if (initial_refinement == max_refinement_)
        {
          std::cout << "Can't resolve overlapping " << fabs(v_before-v_after) << "\n";
          break;
        } else {
          ++initial_refinement;
          std::cout << "Overlapping " << fabs(v_before-v_after) << "\n";
          vtxs_ = vtxs_initial;
          edgs_ = edgs_initial;
          tris_ = tris_initial;
          tets_ = tets_initial;
//          break;
        }
      } else {
        break;
      }
    } else {
      break;
    }
  }

  vtxs_tmp_.clear();
  edgs_tmp_.clear();
  tris_tmp_.clear();
  tets_tmp_.clear();
}




//--------------------------------------------------
// Splitting
//--------------------------------------------------
void simplex3_mls_q_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx3_t *vtx = &vtxs_[n_vtx];

  if (vtx->is_recycled) return;

  switch (action)
  {
    case INTERSECTION:  if (vtx->value > 0)                                                       vtx->set(OUT, -1, -1, -1);  break;
    case ADDITION:      if (vtx->value < 0)                                                       vtx->set(INS, -1, -1, -1);  break;
    case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==LNE || vtx->loc==PNT)  vtx->set(FCE, cn, -1, -1);  break;
  }
}

void simplex3_mls_q_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg3_t *edg = &edgs_[n_edg];

  int c0 = edg->c0;
  int c1 = edg->c1;

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  int num_negatives = 0;
  if (vtxs_[edg->vtx0].value < 0) num_negatives++;
  if (vtxs_[edg->vtx2].value < 0) num_negatives++;

#ifdef SIMPLEX3_MLS_Q_DEBUG
  edg->type = num_negatives;
#endif

  // auxiliary variables
  vtx3_t *c_vtx_x, *c_vtx_0x, *c_vtx_x2;
  edg3_t *c_edg0, *c_edg1;

  switch (num_negatives)
  {
    case 0: // ++
      /* split an edge */
      // no need to split

      /* apply rules */
      switch (action)
      {
        case INTERSECTION:  edg->set(OUT,-1,-1);  break;
        case ADDITION:      /* do nothig */       break;
        case COLORATION:    /* do nothig */       break;
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
      double xyz_0[3];
      double xyz_m[3];
      double xyz_p[3];
      mapping_edg(xyz_0, n_edg, a);
      mapping_edg(xyz_m, n_edg, .5*a);
      mapping_edg(xyz_p, n_edg, a + .5*(1.-a));

      // create new vertices
      vtxs_.push_back(vtx3_t(xyz_m[0], xyz_m[1], xyz_m[2])); int n_vtx_0x = vtxs_.size()-1;
      vtxs_.push_back(vtx3_t(xyz_p[0], xyz_p[1], xyz_p[2])); int n_vtx_x2 = vtxs_.size()-1;
      vtxs_.push_back(vtx3_t(xyz_0[0], xyz_0[1], xyz_0[2]));

      edg->c_vtx_x = vtxs_.size()-1;

      // new edges
      edgs_.push_back(edg3_t(edg->vtx0,    n_vtx_0x, edg->c_vtx_x)); edg = &edgs_[n_edg]; // edges might have changed their addresses
      edgs_.push_back(edg3_t(edg->c_vtx_x, n_vtx_x2, edg->vtx2   )); edg = &edgs_[n_edg];

      edg->c_edg0 = edgs_.size()-2;
      edg->c_edg1 = edgs_.size()-1;

      /* apply rules */
      c_vtx_x  = &vtxs_[edg->c_vtx_x];
      c_vtx_0x = &vtxs_[n_vtx_0x];
      c_vtx_x2 = &vtxs_[n_vtx_x2];

      c_edg0  = &edgs_[edg->c_edg0];
      c_edg1  = &edgs_[edg->c_edg1];

#ifdef SIMPLEX3_MLS_Q_DEBUG
      c_vtx_x->p_edg = n_edg;
      c_edg0->p_edg  = n_edg;
      c_edg1->p_edg  = n_edg;
#endif

      switch (action)
      {
        case INTERSECTION:
          switch (edg->loc)
          {
            case OUT: c_vtx_x->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_vtx_0x->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
            case INS: c_vtx_x->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
            case FCE: c_vtx_x->set(LNE, c0, cn, -1); c_edg0->set(FCE, c0, -1); c_vtx_0x->set(FCE, c0, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1); if (c0==cn)            c_vtx_x->set(FCE, c0, -1, -1);  break;
            case LNE: c_vtx_x->set(PNT, c0, c1, cn); c_edg0->set(LNE, c0, c1); c_vtx_0x->set(LNE, c0, c1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1); if (c0==cn || c1==cn)  c_vtx_x->set(LNE, c0, c1, -1);  break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          }
          break;
        case ADDITION:
          switch (edg->loc)
          {
            case OUT: c_vtx_x->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
            case INS: c_vtx_x->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); c_vtx_x2->set(INS, -1, -1, -1);                                                        break;
            case FCE: c_vtx_x->set(LNE, c0, cn, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(FCE, c0, -1); c_vtx_x2->set(FCE, c0, -1, -1); if (c0==cn)            c_vtx_x->set(FCE, c0, -1, -1);  break;
            case LNE: c_vtx_x->set(PNT, c0, c1, cn); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(LNE, c0, c1); c_vtx_x2->set(LNE, c0, c1, -1); if (c0==cn || c1==cn)  c_vtx_x->set(LNE, c0, c1, -1);  break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          }
          break;
        case COLORATION:
          switch (edg->loc)
          {
            case OUT: c_vtx_x->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_vtx_0x->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); c_vtx_x2->set(OUT, -1, -1, -1);                                                        break;
            case INS: c_vtx_x->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_vtx_0x->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); c_vtx_x2->set(INS, -1, -1, -1);                                                        break;
            case FCE: c_vtx_x->set(LNE, c0, cn, -1); c_edg0->set(FCE, cn, -1); c_vtx_0x->set(FCE, cn, -1, -1); c_edg1->set(FCE, c0, -1); c_vtx_x2->set(FCE, c0, -1, -1); if (c0==cn)            c_vtx_x->set(FCE, c0, -1, -1);  break;
            case LNE: c_vtx_x->set(PNT, c0, c1, cn); c_edg0->set(FCE, cn, -1); c_vtx_0x->set(FCE, cn, -1, -1); c_edg1->set(LNE, c0, c1); c_vtx_x2->set(LNE, c0, c1, -1); if (c0==cn || c1==cn)  c_vtx_x->set(LNE, c0, c1, -1);  break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
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
      switch (action)
      {
        case INTERSECTION:  /* do nothing */                                            break;
        case ADDITION:                                          edg->set(INS, -1, -1);  break;
        case COLORATION:    if (edg->loc==FCE || edg->loc==LNE) edg->set(FCE, cn, -1);  break;
      }
      break;
  }
}

void simplex3_mls_q_t::do_action_tri(int n_tri, int cn, action_t action)
{
  tri3_t *tri = &tris_[n_tri];

  if (tri->is_split) return;

  int cc = tri->c;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Determine type */
  int num_negatives = 0;
  if (vtxs_[tri->vtx0].value < 0) num_negatives++;
  if (vtxs_[tri->vtx1].value < 0) num_negatives++;
  if (vtxs_[tri->vtx2].value < 0) num_negatives++;

#ifdef SIMPLEX3_MLS_Q_DEBUG
  tri->type = num_negatives;

  /* check whether vertices of a triangle and edges coincide */
  if (tri->vtx0 != edgs_[tri->edg1].vtx0 || tri->vtx0 != edgs_[tri->edg2].vtx0 ||
      tri->vtx1 != edgs_[tri->edg0].vtx0 || tri->vtx1 != edgs_[tri->edg2].vtx2 ||
      tri->vtx2 != edgs_[tri->edg0].vtx2 || tri->vtx2 != edgs_[tri->edg1].vtx2)
  {
    std::cout << vtxs_[tri->vtx0].value << " " << vtxs_[tri->vtx1].value << " " << vtxs_[tri->vtx2].value << std::endl;
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Vertices of a triangle and edges do not coincide after sorting.");
  }

  /* check whether appropriate edges have been splitted */
  int e0_type_expect, e1_type_expect, e2_type_expect;

  switch (num_negatives)
  {
    case 0: e0_type_expect = 0; e1_type_expect = 0; e2_type_expect = 0; break;
    case 1: e0_type_expect = 0; e1_type_expect = 1; e2_type_expect = 1; break;
    case 2: e0_type_expect = 1; e1_type_expect = 1; e2_type_expect = 2; break;
    case 3: e0_type_expect = 2; e1_type_expect = 2; e2_type_expect = 2; break;
  }

  if (edgs_[tri->edg0].type != e0_type_expect || edgs_[tri->edg1].type != e1_type_expect || edgs_[tri->edg2].type != e2_type_expect)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a triangle one of edges has an unexpected type.");
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
      switch (action)
      {
        case INTERSECTION:  tri->set(OUT, -1);  break;
        case ADDITION:      /* do nothing */    break;
        case COLORATION:    /* do nothing */    break;
      }
      break;

    case 1: // (-++)
      {
        /* split a triangle */
        tri->is_split = true;

        // new vertices
        tri->c_vtx01 = edgs_[tri->edg2].c_vtx_x;
        tri->c_vtx02 = edgs_[tri->edg1].c_vtx_x;

        double length_edg = length(tri->c_vtx01, tri->c_vtx02);

        // coordinates of new vertices in reference element
        double abc_v01[] = { edgs_[tri->edg2].a, 0. };
        double abc_v02[] = { 0., edgs_[tri->edg1].a };

        // midpoint along interface
        double abc_u0_lin[2] = { .5*(abc_v01[0] + abc_v02[0]), .5*(abc_v01[1] + abc_v02[1]) };
        double abc_u0[2];
        double t[2];
        bool reconstruction_is_good = true;

        if (length_edg > eps_)
          reconstruction_is_good = find_middle_node(abc_u0, abc_v02, abc_v01, n_tri, t);
        else
        {
          abc_u0[0] = abc_u0_lin[0];
          abc_u0[1] = abc_u0_lin[1];
        }

        // midpoint of the auxiliary edge
        double abc_u1[2] = { 0.5*edgs_[tri->edg2].a, 0.5 };

        // slightly move the midpoint of the auxiliary edge based on the deformation of a quadrilateral
        // to reduce probability of crossing edges
        if (adjust_auxiliary_midpoint_)
        {
          double *quad_node0   = abc_v01;
          double *quad_node1   = abc_v02;
          double  quad_node2[] = { 0., 1. };
          double  quad_node3[] = { 1., 0. };
          double *quad_node4   = abc_u0;

          adjust_middle_node(abc_u1, abc_u1, quad_node0, quad_node1, quad_node2, quad_node3, quad_node4);
        }

        // map midpoints to physical space
        double xyz_u0[3]; mapping_tri(xyz_u0, n_tri, abc_u0);
        double xyz_u1[3]; mapping_tri(xyz_u1, n_tri, abc_u1);

        // check for an intersection with an auxiliary straight edge
        if (check_for_edge_intersections_ && reconstruction_is_good && length_edg > eps_)
        {
          // interpolate level-set function into the new point
          double phi1 = interpolate_from_parent(xyz_u1);
          double phi2 = vtxs_[tri->vtx2].value;

          // calculate slope at the endpoint where level-set function is zero
          double c1 = 4.*phi1 - phi2;

          // and check whether the slope and the value at the other end of the same sign
          if (c1*phi2 < 0)
          {
            reconstruction_is_good = false;

            // use linear recontruction in case the max level of refinement is reached
            mapping_tri(xyz_u0, n_tri, abc_u0_lin);
          }
        }

        // new vertices
        vtxs_.push_back(vtx3_t(xyz_u0[0], xyz_u0[1], xyz_u0[2]));
        vtxs_.push_back(vtx3_t(xyz_u1[0], xyz_u1[1], xyz_u1[2]));

        int u0 = vtxs_.size()-2;
        int u1 = vtxs_.size()-1;

        // check if deformation is not too high
        if (check_for_curvature_ && reconstruction_is_good && length_edg > eps_)
        {
          // compute curvature of the curved edge
          vtx3_t *v0 = &vtxs_[tri->c_vtx01];
          vtx3_t *v1 = &vtxs_[u0];
          vtx3_t *v2 = &vtxs_[tri->c_vtx02];

          double xa = v2->x - v0->x;
          double ya = v2->y - v0->y;
          double za = v2->z - v0->z;

          double max_x = fabs(v0->x) < fabs(v1->x) ? (fabs(v1->x) < fabs(v2->x) ? fabs(v2->x) : fabs(v1->x)) :
                                                     (fabs(v0->x) < fabs(v2->x) ? fabs(v2->x) : fabs(v0->x)) ;
          double max_y = fabs(v0->y) < fabs(v1->y) ? (fabs(v1->y) < fabs(v2->y) ? fabs(v2->y) : fabs(v1->y)) :
                                                     (fabs(v0->y) < fabs(v2->y) ? fabs(v2->y) : fabs(v0->y)) ;
          double max_z = fabs(v0->z) < fabs(v1->z) ? (fabs(v1->z) < fabs(v2->z) ? fabs(v2->z) : fabs(v1->z)) :
                                                     (fabs(v0->z) < fabs(v2->z) ? fabs(v2->z) : fabs(v0->z)) ;

          double xaa = 4.*(v0->x - 2.*v1->x + v2->x); if (fabs(xaa) < eps_*max_x) xaa = 0;
          double yaa = 4.*(v0->y - 2.*v1->y + v2->y); if (fabs(yaa) < eps_*max_y) yaa = 0;
          double zaa = 4.*(v0->z - 2.*v1->z + v2->z); if (fabs(zaa) < eps_*max_z) zaa = 0;

          double kappa_edg = fabs( sqrt( pow(zaa*ya-yaa*za, 2.) +
                                         pow(xaa*za-zaa*xa, 2.) +
                                         pow(yaa*xa-xaa*ya, 2.) )
                                   / pow( xa*xa + ya*ya + za*za , 1.5) );

          if (kappa_edg*length_edg > kappa_scale_*kappa_*lmin_ && kappa_edg*length_edg > kappa_eps_)
          {
            reconstruction_is_good = false;

          }
        }

        // refine edges if any of the above tests failed
        if (!reconstruction_is_good && try_to_fix_outside_vertices_)
        {
          invalid_reconstruction_ = true;
          bool at_least_one = false;

          // split edges of the triangle by a straight line that is perpendicular to the linear
          // representation and crosses it in the middle
          if (refine_in_normal_dir_)
          {
            double A, B;
            A = 0; B = 0; double phi_line_0 = (A-abc_u0_lin[0])*t[0] + (B-abc_u0_lin[1])*t[1];
            A = 1; B = 0; double phi_line_1 = (A-abc_u0_lin[0])*t[0] + (B-abc_u0_lin[1])*t[1];
            A = 0; B = 1; double phi_line_2 = (A-abc_u0_lin[0])*t[0] + (B-abc_u0_lin[1])*t[1];

            for (int i = 0; i < 3; ++i)
            {
              double p0, p1;
              int edg_idx;

              switch(i)
              {
                case 0: p0 = phi_line_1; p1 = phi_line_2; edg_idx = tri->edg0; break;
                case 1: p0 = phi_line_0; p1 = phi_line_2; edg_idx = tri->edg1; break;
                case 2: p0 = phi_line_0; p1 = phi_line_1; edg_idx = tri->edg2; break;
                default: throw;
              }

              if (p0*p1 < 0)
              {
                double root = fabs(p0)/fabs(p0-p1);
                if (not_finite(root))
                  throw;
                // snap intersection to an existing vertex if too close
                if (root > snap_limit_ && root < 1.-snap_limit_)
                {
                  edgs_tmp_[edg_idx].to_refine = true;
                  edgs_tmp_[edg_idx].a = root;
                  at_least_one = true;
                }
              }
            }
          }

          // if the above refinement failed, split triangle by a line passing through
          // vertex no. 0 (a=b=0) and the midpoint of the curved edge
          if (!at_least_one)
          {
            edgs_tmp_[tri->edg0].to_refine = true;
            edgs_tmp_[tri->edg0].a = abc_u0[1] / (abc_u0[0]+abc_u0[1]);
            if (not_finite(edgs_tmp_[tri->edg0].a))
              throw;
          }
        }

        // new edges
        edgs_.push_back(edg3_t(tri->c_vtx01, u0, tri->c_vtx02));
        edgs_.push_back(edg3_t(tri->c_vtx01, u1, tri->vtx2   ));

        // edges might have changed their addresses
        edg0 = &edgs_[tri->edg0];
        edg1 = &edgs_[tri->edg1];
        edg2 = &edgs_[tri->edg2];

        tri->c_edg0 = edgs_.size()-2;
        tri->c_edg1 = edgs_.size()-1;

        // new triangles
        tris_.push_back(tri3_t(tri->vtx0,    tri->c_vtx01, tri->c_vtx02, tri->c_edg0,  edg1->c_edg0, edg2->c_edg0)); tri = &tris_[n_tri];
        tris_.push_back(tri3_t(tri->c_vtx01, tri->c_vtx02, tri->vtx2,    edg1->c_edg1, tri->c_edg1,  tri->c_edg0));  tri = &tris_[n_tri];
        tris_.push_back(tri3_t(tri->c_vtx01, tri->vtx1,    tri->vtx2,    tri->edg0,    tri->c_edg1,  edg2->c_edg1)); tri = &tris_[n_tri];

        tri->c_tri0 = tris_.size()-3;
        tri->c_tri1 = tris_.size()-2;
        tri->c_tri2 = tris_.size()-1;

        /* apply rules */
        c_edg0 = &edgs_[tri->c_edg0];
        c_edg1 = &edgs_[tri->c_edg1];

        c_tri0 = &tris_[tri->c_tri0];  c_tri0->dir = tri->dir; c_tri0->p_lsf = tri->p_lsf;
        c_tri1 = &tris_[tri->c_tri1];  c_tri1->dir = tri->dir; c_tri1->p_lsf = tri->p_lsf;
        c_tri2 = &tris_[tri->c_tri2];  c_tri2->dir = tri->dir; c_tri2->p_lsf = tri->p_lsf;

        vtx_u0 = &vtxs_[u0];
        vtx_u1 = &vtxs_[u1];

#ifdef SIMPLEX3_MLS_Q_DEBUG
        if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
          throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a triangle one of child triangles is not consistent.");

        // track the parent
        c_edg0->p_tri = n_tri;
        c_edg1->p_tri = n_tri;
        c_tri0->p_tri = n_tri;
        c_tri1->p_tri = n_tri;
        c_tri2->p_tri = n_tri;
#endif

        switch (action)
        {
          case INTERSECTION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
              case INS: c_edg0->set(FCE, cn, -1); vtx_u0->set(FCE, cn, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
              case FCE: c_edg0->set(LNE, cc, cn); vtx_u0->set(LNE, cc, cn, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(FCE, cc); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
              default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
          case ADDITION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(FCE, cn, -1); vtx_u0->set(FCE, cn, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
              case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
              case FCE: c_edg0->set(LNE, cc, cn); vtx_u0->set(LNE, cc, cn, -1); c_edg1->set(FCE, cc, -1); vtx_u1->set(FCE, cc, -1, -1); c_tri0->set(INS, -1); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
              default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
          case COLORATION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
              case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
              case FCE: c_edg0->set(LNE, cc, cn); vtx_u0->set(LNE, cc, cn, -1); c_edg1->set(FCE, cc, -1); vtx_u1->set(FCE, cc, -1, -1); c_tri0->set(FCE, cn); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
              default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
        }
        break;
      }
    case 2: // (--+)
      {
        /* split a triangle */
        tri->is_split = true;

        // new vertices
        tri->c_vtx02 = edgs_[tri->edg1].c_vtx_x;
        tri->c_vtx12 = edgs_[tri->edg0].c_vtx_x;

        double length_edg = length(tri->c_vtx02, tri->c_vtx12);

        // coordinates of new vertices in reference element
        double abc_v02[] = { 0.,                    edgs_[tri->edg1].a };
        double abc_v12[] = { 1.-edgs_[tri->edg0].a, edgs_[tri->edg0].a };

        // midpoint along interface
        double abc_u1_lin[2] = { .5*(abc_v02[0] + abc_v12[0]), .5*(abc_v02[1] + abc_v12[1]) };
        double abc_u1[2];
        double t[2];
        bool reconstruction_is_good = true;

        if (length_edg > eps_)
          reconstruction_is_good = find_middle_node(abc_u1, abc_v12, abc_v02, n_tri, t);
        else
        {
          abc_u1[0] = abc_u1_lin[0];
          abc_u1[1] = abc_u1_lin[1];
        }

        // midpoint of the auxiliary edge
        double abc_u0[] = { 0.5*(1.-edgs_[tri->edg0].a), 0.5*edgs_[tri->edg0].a };

        // slightly move the midpoint of the auxiliary edge based on the deformation of a quadrilateral
        // to reduce probability of crossing edges
        if (adjust_auxiliary_midpoint_)
        {
          double quad_node0[] = { abc_v12[0], abc_v12[1] };
          double quad_node1[] = { abc_v02[0], abc_v02[1] };
          double quad_node2[] = { 0., 0. };
          double quad_node3[] = { 1., 0. };
          double quad_node4[] = { abc_u1[0], abc_u1[1] };

          adjust_middle_node(abc_u0, abc_u0, quad_node0, quad_node1, quad_node2, quad_node3, quad_node4);
        }

        // map midpoints to physical space
        double xyz_u0[3]; mapping_tri(xyz_u0, n_tri, abc_u0);
        double xyz_u1[3]; mapping_tri(xyz_u1, n_tri, abc_u1);

        // check for an intersection with an auxiliary straight edge
        if (check_for_edge_intersections_ && reconstruction_is_good && length_edg > eps_)
        {
          // interpolate level-set function into the new point
          double phi1 = interpolate_from_parent(xyz_u0);
          double phi2 = vtxs_[tri->vtx0].value;

          // calculate slope at the endpoint where level-set function is zero
          double c1 = 4.*phi1 - phi2;

          // and check whether the slope and the value at the other end of the same sign
          if (c1*phi2 < 0)
          {
            reconstruction_is_good = false;

            // use linear recontruction in case the max level of refinement is reached
            mapping_tri(xyz_u1, n_tri, abc_u1_lin);
          }
        }

        // new vertices
        vtxs_.push_back(vtx3_t(xyz_u0[0], xyz_u0[1], xyz_u0[2]));
        vtxs_.push_back(vtx3_t(xyz_u1[0], xyz_u1[1], xyz_u1[2]));

        int u0 = vtxs_.size()-2;
        int u1 = vtxs_.size()-1;

        // check if deformation is not too high
        if (check_for_curvature_ && reconstruction_is_good && length_edg > eps_)
        {
          // compute curvature
          vtx3_t *v0 = &vtxs_[tri->c_vtx02];
          vtx3_t *v1 = &vtxs_[u1];
          vtx3_t *v2 = &vtxs_[tri->c_vtx12];

          double xa = v2->x - v0->x;
          double ya = v2->y - v0->y;
          double za = v2->z - v0->z;

          double max_x = fabs(v0->x) < fabs(v1->x) ? (fabs(v1->x) < fabs(v2->x) ? fabs(v2->x) : fabs(v1->x)) :
                                                     (fabs(v0->x) < fabs(v2->x) ? fabs(v2->x) : fabs(v0->x)) ;
          double max_y = fabs(v0->y) < fabs(v1->y) ? (fabs(v1->y) < fabs(v2->y) ? fabs(v2->y) : fabs(v1->y)) :
                                                     (fabs(v0->y) < fabs(v2->y) ? fabs(v2->y) : fabs(v0->y)) ;
          double max_z = fabs(v0->z) < fabs(v1->z) ? (fabs(v1->z) < fabs(v2->z) ? fabs(v2->z) : fabs(v1->z)) :
                                                     (fabs(v0->z) < fabs(v2->z) ? fabs(v2->z) : fabs(v0->z)) ;

          double xaa = 4.*(v0->x - 2.*v1->x + v2->x); if (fabs(xaa) < eps_*max_x) xaa = 0;
          double yaa = 4.*(v0->y - 2.*v1->y + v2->y); if (fabs(yaa) < eps_*max_y) yaa = 0;
          double zaa = 4.*(v0->z - 2.*v1->z + v2->z); if (fabs(zaa) < eps_*max_z) zaa = 0;

          double kappa_edg = fabs( sqrt( pow(zaa*ya-yaa*za, 2.) +
                                         pow(xaa*za-zaa*xa, 2.) +
                                         pow(yaa*xa-xaa*ya, 2.) )
                                   / pow( xa*xa + ya*ya + za*za , 1.5) );

          if (kappa_edg*length_edg > kappa_scale_*kappa_*lmin_ && kappa_edg*length_edg > kappa_eps_)
          {
            reconstruction_is_good = false;
          }
        }

        // refine edges if any of the above tests were failed
        if (!reconstruction_is_good && try_to_fix_outside_vertices_)
        {
          invalid_reconstruction_ = true;
          bool at_least_one = false;

          // split edges of the triangle by a straight line that is perpendicular to the linear
          // representation and crosses it in the middle
          if (refine_in_normal_dir_)
          {
            double A, B;
            A = 0; B = 0; double phi_line_0 = (A-abc_u1_lin[0])*t[0] + (B-abc_u1_lin[1])*t[1];
            A = 1; B = 0; double phi_line_1 = (A-abc_u1_lin[0])*t[0] + (B-abc_u1_lin[1])*t[1];
            A = 0; B = 1; double phi_line_2 = (A-abc_u1_lin[0])*t[0] + (B-abc_u1_lin[1])*t[1];

            for (int i = 0; i < 3; ++i)
            {
              double p0, p1;
              int edg_idx;

              switch(i)
              {
                case 0: p0 = phi_line_1; p1 = phi_line_2; edg_idx = tri->edg0; break;
                case 1: p0 = phi_line_0; p1 = phi_line_2; edg_idx = tri->edg1; break;
                case 2: p0 = phi_line_0; p1 = phi_line_1; edg_idx = tri->edg2; break;
                default: throw;
              }

              if (p0*p1 < 0)
              {
                double root = fabs(p0)/fabs(p0-p1);
                if (not_finite(root))
                  throw;
                // snap intersection to an existing vertex if too close
                if (root > snap_limit_ && root < 1.-snap_limit_)
                {
                  edgs_tmp_[edg_idx].to_refine = true;
                  edgs_tmp_[edg_idx].a = root;
                  at_least_one = true;
                }
              }
            }
          }

          // if the above refinement failed, split triangle by a line passing through
          // vertex no. 0 (a=b=0) and the midpoint of the curved edge
          if (!at_least_one)
          {
            edgs_tmp_[tri->edg2].to_refine = true;
            edgs_tmp_[tri->edg2].a = abc_u1[0]/(1.-abc_u1[1]);
            if (not_finite(edgs_tmp_[tri->edg2].a))
              throw;
          }
        }

        // create new edges
        edgs_.push_back(edg3_t(tri->vtx0,    u0, tri->c_vtx12));
        edgs_.push_back(edg3_t(tri->c_vtx02, u1, tri->c_vtx12));

        // edges might have changed their addresses
        edg0 = &edgs_[tri->edg0];
        edg1 = &edgs_[tri->edg1];
        edg2 = &edgs_[tri->edg2];

        tri->c_edg0 = edgs_.size()-2;
        tri->c_edg1 = edgs_.size()-1;

        tris_.push_back(tri3_t(tri->vtx0,    tri->vtx1,    tri->c_vtx12, edg0->c_edg0, tri->c_edg0,  tri->edg2   )); tri = &tris_[n_tri];
        tris_.push_back(tri3_t(tri->vtx0,    tri->c_vtx02, tri->c_vtx12, tri->c_edg1,  tri->c_edg0,  edg1->c_edg0)); tri = &tris_[n_tri];
        tris_.push_back(tri3_t(tri->c_vtx02, tri->c_vtx12, tri->vtx2,    edg0->c_edg1, edg1->c_edg1, tri->c_edg1 )); tri = &tris_[n_tri];

        tri->c_tri0 = tris_.size()-3;
        tri->c_tri1 = tris_.size()-2;
        tri->c_tri2 = tris_.size()-1;

        /* apply rules */
        c_edg0 = &edgs_[tri->c_edg0];
        c_edg1 = &edgs_[tri->c_edg1];

        c_tri0 = &tris_[tri->c_tri0];  c_tri0->dir = tri->dir; c_tri0->p_lsf = tri->p_lsf;
        c_tri1 = &tris_[tri->c_tri1];  c_tri1->dir = tri->dir; c_tri1->p_lsf = tri->p_lsf;
        c_tri2 = &tris_[tri->c_tri2];  c_tri2->dir = tri->dir; c_tri2->p_lsf = tri->p_lsf;

        vtx_u0 = &vtxs_[u0];
        vtx_u1 = &vtxs_[u1];

#ifdef SIMPLEX3_MLS_Q_DEBUG
        if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
          throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a triangle one of child triangles is not consistent.");

        // track the parent
        c_edg0->p_tri = n_tri;
        c_edg1->p_tri = n_tri;
        c_tri0->p_tri = n_tri;
        c_tri1->p_tri = n_tri;
        c_tri2->p_tri = n_tri;
#endif

        switch (action)
        {
          case INTERSECTION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
              case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(FCE, cn, -1); vtx_u1->set(FCE, cn, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
              case FCE: c_edg0->set(FCE, cc, -1); vtx_u0->set(FCE, cc, -1, -1); c_edg1->set(LNE, cc, cn); vtx_u1->set(LNE, cc, cn, -1); c_tri0->set(FCE, cc); c_tri1->set(FCE, cc); c_tri2->set(OUT, -1); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
              default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
          case ADDITION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(FCE, cn, -1); vtx_u1->set(FCE, cn, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
              case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
              case FCE: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(LNE, cc, cn); vtx_u1->set(LNE, cc, cn, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
              default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
          case COLORATION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(OUT, -1, -1); vtx_u0->set(OUT, -1, -1, -1); c_edg1->set(OUT, -1, -1); vtx_u1->set(OUT, -1, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
              case INS: c_edg0->set(INS, -1, -1); vtx_u0->set(INS, -1, -1, -1); c_edg1->set(INS, -1, -1); vtx_u1->set(INS, -1, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
              case FCE: c_edg0->set(FCE, cn, -1); vtx_u0->set(FCE, cn, -1, -1); c_edg1->set(LNE, cc, cn); vtx_u1->set(LNE, cc, cn, -1); c_tri0->set(FCE, cn); c_tri1->set(FCE, cn); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
              default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
        }
        break;
      }

    case 3: // (---)
      /* split a triangle */
      // no need to split

      /* apply rules */
      switch (action)
      {
        case INTERSECTION:  /* do nothing */                          break;
        case ADDITION:                            tri->set(INS, -1);  break;
        case COLORATION:    if (tri->loc == FCE)  tri->set(FCE, cn);  break;
      }
      break;
  }
}

void simplex3_mls_q_t::do_action_tet(int n_tet, int cn, action_t action)
{
  tet3_t *tet = &tets_[n_tet];

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
  if (vtxs_[tet->vtx0].value < 0) num_negatives++;
  if (vtxs_[tet->vtx1].value < 0) num_negatives++;
  if (vtxs_[tet->vtx2].value < 0) num_negatives++;
  if (vtxs_[tet->vtx3].value < 0) num_negatives++;

#ifdef SIMPLEX3_MLS_Q_DEBUG
  tet->type = num_negatives;

  /* check whether vertices coincide */
  if (tet->vtx0 != tris_[tet->tri1].vtx0 || tet->vtx0 != tris_[tet->tri2].vtx0 || tet->vtx0 != tris_[tet->tri3].vtx0 ||
      tet->vtx1 != tris_[tet->tri0].vtx0 || tet->vtx1 != tris_[tet->tri2].vtx1 || tet->vtx1 != tris_[tet->tri3].vtx1 ||
      tet->vtx2 != tris_[tet->tri1].vtx1 || tet->vtx2 != tris_[tet->tri0].vtx1 || tet->vtx2 != tris_[tet->tri3].vtx2 ||
      tet->vtx3 != tris_[tet->tri1].vtx2 || tet->vtx3 != tris_[tet->tri2].vtx2 || tet->vtx3 != tris_[tet->tri0].vtx2)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Vertices of a tetrahedron do not coincide with vertices of triangles after sorting.");

  /* check whether edges coincide */
  if (tris_[tet->tri0].edg0 != tris_[tet->tri1].edg0 || tris_[tet->tri0].edg1 != tris_[tet->tri2].edg0 || tris_[tet->tri0].edg2 != tris_[tet->tri3].edg0 ||
      tris_[tet->tri1].edg1 != tris_[tet->tri2].edg1 || tris_[tet->tri1].edg2 != tris_[tet->tri3].edg1 ||
      tris_[tet->tri2].edg2 != tris_[tet->tri3].edg2)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Edges of different triangles in a tetrahedron do not coincide.");

  /* check if appropriate triangles have been splitted */
  int t0_type_expect, t1_type_expect, t2_type_expect, t3_type_expect;

  switch (num_negatives)
  {
    case 0: t0_type_expect = 0; t1_type_expect = 0; t2_type_expect = 0; t3_type_expect = 0; break;
    case 1: t0_type_expect = 0; t1_type_expect = 1; t2_type_expect = 1; t3_type_expect = 1; break;
    case 2: t0_type_expect = 1; t1_type_expect = 1; t2_type_expect = 2; t3_type_expect = 2; break;
    case 3: t0_type_expect = 2; t1_type_expect = 2; t2_type_expect = 2; t3_type_expect = 3; break;
    case 4: t0_type_expect = 3; t1_type_expect = 3; t2_type_expect = 3; t3_type_expect = 3; break;
  }

  if (tris_[tet->tri0].type != t0_type_expect ||
      tris_[tet->tri1].type != t1_type_expect ||
      tris_[tet->tri2].type != t2_type_expect ||
      tris_[tet->tri3].type != t3_type_expect)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of triangles has an unexpected type.");
#endif

  edg3_t *c_edg;
  tri3_t *c_tri0, *c_tri1, *c_tri2, *c_tri3, *c_tri4, *c_tri5;
  tet3_t *c_tet0, *c_tet1, *c_tet2, *c_tet3, *c_tet4, *c_tet5;
  int n_tris_, n_tets_;
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
      /* split a tetrahedron */
      tet->is_split = true;

      // new vertices
      tet->c_vtx01 = tris_[tet->tri2].c_vtx01;
      tet->c_vtx02 = tris_[tet->tri1].c_vtx01;
      tet->c_vtx03 = tris_[tet->tri1].c_vtx02;

      // new triangles
      tris_.push_back(tri3_t(tet->c_vtx01, tet->c_vtx02, tet->c_vtx03, tris_[tet->tri1].c_edg0, tris_[tet->tri2].c_edg0, tris_[tet->tri3].c_edg0));
      tris_.push_back(tri3_t(tet->c_vtx01, tet->c_vtx02, tet->vtx3,    tris_[tet->tri1].c_edg1, tris_[tet->tri2].c_edg1, tris_[tet->tri3].c_edg0));
      tris_.push_back(tri3_t(tet->c_vtx01, tet->vtx2,    tet->vtx3,    tris_[tet->tri1].edg0,   tris_[tet->tri2].c_edg1, tris_[tet->tri3].c_edg1));

      tet->c_tri0 = tris_.size()-3;
      tet->c_tri1 = tris_.size()-2;
      tet->c_tri2 = tris_.size()-1;

      tri0 = &tris_[tet->tri0];
      tri1 = &tris_[tet->tri1];
      tri2 = &tris_[tet->tri2];
      tri3 = &tris_[tet->tri3];

      // new tetrahedra
      tets_.push_back(tet3_t(tet->vtx0,    tet->c_vtx01, tet->c_vtx02, tet->c_vtx03, tet->c_tri0,  tri1->c_tri0, tri2->c_tri0, tri3->c_tri0)); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->c_vtx01, tet->c_vtx02, tet->c_vtx03, tet->vtx3,    tri1->c_tri1, tri2->c_tri1, tet->c_tri1,  tet->c_tri0 )); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->c_vtx01, tet->c_vtx02, tet->vtx2,    tet->vtx3,    tri1->c_tri2, tet->c_tri2,  tet->c_tri1,  tri3->c_tri1)); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->c_vtx01, tet->vtx1,    tet->vtx2,    tet->vtx3,    tet->tri0,    tet->c_tri2,  tri2->c_tri2, tri3->c_tri2)); tet = &tets_[n_tet];

      tet->c_tet0 = tets_.size()-4;
      tet->c_tet1 = tets_.size()-3;
      tet->c_tet2 = tets_.size()-2;
      tet->c_tet3 = tets_.size()-1;

      /* apply rules */
      c_tri0 = &tris_[tet->c_tri0];
      c_tri1 = &tris_[tet->c_tri1];
      c_tri2 = &tris_[tet->c_tri2];

      c_tet0 = &tets_[tet->c_tet0];
      c_tet1 = &tets_[tet->c_tet1];
      c_tet2 = &tets_[tet->c_tet2];
      c_tet3 = &tets_[tet->c_tet3];

#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child triangles is not consistent.");

      if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) || !tet_is_ok(tet->c_tet3))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");

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
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(FCE, cn); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(INS); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(FCE, cn); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(INS); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          } break;
      }
      break;

    case 2: // --++
      {
        /* split a tetrahedron */
        tet->is_split = true;

        // vertices
        tet->c_vtx02 = tris_[tet->tri1].c_vtx01;
        tet->c_vtx03 = tris_[tet->tri1].c_vtx02;
        tet->c_vtx12 = tris_[tet->tri0].c_vtx01;
        tet->c_vtx13 = tris_[tet->tri0].c_vtx02;

        double length_edg = length(tet->c_vtx03, tet->c_vtx12);

        // midpoint aloung the interface
        double r12 = edgs_[tris_[tet->tri0].edg2].a;
        double r03 = edgs_[tris_[tet->tri1].edg1].a;

        double abc_u_lin[3] = { .5*(1.-r12), .5*r12, .5*r03 };
        double abc_u[3];
        double xyz_u[3];
        double t[3];

        bool reconstruction_is_good = true;

        if (length_edg > eps_)
          reconstruction_is_good = find_middle_node_tet(abc_u, n_tet, t);
        else
        {
          abc_u[0] = abc_u_lin[0];
          abc_u[1] = abc_u_lin[1];
          abc_u[2] = abc_u_lin[2];
        }

        mapping_tet(xyz_u, n_tet, abc_u);

        vtxs_.push_back(vtx3_t(xyz_u[0], xyz_u[1], xyz_u[2]));

        int vn = vtxs_.size()-1;

        /* check if deformation is not too high */
        if (check_for_curvature_ && reconstruction_is_good && length_edg > eps_)
        {
          // compute curvature of the curved edge
          vtx3_t *v0 = &vtxs_[tet->c_vtx03];
          vtx3_t *v1 = &vtxs_[vn];
          vtx3_t *v2 = &vtxs_[tet->c_vtx12];

          double xa = v2->x - v0->x;
          double ya = v2->y - v0->y;
          double za = v2->z - v0->z;

          double max_x = fabs(v0->x) < fabs(v1->x) ? (fabs(v1->x) < fabs(v2->x) ? fabs(v2->x) : fabs(v1->x)) :
                                                     (fabs(v0->x) < fabs(v2->x) ? fabs(v2->x) : fabs(v0->x)) ;
          double max_y = fabs(v0->y) < fabs(v1->y) ? (fabs(v1->y) < fabs(v2->y) ? fabs(v2->y) : fabs(v1->y)) :
                                                     (fabs(v0->y) < fabs(v2->y) ? fabs(v2->y) : fabs(v0->y)) ;
          double max_z = fabs(v0->z) < fabs(v1->z) ? (fabs(v1->z) < fabs(v2->z) ? fabs(v2->z) : fabs(v1->z)) :
                                                     (fabs(v0->z) < fabs(v2->z) ? fabs(v2->z) : fabs(v0->z)) ;

          double xaa = 4.*(v0->x - 2.*v1->x + v2->x); if (fabs(xaa) < eps_rel_*max_x) xaa = 0;
          double yaa = 4.*(v0->y - 2.*v1->y + v2->y); if (fabs(yaa) < eps_rel_*max_y) yaa = 0;
          double zaa = 4.*(v0->z - 2.*v1->z + v2->z); if (fabs(zaa) < eps_rel_*max_z) zaa = 0;

          double kappa_edg = fabs( sqrt( pow(zaa*ya-yaa*za, 2.) +
                                         pow(xaa*za-zaa*xa, 2.) +
                                         pow(yaa*xa-xaa*ya, 2.) )
                                   / pow( xa*xa + ya*ya + za*za , 1.5) );

          if (kappa_edg*length_edg > kappa_scale_*kappa_*lmin_ && kappa_edg*length_edg > kappa_eps_ &&
              !edgs_tmp_[tris_[tet->tri0].edg0].to_refine &&
              !edgs_tmp_[tris_[tet->tri0].edg1].to_refine &&
              !edgs_tmp_[tris_[tet->tri0].edg2].to_refine &&
              !edgs_tmp_[tris_[tet->tri1].edg1].to_refine &&
              !edgs_tmp_[tris_[tet->tri1].edg2].to_refine &&
              !edgs_tmp_[tris_[tet->tri2].edg2].to_refine)
          {
            // TODO: print a message
            reconstruction_is_good = false;
          }
        }


        if (!reconstruction_is_good && try_to_fix_outside_vertices_)
        {
          invalid_reconstruction_ = true;

          double A, B, C;
          A = 0; B = 0; C = 0; double phi_line_0 = (A-abc_u_lin[0])*t[0] + (B-abc_u_lin[1])*t[1]+ (C-abc_u_lin[2])*t[2];
          A = 1; B = 0; C = 0; double phi_line_1 = (A-abc_u_lin[0])*t[0] + (B-abc_u_lin[1])*t[1]+ (C-abc_u_lin[2])*t[2];
          A = 0; B = 1; C = 0; double phi_line_2 = (A-abc_u_lin[0])*t[0] + (B-abc_u_lin[1])*t[1]+ (C-abc_u_lin[2])*t[2];
          A = 0; B = 0; C = 1; double phi_line_3 = (A-abc_u_lin[0])*t[0] + (B-abc_u_lin[1])*t[1]+ (C-abc_u_lin[2])*t[2];

          bool at_least_one = false;

          if (refine_in_normal_dir_)
          {
            for (int i = 0; i < 6; ++i)
            {
              double p0, p1;
              int edg_idx;

              switch(i)
              {
                case 0: p0 = phi_line_2; p1 = phi_line_3; edg_idx = tris_[tet->tri0].edg0; break;
                case 1: p0 = phi_line_1; p1 = phi_line_3; edg_idx = tris_[tet->tri0].edg1; break;
                case 2: p0 = phi_line_1; p1 = phi_line_2; edg_idx = tris_[tet->tri0].edg2; break;
                case 3: p0 = phi_line_0; p1 = phi_line_3; edg_idx = tris_[tet->tri1].edg1; break;
                case 4: p0 = phi_line_0; p1 = phi_line_2; edg_idx = tris_[tet->tri1].edg2; break;
                case 5: p0 = phi_line_0; p1 = phi_line_1; edg_idx = tris_[tet->tri2].edg2; break;
                default: throw;
              }

              if (p0*p1 < 0)
              {
                double root = fabs(p0)/fabs(p0-p1);
                if (root > snap_limit_ && root < 1.-snap_limit_)
                {
                  edgs_tmp_[edg_idx].to_refine = true;
                  edgs_tmp_[edg_idx].a = root;
                  at_least_one = true;
                }
              }
            }
          }

          if (!at_least_one)
          {
            double t1[3] = { r12,
                             r12-1.,
                             0.};

            t[0] = abc_u[1]*t1[2] - abc_u[2]*t1[1];
            t[1] = abc_u[2]*t1[0] - abc_u[0]*t1[2];
            t[2] = abc_u[0]*t1[1] - abc_u[1]*t1[0];

            A = 1; B = 0; C = 0; phi_line_1 = A*t[0] + B*t[1] + C*t[2];
            A = 0; B = 1; C = 0; phi_line_2 = A*t[0] + B*t[1] + C*t[2];
            A = 0; B = 0; C = 1; phi_line_3 = A*t[0] + B*t[1] + C*t[2];

            if (phi_line_1*phi_line_3 > 0 || phi_line_2*phi_line_3 > 0)
            {
              std::cout << "bad!\n";
            }

            edgs_tmp_[tris_[tet->tri0].edg1].to_refine = true;
            edgs_tmp_[tris_[tet->tri0].edg1].a = fabs(phi_line_1)/fabs(phi_line_1-phi_line_3);

            edgs_tmp_[tris_[tet->tri0].edg0].to_refine = true;
            edgs_tmp_[tris_[tet->tri0].edg0].a = fabs(phi_line_2)/fabs(phi_line_2-phi_line_3);
          }
        }

        // new edge
        edgs_.push_back(edg3_t(tet->c_vtx03, vn, tet->c_vtx12));
        tet->c_edg = edgs_.size()-1;

        // new triangles
        tris_.push_back(tri3_t(tet->vtx0,    tet->c_vtx12, tet->c_vtx13, tris_[tet->tri0].c_edg0, tris_[tet->tri2].c_edg0,             tris_[tet->tri3].c_edg0             ));
        tris_.push_back(tri3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx12, tet->c_edg,              tris_[tet->tri3].c_edg0,             edgs_[tris_[tet->tri1].edg1].c_edg0 ));
        tris_.push_back(tri3_t(tet->c_vtx02, tet->c_vtx03, tet->c_vtx12, tet->c_edg,              tris_[tet->tri3].c_edg1,             tris_[tet->tri1].c_edg0             ));
        tris_.push_back(tri3_t(tet->c_vtx03, tet->c_vtx12, tet->c_vtx13, tris_[tet->tri0].c_edg0, tris_[tet->tri2].c_edg1,             tet->c_edg                          ));
        tris_.push_back(tri3_t(tet->c_vtx03, tet->c_vtx12, tet->vtx3,    tris_[tet->tri0].c_edg1, edgs_[tris_[tet->tri1].edg1].c_edg1, tet->c_edg                          ));
        tris_.push_back(tri3_t(tet->c_vtx02, tet->c_vtx12, tet->vtx3,    tris_[tet->tri0].c_edg1, tris_[tet->tri1].c_edg1,             tris_[tet->tri3].c_edg1             ));

        n_tris_ = tris_.size();
        tet->c_tri0 = n_tris_-6;
        tet->c_tri1 = n_tris_-5;
        tet->c_tri2 = n_tris_-4;
        tet->c_tri3 = n_tris_-3;
        tet->c_tri4 = n_tris_-2;
        tet->c_tri5 = n_tris_-1;

        tri0 = &tris_[tet->tri0];
        tri1 = &tris_[tet->tri1];
        tri2 = &tris_[tet->tri2];
        tri3 = &tris_[tet->tri3];

        // new tetrahedra
        tets_.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->c_vtx12, tet->c_vtx13, tri0->c_tri0, tet->c_tri0,  tri2->c_tri0, tri3->c_tri0)); tet = &tets_[n_tet];
        tets_.push_back(tet3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx12, tet->c_vtx13, tet->c_tri3,  tet->c_tri0,  tri2->c_tri1, tet->c_tri1 )); tet = &tets_[n_tet];
        tets_.push_back(tet3_t(tet->vtx0,    tet->c_vtx02, tet->c_vtx03, tet->c_vtx12, tet->c_tri2,  tet->c_tri1,  tri3->c_tri1, tri1->c_tri0)); tet = &tets_[n_tet];
        tets_.push_back(tet3_t(tet->c_vtx03, tet->c_vtx12, tet->c_vtx13, tet->vtx3,    tri0->c_tri1, tri2->c_tri2, tet->c_tri4,  tet->c_tri3 )); tet = &tets_[n_tet];
        tets_.push_back(tet3_t(tet->c_vtx02, tet->c_vtx03, tet->c_vtx12, tet->vtx3,    tet->c_tri4,  tet->c_tri5,  tri1->c_tri1, tet->c_tri2 )); tet = &tets_[n_tet];
        tets_.push_back(tet3_t(tet->c_vtx02, tet->c_vtx12, tet->vtx2,    tet->vtx3,    tri0->c_tri2, tri1->c_tri2, tet->c_tri5,  tri3->c_tri2)); tet = &tets_[n_tet];

        n_tets_ = tets_.size();
        tet->c_tet0 = n_tets_-6;
        tet->c_tet1 = n_tets_-5;
        tet->c_tet2 = n_tets_-4;
        tet->c_tet3 = n_tets_-3;
        tet->c_tet4 = n_tets_-2;
        tet->c_tet5 = n_tets_-1;

        //    construct_proper_mapping(tet->c_tri2, -1);
        //    construct_proper_mapping(tet->c_tri3, -1);

        /* apply rules */
        vtx3_t *c_vtx = &vtxs_[vn];

        c_edg = &edgs_[tet->c_edg];

        c_tri0 = &tris_[tet->c_tri0];
        c_tri1 = &tris_[tet->c_tri1];
        c_tri2 = &tris_[tet->c_tri2];
        c_tri3 = &tris_[tet->c_tri3];
        c_tri4 = &tris_[tet->c_tri4];
        c_tri5 = &tris_[tet->c_tri5];

        c_tet0 = &tets_[tet->c_tet0];
        c_tet1 = &tets_[tet->c_tet1];
        c_tet2 = &tets_[tet->c_tet2];
        c_tet3 = &tets_[tet->c_tet3];
        c_tet4 = &tets_[tet->c_tet4];
        c_tet5 = &tets_[tet->c_tet5];


#ifdef SIMPLEX3_MLS_Q_DEBUG
        if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2) ||
            !tri_is_ok(tet->c_tri3) || !tri_is_ok(tet->c_tri4) || !tri_is_ok(tet->c_tri5))
          throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child triangles is not consistent.");

        if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) ||
            !tet_is_ok(tet->c_tet3) || !tet_is_ok(tet->c_tet4) || !tet_is_ok(tet->c_tet5))
          throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");

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
            switch (tet->loc)
            {
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
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
          case ADDITION:
            switch (tet->loc)
            {
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
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
          case COLORATION:
            switch (tet->loc)
            {
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
#ifdef SIMPLEX3_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
            } break;
        }
        break;
      }
    case 3: // ---+
      /* split a tetrahedron */
      tet->is_split = true;

      // vertices
      tet->c_vtx03 = tris_[tet->tri1].c_vtx02;
      tet->c_vtx13 = tris_[tet->tri0].c_vtx02;
      tet->c_vtx23 = tris_[tet->tri0].c_vtx12;

      // new triangles
      tris_.push_back(tri3_t(tet->vtx0,    tet->vtx1,    tet->c_vtx23, tris_[tet->tri0].c_edg0, tris_[tet->tri1].c_edg0, tris_[tet->tri2].edg2  ));
      tris_.push_back(tri3_t(tet->vtx0,    tet->c_vtx13, tet->c_vtx23, tris_[tet->tri0].c_edg1, tris_[tet->tri1].c_edg0, tris_[tet->tri2].c_edg0));
      tris_.push_back(tri3_t(tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tris_[tet->tri0].c_edg1, tris_[tet->tri1].c_edg1, tris_[tet->tri2].c_edg1));

      n_tris_ = tris_.size();
      tet->c_tri0 = n_tris_ - 3;
      tet->c_tri1 = n_tris_ - 2;
      tet->c_tri2 = n_tris_ - 1;

      tri0 = &tris_[tet->tri0];
      tri1 = &tris_[tet->tri1];
      tri2 = &tris_[tet->tri2];
      tri3 = &tris_[tet->tri3];

      // new tetrahedra
      tets_.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->vtx2,    tet->c_vtx23, tri0->c_tri0, tri1->c_tri0, tet->c_tri0,  tet->tri3   )); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->c_vtx13, tet->c_vtx23, tri0->c_tri1, tet->c_tri1,  tet->c_tri0,  tri2->c_tri0)); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tet->c_tri2,  tet->c_tri1,  tri1->c_tri1, tri2->c_tri1)); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tet->vtx3,    tri0->c_tri2, tri1->c_tri2, tri2->c_tri2, tet->c_tri2 )); tet = &tets_[n_tet];

      n_tets_ = tets_.size();
      tet->c_tet0 = n_tets_-4;
      tet->c_tet1 = n_tets_-3;
      tet->c_tet2 = n_tets_-2;
      tet->c_tet3 = n_tets_-1;

      /* apply rules */
      c_tri0 = &tris_[tet->c_tri0];
      c_tri1 = &tris_[tet->c_tri1];
      c_tri2 = &tris_[tet->c_tri2];

      c_tet0 = &tets_[tet->c_tet0];
      c_tet1 = &tets_[tet->c_tet1];
      c_tet2 = &tets_[tet->c_tet2];
      c_tet3 = &tets_[tet->c_tet3];

#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child triangles is not consistent.");

      if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) || !tet_is_ok(tet->c_tet3))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");

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
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT,-1); c_tri1->set(OUT,-1); c_tri2->set(OUT,-1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(FCE,cn); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(OUT); break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(FCE,cn); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT,-1); c_tri1->set(OUT,-1); c_tri2->set(OUT,-1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default: ;
#ifdef SIMPLEX3_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) An element has wrong location.");
#endif
          } break;
      }
      break;

    case 4: // ----
      // no need to split
      switch (action)
      {
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
double simplex3_mls_q_t::find_intersection_quadratic(int e)
{
  double f0 = vtxs_[edgs_[e].vtx0].value;
  double f1 = vtxs_[edgs_[e].vtx1].value;
  double f2 = vtxs_[edgs_[e].vtx2].value;

#ifdef SIMPLEX3_MLS_Q_DEBUG
  if (same_sign(f0, f2)) throw std::invalid_argument("[CASL_ERROR]: (simplex3_mls_q_t) Cannot find an intersection with an edge, values of a level-set function are of the same sign at end points.");
#endif

//  double l = length(edgs_[e].vtx0, edgs_[e].vtx2);
//  double l = length(e);
//  if (l <= 2.1*eps_) return .5;
//  double ratio = eps_/l;

//  bool f0_close = fabs(f0) < phi_tolerance_;
//  bool f1_close = fabs(f1) < phi_tolerance_;
//  bool f2_close = fabs(f2) < phi_tolerance_;

//  if (f0_close && f1_close && f2_close) return .5;
//  if (f0_close)                         return ratio;
//  if (f1_close)                         return .5;
//  if (f2_close)                         return 1.-ratio;

  double fdd = (f2+f0-2.*f1)/0.25;

  double c2 = 0.5*fdd;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = (f2-f0)/1.;   // the expansion of f at the center of (0,1)
  double c0 = f1;

  double det = c1*c1-4.*c2*c0;

  if (det < 0)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) intersection is not found.");

  double q = c1 > 0 ? c1 + sqrt(det) : c1 - sqrt(det);

  if (2.*fabs(c0) > .5*fabs(q))
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Intersection with edge is not correct.");

  // we are interested only in the closest root
  double x = -2.*c0/q;

  if (not_finite(x))
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");

  if (x < -0.5 || x > 0.5)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Intersection with edge is not correct.");

  x += .5;

//  if (x < ratio)    x = ratio;
//  if (x > 1.-ratio) x = 1.-ratio;

  return x;
}


bool simplex3_mls_q_t::find_middle_node(double *xyz_out, double *xyz0, double *xyz1, int n_tri, double *t)
{
  tri3_t *tri = &tris_[n_tri];

  // fetch all six points of the triangle
  int nv[] = { tri->vtx0,
               tri->vtx1,
               tri->vtx2,
               edgs_[tri->edg2].vtx1,
               edgs_[tri->edg0].vtx1,
               edgs_[tri->edg1].vtx1 };

  // for better reconstruction we calculate normal to the linear reconstruction in real space
  // based on linear approximation of the underlying triangle
  vtx3_t *v0 = &vtxs_[nv[0]];
  vtx3_t *v1 = &vtxs_[nv[1]];
  vtx3_t *v2 = &vtxs_[nv[2]];

  // put triangle on a plane (below we calculate coordinates of vertices in this new plane)
  // vertex v0: at the origin
  double xyz_vtx0[2] = { 0, 0 };

  // vertex v1: on the x-axis
  double x_dir[3] = { v1->x - v0->x,
                      v1->y - v0->y,
                      v1->z - v0->z };

  double xyz_vtx1[2] = { sqrt(pow(x_dir[0],2.) +
                              pow(x_dir[1],2.) +
                              pow(x_dir[2],2.)), 0 };
  x_dir[0] /= xyz_vtx1[0];
  x_dir[1] /= xyz_vtx1[0];
  x_dir[2] /= xyz_vtx1[0];

  // vertex v2: wherever it happens to fall
  double xyz_vtx2[2];

  xyz_vtx2[0] = x_dir[0] * (v2->x - v0->x) +
                x_dir[1] * (v2->y - v0->y) +
                x_dir[2] * (v2->z - v0->z);

  xyz_vtx2[1] = sqrt(pow(v2->x - v0->x,2.) +
                     pow(v2->y - v0->y,2.) +
                     pow(v2->z - v0->z,2.) - pow(xyz_vtx2[0], 2));

  // calculate coordinates of the endpoints of the linear reconstruction in this new plane
  double XYZ0[2] = { xyz_vtx0[0] * (1. - xyz0[0] - xyz0[1]) + xyz_vtx1[0] * xyz0[0] + xyz_vtx2[0] * xyz0[1],
                     xyz_vtx0[1] * (1. - xyz0[0] - xyz0[1]) + xyz_vtx1[1] * xyz0[0] + xyz_vtx2[1] * xyz0[1] };

  double XYZ1[2] = { xyz_vtx0[0] * (1. - xyz1[0] - xyz1[1]) + xyz_vtx1[0] * xyz1[0] + xyz_vtx2[0] * xyz1[1],
                     xyz_vtx0[1] * (1. - xyz1[0] - xyz1[1]) + xyz_vtx1[1] * xyz1[0] + xyz_vtx2[1] * xyz1[1] };

  // vector parallel to the linear reconstruction
  double tx = XYZ1[0]-XYZ0[0];
  double ty = XYZ1[1]-XYZ0[1];
  double norm = sqrt(tx*tx+ty*ty);
  tx /= norm;
  ty /= norm;

  // vector perpendicular to the linear reconstruction
  double Nx =-ty;
  double Ny = tx;

  // map normal vector back to the reference triangle
  double det = (xyz_vtx1[0]-xyz_vtx0[0])*(xyz_vtx2[1]-xyz_vtx0[1]) - (xyz_vtx1[1]-xyz_vtx0[1])*(xyz_vtx2[0]-xyz_vtx0[0]);

  double nx = ( (Nx)*(xyz_vtx2[1]-xyz_vtx0[1]) - (Ny)*(xyz_vtx2[0]-xyz_vtx0[0]) ) / det;
  double ny =-( (Nx)*(xyz_vtx1[1]-xyz_vtx0[1]) - (Ny)*(xyz_vtx1[0]-xyz_vtx0[0]) ) / det;

  norm = sqrt(nx*nx+ny*ny);

  nx /= norm;
  ny /= norm;

  if (not_finite(nx) || not_finite(ny))
  {
    // if the above procedure failed, compute normal vector in the reference triangle
    // TODO: print warning
    tx = xyz1[0]-xyz0[0];
    ty = xyz1[1]-xyz0[1];
    norm = sqrt(tx*tx+ty*ty);
    tx /= norm;
    ty /= norm;
    nx =-ty;
    ny = tx;
  }

  // return vector perpendicular to the normal
  if (t != NULL) { t[0] = -ny; t[1] = nx; }

  // starting point
  double a = 0.5*(xyz0[0]+xyz1[0]);
  double b = 0.5*(xyz0[1]+xyz1[1]);

  // compute values of a level-set functions and its normal derivatives using shape functions
  double N  [nodes_per_tri_] = { (1.-a-b)*(1.-2.*a-2.*b), a*(2.*a-1.), b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b,  4.*b*(1.-a-b) };
  double Na [nodes_per_tri_] = { -3.+4.*a+4.*b,           4.*a-1.,     0,            4.-8.*a-4.*b,   4.*b,   -4.*b          };
  double Nb [nodes_per_tri_] = { -3.+4.*a+4.*b,           0,           4.*b-1.,     -4.*a,           4.*a,    4.-4.*a-8.*b  };
  double Naa[nodes_per_tri_] = { 4, 4, 0,-8, 0, 0 };
  double Nab[nodes_per_tri_] = { 4, 0, 0,-4, 4,-4 };
  double Nbb[nodes_per_tri_] = { 4, 0, 4, 0, 0,-8 };

  double F = 0, Fn = 0, Fnn = 0;
  double f;
  for (short i = 0; i < nodes_per_tri_; ++i)
  {
    f = vtxs_[nv[i]].value;
    F   += f*N[i];
    Fn  += f*(Na[i]*nx+Nb[i]*ny);
    Fnn += f*(Naa[i]*nx*nx + 2.*Nab[i]*nx*ny + Nbb[i]*ny*ny);
  }

  double alpha = 0;

  if (fabs(F) > phi_eps_)
  {
    // solve quadratic equation
    double c2 = .5*Fnn;
    double c1 = Fn;
    double c0 = F;

    det = c1*c1-4.*c2*c0;

    if (det < 0)
      throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) intersection is not found.");

    // we are interested only in the closest root
    alpha = -2.*c0/(c1 + signum(c1)*sqrt(det));

    if (not_finite(alpha))
      throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
  }

  // compute reference coordinates of the midpoint
  xyz_out[0] = a + alpha*nx;
  xyz_out[1] = b + alpha*ny;

  // check whether found midpoint falls outside of the triangle
  if (!edgs_tmp_[tri->edg2].to_refine &&
      !edgs_tmp_[tri->edg0].to_refine &&
      !edgs_tmp_[tri->edg1].to_refine)
  {
    if (xyz_out[0] + xyz_out[1] > 1. || xyz_out[0] < 0. || xyz_out[1] < 0.)
    {
      // TODO: print warning
      std::cout << "Warning: midpoint falls outside of a triangle!\n";
      std::cout << xyz_out[0] << " " << xyz_out[1] << " " << xyz_out[0] + xyz_out[1] << "\n";

      // use linear reconstruction in case the max level of refinement is reached
      xyz_out[0] = a;
      xyz_out[1] = b;

      // notify the calling function that reconstruction is not successfull
      return false;
    }
  }

  return true;
}

bool simplex3_mls_q_t::find_middle_node_tet(double abc_out[3], int n_tet, double *t)
{
  tet3_t *tet = &tets_[n_tet];

  // fetch edges
  int e01 = tris_[tet->tri3].edg2;
  int e02 = tris_[tet->tri3].edg1;
  int e03 = tris_[tet->tri1].edg1;
  int e12 = tris_[tet->tri0].edg2;
  int e23 = tris_[tet->tri0].edg0;
  int e13 = tris_[tet->tri0].edg1;

  // fetch all 10 vertices
  int nv[] = { tet->vtx0,
               tet->vtx1,
               tet->vtx2,
               tet->vtx3,
               edgs_[e01].vtx1,
               edgs_[e12].vtx1,
               edgs_[e02].vtx1,
               edgs_[e03].vtx1,
               edgs_[e13].vtx1,
               edgs_[e23].vtx1 };

  // get coordinates of intersection points
  double r13 = edgs_[e13].a;
  double r12 = edgs_[e12].a;
  double r03 = edgs_[e03].a;
  double r02 = edgs_[e02].a;

  double abc03[3] = { 0.,     0.,   r03 };
  double abc02[3] = { 0.,     r02,  0. };
  double abc12[3] = { 1.-r12, r12,  0. };
  double abc13[3] = { 1.-r13, 0.,   r13 };

  // for better reconstruction we calculate normal to the linear reconstruction in real space
  // based on linear approximation of the underlying triangle
  vtx3_t *v0 = &vtxs_[nv[0]];
  vtx3_t *v1 = &vtxs_[edgs_[e12].c_vtx_x];
  vtx3_t *v2 = &vtxs_[nv[3]];

  // put triangle on a plane (below we calculate coordinates of vertices in this new plane)
  // vertex v0: at the origin
  double xyz_vtx0[2] = { 0, 0 };

  // vertex v1: on the x-axis
  double x_dir[3] = { v1->x - v0->x,
                      v1->y - v0->y,
                      v1->z - v0->z};

  double xyz_vtx1[2] = { sqrt(pow(x_dir[0],2.) +
                              pow(x_dir[1],2.) +
                              pow(x_dir[2],2.)), 0 };

  x_dir[0] /= xyz_vtx1[0];
  x_dir[1] /= xyz_vtx1[0];
  x_dir[2] /= xyz_vtx1[0];

  // vertex v2: wherever it happens to fall
  double xyz_vtx2[2];

  xyz_vtx2[0] = x_dir[0] * (v2->x - v0->x) +
                x_dir[1] * (v2->y - v0->y) +
                x_dir[2] * (v2->z - v0->z);

  xyz_vtx2[1] = sqrt(pow(v2->x - v0->x, 2.) +
                     pow(v2->y - v0->y, 2.) +
                     pow(v2->z - v0->z, 2.) - pow(xyz_vtx2[0], 2));

  // coordinates of the endpoints of the linear reconstruction in the reference tetrahedron
  double xyz0[2] = { 1, 0 };
  double xyz1[2] = { 0, r03 };

  // map coordinates of the endpoints of the linear reconstruction into that new plane
  double XYZ0[2] = { xyz_vtx0[0] * (1. - xyz0[0] - xyz0[1]) + xyz_vtx1[0] * xyz0[0] + xyz_vtx2[0] * xyz0[1],
                     xyz_vtx0[1] * (1. - xyz0[0] - xyz0[1]) + xyz_vtx1[1] * xyz0[0] + xyz_vtx2[1] * xyz0[1] };

  double XYZ1[2] = { xyz_vtx0[0] * (1. - xyz1[0] - xyz1[1]) + xyz_vtx1[0] * xyz1[0] + xyz_vtx2[0] * xyz1[1],
                     xyz_vtx0[1] * (1. - xyz1[0] - xyz1[1]) + xyz_vtx1[1] * xyz1[0] + xyz_vtx2[1] * xyz1[1] };

  // vector parallel to the linear reconstruction
  double tx = XYZ1[0]-XYZ0[0];
  double ty = XYZ1[1]-XYZ0[1];
//  double norm = sqrt(tx*tx+ty*ty);
//  tx /= norm;
//  ty /= norm;

  // vector perpendicular to the linear reconstruction
  double Nx =-ty;
  double Ny = tx;

  // map normal vector back to the triangle
  double det = ( (xyz_vtx1[0]-xyz_vtx0[0])*(xyz_vtx2[1]-xyz_vtx0[1]) - (xyz_vtx1[1]-xyz_vtx0[1])*(xyz_vtx2[0]-xyz_vtx0[0]) );

  double nx_2d = ( (Nx)*(xyz_vtx2[1]-xyz_vtx0[1]) - (Ny)*(xyz_vtx2[0]-xyz_vtx0[0]) ) / det;
  double ny_2d =-( (Nx)*(xyz_vtx1[1]-xyz_vtx0[1]) - (Ny)*(xyz_vtx1[0]-xyz_vtx0[0]) ) / det;

  // calculate coordinates of the normal in the reference tetrahedron
  double cos_theta = (1.-r12)/sqrt( pow(1.-r12, 2.) + pow(r12, 2.) );

  double nx = nx_2d*cos_theta;
  double ny = nx_2d*sqrt(1.-pow(cos_theta, 2.));
  double nz = ny_2d;

  double norm = sqrt(nx*nx + ny*ny + nz*nz);

  nx /= norm;
  ny /= norm;
  nz /= norm;

  if (not_finite(nx) || not_finite(ny) || not_finite(nz))
  {
    // if the above procedure failed, compute normal vector in the reference triangle
    // TODO: print warning
    double t0[3] = { abc03[0]-abc12[0],
                     abc03[1]-abc12[1],
                     abc03[2]-abc12[2] };

    double t1[3] = {  abc12[1],
                     -abc12[0],
                     0.};

    nx = t0[1]*t1[2] - t0[2]*t1[1];
    ny = t0[2]*t1[0] - t0[0]*t1[2];
    nz = t0[0]*t1[1] - t0[1]*t1[0];
  }

  // return vector perpendicular to the plane 'perlendicular' to the linear reconstruction
  if (t != NULL)
  {
    double t1[3] = { abc12[1],
                    -abc12[0],
                     0.};

    t[0] = ny*t1[2] - nz*t1[1];
    t[1] = nz*t1[0] - nx*t1[2];
    t[2] = nx*t1[1] - ny*t1[0];

  }

  // starting point
  double a = 0.5*(abc12[0]+abc03[0]);
  double b = 0.5*(abc12[1]+abc03[1]);
  double c = 0.5*(abc12[2]+abc03[2]);

  // compute values of a level-set functions and its normal derivatives using shape functions
  double d = 1.-a-b-c;
  double N[nodes_per_tet_]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  double Na[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
  double Nb[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
  double Nc[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

  double Naa[nodes_per_tet_] = { 4, 4, 0, 0,-8, 0, 0, 0, 0, 0 };
  double Nbb[nodes_per_tet_] = { 4, 0, 4, 0, 0, 0,-8, 0, 0, 0 };
  double Ncc[nodes_per_tet_] = { 4, 0, 0, 4, 0, 0, 0,-8, 0, 0 };
  double Nab[nodes_per_tet_] = { 4, 0, 0, 0,-4, 4,-4, 0, 0, 0 };
  double Nbc[nodes_per_tet_] = { 4, 0, 0, 0, 0, 0,-4,-4, 0, 4 };
  double Nca[nodes_per_tet_] = { 4, 0, 0, 0,-4, 0, 0,-4, 4, 0 };


  double F = 0, Fx = 0, Fy = 0, Fz = 0, Fxx = 0, Fyy = 0, Fzz = 0, Fxy = 0, Fyz = 0, Fzx = 0;
  double f;
  for (short i = 0; i < nodes_per_tet_; ++i)
  {
    f = vtxs_[nv[i]].value;

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

  double alpha = 0;

  if (fabs(F) > phi_eps_)
  {
    double Fn  = Fx*nx + Fy*ny + Fz*nz;
    double Fnn = Fxx*nx*nx + Fyy*ny*ny + Fzz*nz*nz + 2.*Fxy*nx*ny + 2.*Fyz*ny*nz + 2.*Fzx*nz*nx;

    // solve quadratic equation
    double c2 = 0.5*Fnn;      // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
    double c1 = Fn;   // the expansion of f at the center of (0,1)
    double c0 = F;

    det = c1*c1-4.*c2*c0;

    if (det < 0) det = c1*c1;
    //    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) intersection is not found.");

    // we are interested only in the closest root
    alpha = -2.*c0/(c1 + signum(c1)*sqrt(det));

    if (not_finite(alpha))
      throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
  }

  abc_out[0] = a + alpha*nx;
  abc_out[1] = b + alpha*ny;
  abc_out[2] = c + alpha*nz;

  if (!edgs_tmp_[e01].to_refine ||
      !edgs_tmp_[e02].to_refine ||
      !edgs_tmp_[e03].to_refine ||
      !edgs_tmp_[e12].to_refine ||
      !edgs_tmp_[e13].to_refine ||
      !edgs_tmp_[e23].to_refine )
  {
    if (abc_out[0] + abc_out[1] + abc_out[2] > 1. || abc_out[0] < 0. || abc_out[1] < 0. || abc_out[2] < 0.)
    {
      // TODO: print warning
//      std::cout << "Warning: midpoint falls outside of a tetrahedron!\n";
//      std::cout <<  a+b+c
//          << " " << a
//          << " " << b
//          << " " << c << "\n";
//      std::cout << abc_out[0] + abc_out[1] + abc_out[2]
//          << " " << abc_out[0]
//          << " " << abc_out[1]
//          << " " << abc_out[2] << "\n";

      // use linear reconstruction in case the max level of refinement is reached
      abc_out[0] = a;
      abc_out[1] = b;
      abc_out[2] = c;

      // notify the calling function that reconstruction is not successfull
      return false;
    }
  }

  return true;
}


void simplex3_mls_q_t::adjust_middle_node(double *xyz_out,
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

  if (fabs(c2) < 1.e-15) b = -c0/c1;
  else
  {
    double b1 = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    double b2 = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));

    if      (b1 >= 0. && b1 <= 1.) b = b1;
    else if (b2 >= 0. && b2 <= 1.) b = b2;
    else
    {
      std::cout << "Warning: inverse mapping is incorrect! (" << c0 << " " << c1 << " " << c2 << " " << b1 << " " << b2 << ")\n";
      b = b1;
      return;
    }
  }

  double a = (xyz_in[0]-xyz0[0]-b*Xb)/(Xa+b*Xab);

  xyz_out[0] = xyz0[0] + a*Xa + b*Xb + a*b*Xab + (1.-b)*((1.-a)*xyz0[0] + a*xyz1[0])*2.*(1.-a)*a*(-xyz0[0]+2.*xyz01[0]-xyz1[0]);
  xyz_out[1] = xyz0[1] + a*Ya + b*Yb + a*b*Yab + (1.-b)*((1.-a)*xyz0[1] + a*xyz1[1])*2.*(1.-a)*a*(-xyz0[1]+2.*xyz01[1]-xyz1[1]);
}





//--------------------------------------------------
// Simple Refinement
//--------------------------------------------------
void simplex3_mls_q_t::refine_edg(int n_edg)
{
  edg3_t *edg = &edgs_[n_edg];

  if (edg->is_split) return;
  else edg->is_split = true;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  /* Create two new vertices */
  double xyz_v01[3];
  double xyz_v12[3];

  mapping_edg(xyz_v01, n_edg, 0.25);
  mapping_edg(xyz_v12, n_edg, 0.75);

  vtxs_.push_back(vtx3_t(xyz_v01[0], xyz_v01[1], xyz_v01[2]));
  vtxs_.push_back(vtx3_t(xyz_v12[0], xyz_v12[1], xyz_v12[2]));

  int n_vtx01 = vtxs_.size()-2;
  int n_vtx12 = vtxs_.size()-1;

  /* Create two new edges */
  edgs_.push_back(edg3_t(edg->vtx0, n_vtx01, edg->vtx1)); edg = &edgs_[n_edg];
  edgs_.push_back(edg3_t(edg->vtx1, n_vtx12, edg->vtx2)); edg = &edgs_[n_edg];

  edg->c_edg0 = edgs_.size()-2;
  edg->c_edg1 = edgs_.size()-1;

  /* Transfer properties to new objects */
  loc_t loc = edg->loc;
  int c0 = edg->c0;
  int c1 = edg->c1;

  vtxs_[n_vtx01].set(loc, c0, c1, -1);
  vtxs_[n_vtx12].set(loc, c0, c1, -1);

  edgs_[edg->c_edg0].set(loc, c0, c1);
  edgs_[edg->c_edg1].set(loc, c0, c1);
}

void simplex3_mls_q_t::refine_tri(int n_tri)
{
  tri3_t *tri = &tris_[n_tri];

  if (tri->is_split) return;
  else tri->is_split = true;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Create 3 new vertices */
  double xyz[3];
  double ab[2];
  ab[0] = .25; ab[1] = .25; mapping_tri(xyz, n_tri, ab); vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  ab[0] = .50; ab[1] = .25; mapping_tri(xyz, n_tri, ab); vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  ab[0] = .25; ab[1] = .50; mapping_tri(xyz, n_tri, ab); vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));

  int n_u0 = vtxs_.size()-3;
  int n_u1 = vtxs_.size()-2;
  int n_u2 = vtxs_.size()-1;

  /* Create 3 new edges */
  int n_v0 = tri->vtx0;
  int n_v1 = tri->vtx1;
  int n_v2 = tri->vtx2;
  int n_v01 = edgs_[tri->edg2].vtx1;
  int n_v12 = edgs_[tri->edg0].vtx1;
  int n_v02 = edgs_[tri->edg1].vtx1;

  edgs_.push_back(edg3_t(n_v02, n_u0, n_v01));
  edgs_.push_back(edg3_t(n_v01, n_u1, n_v12));
  edgs_.push_back(edg3_t(n_v02, n_u2, n_v12));

  /* Create 4 new triangles */
  int n_edg0 = edgs_.size()-3;
  int n_edg1 = edgs_.size()-2;
  int n_edg2 = edgs_.size()-1;

  tris_.push_back(tri3_t(n_v0,  n_v01, n_v02, n_edg0, edgs_[tri->edg1].c_edg0, edgs_[tri->edg2].c_edg0)); tri = &tris_[n_tri];
  tris_.push_back(tri3_t(n_v1,  n_v01, n_v12, n_edg1, edgs_[tri->edg0].c_edg0, edgs_[tri->edg2].c_edg1)); tri = &tris_[n_tri];
  tris_.push_back(tri3_t(n_v2,  n_v02, n_v12, n_edg2, edgs_[tri->edg0].c_edg1, edgs_[tri->edg1].c_edg1)); tri = &tris_[n_tri];
  tris_.push_back(tri3_t(n_v01, n_v02, n_v12, n_edg2, n_edg1,                 n_edg0));                 tri = &tris_[n_tri];

  int n_tri0 = tris_.size()-4;
  int n_tri1 = tris_.size()-3;
  int n_tri2 = tris_.size()-2;
  int n_tri3 = tris_.size()-1;

#ifdef SIMPLEX3_MLS_Q_DEBUG
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

  vtxs_[n_u0].set(loc, c, -1, -1);
  vtxs_[n_u1].set(loc, c, -1, -1);
  vtxs_[n_u2].set(loc, c, -1, -1);

  edgs_[n_edg0].set(loc, c, -1);
  edgs_[n_edg1].set(loc, c, -1);
  edgs_[n_edg2].set(loc, c, -1);

  tris_[n_tri0].set(loc, c); tris_[n_tri0].dir = dir;
  tris_[n_tri1].set(loc, c); tris_[n_tri1].dir = dir;
  tris_[n_tri2].set(loc, c); tris_[n_tri2].dir = dir;
  tris_[n_tri3].set(loc, c); tris_[n_tri3].dir = dir;
}

void simplex3_mls_q_t::refine_tet(int n_tet)
{
  tet3_t *tet = &tets_[n_tet];

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
  tri3_t *t0 = &tris_[tet->tri0];
  tri3_t *t1 = &tris_[tet->tri1];
  tri3_t *t2 = &tris_[tet->tri2];
  tri3_t *t3 = &tris_[tet->tri3];

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
  int nv01 = edgs_[e01].vtx1;
  int nv12 = edgs_[e12].vtx1;
  int nv02 = edgs_[e02].vtx1;
  int nv03 = edgs_[e03].vtx1;
  int nv13 = edgs_[e13].vtx1;
  int nv23 = edgs_[e23].vtx1;

  // create one more vertex
  double abc[3] = { .25, .25, .25 };
  double xyz[3];

  mapping_tet(xyz, n_tet, abc);

  vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  int nv = vtxs_.size() - 1;

  // create one more edge
  edgs_.push_back(edg3_t(nv12, nv, nv03));
  int ne = edgs_.size() - 1;

  // create 8 more triagnles
  tris_.push_back(tri3_t(nv01, nv02, nv03, t1->c_edg0, t2->c_edg0, t3->c_edg0));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];
  tris_.push_back(tri3_t(nv01, nv12, nv13, t0->c_edg0, t2->c_edg1, t3->c_edg1));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];
  tris_.push_back(tri3_t(nv02, nv12, nv23, t0->c_edg1, t1->c_edg1, t3->c_edg2));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];
  tris_.push_back(tri3_t(nv03, nv13, nv23, t0->c_edg2, t1->c_edg2, t2->c_edg2));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];
  tris_.push_back(tri3_t(nv01, nv03, nv12, ne,         t3->c_edg1, t2->c_edg0));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];
  tris_.push_back(tri3_t(nv13, nv03, nv12, ne,         t0->c_edg0, t2->c_edg2));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];
  tris_.push_back(tri3_t(nv23, nv03, nv12, ne,         t0->c_edg1, t1->c_edg2));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];
  tris_.push_back(tri3_t(nv02, nv03, nv12, ne,         t3->c_edg2, t1->c_edg0));  t0 = &tris_[tet->tri0]; t1 = &tris_[tet->tri1]; t2 = &tris_[tet->tri2]; t3 = &tris_[tet->tri3];

  int cf0 = tris_.size() - 8;
  int cf1 = tris_.size() - 7;
  int cf2 = tris_.size() - 6;
  int cf3 = tris_.size() - 5;
  int cf4 = tris_.size() - 4;
  int cf5 = tris_.size() - 3;
  int cf6 = tris_.size() - 2;
  int cf7 = tris_.size() - 1;

  // create 8 more tetrahedra
  tets_.push_back(tet3_t(nv0,  nv01, nv02, nv03, cf0, t1->c_tri0, t2->c_tri0, t3->c_tri0));
  tets_.push_back(tet3_t(nv1,  nv01, nv12, nv13, cf1, t0->c_tri0, t2->c_tri1, t3->c_tri1));
  tets_.push_back(tet3_t(nv2,  nv02, nv12, nv23, cf2, t0->c_tri1, t1->c_tri1, t3->c_tri2));
  tets_.push_back(tet3_t(nv3,  nv03, nv13, nv23, cf3, t0->c_tri2, t1->c_tri2, t2->c_tri2));
  tets_.push_back(tet3_t(nv01, nv02, nv03, nv12, cf7, cf4,        t3->c_tri3, cf0));
  tets_.push_back(tet3_t(nv03, nv12, nv13, nv23, t0->c_tri3, cf3, cf6,        cf5));
  tets_.push_back(tet3_t(nv01, nv03, nv12, nv13, cf5, cf1,        t2->c_tri3, cf4));
  tets_.push_back(tet3_t(nv02, nv03, nv12, nv23, cf6, cf2,        t1->c_tri3, cf7));

  int n_tet0 = tets_.size() - 8;
  int n_tet1 = tets_.size() - 7;
  int n_tet2 = tets_.size() - 6;
  int n_tet3 = tets_.size() - 5;
  int n_tet4 = tets_.size() - 4;
  int n_tet5 = tets_.size() - 3;
  int n_tet6 = tets_.size() - 2;
  int n_tet7 = tets_.size() - 1;

  // transfer properties
  loc_t loc = tets_[n_tet].loc;

  vtxs_[nv].set(loc, -1, -1, -1);

  edgs_[ne].set(loc, -1, -1);

  tris_[cf0].set(loc, -1);
  tris_[cf1].set(loc, -1);
  tris_[cf2].set(loc, -1);
  tris_[cf3].set(loc, -1);
  tris_[cf4].set(loc, -1);
  tris_[cf5].set(loc, -1);
  tris_[cf6].set(loc, -1);
  tris_[cf7].set(loc, -1);

  tets_[n_tet0].set(loc);
  tets_[n_tet1].set(loc);
  tets_[n_tet2].set(loc);
  tets_[n_tet3].set(loc);
  tets_[n_tet4].set(loc);
  tets_[n_tet5].set(loc);
  tets_[n_tet6].set(loc);
  tets_[n_tet7].set(loc);


#ifdef SIMPLEX3_MLS_Q_DEBUG
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
// Geometry Aware Refinement
//--------------------------------------------------

void simplex3_mls_q_t::smart_refine_edg(int n_edg)
{
  edg3_t *edg = &edgs_[n_edg];

  if (edg->to_refine)
  {
    if (edg->is_split) return;
    else edg->is_split = true;
  } else { return; }

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  if (fabs(edg->a-0.5) < eps_rel_)
  {
    /* Create two new vertices */
    double xyz_v01[3]; mapping_edg(xyz_v01, n_edg, 0.25);
    double xyz_v12[3]; mapping_edg(xyz_v12, n_edg, 0.75);

    vtxs_.push_back(vtx3_t(xyz_v01[0], xyz_v01[1], xyz_v01[2]));
    vtxs_.push_back(vtx3_t(xyz_v12[0], xyz_v12[1], xyz_v12[2]));

    int n_vtx01 = vtxs_.size()-2;
    int n_vtx12 = vtxs_.size()-1;

    /* Create two new edges */
    edgs_.push_back(edg3_t(edg->vtx0, n_vtx01, edg->vtx1)); edg = &edgs_[n_edg];
    edgs_.push_back(edg3_t(edg->vtx1, n_vtx12, edg->vtx2)); edg = &edgs_[n_edg];

    edg->c_vtx_x = edg->vtx1;
    edg->c_edg0 = edgs_.size()-2;
    edg->c_edg1 = edgs_.size()-1;

    /* Transfer properties to new objects */
    loc_t loc = edg->loc;
    int c0 = edg->c0;
    int c1 = edg->c1;

    vtxs_[n_vtx01].set(loc, c0, c1, -1);
    vtxs_[n_vtx12].set(loc, c0, c1, -1);

    edgs_[edg->c_edg0].set(loc, c0, c1);
    edgs_[edg->c_edg1].set(loc, c0, c1);

  } else {

    /* Create three new vertices */
    double xyz_v01[3]; mapping_edg(xyz_v01, n_edg, .5*edg->a);
    double xyz_v1 [3]; mapping_edg(xyz_v1,  n_edg, edg->a);
    double xyz_v12[3]; mapping_edg(xyz_v12, n_edg, edg->a + .5*(1.-edg->a));

    vtxs_.push_back(vtx3_t(xyz_v01[0], xyz_v01[1], xyz_v01[2]));
    vtxs_.push_back(vtx3_t(xyz_v1 [0], xyz_v1 [1], xyz_v1 [2]));
    vtxs_.push_back(vtx3_t(xyz_v12[0], xyz_v12[1], xyz_v12[2]));

    int n_vtx01 = vtxs_.size()-3;
    int n_vtx1  = vtxs_.size()-2;
    int n_vtx12 = vtxs_.size()-1;

    /* Create two new edges */
    edgs_.push_back(edg3_t(edg->vtx0, n_vtx01, n_vtx1   )); edg = &edgs_[n_edg];
    edgs_.push_back(edg3_t(n_vtx1,    n_vtx12, edg->vtx2)); edg = &edgs_[n_edg];

    edg->c_vtx_x = n_vtx1;
    edg->c_edg0 = edgs_.size()-2;
    edg->c_edg1 = edgs_.size()-1;

    /* Transfer properties to new objects */
    loc_t loc = edg->loc;
    int c0 = edg->c0;
    int c1 = edg->c1;

    vtxs_[edg->vtx1].is_recycled = true;
    vtxs_[n_vtx01].set(loc, c0, c1, -1);
    vtxs_[n_vtx1 ].set(loc, c0, c1, -1);
    vtxs_[n_vtx12].set(loc, c0, c1, -1);

    edgs_[edg->c_edg0].set(loc, c0, c1);
    edgs_[edg->c_edg1].set(loc, c0, c1);

  }
}

void simplex3_mls_q_t::smart_refine_tri(int n_tri)
{
  tri3_t *tri = &tris_[n_tri];

  if (tri->is_split) return;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  if (edgs_[tri->edg0].is_split ||
      edgs_[tri->edg1].is_split ||
      edgs_[tri->edg2].is_split )
  {
    tri->is_split = true;

    // determine which edge was split the first by comparing numbers of splitting vertices
    int n_child_vtx0 = (edgs_[tri->edg0].is_split ? edgs_[tri->edg0].c_vtx_x : INT_MAX);
    int n_child_vtx1 = (edgs_[tri->edg1].is_split ? edgs_[tri->edg1].c_vtx_x : INT_MAX);
    int n_child_vtx2 = (edgs_[tri->edg2].is_split ? edgs_[tri->edg2].c_vtx_x : INT_MAX);

    int v0, v1, v2;
    int e0, e1, e2;
    int split_case;

    if (n_child_vtx0 < n_child_vtx1 &&
        n_child_vtx0 < n_child_vtx2)
    {
      v0 = tri->vtx0; e0 = tri->edg0;
      v1 = tri->vtx1; e1 = tri->edg1;
      v2 = tri->vtx2; e2 = tri->edg2;
      split_case = 0;
    }
    else if (n_child_vtx1 < n_child_vtx0 &&
             n_child_vtx1 < n_child_vtx2)
    {
      v0 = tri->vtx1; e0 = tri->edg1;
      v1 = tri->vtx0; e1 = tri->edg0;
      v2 = tri->vtx2; e2 = tri->edg2;
      split_case = 1;
    }
    else if (n_child_vtx2 < n_child_vtx0 &&
             n_child_vtx2 < n_child_vtx1)
    {
      v0 = tri->vtx2; e0 = tri->edg2;
      v1 = tri->vtx0; e1 = tri->edg0;
      v2 = tri->vtx1; e2 = tri->edg1;
      split_case = 2;
    }

//    std::cout << v0 << " " << v1 << " " << v2 << "\n";

    /* Create one new vertex */

    double xyz[2];
    double abc[2];

    switch (split_case)
    {
      case 0:
        abc[0] = .5*(1.-edgs_[e0].a);
        abc[1] = .5*edgs_[e0].a;
        break;
      case 1:
        abc[0] = .5;
        abc[1] = .5*edgs_[e0].a;
        break;
      case 2:
        abc[0] = .5*edgs_[e0].a;
        abc[1] = .5;
        break;
    }

    mapping_tri(xyz, n_tri, abc);
    vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));

//    std::cout << abc[0] << " " << abc[1] << "\n";

    int v0x = vtxs_.size()-1;

    /* Create one new edge */
    edgs_.push_back(edg3_t(v0, v0x, edgs_[e0].c_vtx_x));

    int e0x = edgs_.size()-1;

    /* Create two new triangles */
    tris_.push_back(tri3_t(v0, v1, edgs_[e0].c_vtx_x, edgs_[e0].c_edg0, e0x, e2));
    tris_.push_back(tri3_t(v0, v2, edgs_[e0].c_vtx_x, edgs_[e0].c_edg1, e0x, e1));

    int ct0 = tris_.size()-2;
    int ct1 = tris_.size()-1;

    sort_tri(ct0);
    sort_tri(ct1);

    tri_is_ok(ct0);
    tri_is_ok(ct1);

    tris_[n_tri].c_tri0 = ct0;
    tris_[n_tri].c_tri1 = ct1;
    tris_[n_tri].c_edg0 = e0x;
    tris_[n_tri].c_vtx01 = v0x;

    /* Transfer properties */
    loc_t loc = tris_[n_tri].loc;
    int c     = tris_[n_tri].c;
    int dir   = tris_[n_tri].dir;

    vtxs_[v0x].set(loc, c, -1, -1);

    edgs_[e0x].set(loc, c, -1);

    tris_[ct0].set(loc, c); tris_[ct0].dir = dir;
    tris_[ct1].set(loc, c); tris_[ct1].dir = dir;
  } else if (tri->to_refine) {
    smart_refine_tri(n_tri, tri->a, tri->b);
  }

}

void simplex3_mls_q_t::smart_refine_tri(int n_tri, double a, double b)
{
  tri3_t *tri = &tris_[n_tri];

  if (tri->is_split) return;

  if (!tri->to_refine) return;

  if (edgs_[tri->edg0].is_split ||
      edgs_[tri->edg1].is_split ||
      edgs_[tri->edg2].is_split )
    throw;

  tri->is_split = true;
  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  int v0 = tri->vtx0;
  int v1 = tri->vtx1;
  int v2 = tri->vtx2;

  /* Create four new vertices */
  double xyz[3];
  double abc[2];

  abc[0] = .5*(a+0.); abc[1] = .5*(b+0.); mapping_tri(xyz, n_tri, abc); vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  abc[0] = .5*(a+1.); abc[1] = .5*(b+0.); mapping_tri(xyz, n_tri, abc); vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  abc[0] = .5*(a+0.); abc[1] = .5*(b+1.); mapping_tri(xyz, n_tri, abc); vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));
  abc[0] = a;         abc[1] = b;         mapping_tri(xyz, n_tri, abc); vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));

  int v03 = vtxs_.size()-4;
  int v13 = vtxs_.size()-3;
  int v23 = vtxs_.size()-2;
  int v3  = vtxs_.size()-1;

  /* Create three new edges */
  edgs_.push_back(edg3_t(v0, v03, v3));
  edgs_.push_back(edg3_t(v1, v13, v3));
  edgs_.push_back(edg3_t(v2, v23, v3));

  int e03 = edgs_.size()-3;
  int e13 = edgs_.size()-2;
  int e23 = edgs_.size()-1;

  /* Create three new triangles */
  tris_.push_back(tri3_t(v3, v1, v2, tris_[n_tri].edg0, e23, e13));
  tris_.push_back(tri3_t(v3, v2, v0, tris_[n_tri].edg1, e03, e23));
  tris_.push_back(tri3_t(v3, v0, v1, tris_[n_tri].edg2, e13, e03));

  int ct0 = tris_.size()-3;
  int ct1 = tris_.size()-2;
  int ct2 = tris_.size()-1;

  tri_is_ok(ct0);
  tri_is_ok(ct1);
  tri_is_ok(ct2);

  tris_[n_tri].c_tri0 = ct0;
  tris_[n_tri].c_tri1 = ct1;
  tris_[n_tri].c_tri2 = ct2;

  tris_[n_tri].c_edg0 = e03;
  tris_[n_tri].c_edg1 = e13;
  tris_[n_tri].c_edg2 = e23;

  tris_[n_tri].c_vtx01 = v3;

  /* Transfer properties */
  loc_t loc = tris_[n_tri].loc;
  int c     = tris_[n_tri].c;
  int dir   = tris_[n_tri].dir;

  vtxs_[v03].set(loc, c, -1, -1);
  vtxs_[v13].set(loc, c, -1, -1);
  vtxs_[v23].set(loc, c, -1, -1);
  vtxs_[v3 ].set(loc, c, -1, -1);

  edgs_[e03].set(loc, c, -1);
  edgs_[e13].set(loc, c, -1);
  edgs_[e23].set(loc, c, -1);

  tris_[ct0].set(loc, c); tris_[ct0].dir = dir;
  tris_[ct1].set(loc, c); tris_[ct1].dir = dir;
  tris_[ct2].set(loc, c); tris_[ct2].dir = dir;
}

void simplex3_mls_q_t::smart_refine_tet(int n_tet)
{
  tet3_t *tet = &tets_[n_tet];

  if (tet->is_split) return;

  // Sort vertices
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx2, tet->vtx3)) {swap(tet->vtx2, tet->vtx3); swap(tet->tri2, tet->tri3);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}


  if (tris_[tet->tri0].is_split ||
      tris_[tet->tri1].is_split ||
      tris_[tet->tri2].is_split ||
      tris_[tet->tri3].is_split )
  {
    tet->is_split = true;


    tri3_t *t0 = &tris_[tet->tri0];
    tri3_t *t1 = &tris_[tet->tri1];
    tri3_t *t2 = &tris_[tet->tri2];
    tri3_t *t3 = &tris_[tet->tri3];

    int e01 = t3->edg2;
    int e02 = t3->edg1;
    int e03 = t1->edg1;
    int e12 = t0->edg2;
    int e23 = t0->edg0;
    int e13 = t0->edg1;

    if (edgs_[e01].is_split ||
        edgs_[e02].is_split ||
        edgs_[e03].is_split ||
        edgs_[e12].is_split ||
        edgs_[e13].is_split ||
        edgs_[e23].is_split )
    {
      int n_child_vtx12 = (edgs_[e12].is_split ? edgs_[e12].c_vtx_x : INT_MAX); int split_case = 0; int n_child_min = n_child_vtx12;
      int n_child_vtx13 = (edgs_[e13].is_split ? edgs_[e13].c_vtx_x : INT_MAX); if (n_child_vtx13 < n_child_min) { split_case = 1; n_child_min = n_child_vtx13; }
      int n_child_vtx23 = (edgs_[e23].is_split ? edgs_[e23].c_vtx_x : INT_MAX); if (n_child_vtx23 < n_child_min) { split_case = 2; n_child_min = n_child_vtx23; }
      int n_child_vtx01 = (edgs_[e01].is_split ? edgs_[e01].c_vtx_x : INT_MAX); if (n_child_vtx01 < n_child_min) { split_case = 3; n_child_min = n_child_vtx01; }
      int n_child_vtx02 = (edgs_[e02].is_split ? edgs_[e02].c_vtx_x : INT_MAX); if (n_child_vtx02 < n_child_min) { split_case = 4; n_child_min = n_child_vtx02; }
      int n_child_vtx03 = (edgs_[e03].is_split ? edgs_[e03].c_vtx_x : INT_MAX); if (n_child_vtx03 < n_child_min) { split_case = 5; n_child_min = n_child_vtx03; }

      int v0, f0;
      int v1, f1;
      int v2, f2;
      int v3, f3;
      int E12;
      int E03;

      switch (split_case)
      {
        case 0:
          v0 = tet->vtx0; f0 = tet->tri0;
          v1 = tet->vtx1; f1 = tet->tri1;
          v2 = tet->vtx2; f2 = tet->tri2;
          v3 = tet->vtx3; f3 = tet->tri3;
          E12 = e12;
          E03 = e03;
          break;
        case 1:
          v0 = tet->vtx0; f0 = tet->tri0;
          v1 = tet->vtx1; f1 = tet->tri1;
          v2 = tet->vtx3; f2 = tet->tri3;
          v3 = tet->vtx2; f3 = tet->tri2;
          E12 = e13;
          E03 = e02;
          break;
        case 2:
          v0 = tet->vtx0; f0 = tet->tri0;
          v1 = tet->vtx2; f1 = tet->tri2;
          v2 = tet->vtx3; f2 = tet->tri3;
          v3 = tet->vtx1; f3 = tet->tri1;
          E12 = e23;
          E03 = e01;
          break;
        case 3:
          v0 = tet->vtx2; f0 = tet->tri2;
          v1 = tet->vtx0; f1 = tet->tri0;
          v2 = tet->vtx1; f2 = tet->tri1;
          v3 = tet->vtx3; f3 = tet->tri3;
          E12 = e01;
          E03 = e23;
          break;
        case 4:
          v0 = tet->vtx1; f0 = tet->tri1;
          v1 = tet->vtx0; f1 = tet->tri0;
          v2 = tet->vtx2; f2 = tet->tri2;
          v3 = tet->vtx3; f3 = tet->tri3;
          E12 = e02;
          E03 = e13;
          break;
        case 5:
          v0 = tet->vtx1; f0 = tet->tri1;
          v1 = tet->vtx0; f1 = tet->tri0;
          v2 = tet->vtx3; f2 = tet->tri3;
          v3 = tet->vtx2; f3 = tet->tri2;
          E12 = e03;
          E03 = e12;
          break;
      }

      // create one triagnle
      tris_.push_back(tri3_t(v0, edgs_[E12].c_vtx_x, v3, tris_[f0].c_edg0, E03, tris_[f3].c_edg0));

      int cf0 = tris_.size() - 1;

//      sort_tri(cf0);
      smart_refine_tri(cf0);

      // create two tetrahedra
      tets_.push_back(tet3_t(v0, v1, edgs_[E12].c_vtx_x, v3, tris_[f0].c_tri0, cf0, f2, tris_[f3].c_tri0));
      tets_.push_back(tet3_t(v0, v2, edgs_[E12].c_vtx_x, v3, tris_[f0].c_tri1, cf0, f1, tris_[f3].c_tri1));

      int n_tet0 = tets_.size() - 2;
      int n_tet1 = tets_.size() - 1;

      // transfer properties
      loc_t loc = tets_[n_tet].loc;

      tris_[cf0].set(loc, -1);

      tets_[n_tet0].set(loc);
      tets_[n_tet1].set(loc);


#ifdef SIMPLEX3_MLS_Q_DEBUG
      tri_is_ok(cf0);

      tet_is_ok(n_tet0);
      tet_is_ok(n_tet1);
#endif
    } else {

//      std::cout << "visited!\n";
      int n_child_vtx0 = (tris_[tet->tri0].is_split ? tris_[tet->tri0].c_vtx01 : INT_MAX); int split_case = 0; int v4 = n_child_vtx0;
      int n_child_vtx1 = (tris_[tet->tri1].is_split ? tris_[tet->tri1].c_vtx01 : INT_MAX); if (n_child_vtx1 < v4) { split_case = 1; v4 = n_child_vtx1; }
      int n_child_vtx2 = (tris_[tet->tri2].is_split ? tris_[tet->tri2].c_vtx01 : INT_MAX); if (n_child_vtx2 < v4) { split_case = 2; v4 = n_child_vtx2; }
      int n_child_vtx3 = (tris_[tet->tri3].is_split ? tris_[tet->tri3].c_vtx01 : INT_MAX); if (n_child_vtx3 < v4) { split_case = 3; v4 = n_child_vtx3; }

      int v0, f0;
      int v1, f1;
      int v2, f2;
      int v3, f3;
      int E01, E02, E03;

      double A0, A4;
      double B0, B4;
      double C0, C4;


      switch (split_case)
      {
        case 0:
          {
          v0 = tet->vtx0; f0 = tet->tri0;
          v1 = tet->vtx1; f1 = tet->tri1;
          v2 = tet->vtx2; f2 = tet->tri2;
          v3 = tet->vtx3; f3 = tet->tri3;
          E01 = e01;
          E02 = e02;
          E03 = e03;
          double a = tris_[f0].a;
          double b = tris_[f0].b;
          A0 = 0; A4 = 1.-a-b;
          B0 = 0; B4 = a;
          C0 = 0; C4 = b;
          } break;
        case 1:
          {
          v0 = tet->vtx1; f0 = tet->tri1;
          v1 = tet->vtx0; f1 = tet->tri0;
          v2 = tet->vtx2; f2 = tet->tri2;
          v3 = tet->vtx3; f3 = tet->tri3;
          E01 = e01;
          E02 = e12;
          E03 = e13;
          double a = tris_[f0].a;
          double b = tris_[f0].b;
          A0 = 1; A4 = 0;
          B0 = 0; B4 = a;
          C0 = 0; C4 = b;
          } break;
        case 2:
          {
          v0 = tet->vtx2; f0 = tet->tri2;
          v1 = tet->vtx0; f1 = tet->tri0;
          v2 = tet->vtx1; f2 = tet->tri1;
          v3 = tet->vtx3; f3 = tet->tri3;
          E01 = e02;
          E02 = e12;
          E03 = e23;
          double a = tris_[f0].a;
          double b = tris_[f0].b;
          A0 = 0; A4 = a;
          B0 = 1; B4 = 0;
          C0 = 0; C4 = b;
          } break;
        case 3:
          {
          v0 = tet->vtx3; f0 = tet->tri3;
          v1 = tet->vtx0; f1 = tet->tri0;
          v2 = tet->vtx1; f2 = tet->tri1;
          v3 = tet->vtx2; f3 = tet->tri2;
          E01 = e03;
          E02 = e13;
          E03 = e23;
          double a = tris_[f0].a;
          double b = tris_[f0].b;
          A0 = 0; A4 = a;
          B0 = 0; B4 = b;
          C0 = 1; C4 = 0;
          } break;
      }

      // create one vertex
      double abc[3] = { .5*(A0+A4), .5*(B0+B4), .5*(C0+C4) };
      double xyz[3];
      mapping_tet(xyz, n_tet, abc);

      vtxs_.push_back(vtx3_t(xyz[0], xyz[1], xyz[2]));

      int v04 = vtxs_.size() - 1;

      // create one edge
      edgs_.push_back(edg3_t(v0, v04, v4));

      int e04 = edgs_.size() - 1;

      // create three triangles
      tris_.push_back(tri3_t(v0, v1, v4, tris_[f0].c_edg0, e04, E01));
      tris_.push_back(tri3_t(v0, v2, v4, tris_[f0].c_edg1, e04, E02));
      tris_.push_back(tri3_t(v0, v3, v4, tris_[f0].c_edg2, e04, E03));

      int cf1 = tris_.size() - 3;
      int cf2 = tris_.size() - 2;
      int cf3 = tris_.size() - 1;

      smart_refine_tri(cf1);
      smart_refine_tri(cf2);
      smart_refine_tri(cf3);

      // create three tetrahedra
      tets_.push_back(tet3_t(v0, v4, v2, v3, tris_[f0].c_tri0, f1, cf3, cf2));
      tets_.push_back(tet3_t(v0, v4, v3, v1, tris_[f0].c_tri1, f2, cf1, cf3));
      tets_.push_back(tet3_t(v0, v4, v1, v2, tris_[f0].c_tri2, f3, cf2, cf1));

      int n_tet1 = tets_.size() - 3;
      int n_tet2 = tets_.size() - 2;
      int n_tet3 = tets_.size() - 1;

      // transfer properties
      loc_t loc = tets_[n_tet].loc;

      vtxs_[v04].set(loc, -1, -1, -1);

      edgs_[e04].set(loc, -1, -1);

      tris_[cf1].set(loc, -1);
      tris_[cf2].set(loc, -1);
      tris_[cf3].set(loc, -1);

      tets_[n_tet1].set(loc);
      tets_[n_tet2].set(loc);
      tets_[n_tet3].set(loc);

#ifdef SIMPLEX3_MLS_Q_DEBUG
      tri_is_ok(cf1);
      tri_is_ok(cf2);
      tri_is_ok(cf3);

      tet_is_ok(n_tet1);
      tet_is_ok(n_tet2);
      tet_is_ok(n_tet3);
#endif

    }
  }

}

void simplex3_mls_q_t::sort_edg(int n_edg)
{
  edg3_t *edg = &edgs_[n_edg];

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);
}

void simplex3_mls_q_t::sort_tri(int n_tri)
{
  tri3_t *tri = &tris_[n_tri];

  if (tri->is_split) return;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
}

void simplex3_mls_q_t::sort_tet(int n_tet)
{
  tet3_t *tet = &tets_[n_tet];

  if (tet->is_split) return;

  // Sort vertices
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx2, tet->vtx3)) {swap(tet->vtx2, tet->vtx3); swap(tet->tri2, tet->tri3);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
  if (need_swap(tet->vtx1, tet->vtx2)) {swap(tet->vtx1, tet->vtx2); swap(tet->tri1, tet->tri2);}
  if (need_swap(tet->vtx0, tet->vtx1)) {swap(tet->vtx0, tet->vtx1); swap(tet->tri0, tet->tri1);}
}

//--------------------------------------------------
// Quadrature points
//--------------------------------------------------
void simplex3_mls_q_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
//  return 0;
  double xyz[3];

  static double scale = 0.125/3.;

  // quadrature points
  static double alph = (5.+3.*sqrt(5.))/20.;
  static double beta = (5.-   sqrt(5.))/20.;

  static double abc0[3] = { beta, beta, beta };
  static double abc1[3] = { alph, beta, beta };
  static double abc2[3] = { beta, alph, beta };
  static double abc3[3] = { beta, beta, alph };

  /* integrate over tetrahedra */
  for (unsigned int i = 0; i < tets_.size(); i++)
    if (!tets_[i].is_split && tets_[i].loc == INS)
    {
      mapping_tet(xyz, i, abc0); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tet(i, abc0)*scale);
      mapping_tet(xyz, i, abc1); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tet(i, abc1)*scale);
      mapping_tet(xyz, i, abc2); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tet(i, abc2)*scale);
      mapping_tet(xyz, i, abc3); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tet(i, abc3)*scale);
    }
}

void simplex3_mls_q_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  bool integrate_specific = (num != -1);

  double xyz[3];

  // quadrature points
  //  double ab0[2] = { 1./6., 1./6. };
  //  double ab1[2] = { 2./3., 1./6. };
  //  double ab2[2] = { 1./6., 2./3. };
  static double abc0[2] = { .0, .5 };
  static double abc1[2] = { .5, .0 };
  static double abc2[2] = { .5, .5 };

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris_.size(); i++)
  {
    tri3_t *t = &tris_[i];
    if (!t->is_split && t->loc == FCE)
      if (!integrate_specific
          || (integrate_specific && t->c == num))
      {
        mapping_tri(xyz, i, abc0); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tri(i, abc0)/6.);
        mapping_tri(xyz, i, abc1); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tri(i, abc1)/6.);
        mapping_tri(xyz, i, abc2); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tri(i, abc2)/6.);
      }
  }
}

void simplex3_mls_q_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  bool integrate_specific = (num0 != -1 && num1 != -1);

  double xyz[3];

  // quadrature points
  //  double a0 = .5*(1.-1./sqrt(3.));
  //  double a1 = .5*(1.+1./sqrt(3.));

  //  double a0 = .5*(1.-sqrt(.6));
  //  double a1 = .5;
  //  double a2 = .5*(1.+sqrt(.6));

  static double a0 = 0.;
  static double a1 = .5;
  static double a2 = 1.;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs_.size(); i++)
  {
    edg3_t *e = &edgs_[i];
    if (!e->is_split && e->loc == LNE)
      if ( !integrate_specific
           || (integrate_specific
               && (e->c0 == num0 || e->c1 == num0)
               && (e->c0 == num1 || e->c1 == num1)) )
      {
        mapping_edg(xyz, i, a0); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_edg(i, a0)/6.);
        mapping_edg(xyz, i, a1); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_edg(i, a1)*2./3.);
        mapping_edg(xyz, i, a2); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_edg(i, a2)/6.);
      }
  }
}

void simplex3_mls_q_t::quadrature_over_intersection(int num0, int num1, int num2, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  bool integrate_specific = (num0 != -1 && num1 != -1 && num2 != -1);

  for (unsigned int i = 0; i < vtxs_.size(); i++)
  {
    vtx3_t *v = &vtxs_[i];
    if (v->loc == PNT)
      if ( !integrate_specific
           || (integrate_specific
               && (v->c0 == num0 || v->c1 == num0 || v->c2 == num0)
               && (v->c0 == num1 || v->c1 == num1 || v->c2 == num1)
               && (v->c0 == num2 || v->c1 == num2 || v->c2 == num2)) )
      {
        X.push_back(v->x); Y.push_back(v->y); Z.push_back(v->z); weights.push_back(1.);
      }
  }
}

void simplex3_mls_q_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  double xyz[3];

  // quadrature points
  //  double abc0[2] = { 1./6., 1./6. };
  //  double abc1[2] = { 2./3., 1./6. };
  //  double abc2[2] = { 1./6., 2./3. };
  static double abc0[2] = { .0, .5 };
  static double abc1[2] = { .5, .0 };
  static double abc2[2] = { .5, .5 };

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris_.size(); i++)
  {
    tri3_t *t = &tris_[i];
    if (!t->is_split && t->loc == INS)
      if (t->dir == dir)
      {
        mapping_tri(xyz, i, abc0); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tri(i, abc0)/6.);
        mapping_tri(xyz, i, abc1); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tri(i, abc1)/6.);
        mapping_tri(xyz, i, abc2); X.push_back(xyz[0]); Y.push_back(xyz[1]); Z.push_back(xyz[2]); weights.push_back(jacobian_tri(i, abc2)/6.);
      }
  }
}



//--------------------------------------------------
// Jacobians
//--------------------------------------------------
double simplex3_mls_q_t::jacobian_edg(int n_edg, double a)
{
  edg3_t *edg = &edgs_[n_edg];

  double Na[3] = {-3.+4.*a, 4.-8.*a, -1.+4.*a};

  double X = vtxs_[edg->vtx0].x * Na[0] + vtxs_[edg->vtx1].x * Na[1] + vtxs_[edg->vtx2].x * Na[2];
  double Y = vtxs_[edg->vtx0].y * Na[0] + vtxs_[edg->vtx1].y * Na[1] + vtxs_[edg->vtx2].y * Na[2];
  double Z = vtxs_[edg->vtx0].z * Na[0] + vtxs_[edg->vtx1].z * Na[1] + vtxs_[edg->vtx2].z * Na[2];

  return sqrt(X*X+Y*Y+Z*Z);
}

double simplex3_mls_q_t::jacobian_tri(int n_tri, double *ab)
{
  tri3_t *tri = &tris_[n_tri];

  if (tri->is_curved) { // if triangle is curved, then use the two-stage mapping
    // first, map a reference element into a 2D triangle on the surface
    double a = ab[0];
    double b = ab[1];

    double N_2d[nodes_per_tri_]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};
    double Na_2d[nodes_per_tri_] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
    double Nb_2d[nodes_per_tri_] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

    double A[nodes_per_tri_] = { 0, 1, 0, tri->ab01[0], tri->ab12[0], tri->ab02[0] };
    double B[nodes_per_tri_] = { 0, 0, 1, tri->ab01[1], tri->ab12[1], tri->ab02[1] };

    double Aa = 0, Ba = 0;
    double Ab = 0, Bb = 0;

    a = 0;
    b = 0;

    for (int i = 0; i < nodes_per_tri_; ++i)
    {
      a += A[i]*N_2d[i];   Aa += A[i]*Na_2d[i];   Ba += B[i]*Na_2d[i];
      b += B[i]*N_2d[i];   Ab += A[i]*Nb_2d[i];   Bb += B[i]*Nb_2d[i];
    }

    double jacobian_2d = fabs(Aa*Bb-Ab*Ba);

    // second, map the 2d surface triangle into 3D
//    a = ab[0];
//    b = ab[1];

    double Na[nodes_per_tri_] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
    double Nb[nodes_per_tri_] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

    int nv0 = tri->vtx0;
    int nv1 = tri->vtx1;
    int nv2 = tri->vtx2;

    double X[nodes_per_tri_] = { vtxs_[nv0].x, vtxs_[nv1].x, vtxs_[nv2].x, tri->g_vtx01[0], tri->g_vtx12[0], tri->g_vtx02[0] };
    double Y[nodes_per_tri_] = { vtxs_[nv0].y, vtxs_[nv1].y, vtxs_[nv2].y, tri->g_vtx01[1], tri->g_vtx12[1], tri->g_vtx02[1] };
    double Z[nodes_per_tri_] = { vtxs_[nv0].z, vtxs_[nv1].z, vtxs_[nv2].z, tri->g_vtx01[2], tri->g_vtx12[2], tri->g_vtx02[2] };

    double Xa = 0, Ya = 0, Za = 0;
    double Xb = 0, Yb = 0, Zb = 0;

    for (int i = 0; i < nodes_per_tri_; ++i)
    {
      Xa += X[i]*Na[i];   Ya += Y[i]*Na[i];   Za += Z[i]*Na[i];
      Xb += X[i]*Nb[i];   Yb += Y[i]*Nb[i];   Zb += Z[i]*Nb[i];
    }

//    return 1.*sqrt((Xa*Xa+Ya*Ya+Za*Za)*(Xb*Xb+Yb*Yb+Zb*Zb) - pow(Xa*Xb+Ya*Yb+Za*Zb, 2.));
    double result = jacobian_2d*sqrt((Xa*Xa+Ya*Ya+Za*Za)*(Xb*Xb+Yb*Yb+Zb*Zb) - pow(Xa*Xb+Ya*Yb+Za*Zb, 2.));

#ifdef SIMPLEX3_MLS_Q_DEBUG
    if (result != result)
      throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
#endif

    return result;

  } else { // if triangle is not curved, then a one-stage mapping suffies

    double a = ab[0];
    double b = ab[1];

    double Na[6] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
    double Nb[6] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

    double X1 = vtxs_[tri->vtx0].x*Na[0] + vtxs_[tri->vtx1].x*Na[1] + vtxs_[tri->vtx2].x*Na[2] + vtxs_[edgs_[tri->edg2].vtx1].x*Na[3] + vtxs_[edgs_[tri->edg0].vtx1].x*Na[4] + vtxs_[edgs_[tri->edg1].vtx1].x*Na[5];
    double Y1 = vtxs_[tri->vtx0].y*Na[0] + vtxs_[tri->vtx1].y*Na[1] + vtxs_[tri->vtx2].y*Na[2] + vtxs_[edgs_[tri->edg2].vtx1].y*Na[3] + vtxs_[edgs_[tri->edg0].vtx1].y*Na[4] + vtxs_[edgs_[tri->edg1].vtx1].y*Na[5];
    double Z1 = vtxs_[tri->vtx0].z*Na[0] + vtxs_[tri->vtx1].z*Na[1] + vtxs_[tri->vtx2].z*Na[2] + vtxs_[edgs_[tri->edg2].vtx1].z*Na[3] + vtxs_[edgs_[tri->edg0].vtx1].z*Na[4] + vtxs_[edgs_[tri->edg1].vtx1].z*Na[5];

    double X2 = vtxs_[tri->vtx0].x*Nb[0] + vtxs_[tri->vtx1].x*Nb[1] + vtxs_[tri->vtx2].x*Nb[2] + vtxs_[edgs_[tri->edg2].vtx1].x*Nb[3] + vtxs_[edgs_[tri->edg0].vtx1].x*Nb[4] + vtxs_[edgs_[tri->edg1].vtx1].x*Nb[5];
    double Y2 = vtxs_[tri->vtx0].y*Nb[0] + vtxs_[tri->vtx1].y*Nb[1] + vtxs_[tri->vtx2].y*Nb[2] + vtxs_[edgs_[tri->edg2].vtx1].y*Nb[3] + vtxs_[edgs_[tri->edg0].vtx1].y*Nb[4] + vtxs_[edgs_[tri->edg1].vtx1].y*Nb[5];
    double Z2 = vtxs_[tri->vtx0].z*Nb[0] + vtxs_[tri->vtx1].z*Nb[1] + vtxs_[tri->vtx2].z*Nb[2] + vtxs_[edgs_[tri->edg2].vtx1].z*Nb[3] + vtxs_[edgs_[tri->edg0].vtx1].z*Nb[4] + vtxs_[edgs_[tri->edg1].vtx1].z*Nb[5];

//    double xyz0[3] = { vtxs_[tri->vtx0].x, vtxs_[tri->vtx0].y, vtxs_[tri->vtx0].z };
//    double xyz1[3] = { vtxs_[tri->vtx1].x, vtxs_[tri->vtx1].y, vtxs_[tri->vtx1].z };
//    double xyz2[3] = { vtxs_[tri->vtx2].x, vtxs_[tri->vtx2].y, vtxs_[tri->vtx2].z };

//    double xyz3[3] = { vtxs_[edgs_[tri->edg2].vtx1].x, vtxs_[edgs_[tri->edg2].vtx1].y, vtxs_[edgs_[tri->edg2].vtx1].z };
//    double xyz4[3] = { vtxs_[edgs_[tri->edg0].vtx1].x, vtxs_[edgs_[tri->edg0].vtx1].y, vtxs_[edgs_[tri->edg0].vtx1].z };
//    double xyz5[3] = { vtxs_[edgs_[tri->edg1].vtx1].x, vtxs_[edgs_[tri->edg1].vtx1].y, vtxs_[edgs_[tri->edg1].vtx1].z };

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

#ifdef SIMPLEX3_MLS_Q_DEBUG
    if (result != result)
      throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
#endif

    return result;
  }
}

double simplex3_mls_q_t::jacobian_tet(int n_tet, double *abc)
{
  tet3_t *tet = &tets_[n_tet];

  double a = abc[0];
  double b = abc[1];
  double c = abc[2];

  double Na[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
  double Nb[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
  double Nc[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

  int e03 = tris_[tet->tri1].edg1;
  int e01 = tris_[tet->tri3].edg2;
  int e02 = tris_[tet->tri3].edg1;
  int e12 = tris_[tet->tri0].edg2;
  int e23 = tris_[tet->tri0].edg0;
  int e13 = tris_[tet->tri0].edg1;

  std::vector<int> nv(nodes_per_tet_, -1);

  nv[0] = tet->vtx0;
  nv[1] = tet->vtx1;
  nv[2] = tet->vtx2;
  nv[3] = tet->vtx3;
  nv[4] = edgs_[e01].vtx1;
  nv[5] = edgs_[e12].vtx1;
  nv[6] = edgs_[e02].vtx1;
  nv[7] = edgs_[e03].vtx1;
  nv[8] = edgs_[e13].vtx1;
  nv[9] = edgs_[e23].vtx1;

  double jac[3][3];

  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      jac[i][j] = 0;

  for (int i = 0; i < nodes_per_tet_; ++i)
  {
    jac[0][0] += vtxs_[nv[i]].x*Na[i];    jac[0][1] += vtxs_[nv[i]].x*Nb[i];    jac[0][2] += vtxs_[nv[i]].x*Nc[i];
    jac[1][0] += vtxs_[nv[i]].y*Na[i];    jac[1][1] += vtxs_[nv[i]].y*Nb[i];    jac[1][2] += vtxs_[nv[i]].y*Nc[i];
    jac[2][0] += vtxs_[nv[i]].z*Na[i];    jac[2][1] += vtxs_[nv[i]].z*Nb[i];    jac[2][2] += vtxs_[nv[i]].z*Nc[i];
  }

  return fabs( jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1]) - jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0]) + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]) );
}





////--------------------------------------------------
//// Interpolation
////--------------------------------------------------
//double simplex3_mls_q_t::interpolate_from_parent(std::vector<double> &f, double* xyz)
//{
//  // map real point to reference element
//  vtx3_t *v0 = &vtxs_[0];
//  vtx3_t *v1 = &vtxs_[1];
//  vtx3_t *v2 = &vtxs_[2];
//  vtx3_t *v3 = &vtxs_[3];

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
//  double N[nodes_per_tet_]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

//  double result = 0;

//  for (short i = 0; i < nodes_per_tet_; ++i)
//  {
//    result += N[i]*f[i];
//  }

//  return result;
//}

//void simplex3_mls_q_t::inv_mat3(double *in, double *out)
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
void simplex3_mls_q_t::mapping_edg(double* xyz, int n_edg, double a)
{
  edg3_t *edg = &edgs_[n_edg];

  double N0 = 1.-3.*a+2.*a*a;
  double N1 = 4.*a-4.*a*a;
  double N2 = -a+2.*a*a;

  xyz[0] = vtxs_[edg->vtx0].x * N0 + vtxs_[edg->vtx1].x * N1 + vtxs_[edg->vtx2].x * N2;
  xyz[1] = vtxs_[edg->vtx0].y * N0 + vtxs_[edg->vtx1].y * N1 + vtxs_[edg->vtx2].y * N2;
  xyz[2] = vtxs_[edg->vtx0].z * N0 + vtxs_[edg->vtx1].z * N1 + vtxs_[edg->vtx2].z * N2;

#ifdef SIMPLEX3_MLS_Q_DEBUG
  if (not_finite(xyz[0]) ||
      not_finite(xyz[1]) ||
      not_finite(xyz[2]) )
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
#endif
}

void simplex3_mls_q_t::mapping_tri(double* xyz, int n_tri, double* ab)
{
  tri3_t *tri = &tris_[n_tri];

  double a = ab[0];
  double b = ab[1];

  double N[nodes_per_tri_]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

  int nv0 = tri->vtx0;
  int nv1 = tri->vtx1;
  int nv2 = tri->vtx2;
  int nv3 = edgs_[tri->edg2].vtx1;
  int nv4 = edgs_[tri->edg0].vtx1;
  int nv5 = edgs_[tri->edg1].vtx1;

  xyz[0] = vtxs_[nv0].x * N[0] + vtxs_[nv1].x * N[1] + vtxs_[nv2].x * N[2] + vtxs_[nv3].x * N[3] + vtxs_[nv4].x * N[4] + vtxs_[nv5].x * N[5];
  xyz[1] = vtxs_[nv0].y * N[0] + vtxs_[nv1].y * N[1] + vtxs_[nv2].y * N[2] + vtxs_[nv3].y * N[3] + vtxs_[nv4].y * N[4] + vtxs_[nv5].y * N[5];
  xyz[2] = vtxs_[nv0].z * N[0] + vtxs_[nv1].z * N[1] + vtxs_[nv2].z * N[2] + vtxs_[nv3].z * N[3] + vtxs_[nv4].z * N[4] + vtxs_[nv5].z * N[5];

#ifdef SIMPLEX3_MLS_Q_DEBUG
  if (not_finite(xyz[0]) ||
      not_finite(xyz[1]) ||
      not_finite(xyz[2]) )
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
#endif
}

void simplex3_mls_q_t::mapping_tet(double *xyz, int n_tet, double* abc)
{
  tet3_t *tet = &tets_[n_tet];

  int e01 = tris_[tet->tri3].edg2;
  int e02 = tris_[tet->tri3].edg1;
  int e03 = tris_[tet->tri1].edg1;
  int e12 = tris_[tet->tri0].edg2;
  int e13 = tris_[tet->tri0].edg1;
  int e23 = tris_[tet->tri0].edg0;

  std::vector<int> nv(nodes_per_tet_, -1);

  nv[0] = tet->vtx0;
  nv[1] = tet->vtx1;
  nv[2] = tet->vtx2;
  nv[3] = tet->vtx3;
  nv[4] = edgs_[e01].vtx1;
  nv[5] = edgs_[e12].vtx1;
  nv[6] = edgs_[e02].vtx1;
  nv[7] = edgs_[e03].vtx1;
  nv[8] = edgs_[e13].vtx1;
  nv[9] = edgs_[e23].vtx1;

  double a = abc[0];
  double b = abc[1];
  double c = abc[2];
  double d = 1.-a-b-c;
  double N[nodes_per_tet_]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  xyz[0] = 0.;
  xyz[1] = 0.;
  xyz[2] = 0.;

  for (short i = 0; i < nodes_per_tet_; ++i)
  {
    xyz[0] += N[i]*vtxs_[nv[i]].x;
    xyz[1] += N[i]*vtxs_[nv[i]].y;
    xyz[2] += N[i]*vtxs_[nv[i]].z;
  }

#ifdef SIMPLEX3_MLS_Q_DEBUG
  if (not_finite(xyz[0]) ||
      not_finite(xyz[1]) ||
      not_finite(xyz[2]) )
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
#endif
}





//--------------------------------------------------
// Computation tools
//--------------------------------------------------
double simplex3_mls_q_t::length(int vtx0, int vtx1)
{
  return sqrt(pow(vtxs_[vtx0].x - vtxs_[vtx1].x, 2.0)
            + pow(vtxs_[vtx0].y - vtxs_[vtx1].y, 2.0)
            + pow(vtxs_[vtx0].z - vtxs_[vtx1].z, 2.0));
}

double simplex3_mls_q_t::length(int e)
{
  return ( jacobian_edg(e, 0.)    +
           jacobian_edg(e, .5)*4. +
           jacobian_edg(e, 1.) )/6.;
}

double simplex3_mls_q_t::area(int vtx0, int vtx1, int vtx2)
{
  double x01 = vtxs_[vtx1].x - vtxs_[vtx0].x; double x02 = vtxs_[vtx2].x - vtxs_[vtx0].x;
  double y01 = vtxs_[vtx1].y - vtxs_[vtx0].y; double y02 = vtxs_[vtx2].y - vtxs_[vtx0].y;
  double z01 = vtxs_[vtx1].z - vtxs_[vtx0].z; double z02 = vtxs_[vtx2].z - vtxs_[vtx0].z;

  return 0.5*sqrt(pow(y01*z02-z01*y02,2.0) + pow(z01*x02-x01*z02,2.0) + pow(x01*y02-y01*x02,2.0));
}

double simplex3_mls_q_t::volume(int vtx0, int vtx1, int vtx2, int vtx3)
{
  double a11 = vtxs_[vtx1].x-vtxs_[vtx0].x; double a12 = vtxs_[vtx1].y-vtxs_[vtx0].y; double a13 = vtxs_[vtx1].z-vtxs_[vtx0].z;
  double a21 = vtxs_[vtx2].x-vtxs_[vtx0].x; double a22 = vtxs_[vtx2].y-vtxs_[vtx0].y; double a23 = vtxs_[vtx2].z-vtxs_[vtx0].z;
  double a31 = vtxs_[vtx3].x-vtxs_[vtx0].x; double a32 = vtxs_[vtx3].y-vtxs_[vtx0].y; double a33 = vtxs_[vtx3].z-vtxs_[vtx0].z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}

double simplex3_mls_q_t::volume(vtx3_t &vtx0, vtx3_t &vtx1, vtx3_t &vtx2, vtx3_t &vtx3)
{
  double a11 = vtx1.x-vtx0.x; double a12 = vtx1.y-vtx0.y; double a13 = vtx1.z-vtx0.z;
  double a21 = vtx2.x-vtx0.x; double a22 = vtx2.y-vtx0.y; double a23 = vtx2.z-vtx0.z;
  double a31 = vtx3.x-vtx0.x; double a32 = vtx3.y-vtx0.y; double a33 = vtx3.z-vtx0.z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}

void simplex3_mls_q_t::inv_mat3(double *in, double *out)
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





//--------------------------------------------------
// Interpolation
//--------------------------------------------------
double simplex3_mls_q_t::interpolate_from_parent(std::vector<double> &f, double* xyz)
{
  // map real point to reference element
  vtx3_t *v0 = &vtxs_[0];

  double D[3] = { xyz[0] - v0->x,
                  xyz[1] - v0->y,
                  xyz[2] - v0->z };

  double a = map_parent_to_ref_[3*0+0]*D[0] + map_parent_to_ref_[3*0+1]*D[1] + map_parent_to_ref_[3*0+2]*D[2];
  double b = map_parent_to_ref_[3*1+0]*D[0] + map_parent_to_ref_[3*1+1]*D[1] + map_parent_to_ref_[3*1+2]*D[2];
  double c = map_parent_to_ref_[3*2+0]*D[0] + map_parent_to_ref_[3*2+1]*D[1] + map_parent_to_ref_[3*2+2]*D[2];

  // compute nodal functions
  double d = 1.-a-b-c;
  double N[nodes_per_tet_]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  double result = 0;

  for (short i = 0; i < nodes_per_tet_; ++i)
  {
    result += N[i]*f[i];
  }

  return result;
}

double simplex3_mls_q_t::interpolate_from_parent(double* xyz)
{
  // map real point to reference element
  vtx3_t *v0 = &vtxs_[0];

  double D[3] = { xyz[0] - v0->x,
                  xyz[1] - v0->y,
                  xyz[2] - v0->z };

  double a = map_parent_to_ref_[3*0+0]*D[0] + map_parent_to_ref_[3*0+1]*D[1] + map_parent_to_ref_[3*0+2]*D[2];
  double b = map_parent_to_ref_[3*1+0]*D[0] + map_parent_to_ref_[3*1+1]*D[1] + map_parent_to_ref_[3*1+2]*D[2];
  double c = map_parent_to_ref_[3*2+0]*D[0] + map_parent_to_ref_[3*2+1]*D[1] + map_parent_to_ref_[3*2+2]*D[2];

  // compute nodal functions
  double d = 1.-a-b-c;
  double N[nodes_per_tet_]  = { d*(2.*d-1.), a*(2.*a-1.), b*(2.*b-1.), c*(2.*c-1.), 4.*d*a, 4.*a*b, 4.*b*d, 4.*d*c, 4.*a*c, 4.*b*c };

  double result = 0;

  for (short i = 0; i < nodes_per_tet_; ++i)
  {
    result += N[i]*vtxs_[i].value;
  }

  return result;
}


void simplex3_mls_q_t::compute_curvature()
{
  // map real point to reference element
  double a_x = map_parent_to_ref_[3*0+0]; double a_y = map_parent_to_ref_[3*0+1]; double a_z = map_parent_to_ref_[3*0+2];
  double b_x = map_parent_to_ref_[3*1+0]; double b_y = map_parent_to_ref_[3*1+1]; double b_z = map_parent_to_ref_[3*1+2];
  double c_x = map_parent_to_ref_[3*2+0]; double c_y = map_parent_to_ref_[3*2+1]; double c_z = map_parent_to_ref_[3*2+2];

  // centroid
  double a = .25;
  double b = .25;
  double c = .25;

  double Na[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   4.*a-1., 0.,      0.,       4.*(1.-2.*a-b-c), 4.*b, -4.*b,             -4.*c,             4.*c, 0. };
  double Nb[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   0.,      4.*b-1., 0.,      -4.*a,             4.*a,  4.*(1.-a-2.*b-c), -4.*c,             0.,   4.*c };
  double Nc[nodes_per_tet_] = { -3.+4.*a+4.*b+4.*c,   0.,      0.,      4.*c-1., -4.*a,             0.,   -4.*b,              4.*(1.-a-b-2.*c), 4.*a, 4.*b };

  double Naa[nodes_per_tet_] = { 4, 4, 0, 0,-8, 0, 0, 0, 0, 0 };
  double Nbb[nodes_per_tet_] = { 4, 0, 4, 0, 0, 0,-8, 0, 0, 0 };
  double Ncc[nodes_per_tet_] = { 4, 0, 0, 4, 0, 0, 0,-8, 0, 0 };
  double Nab[nodes_per_tet_] = { 4, 0, 0, 0,-4, 4,-4, 0, 0, 0 };
  double Nbc[nodes_per_tet_] = { 4, 0, 0, 0, 0, 0,-4,-4, 0, 4 };
  double Nca[nodes_per_tet_] = { 4, 0, 0, 0,-4, 0, 0,-4, 4, 0 };

  double phi_a  = 0;
  double phi_b  = 0;
  double phi_c  = 0;

  double phi_aa = 0;
  double phi_bb = 0;
  double phi_cc = 0;

  double phi_ab = 0;
  double phi_bc = 0;
  double phi_ca = 0;

  double f;
  for (short i = 0; i < nodes_per_tet_; ++i)
  {
    f = vtxs_[i].value;

    phi_a  += f*Na[i];
    phi_b  += f*Nb[i];
    phi_c  += f*Nc[i];

    phi_aa += f*Naa[i];
    phi_bb += f*Nbb[i];
    phi_cc += f*Ncc[i];

    phi_ab += f*Nab[i];
    phi_bc += f*Nbc[i];
    phi_ca += f*Nca[i];
  }

  double phi_x = phi_a*a_x + phi_b*b_x + phi_c*c_x;
  double phi_y = phi_a*a_y + phi_b*b_y + phi_c*c_y;
  double phi_z = phi_a*a_z + phi_b*b_z + phi_c*c_z;

  double phi_xx = phi_aa*a_x*a_x + phi_bb*b_x*b_x + phi_cc*c_x*c_x + 2.*phi_ab*a_x*b_x + 2.*phi_bc*b_x*c_x + 2.*phi_ca*c_x*a_x;
  double phi_yy = phi_aa*a_y*a_y + phi_bb*b_y*b_y + phi_cc*c_y*c_y + 2.*phi_ab*a_y*b_y + 2.*phi_bc*b_y*c_y + 2.*phi_ca*c_y*a_y;
  double phi_zz = phi_aa*a_z*a_z + phi_bb*b_z*b_z + phi_cc*c_z*c_z + 2.*phi_ab*a_z*b_z + 2.*phi_bc*b_z*c_z + 2.*phi_ca*c_z*a_z;

  double phi_xy = phi_aa*a_x*a_y + phi_bb*b_x*b_y + phi_cc*c_x*c_y + phi_ab*(a_x*b_y+a_y*b_x) + phi_bc*(b_x*c_y+b_y*c_x) + phi_ca*(c_x*a_y+c_y*a_x);
  double phi_yz = phi_aa*a_y*a_z + phi_bb*b_y*b_z + phi_cc*c_y*c_z + phi_ab*(a_y*b_z+a_z*b_y) + phi_bc*(b_y*c_z+b_z*c_y) + phi_ca*(c_y*a_z+c_z*a_y);
  double phi_zx = phi_aa*a_z*a_x + phi_bb*b_z*b_x + phi_cc*c_z*c_x + phi_ab*(a_z*b_x+a_x*b_z) + phi_bc*(b_z*c_x+b_x*c_z) + phi_ca*(c_z*a_x+c_x*a_z);

  double kappa_mean = ( (phi_x*phi_x*phi_yy - 2.*phi_x*phi_y*phi_xy + phi_y*phi_y*phi_xx +
                             phi_x*phi_x*phi_zz - 2.*phi_x*phi_z*phi_zx + phi_z*phi_z*phi_xx +
                             phi_z*phi_z*phi_yy - 2.*phi_z*phi_y*phi_yz + phi_y*phi_y*phi_zz)
                            / pow( phi_x*phi_x + phi_y*phi_y + phi_z*phi_z, 1.5) );

  double kappa_gauss = ( phi_x*phi_x*(phi_yy*phi_zz - phi_yz*phi_yz) +
                             phi_y*phi_y*(phi_zz*phi_xx - phi_zx*phi_zx) +
                             phi_z*phi_z*(phi_xx*phi_yy - phi_xy*phi_xy) +
                             2.*( phi_x*phi_y*(phi_zx*phi_yz - phi_xy*phi_zz) +
                                  phi_y*phi_z*(phi_xy*phi_zx - phi_yz*phi_xx) +
                                  phi_z*phi_x*(phi_xy*phi_yz - phi_zx*phi_yy) ) )
      / pow( phi_x*phi_x + phi_y*phi_y + phi_z*phi_z, 2.);

//  if (kappa_mean*kappa_mean - 4.*kappa_gauss < 0.) throw;
  if (kappa_mean*kappa_mean - 4.*kappa_gauss < 0.) kappa_gauss = 0;

  double kappa_1st = .5*(kappa_mean + sqrt(kappa_mean*kappa_mean - 4.*kappa_gauss));
  double kappa_2nd = .5*(kappa_mean - sqrt(kappa_mean*kappa_mean - 4.*kappa_gauss));

  kappa_ = fabs(kappa_1st) > fabs(kappa_2nd) ? fabs(kappa_1st) : fabs(kappa_2nd);
}

//--------------------------------------------------
// Debugging
//--------------------------------------------------
#ifdef SIMPLEX3_MLS_Q_DEBUG
bool simplex3_mls_q_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = (edgs_[e0].vtx0 == v1 || edgs_[e0].vtx2 == v1) && (edgs_[e0].vtx0 == v2 || edgs_[e0].vtx2 == v2);
  result = result && (edgs_[e1].vtx0 == v0 || edgs_[e1].vtx2 == v0) && (edgs_[e1].vtx0 == v2 || edgs_[e1].vtx2 == v2);
  result = result && (edgs_[e2].vtx0 == v0 || edgs_[e2].vtx2 == v0) && (edgs_[e2].vtx0 == v1 || edgs_[e2].vtx2 == v1);
  if (!result) throw std::domain_error("Inconsistent triangle!\n");
  return result;
}

bool simplex3_mls_q_t::tri_is_ok(int t)
{
  tri3_t *tri = &tris_[t];
  bool result = tri_is_ok(tri->vtx0, tri->vtx1, tri->vtx2, tri->edg0, tri->edg1, tri->edg2);
  if (!result) throw std::domain_error("Inconsistent triangle!\n");
  return result;
}

bool simplex3_mls_q_t::tet_is_ok(int s)
{
  bool result = true;
  tet3_t *tet = &tets_[s];

  tri3_t *tri;

  tri = &tris_[tet->tri0];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);
  if (!result)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");

  tri = &tris_[tet->tri1];
  result = result && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);
  if (!result)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");

  tri = &tris_[tet->tri2];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);
  if (!result)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");

  tri = &tris_[tet->tri3];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0);

  if (!result)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");
  return result;
}
#endif
