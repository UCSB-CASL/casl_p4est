#include "simplex2_mls_q.h"

//--------------------------------------------------
// Constructors
//--------------------------------------------------
simplex2_mls_q_t::simplex2_mls_q_t(double x0, double y0,
                                   double x1, double y1,
                                   double x2, double y2,
                                   double x3, double y3,
                                   double x4, double y4,
                                   double x5, double y5)
{
  // usually there will be only one cut
  vtxs_.reserve(15);
  edgs_.reserve(10);
  tris_.reserve(4);

  /* fill the vectors with the initial structure */
  /* 2
   * |\
   * 5 4
   * |  \
   * 0-3-1
   */
  vtxs_.push_back(vtx2_t(x0,y0));
  vtxs_.push_back(vtx2_t(x1,y1));
  vtxs_.push_back(vtx2_t(x2,y2));
  vtxs_.push_back(vtx2_t(x3,y3));
  vtxs_.push_back(vtx2_t(x4,y4));
  vtxs_.push_back(vtx2_t(x5,y5));

  edgs_.push_back(edg2_t(1,4,2));
  edgs_.push_back(edg2_t(0,5,2));
  edgs_.push_back(edg2_t(0,3,1));

  tris_.push_back(tri2_t(0,1,2,0,1,2));

  // pre-compute inverse matrix for mapping of the original simplex onto the reference simplex
  vtx2_t *v0 = &vtxs_[0];
  vtx2_t *v1 = &vtxs_[1];
  vtx2_t *v2 = &vtxs_[2];

  double det = ( (v1->x-v0->x)*(v2->y-v0->y) - (v1->y-v0->y)*(v2->x-v0->x) );

  map_parent_to_ref_[2*0+0] =   (v2->y-v0->y) / det;
  map_parent_to_ref_[2*0+1] = - (v2->x-v0->x) / det;
  map_parent_to_ref_[2*1+0] = - (v1->y-v0->y) / det;
  map_parent_to_ref_[2*1+1] =   (v1->x-v0->x) / det;

  // compute resolution limit (all edges with legnth < 2*eps_ will be split at the middle)
  double l01 = length(0, 1);
  double l02 = length(0, 2);
  double l12 = length(1, 2);

  lmin_ = l01;
  lmin_ = lmin_ < l02 ? lmin_ : l02;
  lmin_ = lmin_ < l12 ? lmin_ : l12;

  eps_ = eps_rel_*lmin_;
}




//--------------------------------------------------
// Constructing domain
//--------------------------------------------------
void simplex2_mls_q_t::construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  num_phi_ = acn.size();

  if (clr.size() != num_phi_) std::invalid_argument("[CASL_ERROR]: (simplex2_mls_q_t) (simplex2_mls_q_t) Numbers of actions and colors are not equal.");
  if (phi.size() != num_phi_*nodes_per_tri_) std::invalid_argument("[CASL_ERROR]: (simplex2_mls_q_t) (simplex2_mls_q_t) Numbers of actions and colors are not equal.");

  double phi0, phi1, phi2;
  double c0, c1, c2;
  double a_ext, phi_ext;

  int initial_refinement = 0;
  int n;

  std::vector<double> phi_current(nodes_per_tri_, -1);

  std::vector<vtx2_t> vtxs_initial = vtxs_;
  std::vector<edg2_t> edgs_initial = edgs_;
  std::vector<tri2_t> tris_initial = tris_;

  while(1)
  {
    for (int i = 0; i < initial_refinement; ++i)
    {
      n = edgs_.size(); for (int i = 0; i < n; i++) refine_edg(i);
      n = tris_.size(); for (int i = 0; i < n; i++) refine_tri(i);
    }

    // loop over LSFs
    int refine_level = 0;

    for (short phi_idx = 0; phi_idx < num_phi_; ++phi_idx)
    {
      phi_max_ = 0;

      for (int i = 0; i < nodes_per_tri_; ++i)
      {
        vtxs_[i].value = phi[nodes_per_tri_*phi_idx + i];
        phi_current[i] = phi[nodes_per_tri_*phi_idx + i];
        phi_max_ = phi_max_ > fabs(phi_current[i]) ? phi_max_ : fabs(phi_current[i]);
      }

      phi_eps_ = phi_max_*eps_rel_;

      for (int i = 0; i < nodes_per_tri_; ++i)
        perturb(vtxs_[i].value, phi_eps_);

      int last_vtxs_size = nodes_per_tri_;

      invalid_reconstruction_ = true;

      while (invalid_reconstruction_)
      {
        bool needs_refinement = true;

        while (needs_refinement)
        {
          // interpolate to all vertices
          for (int i = last_vtxs_size; i < vtxs_.size(); ++i)
            if (!vtxs_[i].is_recycled)
            {
              vtxs_[i].value = interpolate_from_parent (phi_current, vtxs_[i].x, vtxs_[i].y );
              perturb(vtxs_[i].value, phi_eps_);
            }

          last_vtxs_size = vtxs_.size();

          // check validity of data on each edge
          needs_refinement = false;
          int n = edgs_.size();
          for (int i = 0; i < n; ++i)
            if (!edgs_[i].is_split)
            {
              edg2_t *e = &edgs_[i];

              phi0 = vtxs_[e->vtx0].value;
              phi1 = vtxs_[e->vtx1].value;
              phi2 = vtxs_[e->vtx2].value;

              if (!same_sign(phi0, phi1) && same_sign(phi0, phi2))
              {
                needs_refinement = true;
                e->to_refine = true;
                e->a = .5;
                smart_refine_edg(i);
              }

              if (!e->to_refine && same_sign(phi0, phi2))
              {
                c0 = phi0;
                c1 = -3.*phi0 + 4.*phi1 -    phi2;
                c2 =  2.*phi0 - 4.*phi1 + 2.*phi2;

                if (fabs(c1) > DBL_MIN && fabs(c1) < 2.*fabs(c2))
                {
                  a_ext = -.5*c1/c2;

                  if (a_ext > 0. && a_ext < 1.)
                  {
                    phi_ext = c0 + c1*a_ext + c2*a_ext*a_ext;

                    if (!same_sign(phi0, phi_ext))
                    {
                      needs_refinement = true;
                      e->to_refine = true;
                      e->a = need_swap(e->vtx0, e->vtx2) ? 1.-a_ext : a_ext;
                      smart_refine_edg(i);
                    }
                  }
                }
              }
            }


          // refine if necessary
          if (needs_refinement && refine_level < max_refinement_ - initial_refinement)
          {
//            for (int i = 0; i < edgs_.size(); i++) smart_refine_edg(i);
            for (int i = 0; i < tris_.size(); i++) smart_refine_tri(i);
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

        int n;
        n = vtxs_.size(); for (int i = 0; i < n; i++) do_action_vtx(i, clr[phi_idx], acn[phi_idx]);
        n = edgs_.size(); for (int i = 0; i < n; i++) do_action_edg(i, clr[phi_idx], acn[phi_idx]);
        n = tris_.size(); for (int i = 0; i < n; i++) do_action_tri(i, clr[phi_idx], acn[phi_idx]);

//        invalid_reconstruction_ = false;
        if (invalid_reconstruction_ && refine_level < max_refinement_ - initial_refinement)
        {
          vtxs_ = vtxs_tmp_;
          edgs_ = edgs_tmp_;
          tris_ = tris_tmp_;

          for (int i = 0; i < edgs_.size(); i++) smart_refine_edg(i);
          for (int i = 0; i < tris_.size(); i++) smart_refine_tri(i);
          refine_level++;
        } else {
//          if (invalid_reconstruction_) std::cout << "Cannot resolve invalid geometry\n";
          invalid_reconstruction_ = false;
        }
      }
    }

    // sort everything before integration
    for (int i = 0; i < edgs_.size(); i++)
    {
      edg2_t *edg = &edgs_[i];
      if (need_swap(edg->vtx0, edg->vtx2)) { swap(edg->vtx0, edg->vtx2); }
    }

    for (int i = 0; i < tris_.size(); i++)
    {
      tri2_t *tri = &tris_[i];
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); }
      if (need_swap(tri->vtx1, tri->vtx2)) { swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2); }
      if (need_swap(tri->vtx0, tri->vtx1)) { swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1); }
    }

    // check for overlapping volumes
    if (check_for_overlapping_)
    {
      double s_before = area(tris_[0].vtx0, tris_[0].vtx1, tris_[0].vtx2);
      double s_after  = 0;

      // compute volume after using linear representation
      for (int i = 0; i < tris_.size(); ++i)
        if (!tris_[i].is_split)
          s_after += area(tris_[i].vtx0, tris_[i].vtx1, tris_[i].vtx2);

      if (fabs(s_before-s_after)/s_before > eps_rel_)
      {
        if (initial_refinement == max_refinement_)
        {
          std::cout << "Can't resolve overlapping " << fabs(s_before-s_after) << "\n";
          break;
        } else {
          ++initial_refinement;
          std::cout << "Overlapping " << fabs(s_before-s_after) << "\n";
          vtxs_ = vtxs_initial;
          edgs_ = edgs_initial;
          tris_ = tris_initial;
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
}







//--------------------------------------------------
// Splitting
//--------------------------------------------------
void simplex2_mls_q_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx2_t *vtx = &vtxs_[n_vtx];

  if (vtx->is_recycled) return;

  switch (action)
  {
    case INTERSECTION:  if (vtx->value > 0)                                      vtx->set(OUT, -1, -1);  break;
    case ADDITION:      if (vtx->value < 0)                                      vtx->set(INS, -1, -1);  break;
    case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==PNT)  vtx->set(FCE, cn, -1);  break;
  }
}

void simplex2_mls_q_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg2_t *edg = &edgs_[n_edg];

  int c0 = edg->c0;

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  int num_negatives = 0;
  if (vtxs_[edg->vtx0].value < 0) num_negatives++;
  if (vtxs_[edg->vtx2].value < 0) num_negatives++;

#ifdef SIMPLEX2_MLS_Q_DEBUG
  edg->type = num_negatives;
#endif

  // auxiliary variables
  vtx2_t *c_vtx_x, *c_vtx_0x, *c_vtx_x2;
  edg2_t *c_edg0, *c_edg1;

  switch (num_negatives)
  {
    case 0: // ++
      /* split an edge */
      // no need to split

      /* apply rules */
      switch (action)
      {
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
      double xyz_0[2];
      double xyz_m[2];
      double xyz_p[2];
      mapping_edg(xyz_0, n_edg, a);
      mapping_edg(xyz_m, n_edg, .5*a);
      mapping_edg(xyz_p, n_edg, a + .5*(1.-a));

      // create new vertices
      vtxs_.push_back(vtx2_t(xyz_m[0], xyz_m[1])); int n_vtx_0x = vtxs_.size()-1;
      vtxs_.push_back(vtx2_t(xyz_p[0], xyz_p[1])); int n_vtx_x2 = vtxs_.size()-1;
      vtxs_.push_back(vtx2_t(xyz_0[0], xyz_0[1]));

      edg->c_vtx_x = vtxs_.size()-1;

      // new edges
      edgs_.push_back(edg2_t(edg->vtx0,    n_vtx_0x, edg->c_vtx_x)); edg = &edgs_[n_edg]; // edges might have changed their addresses
      edgs_.push_back(edg2_t(edg->c_vtx_x, n_vtx_x2, edg->vtx2   )); edg = &edgs_[n_edg];

      edg->c_edg0 = edgs_.size()-2;
      edg->c_edg1 = edgs_.size()-1;

      /* apply rules */
      vtxs_[edg->vtx1].is_recycled = true;

      c_vtx_x  = &vtxs_[edg->c_vtx_x];
      c_vtx_0x = &vtxs_[n_vtx_0x];
      c_vtx_x2 = &vtxs_[n_vtx_x2];
      c_edg0  = &edgs_[edg->c_edg0];
      c_edg1  = &edgs_[edg->c_edg1];

      c_edg0->dir = edg->dir;
      c_edg1->dir = edg->dir;

      c_edg0->p_lsf = edg->p_lsf;
      c_edg1->p_lsf = edg->p_lsf;

#ifdef SIMPLEX2_MLS_Q_DEBUG
      c_vtx_x->p_edg = n_edg;
      c_edg0->p_edg  = n_edg;
      c_edg1->p_edg  = n_edg;
#endif
      switch (action)
      {
        case INTERSECTION:
          switch (edg->loc)
          {
            case OUT: c_vtx_x->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_vtx_0x->set(OUT, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
            case INS: c_vtx_x->set(FCE, cn, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
            case FCE: c_vtx_x->set(PNT, c0, cn); c_edg0->set(FCE, c0); c_vtx_0x->set(FCE, c0, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); if (c0==cn)  c_vtx_x->set(FCE, c0, -1);  break;
            default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) (simplex2_mls_q_t) An element has wrong location.");
#endif
          }
          break;
        case ADDITION:
          switch (edg->loc)
          {
            case OUT: c_vtx_x->set(FCE, cn, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
            case INS: c_vtx_x->set(INS, -1, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(INS, -1); c_vtx_x2->set(INS, -1, -1); break;
            case FCE: c_vtx_x->set(PNT, c0, cn); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(FCE, c0); c_vtx_x2->set(FCE, c0, -1); if (c0==cn)  c_vtx_x->set(FCE, c0, -1);  break;
            default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) (simplex2_mls_q_t) An element has wrong location.");
#endif
          }
          break;
        case COLORATION:
          switch (edg->loc)
          {
            case OUT: c_vtx_x->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_vtx_0x->set(OUT, -1, -1); c_edg1->set(OUT, -1); c_vtx_x2->set(OUT, -1, -1); break;
            case INS: c_vtx_x->set(INS, -1, -1); c_edg0->set(INS, -1); c_vtx_0x->set(INS, -1, -1); c_edg1->set(INS, -1); c_vtx_x2->set(INS, -1, -1); break;
            case FCE: c_vtx_x->set(PNT, c0, cn); c_edg0->set(FCE, cn); c_vtx_0x->set(FCE, cn, -1); c_edg1->set(FCE, c0); c_vtx_x2->set(FCE, c0, -1); if (c0==cn)  c_vtx_x->set(FCE, c0, -1);  break;
            default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) (simplex2_mls_q_t) An element has wrong location.");
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
      switch (action)
      {
        case INTERSECTION:  /* do nothing */                        break;
        case ADDITION:                          edg->set(INS, -1);  break;
        case COLORATION:    if (edg->loc==FCE)  edg->set(FCE, cn);  break;
      }
      break;
  }
}

void simplex2_mls_q_t::do_action_tri(int n_tri, int cn, action_t action)
{
  tri2_t *tri = &tris_[n_tri];

  if (tri->is_split) return;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Determine type */
  int num_negatives = 0;
  if (vtxs_[tri->vtx0].value < 0) num_negatives++;
  if (vtxs_[tri->vtx1].value < 0) num_negatives++;
  if (vtxs_[tri->vtx2].value < 0) num_negatives++;

#ifdef SIMPLEX2_MLS_Q_DEBUG
  tri->type = num_negatives;

  /* check whether vertices of a triangle and edges coincide */
  if (tri->vtx0 != edgs_[tri->edg1].vtx0 || tri->vtx0 != edgs_[tri->edg2].vtx0 ||
      tri->vtx1 != edgs_[tri->edg0].vtx0 || tri->vtx1 != edgs_[tri->edg2].vtx2 ||
      tri->vtx2 != edgs_[tri->edg0].vtx2 || tri->vtx2 != edgs_[tri->edg1].vtx2)
  {
    std::cout << vtxs_[tri->vtx0].value << " : " << vtxs_[tri->vtx1].value << " : " << vtxs_[tri->vtx2].value << std::endl;
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) Vertices of a triangle and edges do not coincide after sorting.");
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
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) While splitting a triangle one of edges has an unexpected type.");
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
        tri->c_vtx01 = edgs_[tri->edg2].c_vtx_x;
        tri->c_vtx02 = edgs_[tri->edg1].c_vtx_x;

        // coordinates of new vertices in reference element
        double abc_v01[] = { edgs_[tri->edg2].a, 0. };
        double abc_v02[] = { 0., edgs_[tri->edg1].a };

        /* vertex along interface */
        double abc_u0_lin[2] = { .5*(abc_v01[0] + abc_v02[0]),
                                 .5*(abc_v01[1] + abc_v02[1]) };
        double abc_u0[2];
        double t[2];
        bool reconstruction_is_good = find_middle_node(abc_u0, abc_v02, abc_v01, n_tri, t);

        /* midpoint of the auxiliary edge */
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

        /* map midpoints to physical space */
        double xyz_u0[2]; mapping_tri(xyz_u0, n_tri, abc_u0);
        double xyz_u1[2]; mapping_tri(xyz_u1, n_tri, abc_u1);

        if (check_for_edge_intersections_ && reconstruction_is_good)
        {
          // interpolate level-set function into the new point
          double phi1 = interpolate_from_parent(xyz_u1[0], xyz_u1[1]);
          double phi2 = vtxs_[tri->vtx2].value;

          // calculate slope at the endpoint where level-set function is zero
          double c1 = 4.*phi1 - phi2;

          // and check whether the slope and the value at the other end of the same sign
          if (c1*phi2 < 0)
          {
            invalid_reconstruction_ = true;
            reconstruction_is_good = false;

            // use linear recontruction in case the max level of refinement is reached
            mapping_tri(xyz_u0, n_tri, abc_u0_lin);
          }
        }

        // refine edges if any of the above tests failed
        if (!reconstruction_is_good)
        {
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
                default:
                  throw;
              }

              if (p0*p1 < 0)
              {
                double root = fabs(p0)/fabs(p0-p1);
                // snap intersection to an existing vertex if too close
                if (root > snap_limit_ && root < 1.-snap_limit_)
                {
                  if (edg_idx > edgs_tmp_.size())
                    throw;
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
          }
        }

        // new vertices
        vtxs_.push_back(vtx2_t(xyz_u0[0], xyz_u0[1]));
        vtxs_.push_back(vtx2_t(xyz_u1[0], xyz_u1[1]));

        int u0 = vtxs_.size()-2;
        int u1 = vtxs_.size()-1;

        // new edges
        edgs_.push_back(edg2_t(tri->c_vtx01, u0, tri->c_vtx02));
        edgs_.push_back(edg2_t(tri->c_vtx01, u1, tri->vtx2   ));

        // edges might have changed their addresses
        edg0 = &edgs_[tri->edg0];
        edg1 = &edgs_[tri->edg1];
        edg2 = &edgs_[tri->edg2];

        tri->c_edg0 = edgs_.size()-2;
        tri->c_edg1 = edgs_.size()-1;

        // new triangles
        tris_.push_back(tri2_t(tri->vtx0,    tri->c_vtx01, tri->c_vtx02, tri->c_edg0,  edg1->c_edg0, edg2->c_edg0)); tri = &tris_[n_tri];
        tris_.push_back(tri2_t(tri->c_vtx01, tri->c_vtx02, tri->vtx2,    edg1->c_edg1, tri->c_edg1,  tri->c_edg0));  tri = &tris_[n_tri];
        tris_.push_back(tri2_t(tri->c_vtx01, tri->vtx1,    tri->vtx2,    tri->edg0,    tri->c_edg1,  edg2->c_edg1)); tri = &tris_[n_tri];

        tri->c_tri0 = tris_.size()-3;
        tri->c_tri1 = tris_.size()-2;
        tri->c_tri2 = tris_.size()-1;

        /* apply rules */
        c_edg0 = &edgs_[tri->c_edg0];
        c_edg1 = &edgs_[tri->c_edg1];

        c_tri0 = &tris_[tri->c_tri0];
        c_tri1 = &tris_[tri->c_tri1];
        c_tri2 = &tris_[tri->c_tri2];

        vtx_u0 = &vtxs_[u0];
        vtx_u1 = &vtxs_[u1];

        if (action == INTERSECTION || action == ADDITION) c_edg0->p_lsf = cn;

#ifdef SIMPLEX2_MLS_Q_DEBUG
        if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
          throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) While splitting a triangle one of child triangles is not consistent.");

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
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(FCE, cn); vtx_u0->set(FCE, cn, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
              default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) An element has wrong location.");
#endif
            } break;
          case ADDITION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(FCE, cn); vtx_u0->set(FCE, cn, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) An element has wrong location.");
#endif
            } break;
          case COLORATION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) An element has wrong location.");
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
        tri->c_vtx02 = edgs_[tri->edg1].c_vtx_x;
        tri->c_vtx12 = edgs_[tri->edg0].c_vtx_x;

        // coordinates of new vertices in reference element
        double abc_v02[] = { 0.,                    edgs_[tri->edg1].a };
        double abc_v12[] = { 1.-edgs_[tri->edg0].a, edgs_[tri->edg0].a };

        /* vertex along interface */
        double abc_u1_lin[2] = { .5*(abc_v02[0] + abc_v12[0]),
                                 .5*(abc_v02[1] + abc_v12[1]) };
        double abc_u1[2];
        double t[2];
        bool reconstruction_is_good = find_middle_node(abc_u1, abc_v12, abc_v02, n_tri, t);

        /* midpoint of the auxiliary edge */
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

        /* map midpoints to physical space */
        double xyz_u0[2]; mapping_tri(xyz_u0, n_tri, abc_u0);
        double xyz_u1[2]; mapping_tri(xyz_u1, n_tri, abc_u1);

        // check for an intersection with an auxiliary straight edge
        if (check_for_edge_intersections_ && reconstruction_is_good)
        {
          // interpolate level-set function into the new point
          double phi1 = interpolate_from_parent(xyz_u0[0], xyz_u0[1]);
          double phi2 = vtxs_[tri->vtx0].value;

          // calculate slope at the endpoint where level-set function is zero
          double c1 = 4.*phi1 - phi2;

          // and check whether the slope and the value at the other end of the same sign
          if (c1*phi2 < 0)
          {
            invalid_reconstruction_ = true;
            reconstruction_is_good = false;

            // use linear recontruction in case the max level of refinement is reached
            mapping_tri(xyz_u1, n_tri, abc_u1_lin);
          }
        }

        // refine edges if any of the above tests were failed
        if (!reconstruction_is_good)
        {
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
                default:
                  throw;
              }

              if (p0*p1 < 0)
              {
                double root = fabs(p0)/fabs(p0-p1);
                // snap intersection to an existing vertex if too close
                if (root > snap_limit_ && root < 1.-snap_limit_)
                {
                  if (edg_idx > edgs_tmp_.size())
                    throw;
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
          }
        }

        vtxs_.push_back(vtx2_t(xyz_u0[0], xyz_u0[1]));
        vtxs_.push_back(vtx2_t(xyz_u1[0], xyz_u1[1]));

        int u0 = vtxs_.size()-2;
        int u1 = vtxs_.size()-1;

        // create new edges
        edgs_.push_back(edg2_t(tri->vtx0,    u0, tri->c_vtx12));
        edgs_.push_back(edg2_t(tri->c_vtx02, u1, tri->c_vtx12));

        // edges might have changed their addresses
        edg0 = &edgs_[tri->edg0];
        edg1 = &edgs_[tri->edg1];
        edg2 = &edgs_[tri->edg2];

        tri->c_edg0 = edgs_.size()-2;
        tri->c_edg1 = edgs_.size()-1;

        tris_.push_back(tri2_t(tri->vtx0,    tri->vtx1,    tri->c_vtx12, edg0->c_edg0, tri->c_edg0,  tri->edg2   )); tri = &tris_[n_tri];
        tris_.push_back(tri2_t(tri->vtx0,    tri->c_vtx02, tri->c_vtx12, tri->c_edg1,  tri->c_edg0,  edg1->c_edg0)); tri = &tris_[n_tri];
        tris_.push_back(tri2_t(tri->c_vtx02, tri->c_vtx12, tri->vtx2,    edg0->c_edg1, edg1->c_edg1, tri->c_edg1 )); tri = &tris_[n_tri];

        tri->c_tri0 = tris_.size()-3;
        tri->c_tri1 = tris_.size()-2;
        tri->c_tri2 = tris_.size()-1;

        /* apply rules */
        c_edg0 = &edgs_[tri->c_edg0];
        c_edg1 = &edgs_[tri->c_edg1];

        c_tri0 = &tris_[tri->c_tri0];
        c_tri1 = &tris_[tri->c_tri1];
        c_tri2 = &tris_[tri->c_tri2];

        vtx_u0 = &vtxs_[u0];
        vtx_u1 = &vtxs_[u1];

        if (action == INTERSECTION || action == ADDITION) c_edg1->p_lsf = cn;

#ifdef SIMPLEX2_MLS_Q_DEBUG
        if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
          throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) While splitting a triangle one of child triangles is not consistent.");

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
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(FCE, cn); vtx_u1->set(FCE, cn, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
              default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) An element has wrong location.");
#endif
            } break;
          case ADDITION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(FCE, cn); vtx_u1->set(FCE, cn, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) An element has wrong location.");
#endif
            } break;
          case COLORATION:
            switch (tri->loc)
            {
              case OUT: c_edg0->set(OUT, -1); vtx_u0->set(OUT, -1, -1); c_edg1->set(OUT, -1); vtx_u1->set(OUT, -1, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
              case INS: c_edg0->set(INS, -1); vtx_u0->set(INS, -1, -1); c_edg1->set(INS, -1); vtx_u1->set(INS, -1, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
              default:
#ifdef SIMPLEX2_MLS_Q_DEBUG
                throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) An element has wrong location.");
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
double simplex2_mls_q_t::find_intersection_quadratic(int e)
{
  double f0 = vtxs_[edgs_[e].vtx0].value;
  double f1 = vtxs_[edgs_[e].vtx1].value;
  double f2 = vtxs_[edgs_[e].vtx2].value;

#ifdef SIMPLEX2_MLS_Q_DEBUG
  if(same_sign(f0, f2)) throw std::invalid_argument("[CASL_ERROR]: (simplex2_mls_q_t) Cannot find an intersection with an edge, values of a level-set function are of the same sign at end points.");
#endif

//  double l = length(edgs_[e].vtx0, edgs_[e].vtx2);
//  double l = length(e);
//  if (l <= 2.1*eps_)
//    return .5;
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
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) intersection is not found.");

  double q = c1 > 0 ? c1 + sqrt(det) : c1 - sqrt(det);

  if (2.*fabs(c0) > .5*fabs(q))
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) Intersection with edge is not correct.");

  // we are interested only in the closest root
  double x = -2.*c0/q;

  if (not_finite(x))
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) Something went wrong during integration.");

  if (x < -0.5 || x > 0.5)
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) Intersection with edge is not correct.");

  x += .5;

//  if (x < ratio)    x = ratio;
//  if (x > 1.-ratio) x = 1.-ratio;

  return x;
}

bool simplex2_mls_q_t::find_middle_node(double *xyz_out, double *xyz0, double *xyz1, int n_tri, double *t)
{
  tri2_t *tri = &tris_[n_tri];

  // fetch all six points of the triangle
  int nv[] = { tri->vtx0,
               tri->vtx1,
               tri->vtx2,
               edgs_[tri->edg2].vtx1,
               edgs_[tri->edg0].vtx1,
               edgs_[tri->edg1].vtx1 };

  // for better reconstruction we calculate normal to the linear reconstruction in real space
  // based on linear approximation of the underlying triangle
  vtx2_t *v0 = &vtxs_[nv[0]];
  vtx2_t *v1 = &vtxs_[nv[1]];
  vtx2_t *v2 = &vtxs_[nv[2]];

  // calculate coordinates of the endpoints of the linear reconstruction in real space
  double XYZ0[2] = { vtxs_[nv[0]].x * (1. - xyz0[0] - xyz0[1]) + vtxs_[nv[1]].x * xyz0[0] + vtxs_[nv[2]].x * xyz0[1],
                     vtxs_[nv[0]].y * (1. - xyz0[0] - xyz0[1]) + vtxs_[nv[1]].y * xyz0[0] + vtxs_[nv[2]].y * xyz0[1] };

  double XYZ1[2] = { vtxs_[nv[0]].x * (1. - xyz1[0] - xyz1[1]) + vtxs_[nv[1]].x * xyz1[0] + vtxs_[nv[2]].x * xyz1[1],
                     vtxs_[nv[0]].y * (1. - xyz1[0] - xyz1[1]) + vtxs_[nv[1]].y * xyz1[0] + vtxs_[nv[2]].y * xyz1[1] };

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
  double nx = ( (Nx)*(v2->y-v0->y) - (Ny)*(v2->x-v0->x) ) / ( (v1->x-v0->x)*(v2->y-v0->y) - (v1->y-v0->y)*(v2->x-v0->x) );
  double ny = ( (Nx)*(v1->y-v0->y) - (Ny)*(v1->x-v0->x) ) / ( (v2->x-v0->x)*(v1->y-v0->y) - (v2->y-v0->y)*(v1->x-v0->x) );

  norm = sqrt(nx*nx+ny*ny);

  nx /= norm;
  ny /= norm;

  // return vector perpendicular to the normal
  if (t != NULL) { t[0] = -ny; t[1] = nx; }

  // starting point
  double a = 0.5*(xyz0[0]+xyz1[0]);
  double b = 0.5*(xyz0[1]+xyz1[1]);

  // compute values of a level-set functions and its normal derivatives using shape functions
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
    f = vtxs_[nv[i]].value;
    F   += f*N[i];
    Fn  += f*(Na[i]*nx+Nb[i]*ny);
    Fnn += f*(Naa[i]*nx*nx + 2.*Nab[i]*nx*ny + Nbb[i]*ny*ny);
  }

  // solve quadratic equation
  double c2 = .5*Fnn;
  double c1 = Fn;
  double c0 = F;

  double det = c1*c1-4.*c2*c0;

  if (det < 0)
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) intersection is not found.");

  // we are interested only in the closest root
  double alpha = -2.*c0/(c1 + signum(c1)*sqrt(det));

  if (not_finite(alpha))
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");

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

      invalid_reconstruction_ = true;

      // use linear reconstruction in case the max level of refinement is reached
      xyz_out[0] = a;
      xyz_out[1] = b;

      // notify the calling function that reconstruction is not successfull
      return false;
    }
  }

  return true;
}

void simplex2_mls_q_t::adjust_middle_node(double *xyz_out,
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





//--------------------------------------------------
// Simple Refinement
//--------------------------------------------------
void simplex2_mls_q_t::refine_edg(int n_edg)
{
  edg2_t *edg = &edgs_[n_edg];

  if (edg->is_split) return;
  else edg->is_split = true;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx2)) swap(edg->vtx0, edg->vtx2);

  /* Create two new vertices */
  double xyz_v01[2];
  double xyz_v12[2];
  mapping_edg(xyz_v01, n_edg, 0.25);
  mapping_edg(xyz_v12, n_edg, 0.75);

  vtxs_.push_back(vtx2_t(xyz_v01[0], xyz_v01[1]));
  vtxs_.push_back(vtx2_t(xyz_v12[0], xyz_v12[1]));

  int n_vtx01 = vtxs_.size()-2;
  int n_vtx12 = vtxs_.size()-1;

  /* Create two new edges */
  edgs_.push_back(edg2_t(edg->vtx0, n_vtx01, edg->vtx1)); edg = &edgs_[n_edg];
  edgs_.push_back(edg2_t(edg->vtx1, n_vtx12, edg->vtx2)); edg = &edgs_[n_edg];

  edg->c_edg0 = edgs_.size()-2;
  edg->c_edg1 = edgs_.size()-1;

  /* Transfer properties to new objects */
  loc_t loc = edg->loc;
  int c = edg->c0;

  int dir = edg->dir;

  vtxs_[n_vtx01].set(loc, c, -1);
  vtxs_[n_vtx12].set(loc, c, -1);

  edgs_[edg->c_edg0].set(loc, c);
  edgs_[edg->c_edg1].set(loc, c);

  edgs_[edg->c_edg0].dir = dir;
  edgs_[edg->c_edg1].dir = dir;
}

void simplex2_mls_q_t::refine_tri(int n_tri)
{
  tri2_t *tri = &tris_[n_tri];

  if (tri->is_split) return;
  else tri->is_split = true;

  /* Sort vertices */
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
  if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
  if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

  /* Create 3 new vertices */
  double xyz[2];
  double abc[2];
  abc[0] = .25; abc[1] = .25; mapping_tri(xyz, n_tri, abc); vtxs_.push_back(vtx2_t(xyz[0], xyz[1]));
  abc[0] = .50; abc[1] = .25; mapping_tri(xyz, n_tri, abc); vtxs_.push_back(vtx2_t(xyz[0], xyz[1]));
  abc[0] = .25; abc[1] = .50; mapping_tri(xyz, n_tri, abc); vtxs_.push_back(vtx2_t(xyz[0], xyz[1]));

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

  edgs_.push_back(edg2_t(n_v02, n_u0, n_v01));
  edgs_.push_back(edg2_t(n_v01, n_u1, n_v12));
  edgs_.push_back(edg2_t(n_v02, n_u2, n_v12));

  /* Create 4 new triangles */
  int n_edg0 = edgs_.size()-3;
  int n_edg1 = edgs_.size()-2;
  int n_edg2 = edgs_.size()-1;

  tris_.push_back(tri2_t(n_v0,  n_v01, n_v02, n_edg0, edgs_[tri->edg1].c_edg0, edgs_[tri->edg2].c_edg0)); tri = &tris_[n_tri];
  tris_.push_back(tri2_t(n_v1,  n_v01, n_v12, n_edg1, edgs_[tri->edg0].c_edg0, edgs_[tri->edg2].c_edg1)); tri = &tris_[n_tri];
  tris_.push_back(tri2_t(n_v2,  n_v02, n_v12, n_edg2, edgs_[tri->edg0].c_edg1, edgs_[tri->edg1].c_edg1)); tri = &tris_[n_tri];
  tris_.push_back(tri2_t(n_v01, n_v02, n_v12, n_edg2, n_edg1,                 n_edg0));                 tri = &tris_[n_tri];

  int n_tri0 = tris_.size()-4;
  int n_tri1 = tris_.size()-3;
  int n_tri2 = tris_.size()-2;
  int n_tri3 = tris_.size()-1;

  /* Transfer properties */
  loc_t loc = tri->loc;

  vtxs_[n_u0].set(loc, -1, -1);
  vtxs_[n_u1].set(loc, -1, -1);
  vtxs_[n_u2].set(loc, -1, -1);

  edgs_[n_edg0].set(loc, -1);
  edgs_[n_edg1].set(loc, -1);
  edgs_[n_edg2].set(loc, -1);

  tris_[n_tri0].set(loc);
  tris_[n_tri1].set(loc);
  tris_[n_tri2].set(loc);
  tris_[n_tri3].set(loc);
}








//--------------------------------------------------
// Geometry Aware Refinement
//--------------------------------------------------

void simplex2_mls_q_t::smart_refine_edg(int n_edg)
{
  edg2_t *edg = &edgs_[n_edg];

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
    double xyz_v01[2];
    double xyz_v12[2];

    mapping_edg(xyz_v01, n_edg, 0.25);
    mapping_edg(xyz_v12, n_edg, 0.75);

    vtxs_.push_back(vtx2_t(xyz_v01[0], xyz_v01[1]));
    vtxs_.push_back(vtx2_t(xyz_v12[0], xyz_v12[1]));

    int n_vtx01 = vtxs_.size()-2;
    int n_vtx12 = vtxs_.size()-1;

    /* Create two new edges */
    edgs_.push_back(edg2_t(edg->vtx0, n_vtx01, edg->vtx1)); edg = &edgs_[n_edg];
    edgs_.push_back(edg2_t(edg->vtx1, n_vtx12, edg->vtx2)); edg = &edgs_[n_edg];

    edg->c_vtx_x = edg->vtx1;
    edg->c_edg0 = edgs_.size()-2;
    edg->c_edg1 = edgs_.size()-1;

    /* Transfer properties to new objects */
    loc_t loc = edg->loc;
    int c = edg->c0;

    int dir = edg->dir;

    vtxs_[n_vtx01].set(loc, c, -1);
    vtxs_[n_vtx12].set(loc, c, -1);

    edgs_[edg->c_edg0].set(loc, c);
    edgs_[edg->c_edg1].set(loc, c);

    edgs_[edg->c_edg0].dir = dir;
    edgs_[edg->c_edg1].dir = dir;
  } else {
    /* Create three new vertices */
    double xyz_v01[2];
    double xyz_v1 [2];
    double xyz_v12[2];

    mapping_edg(xyz_v01, n_edg, .5*edg->a);
    mapping_edg(xyz_v1,  n_edg, edg->a);
    mapping_edg(xyz_v12, n_edg, edg->a + .5*(1.-edg->a));

    vtxs_.push_back(vtx2_t(xyz_v01[0], xyz_v01[1]));
    vtxs_.push_back(vtx2_t(xyz_v1 [0], xyz_v1 [1]));
    vtxs_.push_back(vtx2_t(xyz_v12[0], xyz_v12[1]));

    int n_vtx01 = vtxs_.size()-3;
    int n_vtx1  = vtxs_.size()-2;
    int n_vtx12 = vtxs_.size()-1;

    /* Create two new edges */
    edgs_.push_back(edg2_t(edg->vtx0, n_vtx01, n_vtx1   )); edg = &edgs_[n_edg];
    edgs_.push_back(edg2_t(n_vtx1,    n_vtx12, edg->vtx2)); edg = &edgs_[n_edg];

    edg->c_vtx_x = n_vtx1;
    edg->c_edg0 = edgs_.size()-2;
    edg->c_edg1 = edgs_.size()-1;

    /* Transfer properties to new objects */
    loc_t loc = edg->loc;
    int c = edg->c0;

    int dir = edg->dir;

    vtxs_[edg->vtx1].is_recycled = true;;
    vtxs_[n_vtx01].set(loc, c, -1);
    vtxs_[n_vtx1 ].set(loc, c, -1);
    vtxs_[n_vtx12].set(loc, c, -1);

    edgs_[edg->c_edg0].set(loc, c);
    edgs_[edg->c_edg1].set(loc, c);

    edgs_[edg->c_edg0].dir = dir;
    edgs_[edg->c_edg1].dir = dir;
  }
}

void simplex2_mls_q_t::smart_refine_tri(int n_tri)
{
  tri2_t *tri = &tris_[n_tri];

  if (tri->is_split) return;

  if (edgs_[tri->edg0].is_split ||
      edgs_[tri->edg1].is_split ||
      edgs_[tri->edg2].is_split )
  {
    tri->is_split = true;
    /* Sort vertices */
    if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}
    if (need_swap(tri->vtx1, tri->vtx2)) {swap(tri->vtx1, tri->vtx2); swap(tri->edg1, tri->edg2);}
    if (need_swap(tri->vtx0, tri->vtx1)) {swap(tri->vtx0, tri->vtx1); swap(tri->edg0, tri->edg1);}

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
    vtxs_.push_back(vtx2_t(xyz[0], xyz[1]));

//    std::cout << abc[0] << " " << abc[1] << "\n";

    int v0x = vtxs_.size()-1;

    /* Create one new edge */
    edgs_.push_back(edg2_t(v0, v0x, edgs_[e0].c_vtx_x));

    int e0x = edgs_.size()-1;

    /* Create two new triangles */
    tris_.push_back(tri2_t(v0, v1, edgs_[e0].c_vtx_x, edgs_[e0].c_edg0, e0x, e2));
    tris_.push_back(tri2_t(v0, v2, edgs_[e0].c_vtx_x, edgs_[e0].c_edg1, e0x, e1));

    int ct0 = tris_.size()-2;
    int ct1 = tris_.size()-1;

    tri_is_ok(ct0);
    tri_is_ok(ct1);

    tris_[n_tri].c_tri0 = ct0;
    tris_[n_tri].c_tri1 = ct1;
    tris_[n_tri].c_edg0 = e0x;
    tris_[n_tri].c_vtx01 = v0x;

    /* Transfer properties */
    loc_t loc = tris_[n_tri].loc;

    vtxs_[v0x].set(loc, -1, -1);
    edgs_[e0x].set(loc, -1);

    tris_[ct0].set(loc);
    tris_[ct1].set(loc);
  }

}






//--------------------------------------------------
// Quadrature points
//--------------------------------------------------
void simplex2_mls_q_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
  double xyz[3];

  // quadrature points
  static double abc0[] = {.0, .5};
  static double abc1[] = {.5, .0};
  static double abc2[] = {.5, .5};
//  static double abc0[] = {1./6., 1./6.};
//  static double abc1[] = {2./3., 1./6.};
//  static double abc2[] = {1./6., 2./3.};

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris_.size(); i++)
  {
    tri2_t *t = &tris_[i];
    if (!t->is_split && t->loc == INS)
    {
      mapping_tri(xyz, i, abc0); X.push_back(xyz[0]); Y.push_back(xyz[1]); weights.push_back(jacobian_tri(i, abc0[0], abc0[1])/6.);
      mapping_tri(xyz, i, abc1); X.push_back(xyz[0]); Y.push_back(xyz[1]); weights.push_back(jacobian_tri(i, abc1[0], abc1[1])/6.);
      mapping_tri(xyz, i, abc2); X.push_back(xyz[0]); Y.push_back(xyz[1]); weights.push_back(jacobian_tri(i, abc2[0], abc2[1])/6.);
    }
  }
}

void simplex2_mls_q_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
  bool integrate_specific = (num != -1);

  double xyz[2];

  // quadrature points, order 3
  static double a0 = .5*(1.-1./sqrt(3.));
  static double a1 = .5*(1.+1./sqrt(3.));
  //  double a0 = 0.;
  //  double a1 = .5;
  //  double a2 = 1.;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs_.size(); i++)
  {
    edg2_t *e = &edgs_[i];
    if (!e->is_split && e->loc == FCE)
      if ((!integrate_specific && e->c0 >= 0) || (integrate_specific && e->c0 == num))
      {
        mapping_edg(xyz, i, a0); X.push_back(xyz[0]); Y.push_back(xyz[1]); weights.push_back(jacobian_edg(i, a0)/2.);
        mapping_edg(xyz, i, a1); X.push_back(xyz[0]); Y.push_back(xyz[1]); weights.push_back(jacobian_edg(i, a1)/2.);
      }
  }

}

void simplex2_mls_q_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
  bool integrate_specific = (num0 != -1 && num1 != -1);

  for (unsigned int i = 0; i < vtxs_.size(); i++)
  {
    vtx2_t *v = &vtxs_[i];
    if (v->loc == PNT)
      if ( !integrate_specific
           || (integrate_specific
               && (v->c0 == num0 || v->c1 == num0)
               && (v->c0 == num1 || v->c1 == num1)) )
      {
        X.push_back(v->x); Y.push_back(v->y); weights.push_back(1.);
      }
  }
}

void simplex2_mls_q_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
  double xyz[2];

  // quadrature points
  static double a0 = .5*(1.-1./sqrt(3.));
  static double a1 = .5*(1.+1./sqrt(3.));
//  double a0 = 0.;
//  double a1 = .5;
//  double a2 = 1.;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs_.size(); i++)
  {
    edg2_t *e = &edgs_[i];
    if (!e->is_split && e->loc == INS)
      if (e->dir == dir)
      {
        mapping_edg(xyz, i, a0); X.push_back(xyz[0]); Y.push_back(xyz[1]); weights.push_back(jacobian_edg(i, a0)/2.);
        mapping_edg(xyz, i, a1); X.push_back(xyz[0]); Y.push_back(xyz[1]); weights.push_back(jacobian_edg(i, a1)/2.);
      }
  }
}





//--------------------------------------------------
// Jacobians
//--------------------------------------------------
double simplex2_mls_q_t::jacobian_edg(int n_edg, double a)
{
  edg2_t *edg = &edgs_[n_edg];

  double Na[3] = {-3.+4.*a, 4.-8.*a, -1.+4.*a};

  double X = vtxs_[edg->vtx0].x * Na[0] + vtxs_[edg->vtx1].x * Na[1] + vtxs_[edg->vtx2].x * Na[2];
  double Y = vtxs_[edg->vtx0].y * Na[0] + vtxs_[edg->vtx1].y * Na[1] + vtxs_[edg->vtx2].y * Na[2];

  return sqrt(X*X+Y*Y);
}

double simplex2_mls_q_t::jacobian_tri(int n_tri, double a, double b)
{
  tri2_t *tri = &tris_[n_tri];

  double Na[6] = {-3.+4.*a+4.*b,  4.*a-1.,  0,        4.-8.*a-4.*b, 4.*b, -4.*b         };
  double Nb[6] = {-3.+4.*a+4.*b,  0,        4.*b-1., -4.*a,         4.*a,  4.-4.*a-8.*b };

  double j00 = vtxs_[tri->vtx0].x*Na[0] + vtxs_[tri->vtx1].x*Na[1] + vtxs_[tri->vtx2].x*Na[2] + vtxs_[edgs_[tri->edg2].vtx1].x*Na[3] + vtxs_[edgs_[tri->edg0].vtx1].x*Na[4] + vtxs_[edgs_[tri->edg1].vtx1].x*Na[5];
  double j10 = vtxs_[tri->vtx0].y*Na[0] + vtxs_[tri->vtx1].y*Na[1] + vtxs_[tri->vtx2].y*Na[2] + vtxs_[edgs_[tri->edg2].vtx1].y*Na[3] + vtxs_[edgs_[tri->edg0].vtx1].y*Na[4] + vtxs_[edgs_[tri->edg1].vtx1].y*Na[5];
  double j01 = vtxs_[tri->vtx0].x*Nb[0] + vtxs_[tri->vtx1].x*Nb[1] + vtxs_[tri->vtx2].x*Nb[2] + vtxs_[edgs_[tri->edg2].vtx1].x*Nb[3] + vtxs_[edgs_[tri->edg0].vtx1].x*Nb[4] + vtxs_[edgs_[tri->edg1].vtx1].x*Nb[5];
  double j11 = vtxs_[tri->vtx0].y*Nb[0] + vtxs_[tri->vtx1].y*Nb[1] + vtxs_[tri->vtx2].y*Nb[2] + vtxs_[edgs_[tri->edg2].vtx1].y*Nb[3] + vtxs_[edgs_[tri->edg0].vtx1].y*Nb[4] + vtxs_[edgs_[tri->edg1].vtx1].y*Nb[5];

  return fabs(j00*j11-j01*j10);
}





//--------------------------------------------------
// Interpolation
//--------------------------------------------------
double simplex2_mls_q_t::interpolate_from_parent(std::vector<double> &f, double x, double y)
{
  // map real point to reference element
  vtx2_t *v0 = &vtxs_[0];

  double D[3] = { x - v0->x,
                  y - v0->y };

  double a = map_parent_to_ref_[2*0+0]*D[0] + map_parent_to_ref_[2*0+1]*D[1];
  double b = map_parent_to_ref_[2*1+0]*D[0] + map_parent_to_ref_[2*1+1]*D[1];

//  // map real point to reference element
//  vtx2_t *v0 = &vtxs_[0];
//  vtx2_t *v1 = &vtxs_[1];
//  vtx2_t *v2 = &vtxs_[2];

//  double a = ( (x-v0->x)*(v2->y-v0->y) - (y-v0->y)*(v2->x-v0->x) ) / ( (v1->x-v0->x)*(v2->y-v0->y) - (v1->y-v0->y)*(v2->x-v0->x) );
//  double b = ( (x-v0->x)*(v1->y-v0->y) - (y-v0->y)*(v1->x-v0->x) ) / ( (v2->x-v0->x)*(v1->y-v0->y) - (v2->y-v0->y)*(v1->x-v0->x) );


  // compute nodal functions
  double N[nodes_per_tri_]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

  double result = 0;

  for (short i = 0; i < nodes_per_tri_; ++i)
  {
    result += N[i]*f[i];
  }

  return result;
}

double simplex2_mls_q_t::interpolate_from_parent(double x, double y)
{
  // map real point to reference element
  vtx2_t *v0 = &vtxs_[0];

  double D[3] = { x - v0->x,
                  y - v0->y };

  double a = map_parent_to_ref_[2*0+0]*D[0] + map_parent_to_ref_[2*0+1]*D[1];
  double b = map_parent_to_ref_[2*1+0]*D[0] + map_parent_to_ref_[2*1+1]*D[1];

//  // map real point to reference element
//  vtx2_t *v0 = &vtxs_[0];
//  vtx2_t *v1 = &vtxs_[1];
//  vtx2_t *v2 = &vtxs_[2];

//  double a = ( (x-v0->x)*(v2->y-v0->y) - (y-v0->y)*(v2->x-v0->x) ) / ( (v1->x-v0->x)*(v2->y-v0->y) - (v1->y-v0->y)*(v2->x-v0->x) );
//  double b = ( (x-v0->x)*(v1->y-v0->y) - (y-v0->y)*(v1->x-v0->x) ) / ( (v2->x-v0->x)*(v1->y-v0->y) - (v2->y-v0->y)*(v1->x-v0->x) );


  // compute nodal functions
  double N[nodes_per_tri_]  = {(1.-a-b)*(1.-2.*a-2.*b),  a*(2.*a-1.),  b*(2.*b-1.),  4.*a*(1.-a-b),  4.*a*b, 4.*b*(1.-a-b)};

  double result = 0;

  for (short i = 0; i < nodes_per_tri_; ++i)
  {
    result += N[i]*vtxs_[i].value;
  }

  return result;
}





//--------------------------------------------------
// Mapping
//--------------------------------------------------
void simplex2_mls_q_t::mapping_edg(double *xyz, int n_edg, double a)
{
  edg2_t *edg = &edgs_[n_edg];

  double N0 = 1.-3.*a+2.*a*a;
  double N1 = 4.*a-4.*a*a;
  double N2 = -a+2.*a*a;

  xyz[0] = vtxs_[edg->vtx0].x * N0 + vtxs_[edg->vtx1].x * N1 + vtxs_[edg->vtx2].x * N2;
  xyz[1] = vtxs_[edg->vtx0].y * N0 + vtxs_[edg->vtx1].y * N1 + vtxs_[edg->vtx2].y * N2;

#ifdef SIMPLEX2_MLS_Q_DEBUG
  if (not_finite(xyz[0]) ||
      not_finite(xyz[1]) )
        throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) (simplex2_mls_q_t) Something went wrong during integration.");
#endif
}

void simplex2_mls_q_t::mapping_tri(double *xyz, int n_tri, double *ab)
{
  tri2_t *tri = &tris_[n_tri];

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

#ifdef SIMPLEX2_MLS_Q_DEBUG
  if (not_finite(xyz[0]) ||
      not_finite(xyz[1]) )
        throw std::domain_error("[CASL_ERROR]: (simplex2_mls_q_t) (simplex2_mls_q_t) Something went wrong during integration.");
#endif
}




//--------------------------------------------------
// Computation tools
//--------------------------------------------------
double simplex2_mls_q_t::length(int v0, int v1)
{
  return sqrt(pow(vtxs_[v0].x - vtxs_[v1].x, 2.0)
            + pow(vtxs_[v0].y - vtxs_[v1].y, 2.0));
}

double simplex2_mls_q_t::length(int e)
{
  return ( jacobian_edg(e, 0.)    +
           jacobian_edg(e, .5)*4. +
           jacobian_edg(e, 1.) )/6.;
}

double simplex2_mls_q_t::area(int v0, int v1, int v2)
{
  double x01 = vtxs_[v1].x - vtxs_[v0].x; double x02 = vtxs_[v2].x - vtxs_[v0].x;
  double y01 = vtxs_[v1].y - vtxs_[v0].y; double y02 = vtxs_[v2].y - vtxs_[v0].y;

  return 0.5*fabs(x01*y02-y01*x02);
}





//--------------------------------------------------
// Debugging tools
//--------------------------------------------------
#ifdef SIMPLEX2_MLS_Q_DEBUG
bool simplex2_mls_q_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = (edgs_[e0].vtx0 == v1 || edgs_[e0].vtx2 == v1) && (edgs_[e0].vtx0 == v2 || edgs_[e0].vtx2 == v2);
  result = result && (edgs_[e1].vtx0 == v0 || edgs_[e1].vtx2 == v0) && (edgs_[e1].vtx0 == v2 || edgs_[e1].vtx2 == v2);
  result = result && (edgs_[e2].vtx0 == v0 || edgs_[e2].vtx2 == v0) && (edgs_[e2].vtx0 == v1 || edgs_[e2].vtx2 == v1);
  return result;
}

bool simplex2_mls_q_t::tri_is_ok(int t)
{
  tri2_t *tri = &tris_[t];
  bool result = tri_is_ok(tri->vtx0, tri->vtx1, tri->vtx2, tri->edg0, tri->edg1, tri->edg2);
  if (!result)
  {
    std::cout << "Inconsistent triangle!\n";
  }
  return result;
}
#endif
