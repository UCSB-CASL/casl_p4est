#include "simplex3_mls_l.h"

//--------------------------------------------------
// Class Constructors
//--------------------------------------------------
simplex3_mls_l_t::simplex3_mls_l_t(double x0, double y0, double z0,
                                   double x1, double y1, double z1,
                                   double x2, double y2, double z2,
                                   double x3, double y3, double z3, double eps_rel)
{
  // usually there will be only one cut
  vtxs_.reserve(8);
  edgs_.reserve(27);
  tris_.reserve(20);
  tets_.reserve(6);

  // fill the vectors with the initial structure
  vtxs_.push_back(vtx3_t(x0,y0,z0));
  vtxs_.push_back(vtx3_t(x1,y1,z1));
  vtxs_.push_back(vtx3_t(x2,y2,z2));
  vtxs_.push_back(vtx3_t(x3,y3,z3));

  edgs_.push_back(edg3_t(0,1));
  edgs_.push_back(edg3_t(0,2));
  edgs_.push_back(edg3_t(0,3));
  edgs_.push_back(edg3_t(1,2));
  edgs_.push_back(edg3_t(1,3));
  edgs_.push_back(edg3_t(2,3));

  tris_.push_back(tri3_t(1,2,3,5,4,3));
  tris_.push_back(tri3_t(0,2,3,5,2,1));
  tris_.push_back(tri3_t(0,1,3,4,2,0));
  tris_.push_back(tri3_t(0,1,2,3,1,0));

  tets_.push_back(tet3_t(0,1,2,3,0,1,2,3));

  // pre-compute the simplex volume for interpolation
  vol_ = volume(0, 1, 2, 3);

#ifdef SIMPLEX3_MLS_L_T_DEBUG
  if (vol_ < 1000.*DBL_MIN)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) Simplex has zero volume.");
#endif

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

  eps_rel_ = eps_rel;

  eps_ = eps_rel_*lmin_;

  use_linear_ = true;
}





//--------------------------------------------------
// Domain Reconstruction
//--------------------------------------------------
void simplex3_mls_l_t::construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  num_phi_ = acn.size();

  if (clr.size() != num_phi_) std::invalid_argument("[CASL_ERROR]: (simplex3_mls_l_t) Numbers of actions and colors are not equal.");
  if (phi.size() != num_phi_*nodes_per_tet_) std::invalid_argument("[CASL_ERROR]: (simplex3_mls_l_t) Numbers of actions and colors are not equal.");

  // loop over LSFs
  for (short phi_idx = 0; phi_idx < acn.size(); ++phi_idx)
  {
    for (int i = 0; i < nodes_per_tet_; ++i)
    {
      vtxs_[i].value = phi[phi_idx*nodes_per_tet_ + i];
      perturb(vtxs_[i].value, phi_perturbance_);
    }

    // interpolate to all vertices
    for (int i = nodes_per_tet_; i < vtxs_.size(); ++i)
    {
      interpolate_from_parent(i);
      perturb(vtxs_[i].value, phi_perturbance_);
    }

    // split all elements
    int n;
    n = vtxs_.size(); for (int i = 0; i < n; i++) do_action_vtx(i, clr[phi_idx], acn[phi_idx]);
    n = edgs_.size(); for (int i = 0; i < n; i++) do_action_edg(i, clr[phi_idx], acn[phi_idx]);
    n = tris_.size(); for (int i = 0; i < n; i++) do_action_tri(i, clr[phi_idx], acn[phi_idx]);
    n = tets_.size(); for (int i = 0; i < n; i++) do_action_tet(i, clr[phi_idx], acn[phi_idx]);
  }
}





//--------------------------------------------------
// Splitting
//--------------------------------------------------
void simplex3_mls_l_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx3_t *vtx = &vtxs_[n_vtx];

  switch (action)
  {
    case INTERSECTION:  if (vtx->value > 0)                                                       vtx->set(OUT, -1, -1, -1);  break;
    case ADDITION:      if (vtx->value < 0)                                                       vtx->set(INS, -1, -1, -1);  break;
    case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==LNE || vtx->loc==PNT)  vtx->set(FCE, cn, -1, -1);  break;
  }
}

void simplex3_mls_l_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg3_t *edg = &edgs_[n_edg];

  int c0 = edg->c0;
  int c1 = edg->c1;

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx1)) swap(edg->vtx0, edg->vtx1);

  int num_negatives = 0;
  if (vtxs_[edg->vtx0].value < 0) num_negatives++;
  if (vtxs_[edg->vtx1].value < 0) num_negatives++;

#ifdef SIMPLEX3_MLS_L_T_DEBUG
  edg->type = num_negatives;
#endif

  // auxiliary variables
  vtx3_t *c_vtx01;
  edg3_t *c_edg0, *c_edg1;
  double r;

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
      /* split an edge */
      edg->is_split = true;

      // new vertex
      if (use_linear_) r = find_intersection_linear    (edg->vtx0, edg->vtx1);
      else             r = find_intersection_quadratic (n_edg);

      vtxs_.push_back(vtx3_t(vtxs_[edg->vtx0].x*(1.-r) + vtxs_[edg->vtx1].x*r,
                             vtxs_[edg->vtx0].y*(1.-r) + vtxs_[edg->vtx1].y*r,
                             vtxs_[edg->vtx0].z*(1.-r) + vtxs_[edg->vtx1].z*r));

      vtxs_.back().n_vtx0 = edg->vtx0;
      vtxs_.back().n_vtx1 = edg->vtx1;
      vtxs_.back().ratio  = r;

      edg->c_vtx01 = vtxs_.size()-1;

      // new edges
      edgs_.push_back(edg3_t(edg->vtx0,    edg->c_vtx01)); edg = &edgs_[n_edg]; // edges might have changed their addresses
      edgs_.push_back(edg3_t(edg->c_vtx01, edg->vtx1   )); edg = &edgs_[n_edg];

      edg->c_edg0 = edgs_.size()-2;
      edg->c_edg1 = edgs_.size()-1;

      /* apply rules */
      c_vtx01 = &vtxs_[edg->c_vtx01];
      c_edg0  = &edgs_[edg->c_edg0];
      c_edg1  = &edgs_[edg->c_edg1];

#ifdef SIMPLEX3_MLS_L_T_DEBUG
      c_vtx01->p_edg = n_edg;
      c_edg0->p_edg  = n_edg;
      c_edg1->p_edg  = n_edg;
#endif

      switch (action)
      {
        case INTERSECTION:
          switch (edg->loc)
          {
            case OUT: c_vtx01->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
            case INS: c_vtx01->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
            case FCE: c_vtx01->set(LNE, c0, cn, -1); c_edg0->set(FCE, c0, -1); c_edg1->set(OUT, -1, -1); if (c0==cn)            c_vtx01->set(FCE, c0, -1, -1);  break;
            case LNE: c_vtx01->set(PNT, c0, c1, cn); c_edg0->set(LNE, c0, c1); c_edg1->set(OUT, -1, -1); if (c0==cn || c1==cn)  c_vtx01->set(LNE, c0, c1, -1);  break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          }
          break;
        case ADDITION:
          switch (edg->loc)
          {
            case OUT: c_vtx01->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
            case INS: c_vtx01->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1);                                                        break;
            case FCE: c_vtx01->set(LNE, c0, cn, -1); c_edg0->set(INS, -1, -1); c_edg1->set(FCE, c0, -1); if (c0==cn)            c_vtx01->set(FCE, c0, -1, -1);  break;
            case LNE: c_vtx01->set(PNT, c0, c1, cn); c_edg0->set(INS, -1, -1); c_edg1->set(LNE, c0, c1); if (c0==cn || c1==cn)  c_vtx01->set(LNE, c0, c1, -1);  break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          }
          break;
        case COLORATION:
          switch (edg->loc)
          {
            case OUT: c_vtx01->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
            case INS: c_vtx01->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1);                                                        break;
            case FCE: c_vtx01->set(LNE, c0, cn, -1); c_edg0->set(FCE, cn, -1); c_edg1->set(FCE, c0, -1); if (c0==cn)            c_vtx01->set(FCE, c0, -1, -1);  break;
            case LNE: c_vtx01->set(PNT, c0, c1, cn); c_edg0->set(FCE, cn, -1); c_edg1->set(LNE, c0, c1); if (c0==cn || c1==cn)  c_vtx01->set(LNE, c0, c1, -1);  break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          }
          break;
      }
      break;
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

void simplex3_mls_l_t::do_action_tri(int n_tri, int cn, action_t action)
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

#ifdef SIMPLEX3_MLS_L_T_DEBUG
  tri->type = num_negatives;

  /* check whether vertices of a triangle and edges coincide */
  if (tri->vtx0 != edgs_[tri->edg1].vtx0 || tri->vtx0 != edgs_[tri->edg2].vtx0 ||
      tri->vtx1 != edgs_[tri->edg0].vtx0 || tri->vtx1 != edgs_[tri->edg2].vtx1 ||
      tri->vtx2 != edgs_[tri->edg0].vtx1 || tri->vtx2 != edgs_[tri->edg1].vtx1)
  {
    std::cout << vtxs_[tri->vtx0].value << " " << vtxs_[tri->vtx1].value << " " << vtxs_[tri->vtx2].value << std::endl;
    std::cout << vtxs_[tri->vtx0].value - vtxs_[tri->vtx1].value << " "
              << vtxs_[tri->vtx1].value - vtxs_[tri->vtx2].value << " "
              << vtxs_[tri->vtx2].value - vtxs_[tri->vtx0].value << std::endl;
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) Vertices of a triangle and edges do not coincide after sorting.");
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
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a triangle one of edges has an unexpected type.");
#endif

  // auxiliary variables
  edg3_t *c_edg0, *c_edg1;
  edg3_t *edg0, *edg1, *edg2;
  tri3_t *c_tri0, *c_tri1, *c_tri2;

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
      /* split a triangle */
      tri->is_split = true;

      // new vertices
      tri->c_vtx01 = edgs_[tri->edg2].c_vtx01;
      tri->c_vtx02 = edgs_[tri->edg1].c_vtx01;

      // new edges
      edgs_.push_back(edg3_t(tri->c_vtx01, tri->c_vtx02));
      edgs_.push_back(edg3_t(tri->c_vtx01, tri->vtx2   ));

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

#ifdef SIMPLEX3_MLS_L_T_DEBUG
      if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a triangle one of child triangles is not consistent.");

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
            case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
            case INS: c_edg0->set(FCE, cn, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
            case FCE: c_edg0->set(LNE, cc, cn); c_edg1->set(OUT, -1, -1); c_tri0->set(FCE, cc); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tri->loc)
          {
            case OUT: c_edg0->set(FCE, cn, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
            case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
            case FCE: c_edg0->set(LNE, cc, cn); c_edg1->set(FCE, cc, -1); c_tri0->set(INS, -1); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tri->loc)
          {
            case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
            case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
            case FCE: c_edg0->set(LNE, cc, cn); c_edg1->set(FCE, cc, -1); c_tri0->set(FCE, cn); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
      }
      break;

    case 2: // (--+)
      /* split a triangle */
      tri->is_split = true;

      tri->c_vtx02 = edgs_[tri->edg1].c_vtx01;
      tri->c_vtx12 = edgs_[tri->edg0].c_vtx01;

      // create new edges
      edgs_.push_back(edg3_t(tri->vtx0,    tri->c_vtx12));
      edgs_.push_back(edg3_t(tri->c_vtx02, tri->c_vtx12));

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

#ifdef SIMPLEX3_MLS_L_T_DEBUG
      if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a triangle one of child triangles is not consistent.");

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
            case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
            case INS: c_edg0->set(INS, -1, -1); c_edg1->set(FCE, cn, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
            case FCE: c_edg0->set(FCE, cc, -1); c_edg1->set(LNE, cc, cn); c_tri0->set(FCE, cc); c_tri1->set(FCE, cc); c_tri2->set(OUT, -1); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tri->loc)
          {
            case OUT: c_edg0->set(INS, -1, -1); c_edg1->set(FCE, cn, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
            case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
            case FCE: c_edg0->set(INS, -1, -1); c_edg1->set(LNE, cc, cn); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tri->loc)
          {
            case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
            case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
            case FCE: c_edg0->set(FCE, cn, -1); c_edg1->set(LNE, cc, cn); c_tri0->set(FCE, cn); c_tri1->set(FCE, cn); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
      }
      break;

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

void simplex3_mls_l_t::do_action_tet(int n_tet, int cn, action_t action)
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
  
#ifdef SIMPLEX3_MLS_L_T_DEBUG
  tet->type = num_negatives;
  
  /* check whether vertices coincide */
  if (tet->vtx0 != tris_[tet->tri1].vtx0 || tet->vtx0 != tris_[tet->tri2].vtx0 || tet->vtx0 != tris_[tet->tri3].vtx0 ||
      tet->vtx1 != tris_[tet->tri0].vtx0 || tet->vtx1 != tris_[tet->tri2].vtx1 || tet->vtx1 != tris_[tet->tri3].vtx1 ||
      tet->vtx2 != tris_[tet->tri1].vtx1 || tet->vtx2 != tris_[tet->tri0].vtx1 || tet->vtx2 != tris_[tet->tri3].vtx2 ||
      tet->vtx3 != tris_[tet->tri1].vtx2 || tet->vtx3 != tris_[tet->tri2].vtx2 || tet->vtx3 != tris_[tet->tri0].vtx2)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) Vertices of a tetrahedron do not coincide with vertices of triangles after sorting.");
  
  /* check whether edges coincide */
  if (tris_[tet->tri0].edg0 != tris_[tet->tri1].edg0 || tris_[tet->tri0].edg1 != tris_[tet->tri2].edg0 || tris_[tet->tri0].edg2 != tris_[tet->tri3].edg0 ||
      tris_[tet->tri1].edg1 != tris_[tet->tri2].edg1 || tris_[tet->tri1].edg2 != tris_[tet->tri3].edg1 ||
      tris_[tet->tri2].edg2 != tris_[tet->tri3].edg2)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) Edges of different triangles in a tetrahedron do not coincide.");
  
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
    throw std::domain_error("CASL_ERROR]: (simplex3_mls_l_t) While splitting a tetrahedron one of triangles has an unexpected type.");
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
      switch (action)
      {
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
      
#ifdef SIMPLEX3_MLS_L_T_DEBUG
      if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a tetrahedron one of child triangles is not consistent.");
      
      if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) || !tet_is_ok(tet->c_tet3))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");
      
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
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(FCE, cn); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(INS); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
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

      // new edge
      edgs_.push_back(edg3_t(tet->c_vtx03, tet->c_vtx12));
      tet->c_edg = edgs_.size()-1;

      // new triangles
      tris_.push_back(tri3_t(tet->vtx0,    tet->c_vtx12, tet->c_vtx13, tris_[tet->tri0].c_edg0, tris_[tet->tri2].c_edg0,             tris_[tet->tri3].c_edg0             ));
      tris_.push_back(tri3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx12, tet->c_edg,              tris_[tet->tri3].c_edg0,             edgs_[tris_[tet->tri1].edg1].c_edg0 ));
      tris_.push_back(tri3_t(tet->c_vtx02, tet->c_vtx03, tet->c_vtx12, tet->c_edg,              tris_[tet->tri3].c_edg1,             tris_[tet->tri1].c_edg0             ));
      tris_.push_back(tri3_t(tet->c_vtx03, tet->c_vtx12, tet->c_vtx13, tris_[tet->tri0].c_edg0, tris_[tet->tri2].c_edg1,             tet->c_edg                          ));
      tris_.push_back(tri3_t(tet->c_vtx03, tet->c_vtx12, tet->vtx3,    tris_[tet->tri0].c_edg1, edgs_[tris_[tet->tri1].edg1].c_edg1, tet->c_edg                          ));
      tris_.push_back(tri3_t(tet->c_vtx02, tet->c_vtx12, tet->vtx3,    tris_[tet->tri0].c_edg1, tris_[tet->tri1].c_edg1,             tris_[tet->tri3].c_edg1             ));

      n_tris = tris_.size();
      tet->c_tri0 = n_tris-6;
      tet->c_tri1 = n_tris-5;
      tet->c_tri2 = n_tris-4;
      tet->c_tri3 = n_tris-3;
      tet->c_tri4 = n_tris-2;
      tet->c_tri5 = n_tris-1;

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
      
      n_tets = tets_.size();
      tet->c_tet0 = n_tets-6;
      tet->c_tet1 = n_tets-5;
      tet->c_tet2 = n_tets-4;
      tet->c_tet3 = n_tets-3;
      tet->c_tet4 = n_tets-2;
      tet->c_tet5 = n_tets-1;
      
      /* apply rules */
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
      
      
#ifdef SIMPLEX3_MLS_L_T_DEBUG
      if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2) ||
          !tri_is_ok(tet->c_tri3) || !tri_is_ok(tet->c_tri4) || !tri_is_ok(tet->c_tri5))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a tetrahedron one of child triangles is not consistent.");
      
      if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) ||
          !tet_is_ok(tet->c_tet3) || !tet_is_ok(tet->c_tet4) || !tet_is_ok(tet->c_tet5))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");
      
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
      
      if (action == INTERSECTION || action == ADDITION)
      {
        c_tri2->p_lsf = cn;
        c_tri3->p_lsf = cn;
      }
      
      switch (action)
      {
        case INTERSECTION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT,-1);  c_tet0->set(OUT); c_edg->set(OUT,-1,-1);
              c_tri1->set(OUT,-1);  c_tet1->set(OUT);
              c_tri2->set(OUT,-1);  c_tet2->set(OUT);
              c_tri3->set(OUT,-1);  c_tet3->set(OUT);
              c_tri4->set(OUT,-1);  c_tet4->set(OUT);
              c_tri5->set(OUT,-1);  c_tet5->set(OUT);
              break;
            case INS: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(FCE,cn,-1);
              c_tri1->set(INS,-1);  c_tet1->set(INS);
              c_tri2->set(FCE,cn);  c_tet2->set(INS);
              c_tri3->set(FCE,cn);  c_tet3->set(OUT);
              c_tri4->set(OUT,-1);  c_tet4->set(OUT);
              c_tri5->set(OUT,-1);  c_tet5->set(OUT);
              break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(FCE,cn,-1);
              c_tri1->set(INS,-1);  c_tet1->set(INS);
              c_tri2->set(FCE,cn);  c_tet2->set(INS);
              c_tri3->set(FCE,cn);  c_tet3->set(OUT);
              c_tri4->set(OUT,-1);  c_tet4->set(OUT);
              c_tri5->set(OUT,-1);  c_tet5->set(OUT);
              break;
            case INS: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(INS,-1,-1);
              c_tri1->set(INS,-1);  c_tet1->set(INS);
              c_tri2->set(INS,-1);  c_tet2->set(INS);
              c_tri3->set(INS,-1);  c_tet3->set(INS);
              c_tri4->set(INS,-1);  c_tet4->set(INS);
              c_tri5->set(INS,-1);  c_tet5->set(INS);
              break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT,-1);  c_tet0->set(OUT); c_edg->set(OUT,-1,-1);
              c_tri1->set(OUT,-1);  c_tet1->set(OUT);
              c_tri2->set(OUT,-1);  c_tet2->set(OUT);
              c_tri3->set(OUT,-1);  c_tet3->set(OUT);
              c_tri4->set(OUT,-1);  c_tet4->set(OUT);
              c_tri5->set(OUT,-1);  c_tet5->set(OUT);
              break;
            case INS: c_tri0->set(INS,-1);  c_tet0->set(INS); c_edg->set(INS,-1,-1);
              c_tri1->set(INS,-1);  c_tet1->set(INS);
              c_tri2->set(INS,-1);  c_tet2->set(INS);
              c_tri3->set(INS,-1);  c_tet3->set(INS);
              c_tri4->set(INS,-1);  c_tet4->set(INS);
              c_tri5->set(INS,-1);  c_tet5->set(INS);
              break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
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
      
      n_tris = tris_.size();
      tet->c_tri0 = n_tris - 3;
      tet->c_tri1 = n_tris - 2;
      tet->c_tri2 = n_tris - 1;
      
      tri0 = &tris_[tet->tri0];
      tri1 = &tris_[tet->tri1];
      tri2 = &tris_[tet->tri2];
      tri3 = &tris_[tet->tri3];
      
      // new tetrahedra
      tets_.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->vtx2,    tet->c_vtx23, tri0->c_tri0, tri1->c_tri0, tet->c_tri0,  tet->tri3   )); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->vtx0,    tet->vtx1,    tet->c_vtx13, tet->c_vtx23, tri0->c_tri1, tet->c_tri1,  tet->c_tri0,  tri2->c_tri0)); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->vtx0,    tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tet->c_tri2,  tet->c_tri1,  tri1->c_tri1, tri2->c_tri1)); tet = &tets_[n_tet];
      tets_.push_back(tet3_t(tet->c_vtx03, tet->c_vtx13, tet->c_vtx23, tet->vtx3,    tri0->c_tri2, tri1->c_tri2, tri2->c_tri2, tet->c_tri2 )); tet = &tets_[n_tet];
      
      n_tets = tets_.size();
      tet->c_tet0 = n_tets-4;
      tet->c_tet1 = n_tets-3;
      tet->c_tet2 = n_tets-2;
      tet->c_tet3 = n_tets-1;
      
      /* apply rules */
      c_tri0 = &tris_[tet->c_tri0];
      c_tri1 = &tris_[tet->c_tri1];
      c_tri2 = &tris_[tet->c_tri2];
      
      c_tet0 = &tets_[tet->c_tet0];
      c_tet1 = &tets_[tet->c_tet1];
      c_tet2 = &tets_[tet->c_tet2];
      c_tet3 = &tets_[tet->c_tet3];
      
#ifdef SIMPLEX3_MLS_L_T_DEBUG
      if (!tri_is_ok(tet->c_tri0) || !tri_is_ok(tet->c_tri1) || !tri_is_ok(tet->c_tri2))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a tetrahedron one of child triangles is not consistent.");
      
      if (!tet_is_ok(tet->c_tet0) || !tet_is_ok(tet->c_tet1) || !tet_is_ok(tet->c_tet2) || !tet_is_ok(tet->c_tet3))
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) While splitting a tetrahedron one of child tetrahedra is not consistent.");
      
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
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(FCE,cn); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tet->loc)
          {
            case OUT: c_tri0->set(OUT,-1); c_tri1->set(OUT,-1); c_tri2->set(OUT,-1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
            case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
            default:
#ifdef SIMPLEX3_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) An element has wrong location.");
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
// Computation tools
//--------------------------------------------------
double simplex3_mls_l_t::length(int vtx0, int vtx1)
{
  return sqrt(pow(vtxs_[vtx0].x - vtxs_[vtx1].x, 2.0)
            + pow(vtxs_[vtx0].y - vtxs_[vtx1].y, 2.0)
            + pow(vtxs_[vtx0].z - vtxs_[vtx1].z, 2.0));
}

double simplex3_mls_l_t::area(int vtx0, int vtx1, int vtx2)
{
  double x01 = vtxs_[vtx1].x - vtxs_[vtx0].x; double x02 = vtxs_[vtx2].x - vtxs_[vtx0].x;
  double y01 = vtxs_[vtx1].y - vtxs_[vtx0].y; double y02 = vtxs_[vtx2].y - vtxs_[vtx0].y;
  double z01 = vtxs_[vtx1].z - vtxs_[vtx0].z; double z02 = vtxs_[vtx2].z - vtxs_[vtx0].z;

  return 0.5*sqrt(pow(y01*z02-z01*y02,2.0) + pow(z01*x02-x01*z02,2.0) + pow(x01*y02-y01*x02,2.0));
}

double simplex3_mls_l_t::volume(int vtx0, int vtx1, int vtx2, int vtx3)
{
  double a11 = vtxs_[vtx1].x-vtxs_[vtx0].x; double a12 = vtxs_[vtx1].y-vtxs_[vtx0].y; double a13 = vtxs_[vtx1].z-vtxs_[vtx0].z;
  double a21 = vtxs_[vtx2].x-vtxs_[vtx0].x; double a22 = vtxs_[vtx2].y-vtxs_[vtx0].y; double a23 = vtxs_[vtx2].z-vtxs_[vtx0].z;
  double a31 = vtxs_[vtx3].x-vtxs_[vtx0].x; double a32 = vtxs_[vtx3].y-vtxs_[vtx0].y; double a33 = vtxs_[vtx3].z-vtxs_[vtx0].z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}

double simplex3_mls_l_t::volume(vtx3_t &vtx0, vtx3_t &vtx1, vtx3_t &vtx2, vtx3_t &vtx3)
{
  double a11 = vtx1.x-vtx0.x; double a12 = vtx1.y-vtx0.y; double a13 = vtx1.z-vtx0.z;
  double a21 = vtx2.x-vtx0.x; double a22 = vtx2.y-vtx0.y; double a23 = vtx2.z-vtx0.z;
  double a31 = vtx3.x-vtx0.x; double a32 = vtx3.y-vtx0.y; double a33 = vtx3.z-vtx0.z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}





//--------------------------------------------------
// Quadrature Points
//--------------------------------------------------
void simplex3_mls_l_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  for (unsigned int i = 0; i < tets_.size(); i++)
    if (!tets_[i].is_split && tets_[i].loc == INS)
    {
      tet3_t *t = &tets_[i];

      double w = 0.25*volume(t->vtx0, t->vtx1, t->vtx2, t->vtx3);

      weights.push_back(w); X.push_back(vtxs_[t->vtx0].x); Y.push_back(vtxs_[t->vtx0].y); Z.push_back(vtxs_[t->vtx0].z);
      weights.push_back(w); X.push_back(vtxs_[t->vtx1].x); Y.push_back(vtxs_[t->vtx1].y); Z.push_back(vtxs_[t->vtx1].z);
      weights.push_back(w); X.push_back(vtxs_[t->vtx2].x); Y.push_back(vtxs_[t->vtx2].y); Z.push_back(vtxs_[t->vtx2].z);
      weights.push_back(w); X.push_back(vtxs_[t->vtx3].x); Y.push_back(vtxs_[t->vtx3].y); Z.push_back(vtxs_[t->vtx3].z);
    }
}

void simplex3_mls_l_t::quadrature_over_interface(int num0, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  bool integrate_specific = (num0 != -1);

  for (unsigned int i = 0; i < tris_.size(); i++)
  {
    tri3_t *t = &tris_[i];
    if (!t->is_split && t->loc == FCE)
      if (!integrate_specific
          || (integrate_specific && t->c == num0))
      {
        double w = area(t->vtx0, t->vtx1, t->vtx2)/3.0;

        weights.push_back(w); X.push_back(vtxs_[t->vtx0].x); Y.push_back(vtxs_[t->vtx0].y); Z.push_back(vtxs_[t->vtx0].z);
        weights.push_back(w); X.push_back(vtxs_[t->vtx1].x); Y.push_back(vtxs_[t->vtx1].y); Z.push_back(vtxs_[t->vtx1].z);
        weights.push_back(w); X.push_back(vtxs_[t->vtx2].x); Y.push_back(vtxs_[t->vtx2].y); Z.push_back(vtxs_[t->vtx2].z);
      }
  }
}

void simplex3_mls_l_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  bool integrate_specific = (num0 != -1 && num1 != -1);

  for (unsigned int i = 0; i < edgs_.size(); i++)
  {
    edg3_t *e = &edgs_[i];
    if (!e->is_split && e->loc == LNE)
      if ( !integrate_specific
           || (integrate_specific
               && (e->c0 == num0 || e->c1 == num0)
               && (e->c0 == num1 || e->c1 == num1)) )
      {
        double w = length(e->vtx0, e->vtx1)/2.0;

        weights.push_back(w); X.push_back(vtxs_[e->vtx0].x); Y.push_back(vtxs_[e->vtx0].y); Z.push_back(vtxs_[e->vtx0].z);
        weights.push_back(w); X.push_back(vtxs_[e->vtx1].x); Y.push_back(vtxs_[e->vtx1].y); Z.push_back(vtxs_[e->vtx1].z);
      }
  }
}

void simplex3_mls_l_t::quadrature_over_intersection(int num0, int num1, int num2, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
//  /* sort values */
//  if (num0 > num1) swap(num0, num1);
//  if (num1 > num2) swap(num1, num2);
//  if (num0 > num1) swap(num0, num1);

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
        weights.push_back(1); X.push_back(v->x); Y.push_back(v->y); Z.push_back(v->z);
      }
  }
}

void simplex3_mls_l_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  for (unsigned int i = 0; i < tris_.size(); i++)
  {
    tri3_t *t = &tris_[i];
    if (!t->is_split && t->loc == INS)
      if (t->dir == dir)
      {
        double w = area(t->vtx0, t->vtx1, t->vtx2)/3.0;

        weights.push_back(w); X.push_back(vtxs_[t->vtx0].x); Y.push_back(vtxs_[t->vtx0].y); Z.push_back(vtxs_[t->vtx0].z);
        weights.push_back(w); X.push_back(vtxs_[t->vtx1].x); Y.push_back(vtxs_[t->vtx1].y); Z.push_back(vtxs_[t->vtx1].z);
        weights.push_back(w); X.push_back(vtxs_[t->vtx2].x); Y.push_back(vtxs_[t->vtx2].y); Z.push_back(vtxs_[t->vtx2].z);
      }
  }
}





//--------------------------------------------------
// Interpolation
//--------------------------------------------------
void simplex3_mls_l_t::interpolate_from_neighbors(int v)
{
  vtx3_t *vtx = &vtxs_[v];
  vtx->value = (1.-vtx->ratio)*vtxs_[vtx->n_vtx0].value + (vtx->ratio)*vtxs_[vtx->n_vtx1].value;
}

void simplex3_mls_l_t::interpolate_from_parent(int v)
{
  double vol0 = volume(v, 1, 2, 3);
  double vol1 = volume(0, v, 2, 3);
  double vol2 = volume(0, 1, v, 3);
  double vol3 = volume(0, 1, 2, v);

  vtxs_[v].value = (vol0*vtxs_[0].value + vol1*vtxs_[1].value + vol2*vtxs_[2].value + vol3*vtxs_[3].value)/vol_;
}

void simplex3_mls_l_t::interpolate_from_parent(vtx3_t &vertex)
{
  double vol0 = volume(vertex,   vtxs_[1], vtxs_[2], vtxs_[3]);
  double vol1 = volume(vtxs_[0], vertex,   vtxs_[2], vtxs_[3]);
  double vol2 = volume(vtxs_[0], vtxs_[1], vertex,   vtxs_[3]);
  double vol3 = volume(vtxs_[0], vtxs_[1], vtxs_[2], vertex);

  vertex.value = (vol0*vtxs_[0].value + vol1*vtxs_[1].value + vol2*vtxs_[2].value + vol3*vtxs_[3].value)/vol_;
}





//--------------------------------------------------
// Intersections
//--------------------------------------------------
double simplex3_mls_l_t::find_intersection_linear(int v0, int v1)
{
  double f0 = vtxs_[v0].value;
  double f1 = vtxs_[v1].value;

#ifdef SIMPLEX3_MLS_L_T_DEBUG
  if (f0 <= 0 && f1 <= 0 ||
      f0 >= 0 && f1 >= 0)
    throw std::invalid_argument("[CASL_ERROR]: (simplex3_mls_l_t) Cannot find an intersection with an edge, values of a level-set function are of the same sign at end points.");
#endif

//  double l = length(v0, v1);
//  if (l <= 2.1*eps_) return .5;
//  double ratio = eps_/l;
//#ifdef SIMPLEX3_MLS_L_T_DEBUG
//  if(l <= 2.*eps_) throw std::invalid_argument("[CASL_ERROR]: (simplex3_mls_l_t) Vertices are too close.");
//#endif

//  bool f0_close = fabs(f0) < phi_tolerance_;
//  bool f1_close = fabs(f1) < phi_tolerance_;

//  if (f0_close && f1_close) return .5;
//  if (f0_close)             return ratio;
//  if (f1_close)             return 1.-ratio;

  if (fabs(f0) <= DBL_MIN || fabs(f0) >= fabs(f1-f0))
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) Interestion with an edge falls outside of the edge");

  double x = -f0/(f1-f0);

  if (x < 0. || x > 1.) throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) Interestion with an edge falls outside of the edge");

//  if (x < ratio)    x = ratio;
//  if (x > 1.-ratio) x = 1.-ratio;

  return x;
}

double simplex3_mls_l_t::find_intersection_quadratic(int e)
{
  throw std::invalid_argument("[CASL_ERROR]: (simplex3_mls_l_t) Quadratic intersections are not implemented.");

  vtx3_t *vtx0 = &vtxs_[edgs_[e].vtx0];
  vtx3_t *vtx1 = &vtxs_[edgs_[e].vtx1];
  double nx = vtx1->x - vtx0->x;
  double ny = vtx1->y - vtx0->y;
  double nz = vtx1->z - vtx0->z;
  double l = sqrt(nx*nx+ny*ny+nz*nz);
#ifdef SIMPLEX3_MLS_L_T_DEBUG
  if(l < 0.8*eps_) throw std::invalid_argument("[CASL_ERROR]: (simplex3_mls_l_t) Vertices are too close.");
#endif
  nx /= l;
  ny /= l;
  nz /= l;
  double f0 = vtx0->value;
  double f01 = edgs_[e].value;
  double f1 = vtx1->value;

  if (fabs(f0)  < 0.8*eps_) return (l-0.8*eps_)/l;
  if (fabs(f01) < 0.8*eps_) return 0.5;
  if (fabs(f1)  < 0.8*eps_) return (0.+0.8*eps_)/l;

#ifdef SIMPLEX3_MLS_L_T_DEBUG
  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: (simplex3_mls_l_t) Wrong arguments.");
#endif

  double fdd = (f1+f0-2.*f01)/(0.25*l*l);

  double c2 = 0.5*fdd;   // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = (f1-f0)/l; //  the expansion of f at the center of (a,b)
  double c0 = f01;

  double x;

  if(fabs(c2)<eps_) x = -c0/c1;
  else
  {
    if(f1<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    else     x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
  }
#ifdef SIMPLEX3_MLS_L_T_DEBUG
  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
#endif

//  if (x < -0.5*l) return (l-eps_)/l;
//  if (x > 0.5*l) return (0.+eps_)/l;

  return 1.-(x+0.5*l)/l;
}





//--------------------------------------------------
// Debugging
//--------------------------------------------------
#ifdef SIMPLEX3_MLS_L_T_DEBUG
bool simplex3_mls_l_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = (edgs_[e0].vtx0 == v1 || edgs_[e0].vtx1 == v1) && (edgs_[e0].vtx0 == v2 || edgs_[e0].vtx1 == v2);
  result = result && (edgs_[e1].vtx0 == v0 || edgs_[e1].vtx1 == v0) && (edgs_[e1].vtx0 == v2 || edgs_[e1].vtx1 == v2);
  result = result && (edgs_[e2].vtx0 == v0 || edgs_[e2].vtx1 == v0) && (edgs_[e2].vtx0 == v1 || edgs_[e2].vtx1 == v1);
  return result;
}

bool simplex3_mls_l_t::tri_is_ok(int t)
{
  tri3_t *tri = &tris_[t];
  bool result = tri_is_ok(tri->vtx0, tri->vtx1, tri->vtx2, tri->edg0, tri->edg1, tri->edg2);
  if (!result) std::cout << "Inconsistent triangle!\n";
  return result;
}

bool simplex3_mls_l_t::tet_is_ok(int s)
{
  bool result = true;
  tet3_t *tet = &tets_[s];

  tri3_t *tri;

  tri = &tris_[tet->tri0];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);

  tri = &tris_[tet->tri1];
  result = result && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);

  tri = &tris_[tet->tri2];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);

  tri = &tris_[tet->tri3];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0);

  return result;
}
#endif
