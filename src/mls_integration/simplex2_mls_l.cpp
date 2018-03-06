#include "simplex2_mls_l.h"

simplex2_mls_l_t::simplex2_mls_l_t(double x0, double y0,
                                   double x1, double y1,
                                   double x2, double y2,
                                   double eps_rel)
{
  // usually there will be only one cut
  vtxs_.reserve(4);
  edgs_.reserve(9);
  tris_.reserve(4);

  /* fill the vectors with the initial structure */
  vtxs_.push_back(vtx2_t(x0,y0));
  vtxs_.push_back(vtx2_t(x1,y1));
  vtxs_.push_back(vtx2_t(x2,y2));

  edgs_.push_back(edg2_t(1,2));
  edgs_.push_back(edg2_t(0,2));
  edgs_.push_back(edg2_t(0,1));

  tris_.push_back(tri2_t(0,1,2,0,1,2));

  edgs_[0].dir = 0;
  edgs_[1].dir = 1;
  edgs_[2].dir = 2;

  // pre-compute the simplex area for interpolation
  A_ = area(0, 1, 2);

#ifdef SIMPLEX2_MLS_L_T_DEBUG
  if (A_ < 1000.*DBL_MIN)
    throw std::domain_error("[CASL_ERROR]: (simplex3_mls_l_t) Simplex has zero volume.");
#endif

  // compute resolution limit (all edges with legnth < 2*eps_ will be split at the middle)
  double l01 = length(0, 1);
  double l02 = length(0, 2);
  double l12 = length(1, 2);

  lmin_ = l01;
  lmin_ = lmin_ < l02 ? lmin_ : l02;
  lmin_ = lmin_ < l12 ? lmin_ : l12;

  eps_rel_ = eps_rel;

  eps_ = eps_rel_*lmin_;

  use_linear_ = true;
}





//--------------------------------------------------
// Domain Reconstruction
//--------------------------------------------------
void simplex2_mls_l_t::construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  num_phi_ = acn.size();

  if (clr.size() != num_phi_) std::invalid_argument("[CASL_ERROR]: (simplex2_mls_l_t) Numbers of actions and colors are not equal.");
  if (phi.size() != num_phi_*nodes_per_tri_) std::invalid_argument("[CASL_ERROR]: (simplex2_mls_l_t) Numbers of actions and colors are not equal.");

  // loop over LSFs
  for (short phi_idx = 0; phi_idx < num_phi_; ++phi_idx)
  {
    phi_max_ = 0;

    for (int i = 0; i < nodes_per_tri_; ++i)
    {
      vtxs_[i].value = phi[phi_idx*nodes_per_tri_+i];
      phi_max_ = phi_max_ > fabs(vtxs_[i].value) ? phi_max_ : fabs(vtxs_[i].value);
    }

    phi_eps_ = phi_max_*eps_rel_;

    for (int i = 0; i < nodes_per_tri_; ++i)
      perturb(vtxs_[i].value, phi_eps_);

    // interpolate to all vertices
    for (int i = nodes_per_tri_; i < vtxs_.size(); ++i)
    {
      interpolate_from_parent(i);
      perturb(vtxs_[i].value, phi_eps_);
    }

    // split all elements
    int n;
    n = vtxs_.size(); for (int i = 0; i < n; i++) do_action_vtx(i, clr[phi_idx], acn[phi_idx]);
    n = edgs_.size(); for (int i = 0; i < n; i++) do_action_edg(i, clr[phi_idx], acn[phi_idx]);
    n = tris_.size(); for (int i = 0; i < n; i++) do_action_tri(i, clr[phi_idx], acn[phi_idx]);
  }
}





//--------------------------------------------------
// Splitting
//--------------------------------------------------
void simplex2_mls_l_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx2_t *vtx = &vtxs_[n_vtx];

  switch (action)
  {
    case INTERSECTION:  if (vtx->value > 0)                                      vtx->set(OUT, -1, -1);  break;
    case ADDITION:      if (vtx->value < 0)                                      vtx->set(INS, -1, -1);  break;
    case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==PNT)  vtx->set(FCE, cn, -1);  break;
  }
}
void simplex2_mls_l_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg2_t *edg = &edgs_[n_edg];

  int c0 = edg->c0;

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx1)) swap(edg->vtx0, edg->vtx1);

  int num_negatives = 0;
  if (vtxs_[edg->vtx0].value < 0) num_negatives++;
  if (vtxs_[edg->vtx1].value < 0) num_negatives++;

#ifdef SIMPLEX2_MLS_L_T_DEBUG
  edg->type = num_negatives;
#endif

  // auxiliary variables
  vtx2_t *c_vtx01;
  edg2_t *c_edg0, *c_edg1;
  double r;

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
      /* split an edge */
      edg->is_split = true;

      // new vertex
      if (use_linear_) r = find_intersection_linear    (edg->vtx0, edg->vtx1);
      else             r = find_intersection_quadratic (n_edg);

      vtxs_.push_back(vtx2_t(vtxs_[edg->vtx0].x*(1.-r) + vtxs_[edg->vtx1].x*r,
                             vtxs_[edg->vtx0].y*(1.-r) + vtxs_[edg->vtx1].y*r));

      vtxs_.back().n_vtx0 = edg->vtx0;
      vtxs_.back().n_vtx1 = edg->vtx1;
      vtxs_.back().ratio  = r;

      edg->c_vtx01 = vtxs_.size()-1;

      // new edges
      edgs_.push_back(edg2_t(edg->vtx0,    edg->c_vtx01)); edg = &edgs_[n_edg]; // edges might have changed their addresses
      edgs_.push_back(edg2_t(edg->c_vtx01, edg->vtx1   )); edg = &edgs_[n_edg];

      edg->c_edg0 = edgs_.size()-2;
      edg->c_edg1 = edgs_.size()-1;

      /* apply rules */
      c_vtx01 = &vtxs_[edg->c_vtx01];
      c_edg0  = &edgs_[edg->c_edg0];
      c_edg1  = &edgs_[edg->c_edg1];

      c_edg0->dir = edg->dir;
      c_edg1->dir = edg->dir;

      c_edg0->p_lsf = edg->p_lsf;
      c_edg1->p_lsf = edg->p_lsf;

#ifdef SIMPLEX2_MLS_L_T_DEBUG
      c_vtx01->p_edg = n_edg;
      c_edg0->p_edg  = n_edg;
      c_edg1->p_edg  = n_edg;
#endif

      switch (action)
      {
        case INTERSECTION:
          switch (edg->loc)
          {
            case OUT: c_vtx01->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); break;
            case INS: c_vtx01->set(FCE, cn, -1); c_edg0->set(INS, -1); c_edg1->set(OUT, -1); break;
            case FCE: c_vtx01->set(PNT, c0, cn); c_edg0->set(FCE, c0); c_edg1->set(OUT, -1); if (c0==cn)  c_vtx01->set(FCE, c0, -1);  break;
            default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
#endif
          }
          break;
        case ADDITION:
          switch (edg->loc)
          {
            case OUT: c_vtx01->set(FCE, cn, -1); c_edg0->set(INS, -1); c_edg1->set(OUT, -1); break;
            case INS: c_vtx01->set(INS, -1, -1); c_edg0->set(INS, -1); c_edg1->set(INS, -1); break;
            case FCE: c_vtx01->set(PNT, c0, cn); c_edg0->set(INS, -1); c_edg1->set(FCE, c0); if (c0==cn)  c_vtx01->set(FCE, c0, -1);  break;
            default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
#endif
          }
          break;
        case COLORATION:
          switch (edg->loc)
          {
            case OUT: c_vtx01->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); break;
            case INS: c_vtx01->set(INS, -1, -1); c_edg0->set(INS, -1); c_edg1->set(INS, -1); break;
            case FCE: c_vtx01->set(PNT, c0, cn); c_edg0->set(FCE, cn); c_edg1->set(FCE, c0); if (c0==cn)  c_vtx01->set(FCE, c0, -1);  break;
            default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
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
        case INTERSECTION:  /* do nothing */                        break;
        case ADDITION:                          edg->set(INS, -1);  break;
        case COLORATION:    if (edg->loc==FCE)  edg->set(FCE, cn);  break;
      }
      break;
  }
}

void simplex2_mls_l_t::do_action_tri(int n_tri, int cn, action_t action)
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

#ifdef SIMPLEX2_MLS_L_T_DEBUG
  tri->type = num_negatives;

  /* check whether vertices of a triangle and edges coincide */
  if (tri->vtx0 != edgs_[tri->edg1].vtx0 || tri->vtx0 != edgs_[tri->edg2].vtx0 ||
      tri->vtx1 != edgs_[tri->edg0].vtx0 || tri->vtx1 != edgs_[tri->edg2].vtx1 ||
      tri->vtx2 != edgs_[tri->edg0].vtx1 || tri->vtx2 != edgs_[tri->edg1].vtx1)
  {
    std::cout << vtxs_[tri->vtx0].value << " : " << vtxs_[tri->vtx1].value << " : " << vtxs_[tri->vtx2].value << std::endl;
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) Vertices of a triangle and edges do not coincide after sorting.");
  }

  /* check whether appropriate edges have been splitted */
  int e0_type_expect, e1_type_expect, e2_type_expect;

  switch (num_negatives){
  case 0: e0_type_expect = 0; e1_type_expect = 0; e2_type_expect = 0; break;
  case 1: e0_type_expect = 0; e1_type_expect = 1; e2_type_expect = 1; break;
  case 2: e0_type_expect = 1; e1_type_expect = 1; e2_type_expect = 2; break;
  case 3: e0_type_expect = 2; e1_type_expect = 2; e2_type_expect = 2; break;
  }

  if (edgs_[tri->edg0].type != e0_type_expect || edgs_[tri->edg1].type != e1_type_expect || edgs_[tri->edg2].type != e2_type_expect)
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) While splitting a triangle one of edges has an unexpected type.");
#endif

  // auxiliary variables
  edg2_t *c_edg0, *c_edg1;
  edg2_t *edg0, *edg1, *edg2;
  tri2_t *c_tri0, *c_tri1, *c_tri2;

  switch (num_negatives)
  {
  case 0: // (+++)
    /* split a triangle */
    // no need to split

    /* apply rules */
    switch (action)
    {
    case INTERSECTION:  tri->set(OUT);    break;
    case ADDITION:      /* do nothing */  break;
    case COLORATION:    /* do nothing */  break;
    }
    break;

  case 1: // (-++)
    /* split a triangle */
    tri->is_split = true;

    // new vertices
    tri->c_vtx01 = edgs_[tri->edg2].c_vtx01;
    tri->c_vtx02 = edgs_[tri->edg1].c_vtx01;

    // new edges
    edgs_.push_back(edg2_t(tri->c_vtx01, tri->c_vtx02));
    edgs_.push_back(edg2_t(tri->c_vtx01, tri->vtx2   ));

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

    if (action == INTERSECTION || action == ADDITION) c_edg0->p_lsf = cn;

#ifdef SIMPLEX2_MLS_L_T_DEBUG
    if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
      throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) While splitting a triangle one of child triangles is not consistent.");

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
          case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
          case INS: c_edg0->set(FCE, cn); c_edg1->set(OUT, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
          default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
            throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
#endif
        } break;
      case ADDITION:
        switch (tri->loc)
        {
          case OUT: c_edg0->set(FCE, cn); c_edg1->set(OUT, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
          case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
          default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
            throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
#endif
        } break;
      case COLORATION:
        switch (tri->loc)
        {
          case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
          case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
          default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
            throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
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
      edgs_.push_back(edg2_t(tri->vtx0,    tri->c_vtx12));
      edgs_.push_back(edg2_t(tri->c_vtx02, tri->c_vtx12));

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

      if (action == INTERSECTION || action == ADDITION) c_edg1->p_lsf = cn;

#ifdef SIMPLEX2_MLS_L_T_DEBUG
      if (!tri_is_ok(tri->c_tri0) || !tri_is_ok(tri->c_tri1) || !tri_is_ok(tri->c_tri2))
        throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) While splitting a triangle one of child triangles is not consistent.");

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
            case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
            case INS: c_edg0->set(INS, -1); c_edg1->set(FCE, cn); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
            default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
#endif
          } break;
        case ADDITION:
          switch (tri->loc)
          {
            case OUT: c_edg0->set(INS, -1); c_edg1->set(FCE, cn); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
            case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
            default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t)  An element has wrong location.");
#endif
          } break;
        case COLORATION:
          switch (tri->loc)
          {
            case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
            case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
            default:
#ifdef SIMPLEX2_MLS_L_T_DEBUG
              throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) An element has wrong location.");
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
        case INTERSECTION:  /* do nothing */  break;
        case ADDITION:      tri->set(INS);    break;
        case COLORATION:                      break;
      }
      break;
  }
}





//--------------------------------------------------
// Quadrature Points
//--------------------------------------------------
void simplex2_mls_l_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
  for (unsigned int i = 0; i < tris_.size(); i++)
  {
    tri2_t *t = &tris_[i];
    if (!t->is_split && t->loc == INS)
    {
      double w = area(t->vtx0, t->vtx1, t->vtx2)/3.0;

      weights.push_back(w); X.push_back(vtxs_[t->vtx0].x); Y.push_back(vtxs_[t->vtx0].y);
      weights.push_back(w); X.push_back(vtxs_[t->vtx1].x); Y.push_back(vtxs_[t->vtx1].y);
      weights.push_back(w); X.push_back(vtxs_[t->vtx2].x); Y.push_back(vtxs_[t->vtx2].y);
    }
  }
}

void simplex2_mls_l_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
  bool integrate_specific = (num != -1);

  for (unsigned int i = 0; i < edgs_.size(); i++)
  {
    edg2_t *e = &edgs_[i];
    if (!e->is_split && e->loc == FCE)
      if ((!integrate_specific && e->c0 >= 0) || (integrate_specific && e->c0 == num))
      {
        double w = length(e->vtx0, e->vtx1)/2.0;

        weights.push_back(w); X.push_back(vtxs_[e->vtx0].x); Y.push_back(vtxs_[e->vtx0].y);
        weights.push_back(w); X.push_back(vtxs_[e->vtx1].x); Y.push_back(vtxs_[e->vtx1].y);
      }
  }
}

void simplex2_mls_l_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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
        weights.push_back(1); X.push_back(v->x); Y.push_back(v->y);
      }
  }
}

void simplex2_mls_l_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
{
  for (unsigned int i = 0; i < edgs_.size(); i++)
  {
    edg2_t *e = &edgs_[i];
    if (!e->is_split && e->loc == INS)
      if (e->dir == dir)
      {
        double w = length(e->vtx0, e->vtx1)/2.0;

        weights.push_back(w); X.push_back(vtxs_[e->vtx0].x); Y.push_back(vtxs_[e->vtx0].y);
        weights.push_back(w); X.push_back(vtxs_[e->vtx1].x); Y.push_back(vtxs_[e->vtx1].y);
      }
  }
}





//--------------------------------------------------
// Interpolation
//--------------------------------------------------
void simplex2_mls_l_t::interpolate_from_neighbors(int v)
{
  vtx2_t *vtx = &vtxs_[v];
  vtx->value = (1.-vtx->ratio)*vtxs_[vtx->n_vtx0].value + (vtx->ratio)*vtxs_[vtx->n_vtx1].value;
}

void simplex2_mls_l_t::interpolate_from_parent(int v)
{
  double A0 = area(v, 1, 2);
  double A1 = area(0, v, 2);
  double A2 = area(0, 1, v);

  vtxs_[v].value = (A0*vtxs_[0].value + A1*vtxs_[1].value + A2*vtxs_[2].value)/A_;
}

void simplex2_mls_l_t::interpolate_from_parent(vtx2_t &vtx)
{
  double A0 = area(vtx,      vtxs_[1], vtxs_[2]);
  double A1 = area(vtxs_[0], vtx,      vtxs_[2]);
  double A2 = area(vtxs_[0], vtxs_[1], vtx);

  vtx.value = (A0*vtxs_[0].value + A1*vtxs_[1].value + A2*vtxs_[2].value)/A_;
}





//--------------------------------------------------
// Intersections
//--------------------------------------------------
double simplex2_mls_l_t::find_intersection_linear(int v0, int v1)
{
  double f0 = vtxs_[v0].value;
  double f1 = vtxs_[v1].value;

#ifdef SIMPLEX2_MLS_L_T_DEBUG
  if (f0 <= 0 && f1 <= 0 ||
      f0 >= 0 && f1 >= 0)
    throw std::invalid_argument("[CASL_ERROR]: (simplex2_mls_l_t) Cannot find an intersection with an edge, values of a level-set function are of the same sign at end points.");
#endif

//  double l = length(v0, v1);
//  if (l <= 2.1*eps_) return .5;
//  double ratio = eps_/l;
//#ifdef SIMPLEX2_MLS_L_T_DEBUG
//  if(l <= 2.*eps_) throw std::invalid_argument("[CASL_ERROR]: (simplex2_mls_l_t) Vertices are too close.");
//#endif

//  bool f0_close = fabs(f0) < phi_tolerance_;
//  bool f1_close = fabs(f1) < phi_tolerance_;

//  if (f0_close && f1_close) return .5;
//  if (f0_close)             return ratio;
//  if (f1_close)             return 1.-ratio;

  if (fabs(f0) <= DBL_MIN || fabs(f0) >= fabs(f1-f0))
    throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) Interestion with an edge falls outside of the edge");

  double x = -f0/(f1-f0);

  if (x < 0. || x > 1.) throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) Interestion with an edge falls outside of the edge");

//  if (x < ratio)    x = ratio;
//  if (x > 1.-ratio) x = 1.-ratio;

  return x;
}

double simplex2_mls_l_t::find_intersection_quadratic(int e)
{
  throw std::invalid_argument("[CASL_ERROR]: (simplex2_mls_l_t) Quadratic intersections are not implemented.");

  int v0 = edgs_[e].vtx0;
  int v1 = edgs_[e].vtx1;
  double l = length(v0, v1);
#ifdef SIMPLEX2_MLS_L_T_DEBUG
  if(l < eps_) throw std::invalid_argument("[CASL_ERROR]: (simplex2_mls_l_t) Vertices are too close.");
#endif

  double f0 = vtxs_[v0].value;
  double f1 = vtxs_[v1].value;

  // TODO: FIX THIS FOR QUADRATIC INTERSECTIONS
  double f01 = .5*(f0+f1);
//  double xyz[2];
//  get_edge_coords(e, xyz);
//  double f01 = interpolate(xyz);

  if (fabs(f0)  < .8*eps_) return (l-.8*eps_)/l;
  if (fabs(f01) < .8*eps_) return 0.5;
  if (fabs(f1)  < .8*eps_) return (0.+.8*eps_)/l;

#ifdef SIMPLEX2_MLS_L_T_DEBUG
  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: (simplex2_mls_l_t) Cannot find an intersection with an edge, values of a level-set function are of the same sign at end points.");
#endif

  double fdd = (f1+f0-2.*f01)/(0.25*l*l);

  double c2 = 0.5*fdd;    // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = (f1-f0)/l;  //  the expansion of f at the center of the edge
  double c0 = f01;

  double x;

  if(fabs(c2)<eps_) x = -c0/c1;
  else
  {
    if(f1<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    else     x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
  }
#ifdef SIMPLEX2_MLS_L_T_DEBUG
  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: (simplex2_mls_l_t) Interestion with an edge falls outside of the edge");
#endif

  if (x < -0.5*l) return (l-.8*eps_)/l;
  if (x > 0.5*l) return (0.+.8*eps_)/l;

  return 1.-(x+0.5*l)/l;
}





//--------------------------------------------------
// Computation tools
//--------------------------------------------------
double simplex2_mls_l_t::length(int v0, int v1)
{
  return sqrt(pow(vtxs_[v0].x - vtxs_[v1].x, 2.0)
            + pow(vtxs_[v0].y - vtxs_[v1].y, 2.0));
}

double simplex2_mls_l_t::area(int v0, int v1, int v2)
{
  double x01 = vtxs_[v1].x - vtxs_[v0].x; double x02 = vtxs_[v2].x - vtxs_[v0].x;
  double y01 = vtxs_[v1].y - vtxs_[v0].y; double y02 = vtxs_[v2].y - vtxs_[v0].y;

  return 0.5*fabs(x01*y02-y01*x02);
}

double simplex2_mls_l_t::area(vtx2_t &vtx0, vtx2_t &vtx1, vtx2_t &vtx2)
{
  double x01 = vtx1.x - vtx0.x; double x02 = vtx2.x - vtx0.x;
  double y01 = vtx1.y - vtx0.y; double y02 = vtx2.y - vtx0.y;

  return 0.5*fabs(x01*y02-y01*x02);
}





//--------------------------------------------------
// Debugging
//--------------------------------------------------
#ifdef SIMPLEX2_MLS_L_T_DEBUG
bool simplex2_mls_l_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = result && (edgs_[e0].vtx0 == v1 || edgs_[e0].vtx1 == v1) && (edgs_[e0].vtx0 == v2 || edgs_[e0].vtx1 == v2);
  result = result && (edgs_[e1].vtx0 == v0 || edgs_[e1].vtx1 == v0) && (edgs_[e1].vtx0 == v2 || edgs_[e1].vtx1 == v2);
  result = result && (edgs_[e2].vtx0 == v0 || edgs_[e2].vtx1 == v0) && (edgs_[e2].vtx0 == v1 || edgs_[e2].vtx1 == v1);
  return result;
}

bool simplex2_mls_l_t::tri_is_ok(int t)
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
