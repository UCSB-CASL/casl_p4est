#include "simplex3_mls.h"

simplex3_mls_t::simplex3_mls_t()
{
  vtxs.reserve(8);
  edgs.reserve(27);
  tris.reserve(20);
  tets.reserve(6);

  eps = 1.0e-15;
}

simplex3_mls_t::simplex3_mls_t(double x0, double y0, double z0,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2,
                               double x3, double y3, double z3)
//  : x({x0, x1, x2, x3}), y({y0, y1, y2, y3}), z({z0, z1, z2, z3})
{
  if (1) // usually there will be only one cut
  {
    vtxs.reserve(8);
    edgs.reserve(27);
    tris.reserve(20);
    tets.reserve(6);
  }

  /* fill the vectors with the initial structure */
  vtxs.push_back(vtx3_t(x0,y0,z0));
  vtxs.push_back(vtx3_t(x1,y1,z1));
  vtxs.push_back(vtx3_t(x2,y2,z2));
  vtxs.push_back(vtx3_t(x3,y3,z3));

  edgs.push_back(edg3_t(0,1));
  edgs.push_back(edg3_t(0,2));
  edgs.push_back(edg3_t(0,3));
  edgs.push_back(edg3_t(1,2));
  edgs.push_back(edg3_t(1,3));
  edgs.push_back(edg3_t(2,3));

  tris.push_back(tri3_t(1,2,3,5,4,3));
  tris.push_back(tri3_t(0,2,3,5,2,1));
  tris.push_back(tri3_t(0,1,3,4,2,0));
  tris.push_back(tri3_t(0,1,2,3,1,0));

  tets.push_back(tet3_t(0,1,2,3,0,1,2,3));

  use_linear = true;
  eps = 1.0e-15;
}

void simplex3_mls_t::do_action(int cn, action_t action)
{
  /* Process elements */
  int n;
  n = vtxs.size(); for (int i = 0; i < n; i++) do_action_vtx(i, cn, action);
  n = edgs.size(); for (int i = 0; i < n; i++) do_action_edg(i, cn, action);
  n = tris.size(); for (int i = 0; i < n; i++) do_action_tri(i, cn, action);
  n = tets.size(); for (int i = 0; i < n; i++) do_action_tet(i, cn, action);
}

//void simplex3_mls_t::do_action(int num, Action action, CF_3 &func)
//{
//  return do_action(num, action,
//                   func(p0.x, p0.y, p0.z),
//                   func(p1.x, p1.y, p1.z),
//                   func(p2.x, p2.y, p2.z),
//                   func(p3.x, p3.y, p3.z));
//}

//double simplex3_mls_t::interpolate_from_parent(Point3 &r, double f0, double f1, double f2, double f3)
//{
//  double vol0 = Point3::volume(r, p1, p2, p3);
//  double vol1 = Point3::volume(p0, r, p2, p3);
//  double vol2 = Point3::volume(p0, p1, r, p3);
//  double vol3 = Point3::volume(p0, p1, p2, r);
//  double vol = Point3::volume(p0, p1, p2, p3);

//#ifdef CASL_THROWS
//  if (vol < eps)
//    throw std::domain_error("[CASL_ERROR]: Division by zero.");
//#endif

//  return (vol0*f0 + vol1*f1 + vol2*f2 + vol3*f3)/vol;
//}

void simplex3_mls_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx3_t *vtx = &vtxs[n_vtx];

  perturb(vtx->value, eps);

  switch (action){
  case INTERSECTION:  if (vtx->value > 0)                                                       vtx->set(OUT, -1, -1, -1);  break;
  case ADDITION:      if (vtx->value < 0)                                                       vtx->set(INS, -1, -1, -1);  break;
  case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==LNE || vtx->loc==PNT)  vtx->set(FCE, cn, -1, -1);  break;
  }
}

void simplex3_mls_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg3_t *edg = &edgs[n_edg];

  int c0 = edg->c0;
  int c1 = edg->c1;

  if (edg->is_split) return;

  /* Sort vertices */
  if (need_swap(edg->vtx0, edg->vtx1)) swap(edg->vtx0, edg->vtx1);

  int num_negatives = 0;
  if (vtxs[edg->vtx0].value < 0) num_negatives++;
  if (vtxs[edg->vtx1].value < 0) num_negatives++;

#ifdef CASL_THROWS
  edg->type = num_negatives;
#endif

  // auxiliary variables
  vtx3_t *c_vtx01;
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
//    edgs.reserve(edgs.size()+2);
//    edg = &edgs[n_edg];
    /* split an edge */
    edg->is_split = true;

    // new vertex
    if (use_linear) r = find_intersection_linear    (edg->vtx0, edg->vtx1);
//    else            r = find_intersection_quadratic (edg->vtx0, edg->vtx1);
    else            r = find_intersection_quadratic (n_edg);

    vtxs.push_back(vtx3_t(vtxs[edg->vtx0].x*r + vtxs[edg->vtx1].x*(1.0-r),
                          vtxs[edg->vtx0].y*r + vtxs[edg->vtx1].y*(1.0-r),
                          vtxs[edg->vtx0].z*r + vtxs[edg->vtx1].z*(1.0-r)));

    vtxs.back().n_vtx0 = edg->vtx0;
    vtxs.back().n_vtx1 = edg->vtx1;
    vtxs.back().ratio  = r;

    edg->c_vtx01 = vtxs.size()-1;

    // new edges
    edgs.push_back(edg3_t(edg->vtx0,    edg->c_vtx01)); edg = &edgs[n_edg]; // edges might have changed their addresses
    edgs.push_back(edg3_t(edg->c_vtx01, edg->vtx1   )); edg = &edgs[n_edg];

    edg->c_edg0 = edgs.size()-2;
    edg->c_edg1 = edgs.size()-1;

    /* apply rules */
    c_vtx01 = &vtxs[edg->c_vtx01];
    c_edg0  = &edgs[edg->c_edg0];
    c_edg1  = &edgs[edg->c_edg1];

#ifdef CASL_THROWS
    c_vtx01->p_edg = n_edg;
    c_edg0->p_edg  = n_edg;
    c_edg1->p_edg  = n_edg;
#endif

    switch (action){
    case INTERSECTION:
      switch (edg->loc){
      case OUT: c_vtx01->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
      case INS: c_vtx01->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
      case FCE: c_vtx01->set(LNE, c0, cn, -1); c_edg0->set(FCE, c0, -1); c_edg1->set(OUT, -1, -1); if (c0==cn)            c_vtx01->set(FCE, c0, -1, -1);  break;
      case LNE: c_vtx01->set(PNT, c0, c1, cn); c_edg0->set(LNE, c0, c1); c_edg1->set(OUT, -1, -1); if (c0==cn || c1==cn)  c_vtx01->set(LNE, c0, c1, -1);  break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    case ADDITION:
      switch (edg->loc){
      case OUT: c_vtx01->set(FCE, cn, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
      case INS: c_vtx01->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1);                                                        break;
      case FCE: c_vtx01->set(LNE, c0, cn, -1); c_edg0->set(INS, -1, -1); c_edg1->set(FCE, c0, -1); if (c0==cn)            c_vtx01->set(FCE, c0, -1, -1);  break;
      case LNE: c_vtx01->set(PNT, c0, c1, cn); c_edg0->set(INS, -1, -1); c_edg1->set(LNE, c0, c1); if (c0==cn || c1==cn)  c_vtx01->set(LNE, c0, c1, -1);  break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    case COLORATION:
      switch (edg->loc){
      case OUT: c_vtx01->set(OUT, -1, -1, -1); c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1);                                                        break;
      case INS: c_vtx01->set(INS, -1, -1, -1); c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1);                                                        break;
      case FCE: c_vtx01->set(LNE, c0, cn, -1); c_edg0->set(FCE, cn, -1); c_edg1->set(FCE, c0, -1); if (c0==cn)            c_vtx01->set(FCE, c0, -1, -1);  break;
      case LNE: c_vtx01->set(PNT, c0, c1, cn); c_edg0->set(FCE, cn, -1); c_edg1->set(LNE, c0, c1); if (c0==cn || c1==cn)  c_vtx01->set(LNE, c0, c1, -1);  break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      }
      break;
    }
    break;
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

void simplex3_mls_t::do_action_tri(int n_tri, int cn, action_t action)
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

#ifdef CASL_THROWS
  tri->type = num_negatives;

  /* check whether vertices of a triangle and edges coincide */
  if (tri->vtx0 != edgs[tri->edg1].vtx0 || tri->vtx0 != edgs[tri->edg2].vtx0 ||
      tri->vtx1 != edgs[tri->edg0].vtx0 || tri->vtx1 != edgs[tri->edg2].vtx1 ||
      tri->vtx2 != edgs[tri->edg0].vtx1 || tri->vtx2 != edgs[tri->edg1].vtx1)
    throw std::domain_error("[CASL_ERROR]: Vertices of a triangle and edges do not coincide after sorting.");

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
//    edgs.reserve(edgs.size()+2);
//    tris.reserve(tris.size()+3);
//    tri = &tris[n_tri];
    /* split a triangle */
    tri->is_split = true;

    // new vertices
    tri->c_vtx01 = edgs[tri->edg2].c_vtx01;
    tri->c_vtx02 = edgs[tri->edg1].c_vtx01;

    // new edges
    edgs.push_back(edg3_t(tri->c_vtx01, tri->c_vtx02));
    edgs.push_back(edg3_t(tri->c_vtx01, tri->vtx2   ));

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
      case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(FCE, cn, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case FCE: c_edg0->set(LNE, cc, cn); c_edg1->set(OUT, -1, -1); c_tri0->set(FCE, cc); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tri->loc){
      case OUT: c_edg0->set(FCE, cn, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(INS, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(LNE, cc, cn); c_edg1->set(FCE, cc, -1); c_tri0->set(INS, -1); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(LNE, cc, cn); c_edg1->set(FCE, cc, -1); c_tri0->set(FCE, cn); c_tri1->set(FCE, cc); c_tri2->set(FCE, cc); if (cc==cn) c_edg0->set(FCE, cc, -1); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    }
    break;

  case 2: // (--+)
//    edgs.reserve(edgs.size()+2);
//    tris.reserve(tris.size()+3);
//    tri = &tris[n_tri];
    /* split a triangle */
    tri->is_split = true;

    tri->c_vtx02 = edgs[tri->edg1].c_vtx01;
    tri->c_vtx12 = edgs[tri->edg0].c_vtx01;

    // create new edges
    edgs.push_back(edg3_t(tri->vtx0,    tri->c_vtx12));
    edgs.push_back(edg3_t(tri->c_vtx02, tri->c_vtx12));

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
      case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); c_edg1->set(FCE, cn, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
      case FCE: c_edg0->set(FCE, cc, -1); c_edg1->set(LNE, cc, cn); c_tri0->set(FCE, cc); c_tri1->set(FCE, cc); c_tri2->set(OUT, -1); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tri->loc){
      case OUT: c_edg0->set(INS, -1, -1); c_edg1->set(FCE, cn, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(INS, -1, -1); c_edg1->set(LNE, cc, cn); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1, -1); c_edg1->set(OUT, -1, -1); c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); break;
      case INS: c_edg0->set(INS, -1, -1); c_edg1->set(INS, -1, -1); c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); break;
      case FCE: c_edg0->set(FCE, cn, -1); c_edg1->set(LNE, cc, cn); c_tri0->set(FCE, cn); c_tri1->set(FCE, cn); c_tri2->set(FCE, cc); if (cc==cn) c_edg1->set(FCE, cc, -1); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    }
    break;

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
void simplex3_mls_t::do_action_tet(int n_tet, int cn, action_t action)
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

#ifdef CASL_THROWS
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

#ifdef CASL_THROWS
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
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tet->loc){
      case OUT: c_tri0->set(FCE, cn); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(INS); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT, -1); c_tri1->set(OUT, -1); c_tri2->set(OUT, -1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS, -1); c_tri1->set(INS, -1); c_tri2->set(INS, -1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default:
#ifdef CASL_THROWS
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

    vtx3_t vtx_aux0(0.5*(vtxs[tet->c_vtx03].x+vtxs[tet->c_vtx12].x),0.5*(vtxs[tet->c_vtx03].y+vtxs[tet->c_vtx12].y),0.5*(vtxs[tet->c_vtx03].z+vtxs[tet->c_vtx12].z));
    vtx3_t vtx_aux1(0.5*(vtxs[tet->c_vtx02].x+vtxs[tet->c_vtx13].x),0.5*(vtxs[tet->c_vtx02].y+vtxs[tet->c_vtx13].y),0.5*(vtxs[tet->c_vtx02].z+vtxs[tet->c_vtx13].z));

    interpolate_from_parent(vtx_aux0);
    interpolate_from_parent(vtx_aux1);

    if (fabs(vtx_aux0.value) <= fabs(vtx_aux1.value))
    {
      // new edge
      edgs.push_back(edg3_t(tet->c_vtx03, tet->c_vtx12));
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
    } else {
      // new edge
      edgs.push_back(edg3_t(tet->c_vtx02, tet->c_vtx13));
      tet->c_edg = edgs.size()-1;

      // new triangles
      tris.push_back(tri3_t(tet->vtx0,    tet->c_vtx12, tet->c_vtx13, tris[tet->tri0].c_edg0, tris[tet->tri2].c_edg0,             tris[tet->tri3].c_edg0            ));
      tris.push_back(tri3_t(tet->vtx0,    tet->c_vtx02, tet->c_vtx13, tet->c_edg,             tris[tet->tri2].c_edg0,             edgs[tris[tet->tri1].edg2].c_edg0 ));
      tris.push_back(tri3_t(tet->c_vtx02, tet->c_vtx03, tet->c_vtx13, tris[tet->tri2].c_edg1, tet->c_edg,                         tris[tet->tri1].c_edg0            ));
      tris.push_back(tri3_t(tet->c_vtx02, tet->c_vtx12, tet->c_vtx13, tris[tet->tri0].c_edg0, tet->c_edg,                         tris[tet->tri3].c_edg1            ));
      tris.push_back(tri3_t(tet->c_vtx02, tet->c_vtx13, tet->vtx3,    edgs[tris[tet->tri0].edg1].c_edg1, tris[tet->tri1].c_edg1,  tet->c_edg                        ));
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
      tets.push_back(tet3_t(tet->vtx0,    tet->c_vtx02, tet->c_vtx12, tet->c_vtx13, tet->c_tri3,  tet->c_tri0,  tet->c_tri1,  tri3->c_tri1)); tet = &tets[n_tet];
      tets.push_back(tet3_t(tet->vtx0,    tet->c_vtx02, tet->c_vtx03, tet->c_vtx13, tet->c_tri2,  tri2->c_tri1, tet->c_tri1,  tri1->c_tri0)); tet = &tets[n_tet];
      tets.push_back(tet3_t(tet->c_vtx02, tet->c_vtx12, tet->c_vtx13, tet->vtx3,    tri0->c_tri1, tet->c_tri4,  tet->c_tri5,  tet->c_tri3 )); tet = &tets[n_tet];
      tets.push_back(tet3_t(tet->c_vtx02, tet->c_vtx03, tet->c_vtx13, tet->vtx3,    tri2->c_tri2, tet->c_tri4,  tri1->c_tri1, tet->c_tri2 )); tet = &tets[n_tet];
      tets.push_back(tet3_t(tet->c_vtx02, tet->c_vtx12, tet->vtx2,    tet->vtx3,    tri0->c_tri2, tri1->c_tri2, tet->c_tri5,  tri3->c_tri2)); tet = &tets[n_tet];

    }

    n_tets = tets.size();
    tet->c_tet0 = n_tets-6;
    tet->c_tet1 = n_tets-5;
    tet->c_tet2 = n_tets-4;
    tet->c_tet3 = n_tets-3;
    tet->c_tet4 = n_tets-2;
    tet->c_tet5 = n_tets-1;

    /* apply rules */
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


#ifdef CASL_THROWS
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
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tet->loc){
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
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tet->loc){
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
#ifdef CASL_THROWS
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

    /* apply rules */
    c_tri0 = &tris[tet->c_tri0];
    c_tri1 = &tris[tet->c_tri1];
    c_tri2 = &tris[tet->c_tri2];

    c_tet0 = &tets[tet->c_tet0];
    c_tet1 = &tets[tet->c_tet1];
    c_tet2 = &tets[tet->c_tet2];
    c_tet3 = &tets[tet->c_tet3];

#ifdef CASL_THROWS
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
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case ADDITION:
      switch (tet->loc){
      case OUT: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(FCE,cn); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default:
#ifdef CASL_THROWS
          throw std::domain_error("[CASL_ERROR]: An element has wrong location.");
#endif
      } break;
    case COLORATION:
      switch (tet->loc){
      case OUT: c_tri0->set(OUT,-1); c_tri1->set(OUT,-1); c_tri2->set(OUT,-1); c_tet0->set(OUT); c_tet1->set(OUT); c_tet2->set(OUT); c_tet3->set(OUT); break;
      case INS: c_tri0->set(INS,-1); c_tri1->set(INS,-1); c_tri2->set(INS,-1); c_tet0->set(INS); c_tet1->set(INS); c_tet2->set(INS); c_tet3->set(INS); break;
        default:
#ifdef CASL_THROWS
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

bool simplex3_mls_t::need_swap(int v0, int v1)
{
  double dif = vtxs[v0].value - vtxs[v1].value;
  if (fabs(dif) < eps){ // if values are too close, sort vertices by their numbers
    if (v0 > v1) return true;
    else         return false;
  } else if (dif > 0.0){ // otherwise sort by values
    return true;
  } else {
    return false;
  }
}

double simplex3_mls_t::length(int vtx0, int vtx1)
{
  return sqrt(pow(vtxs[vtx0].x - vtxs[vtx1].x, 2.0)
            + pow(vtxs[vtx0].y - vtxs[vtx1].y, 2.0)
            + pow(vtxs[vtx0].z - vtxs[vtx1].z, 2.0));
}
double simplex3_mls_t::area(int vtx0, int vtx1, int vtx2)
{
  double x01 = vtxs[vtx1].x - vtxs[vtx0].x; double x02 = vtxs[vtx2].x - vtxs[vtx0].x;
  double y01 = vtxs[vtx1].y - vtxs[vtx0].y; double y02 = vtxs[vtx2].y - vtxs[vtx0].y;
  double z01 = vtxs[vtx1].z - vtxs[vtx0].z; double z02 = vtxs[vtx2].z - vtxs[vtx0].z;

  return 0.5*sqrt(pow(y01*z02-z01*y02,2.0) + pow(z01*x02-x01*z02,2.0) + pow(x01*y02-y01*x02,2.0));
}

double simplex3_mls_t::volume(int vtx0, int vtx1, int vtx2, int vtx3)
{
  double a11 = vtxs[vtx1].x-vtxs[vtx0].x; double a12 = vtxs[vtx1].y-vtxs[vtx0].y; double a13 = vtxs[vtx1].z-vtxs[vtx0].z;
  double a21 = vtxs[vtx2].x-vtxs[vtx0].x; double a22 = vtxs[vtx2].y-vtxs[vtx0].y; double a23 = vtxs[vtx2].z-vtxs[vtx0].z;
  double a31 = vtxs[vtx3].x-vtxs[vtx0].x; double a32 = vtxs[vtx3].y-vtxs[vtx0].y; double a33 = vtxs[vtx3].z-vtxs[vtx0].z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}

double simplex3_mls_t::volume(vtx3_t &vtx0, vtx3_t &vtx1, vtx3_t &vtx2, vtx3_t &vtx3)
{
  double a11 = vtx1.x-vtx0.x; double a12 = vtx1.y-vtx0.y; double a13 = vtx1.z-vtx0.z;
  double a21 = vtx2.x-vtx0.x; double a22 = vtx2.y-vtx0.y; double a23 = vtx2.z-vtx0.z;
  double a31 = vtx3.x-vtx0.x; double a32 = vtx3.y-vtx0.y; double a33 = vtx3.z-vtx0.z;

  double vol = a11*(a22*a33-a23*a32)
             + a21*(a32*a13-a12*a33)
             + a31*(a12*a23-a22*a13);

  return fabs(vol/6.);
}


double simplex3_mls_t::integrate_over_domain(double f0, double f1, double f2, double f3)
{
  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2, f3);

  double result = 0.0;

  /* integrate over tetrahedra */
  for (unsigned int i = 0; i < tets.size(); i++)
    if (!tets[i].is_split && tets[i].loc == INS)
    {
      tet3_t *s = &tets[i];
      result += volume(s->vtx0, s->vtx1, s->vtx2, s->vtx3) *
          (vtxs[s->vtx0].value + vtxs[s->vtx1].value + vtxs[s->vtx2].value + vtxs[s->vtx3].value)/4.0;
    }

  return result;
}

double simplex3_mls_t::integrate_over_interface(double f0, double f1, double f2, double f3, int num)
{
  bool integrate_specific = (num != -1);

  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2, f3);

  double result = 0.0;

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris.size(); i++)
  {
    tri3_t *t = &tris[i];
    if (!t->is_split && t->loc == FCE)
      if (!integrate_specific
          || (integrate_specific && t->c == num))
      {
        result += area(t->vtx0, t->vtx1, t->vtx2) *
            (vtxs[t->vtx0].value + vtxs[t->vtx1].value + vtxs[t->vtx2].value)/3.0;
      }
  }

  return result;
}

double simplex3_mls_t::integrate_over_colored_interface(double f0, double f1, double f2, double f3, int num0, int num1)
{
  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2, f3);

  double result = 0.0;

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris.size(); i++)
  {
    tri3_t *t = &tris[i];
    if (!t->is_split && t->loc == FCE)
      if (t->p_lsf == num0 && t->c == num1)
      {
        result += area(t->vtx0, t->vtx1, t->vtx2) *
            (vtxs[t->vtx0].value + vtxs[t->vtx1].value + vtxs[t->vtx2].value)/3.0;
      }
  }

  return result;
}

double simplex3_mls_t::integrate_over_intersection(double f0, double f1, double f2, double f3, int num0, int num1)
{
  bool integrate_specific = (num0 != -1 && num1 != -1);

  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2, f3);

  double result = 0.0;

  /* integrate over edges */
  for (unsigned int i = 0; i < edgs.size(); i++)
  {
    edg3_t *e = &edgs[i];
    if (!e->is_split && e->loc == LNE)
      if ( !integrate_specific
           || (integrate_specific
               && (e->c0 == num0 || e->c1 == num0)
               && (e->c0 == num1 || e->c1 == num1)) )
        result += length(e->vtx0, e->vtx1)*(vtxs[e->vtx0].value + vtxs[e->vtx1].value)/2.0;
  }

  return result;
}

double simplex3_mls_t::integrate_over_intersection(double f0, double f1, double f2, double f3, int num0, int num1, int num2)
{
//  /* sort values */
//  if (num0 > num1) swap(num0, num1);
//  if (num1 > num2) swap(num1, num2);
//  if (num0 > num1) swap(num0, num1);

  double result = 0.0;
  bool integrate_specific = (num0 != -1 && num1 != -1 && num2 != -1);

  interpolate_all(f0, f1, f2, f3);

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
        result += v->value;
      }
  }

  return result;
}

double simplex3_mls_t::integrate_in_dir(double f0, double f1, double f2, double f3, int dir)
{
  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2, f3);

  double result = 0.0;

  /* integrate over triangles */
  for (unsigned int i = 0; i < tris.size(); i++)
  {
    tri3_t *t = &tris[i];
    if (!t->is_split && t->loc == INS)
      if (t->dir == dir)
        result += area(t->vtx0, t->vtx1, t->vtx2) *
            (vtxs[t->vtx0].value + vtxs[t->vtx1].value + vtxs[t->vtx2].value)/3.0;
  }

  return result;
}

void simplex3_mls_t::interpolate_from_neighbors(int v)
{
  vtx3_t *vtx = &vtxs[v];
  vtx->value = vtx->ratio*vtxs[vtx->n_vtx0].value + (1.0-vtx->ratio)*vtxs[vtx->n_vtx1].value;
}

void simplex3_mls_t::interpolate_from_parent(int v)
{
  double vol0 = volume(v, 1, 2, 3);
  double vol1 = volume(0, v, 2, 3);
  double vol2 = volume(0, 1, v, 3);
  double vol3 = volume(0, 1, 2, v);
  double vol  = volume(0, 1, 2, 3);

  #ifdef CASL_THROWS
    if (vol < eps)
      throw std::domain_error("[CASL_ERROR]: Division by zero.");
  #endif

  vtxs[v].value = (vol0*vtxs[0].value + vol1*vtxs[1].value + vol2*vtxs[2].value + vol3*vtxs[3].value)/vol;
}

void simplex3_mls_t::interpolate_from_parent(vtx3_t &vertex)
{
  double vol0 = volume(vertex, vtxs[1], vtxs[2], vtxs[3]);
  double vol1 = volume(vtxs[0], vertex, vtxs[2], vtxs[3]);
  double vol2 = volume(vtxs[0], vtxs[1], vertex, vtxs[3]);
  double vol3 = volume(vtxs[0], vtxs[1], vtxs[2], vertex);
  double vol  = volume(vtxs[0], vtxs[1], vtxs[2], vtxs[3]);

  #ifdef CASL_THROWS
    if (vol < eps)
      throw std::domain_error("[CASL_ERROR]: Division by zero.");
  #endif

  vertex.value = (vol0*vtxs[0].value + vol1*vtxs[1].value + vol2*vtxs[2].value + vol3*vtxs[3].value)/vol;
}

void simplex3_mls_t::interpolate_all(double &p0, double &p1, double &p2, double &p3)
{
  vtxs[0].value = p0;
  vtxs[1].value = p1;
  vtxs[2].value = p2;
  vtxs[3].value = p3;

  for (unsigned int i = 4; i < vtxs.size(); i++) interpolate_from_neighbors(i);
//  for (int i = 4; i < vtxs.size(); i++) interpolate_from_parent(i);
}

double simplex3_mls_t::find_intersection_linear(int v0, int v1)
{
  vtx3_t *vtx0 = &vtxs[v0];
  vtx3_t *vtx1 = &vtxs[v1];
  double nx = vtx1->x - vtx0->x;
  double ny = vtx1->y - vtx0->y;
  double nz = vtx1->z - vtx0->z;
  double l = sqrt(nx*nx+ny*ny+nz*nz);
#ifdef CASL_THROWS
  if(l < eps) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
#endif
  nx /= l;
  ny /= l;
  nz /= l;
  double f0 = vtx0->value;
  double f1 = vtx1->value;

  if(fabs(f0)<eps) return 0.+eps;
  if(fabs(f1)<eps) return l-eps;

#ifdef CASL_THROWS
  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
#endif

  double c1 =     (f1-f0)/l;          //  the expansion of f at the center of (a,b)
  double c0 = 0.5*(f1+f0);

  double x = -c0/c1;

#ifdef CASL_THROWS
  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
#endif

  return 1.-(x+0.5*l)/l;
}

double simplex3_mls_t::find_intersection_quadratic(int e)
{
  vtx3_t *vtx0 = &vtxs[edgs[e].vtx0];
  vtx3_t *vtx1 = &vtxs[edgs[e].vtx1];
  double nx = vtx1->x - vtx0->x;
  double ny = vtx1->y - vtx0->y;
  double nz = vtx1->z - vtx0->z;
  double l = sqrt(nx*nx+ny*ny+nz*nz);
#ifdef CASL_THROWS
  if(l < eps) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
#endif
  nx /= l;
  ny /= l;
  nz /= l;
  double f0 = vtx0->value;
  double f01 = edgs[e].value;
  double f1 = vtx1->value;

  if (fabs(f0)  < eps) return (l-eps)/l;
  if (fabs(f01) < eps) return 0.5;
  if (fabs(f1)  < eps) return (0.+eps)/l;

#ifdef CASL_THROWS
  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
#endif

  double fdd = (f1+f0-2.*f01)/(0.25*l*l);

  double c2 = 0.5*fdd;   // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 = (f1-f0)/l; //  the expansion of f at the center of (a,b)
  double c0 = f01;

  double x;

  if(fabs(c2)<eps) x = -c0/c1;
  else
  {
    if(f1<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    else     x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
  }
#ifdef CASL_THROWS
  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
#endif

//  if (x < -0.5*l) return (l-eps)/l;
//  if (x > 0.5*l) return (0.+eps)/l;

  return 1.-(x+0.5*l)/l;
}

void simplex3_mls_t::get_edge_coords(int e, double xyz[])
{
  vtx3_t *vtx0 = &vtxs[edgs[e].vtx0];
  vtx3_t *vtx1 = &vtxs[edgs[e].vtx1];

  xyz[0] = 0.5*(vtx0->x+vtx1->x);
  xyz[1] = 0.5*(vtx0->y+vtx1->y);
  xyz[2] = 0.5*(vtx0->z+vtx1->z);
}


#ifdef CASL_THROWS
bool simplex3_mls_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = (edgs[e0].vtx0 == v1 || edgs[e0].vtx1 == v1) && (edgs[e0].vtx0 == v2 || edgs[e0].vtx1 == v2);
  result = result && (edgs[e1].vtx0 == v0 || edgs[e1].vtx1 == v0) && (edgs[e1].vtx0 == v2 || edgs[e1].vtx1 == v2);
  result = result && (edgs[e2].vtx0 == v0 || edgs[e2].vtx1 == v0) && (edgs[e2].vtx0 == v1 || edgs[e2].vtx1 == v1);
  return result;
}

bool simplex3_mls_t::tri_is_ok(int t)
{
  tri3_t *tri = &tris[t];
  bool result = tri_is_ok(tri->vtx0, tri->vtx1, tri->vtx2, tri->edg0, tri->edg1, tri->edg2);
  if (!result) std::cout << "Inconsistent triangle!\n";
  return result;
}

bool simplex3_mls_t::tet_is_ok(int s)
{
  bool result = true;
  tet3_t *tet = &tets[s];

  tri3_t *tri;

  tri = &tris[tet->tri0];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);

  tri = &tris[tet->tri1];
  result = result && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);

  tri = &tris[tet->tri2];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0)
                  && (tri->vtx0 == tet->vtx3 || tri->vtx1 == tet->vtx3 || tri->vtx2 == tet->vtx3);

  tri = &tris[tet->tri3];
  result = result && (tri->vtx0 == tet->vtx1 || tri->vtx1 == tet->vtx1 || tri->vtx2 == tet->vtx1)
                  && (tri->vtx0 == tet->vtx2 || tri->vtx1 == tet->vtx2 || tri->vtx2 == tet->vtx2)
                  && (tri->vtx0 == tet->vtx0 || tri->vtx1 == tet->vtx0 || tri->vtx2 == tet->vtx0);

  return result;
}
#endif
