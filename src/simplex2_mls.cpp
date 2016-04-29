#include "simplex2_mls.h"
#include "CASL_math.h"

#define EPS 1.0e-20

simplex2_mls_t::simplex2_mls_t()
{
  vtxs.reserve(8);
  edgs.reserve(27);
  tris.reserve(20);
}

simplex2_mls_t::simplex2_mls_t(double x0, double y0,
                               double x1, double y1,
                               double x2, double y2)
//  : x({x0, x1, x2}), y({y0, y1, y2})
{
  // usually there will be only one cut
  vtxs.reserve(4);
  edgs.reserve(9);
  tris.reserve(4);

  /* fill the vectors with the initial structure */
  vtxs.push_back(vtx2_t(x0,y0));
  vtxs.push_back(vtx2_t(x1,y1));
  vtxs.push_back(vtx2_t(x2,y2));

  edgs.push_back(edg2_t(1,2));
  edgs.push_back(edg2_t(0,2));
  edgs.push_back(edg2_t(0,1));

  tris.push_back(tri2_t(0,1,2,0,1,2));

  phi = NULL;
  phi_x = NULL;
  phi_y = NULL;
}

void simplex2_mls_t::do_action(std::vector<double> *phi_, std::vector<double> *phi_x_, std::vector<double> *phi_y_, int cn, action_t action)
{
  phi = phi_;
  phi_x = phi_x_;
  phi_y = phi_y_;

#ifdef CASL_THROWS
  if(phi == NULL) throw std::invalid_argument("[CASL_ERROR]: Values of LSF are not provided.");
#endif
  for (int i = 0; i < vtxs.size(); i++)
  {
    vtxs[i].value = phi->at(i);
  }

  if (phi_x == NULL || phi_y == NULL) use_linear = true;
  else                                use_linear = false;

  /* Process elements */
  int n;
  n = vtxs.size(); for (int i = 0; i < n; i++) do_action_vtx(i, cn, action);
  n = edgs.size(); for (int i = 0; i < n; i++) do_action_edg(i, cn, action);
  n = tris.size(); for (int i = 0; i < n; i++) do_action_tri(i, cn, action);
}

void simplex2_mls_t::do_action_vtx(int n_vtx, int cn, action_t action)
{
  vtx2_t *vtx = &vtxs[n_vtx];

  perturb(vtx->value, EPS);

  switch (action){
  case INTERSECTION:  if (vtx->value > 0)                                      vtx->set(OUT, -1, -1);  break;
  case ADDITION:      if (vtx->value < 0)                                      vtx->set(INS, -1, -1);  break;
  case COLORATION:    if (vtx->value < 0) if (vtx->loc==FCE || vtx->loc==PNT)  vtx->set(FCE, cn, -1);  break;
  }
}
void simplex2_mls_t::do_action_edg(int n_edg, int cn, action_t action)
{
  edg2_t *edg = &edgs[n_edg];

  int c0 = edg->c0;

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
  vtx2_t *c_vtx01;
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
    /* split an edge */
    edg->is_split = true;

    // new vertex
    if (use_linear) r = find_intersection_linear    (edg->vtx0, edg->vtx1);
    else            r = find_intersection_quadratic (edg->vtx0, edg->vtx1);

    vtxs.push_back(vtx2_t(vtxs[edg->vtx0].x*r + vtxs[edg->vtx1].x*(1.0-r),
                          vtxs[edg->vtx0].y*r + vtxs[edg->vtx1].y*(1.0-r)));

    vtxs.back().n_vtx0 = edg->vtx0;
    vtxs.back().n_vtx1 = edg->vtx1;
    vtxs.back().ratio  = r;

    edg->c_vtx01 = vtxs.size()-1;

    // new edges
    edgs.push_back(edg2_t(edg->vtx0,    edg->c_vtx01)); edg = &edgs[n_edg]; // edges might have changed their addresses
    edgs.push_back(edg2_t(edg->c_vtx01, edg->vtx1   )); edg = &edgs[n_edg];

    edg->c_edg0 = edgs.size()-2;
    edg->c_edg1 = edgs.size()-1;


    /* apply rules */
    c_vtx01 = &vtxs[edg->c_vtx01];
    c_edg0  = &edgs[edg->c_edg0];
    c_edg1  = &edgs[edg->c_edg1];


    c_edg0->dir = edg->dir;
    c_edg1->dir = edg->dir;

    c_edg0->p_lsf = edg->p_lsf;
    c_edg1->p_lsf = edg->p_lsf;

#ifdef CASL_THROWS
    c_vtx01->p_edg = n_edg;
    c_edg0->p_edg  = n_edg;
    c_edg1->p_edg  = n_edg;
#endif
    switch (action){
    case INTERSECTION:
      switch (edg->loc){
      case OUT: c_vtx01->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); break;
      case INS: c_vtx01->set(FCE, cn, -1); c_edg0->set(INS, -1); c_edg1->set(OUT, -1); break;
      case FCE: c_vtx01->set(PNT, c0, cn); c_edg0->set(FCE, c0); c_edg1->set(OUT, -1); if (c0==cn)  c_vtx01->set(FCE, c0, -1);  break;
      }
      break;
    case ADDITION:
      switch (edg->loc){
      case OUT: c_vtx01->set(FCE, cn, -1); c_edg0->set(INS, -1); c_edg1->set(OUT, -1); break;
      case INS: c_vtx01->set(INS, -1, -1); c_edg0->set(INS, -1); c_edg1->set(INS, -1); break;
      case FCE: c_vtx01->set(PNT, c0, cn); c_edg0->set(INS, -1); c_edg1->set(FCE, c0); if (c0==cn)  c_vtx01->set(FCE, c0, -1);  break;
      }
      break;
    case COLORATION:
      switch (edg->loc){
      case OUT: c_vtx01->set(OUT, -1, -1); c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); break;
      case INS: c_vtx01->set(INS, -1, -1); c_edg0->set(INS, -1); c_edg1->set(INS, -1); break;
      case FCE: c_vtx01->set(PNT, c0, cn); c_edg0->set(FCE, cn); c_edg1->set(FCE, c0); if (c0==cn)  c_vtx01->set(FCE, c0, -1);  break;
      }
      break;
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

void simplex2_mls_t::do_action_tri(int n_tri, int cn, action_t action)
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
  edg2_t *c_edg0, *c_edg1;
  edg2_t *edg0, *edg1, *edg2;
  tri2_t *c_tri0, *c_tri1, *c_tri2;

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
    /* split a triangle */
    tri->is_split = true;

    // new vertices
    tri->c_vtx01 = edgs[tri->edg2].c_vtx01;
    tri->c_vtx02 = edgs[tri->edg1].c_vtx01;

    // new edges
    edgs.push_back(edg2_t(tri->c_vtx01, tri->c_vtx02));
    edgs.push_back(edg2_t(tri->c_vtx01, tri->vtx2   ));

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
      case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
      case INS: c_edg0->set(FCE, cn); c_edg1->set(OUT, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
      } break;
    case ADDITION:
      switch (tri->loc){
      case OUT: c_edg0->set(FCE, cn); c_edg1->set(OUT, -1); c_tri0->set(INS); c_tri1->set(OUT); c_tri2->set(OUT); break;
      case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
      } break;
    case COLORATION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
      case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
      } break;
    }
    break;

  case 2: // (--+)
    /* split a triangle */
    tri->is_split = true;

    tri->c_vtx02 = edgs[tri->edg1].c_vtx01;
    tri->c_vtx12 = edgs[tri->edg0].c_vtx01;

    // create new edges
    edgs.push_back(edg2_t(tri->vtx0,    tri->c_vtx12));
    edgs.push_back(edg2_t(tri->c_vtx02, tri->c_vtx12));

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
      case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
      case INS: c_edg0->set(INS, -1); c_edg1->set(FCE, cn); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
      } break;
    case ADDITION:
      switch (tri->loc){
      case OUT: c_edg0->set(INS, -1); c_edg1->set(FCE, cn); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(OUT); break;
      case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
      } break;
    case COLORATION:
      switch (tri->loc){
      case OUT: c_edg0->set(OUT, -1); c_edg1->set(OUT, -1); c_tri0->set(OUT); c_tri1->set(OUT); c_tri2->set(OUT); break;
      case INS: c_edg0->set(INS, -1); c_edg1->set(INS, -1); c_tri0->set(INS); c_tri1->set(INS); c_tri2->set(INS); break;
      } break;
    }
    break;

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

bool simplex2_mls_t::need_swap(int v0, int v1)
{
  double dif = vtxs[v0].value - vtxs[v1].value;
  if (fabs(dif) < EPS){ // if values are too close, sort vertices by their numbers
    if (v0 > v1) return true;
    else         return false;
  } else if (dif > 0.0){ // otherwise sort by values
    return true;
  } else {
    return false;
  }
}

double simplex2_mls_t::length(int vtx0, int vtx1)
{
  return sqrt(pow(vtxs[vtx0].x - vtxs[vtx1].x, 2.0)
            + pow(vtxs[vtx0].y - vtxs[vtx1].y, 2.0));
}

double simplex2_mls_t::area(int vtx0, int vtx1, int vtx2)
{
  double x01 = vtxs[vtx1].x - vtxs[vtx0].x; double x02 = vtxs[vtx2].x - vtxs[vtx0].x;
  double y01 = vtxs[vtx1].y - vtxs[vtx0].y; double y02 = vtxs[vtx2].y - vtxs[vtx0].y;

  return 0.5*fabs(x01*y02-y01*x02);
}

double simplex2_mls_t::integrate_over_domain(double f0, double f1, double f2)
{
  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2);

  double result = 0.0;

  /* integrate over triangles */
  for (int i = 0; i < tris.size(); i++)
  {
    tri2_t *t = &tris[i];
    if (!t->is_split && t->loc == INS)
      {
        result += area(t->vtx0, t->vtx1, t->vtx2) *
            (vtxs[t->vtx0].value + vtxs[t->vtx1].value + vtxs[t->vtx2].value)/3.0;
      }
  }

  return result;
}

double simplex2_mls_t::integrate_over_interface(double f0, double f1, double f2, int num)
{
  bool integrate_specific = (num != -1);

  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2);

  double result = 0.0;

  /* integrate over edges */
  for (int i = 0; i < edgs.size(); i++)
  {
    edg2_t *e = &edgs[i];
    if (!e->is_split && e->loc == FCE)
      if ((!integrate_specific && e->c0 >= 0) || (integrate_specific && e->c0 == num))
        result += length(e->vtx0, e->vtx1)*(vtxs[e->vtx0].value + vtxs[e->vtx1].value)/2.0;
  }

  return result;
}

// integrate over colored interfaces (num0 - parental lsf, num1 - coloring lsf)
double simplex2_mls_t::integrate_over_colored_interface(double f0, double f1, double f2, int num0, int num1)
{
  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2);

  double result = 0.0;

  /* integrate over edges */
  for (int i = 0; i < edgs.size(); i++)
  {
    edg2_t *e = &edgs[i];
    if (!e->is_split && e->loc == FCE)
      if (e->p_lsf == num0 && e->c0 == num1)
        result += length(e->vtx0, e->vtx1)*(vtxs[e->vtx0].value + vtxs[e->vtx1].value)/2.0;
  }

  return result;
}

double simplex2_mls_t::integrate_over_intersection(double f0, double f1, double f2, int num0, int num1)
{
  double result = 0.0;
  bool integrate_specific = (num0 != -1 && num1 != -1);

  interpolate_all(f0, f1, f2);

  for (int i = 0; i < vtxs.size(); i++)
  {
    vtx2_t *v = &vtxs[i];
    if (v->loc == PNT)
      if ( !integrate_specific
           || (integrate_specific
               && (v->c0 == num0 || v->c1 == num0)
               && (v->c0 == num1 || v->c1 == num1)) )
      {
        result += v->value;
      }
  }

  return result;
}

double simplex2_mls_t::integrate_in_dir(double f0, double f1, double f2, int dir)
{
  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2);

  double result = 0.0;

  /* integrate over edges */
  for (int i = 0; i < edgs.size(); i++)
  {
    edg2_t *e = &edgs[i];
    if (!e->is_split && e->loc == INS)
      if (e->dir == dir)
        result += length(e->vtx0, e->vtx1)*(vtxs[e->vtx0].value + vtxs[e->vtx1].value)/2.0;
  }

  return result;
}

double simplex2_mls_t::integrate_in_non_cart_dir(double f0, double f1, double f2, int num)
{
  /* interpolate function values to vertices */
  interpolate_all(f0, f1, f2);

  double result = 0.0;

  /* integrate over edges */
  for (int i = 0; i < edgs.size(); i++)
  {
    edg2_t *e = &edgs[i];
    if (!e->is_split && (e->loc == INS || e->loc == FCE))
      if (e->p_lsf == num)
        result += length(e->vtx0, e->vtx1)*(vtxs[e->vtx0].value + vtxs[e->vtx1].value)/2.0;
  }

  return result;
}

void simplex2_mls_t::interpolate_from_neighbors(int v)
{
  vtx2_t *vtx = &vtxs[v];
  vtx->value = vtx->ratio*vtxs[vtx->n_vtx0].value + (1.0-vtx->ratio)*vtxs[vtx->n_vtx1].value;
}

void simplex2_mls_t::interpolate_all(double &p0, double &p1, double &p2)
{
  vtxs[0].value = p0;
  vtxs[1].value = p1;
  vtxs[2].value = p2;

  for (int i = 3; i < vtxs.size(); i++)
  {
    interpolate_from_neighbors(i);
  }
}

double simplex2_mls_t::find_intersection_linear(int v0, int v1)
{
  vtx2_t *vtx0 = &vtxs[v0];
  vtx2_t *vtx1 = &vtxs[v1];
  double nx = vtx1->x - vtx0->x;
  double ny = vtx1->y - vtx0->y;
  double l = sqrt(nx*nx+ny*ny);
#ifdef CASL_THROWS
  if(l < EPS) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
#endif
  nx /= l;
  ny /= l;
  double f0 = vtx0->value;
  double f1 = vtx1->value;

  if(fabs(f0)<EPS) return 0.+EPS;
  if(fabs(f1)<EPS) return l-EPS;

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

double simplex2_mls_t::find_intersection_quadratic(int v0, int v1)
{
#ifdef CASL_THROWS
  if (phi_x == NULL) throw std::invalid_argument("[CASL_ERROR]: Values of x-derivative of LSF are not provided.");
  if (phi_y == NULL) throw std::invalid_argument("[CASL_ERROR]: Values of y-derivative of LSF are not provided.");
#endif

  vtx2_t *vtx0 = &vtxs[v0];
  vtx2_t *vtx1 = &vtxs[v1];
  double nx = vtx1->x - vtx0->x;
  double ny = vtx1->y - vtx0->y;
  double l = sqrt(nx*nx+ny*ny);
#ifdef CASL_THROWS
  if(l < EPS) throw std::invalid_argument("[CASL_ERROR]: Vertices are too close.");
#endif
  nx /= l;
  ny /= l;
  double f0 = vtx0->value;  double fd0 = phi_x->at(v0)*nx + phi_y->at(v0)*ny;
  double f1 = vtx1->value;  double fd1 = phi_x->at(v1)*nx + phi_y->at(v1)*ny;

  if(fabs(f0)<EPS) return (l-EPS)/l;
  if(fabs(f1)<EPS) return (0.+EPS)/l;

#ifdef CASL_THROWS
  if(f0*f1 >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
#endif

  double fdd = (fd1-fd0)/l;
//  double fdd = MINMOD(fdd0,fdd1); // take nonocillating fxx

  double c2 = 0.5*fdd;                // c2*(x-xc)^2 + c1*(x-xc) + c0 = 0, i.e
  double c1 =     (f1-f0)/l;          //  the expansion of f at the center of (a,b)
//  double c1 = 0.5*(fd1+fd0);          //  the expansion of f at the center of (a,b)
//  double c1 = 0.25*(fd1+fd0)+0.5*(f1-f0)/l;          //  the expansion of f at the center of (a,b)
  double c0 = 0.5*(f1+f0)-l*l/8.*fdd;

  double x;

  if(fabs(c2)<EPS) x = -c0/c1;
  else
  {
    if(f1<0) x = (-2.*c0)/(c1 - sqrt(c1*c1-4.*c2*c0));
    else     x = (-2.*c0)/(c1 + sqrt(c1*c1-4.*c2*c0));
  }
#ifdef CASL_THROWS
  if (x < -0.5*l || x > 0.5*l) throw std::domain_error("[CASL_ERROR]: ");
#endif

  if (x < -0.5*l) return (l-EPS)/l;
  if (x > 0.5*l) return (0.+EPS)/l;

  return 1.-(x+0.5*l)/l;
}

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

//  while (fabs(f_b) > EPS && fabs(r_a-r_b) > tol)
//  {
//    f_c = phi_cf(x0 + r_c*(x1-x0), y0 + r_c*(y1-y0));

//    if (fabs(f_a-f_c) > EPS && fabs(f_b-f_c) > EPS)
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

#ifdef CASL_THROWS
bool simplex2_mls_t::tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2)
{
  bool result = true;
  result = (edgs[e0].vtx0 == v1 || edgs[e0].vtx1 == v1) && (edgs[e0].vtx0 == v2 || edgs[e0].vtx1 == v2);
  result = result && (edgs[e1].vtx0 == v0 || edgs[e1].vtx1 == v0) && (edgs[e1].vtx0 == v2 || edgs[e1].vtx1 == v2);
  result = result && (edgs[e2].vtx0 == v0 || edgs[e2].vtx1 == v0) && (edgs[e2].vtx0 == v1 || edgs[e2].vtx1 == v1);
  return result;
}

bool simplex2_mls_t::tri_is_ok(int t)
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
