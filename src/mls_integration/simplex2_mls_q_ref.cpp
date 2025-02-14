#include "simplex2_mls_q_ref.h"

//--------------------------------------------------
// Constructors
//--------------------------------------------------
simplex2_mls_q_ref_t::simplex2_mls_q_ref_t(double x0, double y0,
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

  edgs_[0].dir = 0;
  edgs_[1].dir = 1;
  edgs_[2].dir = 2;

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
void simplex2_mls_q_ref_t::construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{

  for (short idx = 0; idx < simplices_.size(); idx++)
  {
    bool success = simplices_[idx].construct_domain();

    if (!success)
    {

    }

  }
}




//--------------------------------------------------
// Quadrature points
//--------------------------------------------------
void simplex2_mls_q_ref_t::quadrature_over_domain(std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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

void simplex2_mls_q_ref_t::quadrature_over_interface(int num, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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

void simplex2_mls_q_ref_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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

void simplex2_mls_q_ref_t::quadrature_in_dir(int dir, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y)
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
// Interpolation
//--------------------------------------------------
double simplex2_mls_q_ref_t::interpolate_from_parent(std::vector<double> &f, double x, double y)
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

double simplex2_mls_q_ref_t::interpolate_from_parent(double x, double y)
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
