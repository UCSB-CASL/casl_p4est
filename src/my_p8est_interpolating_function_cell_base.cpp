#include <src/my_p4est_to_p8est.h>
#include "my_p4est_interpolating_function_cell_base.cpp"

inline static bool test_tetrahedron(const Point3 p1, const Point3& p2, const Point3& p3, const Point3& px, double *uvw)
{
  double det = compute_det(p1, p2, p3);
  uvw[0] = compute_det(px, p2, p3)/det; if (uvw[0] < 0) return false;
  uvw[1] = compute_det(p1, px, p3)/det; if (uvw[1] < 0) return false;
  uvw[2] = compute_det(p1, p2, px)/det; if (uvw[2] < 0) return false;
  if (uvw[0] + uvw[1] + uvw[2] <= 1)
    return true;
  else
    return false;
}

inline static bool test_edge_tetrahedrons(const p4est_t *p4est_, const Point3& p0, const Point3& px,
                                          std::vector<const InterpolatingFunctionCellBase::quad_info_t*>& ng1,
                                          std::vector<const InterpolatingFunctionCellBase::quad_info_t*>& ng2,
                                          double *uvw, const InterpolatingFunctionCellBase::quad_info_t **qu123)
{
  Point3 p1, p2, p3;

  /* now construct the triangulation */
  if (ng1.size() != 0 && ng2.size() != 0){
    if (ng1.size() > ng2.size())
      std::swap(ng1, ng2);

    // forward
    int s = (ng2.size()-1) / ng1.size();
    int r = (ng2.size()-1) % ng1.size();
    int c = 0;
    std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
    if (ng2.size() > 1){
      for (int i=0; i<r; i++){
        for (int j = 0; j<s+1; j++){
          quad_center(p4est_,ng1[i],     p1); p1 -= p0;
          quad_center(p4est_,ng2[c+j],   p2); p2 -= p0;
          quad_center(p4est_,ng2[c+j+1], p3); p3 -= p0;

          if(test_tetrahedron(p1, p2, p3, px, uvw)){
            qu123[0] = ng1[i];
            qu123[1] = ng2[c+j];
            qu123[2] = ng2[c+j+1];
            return true;
          }
        }
        c += s+1;
        ng2s.push_back(c);
      }
      for (size_t i = r; i<ng1.size(); i++){
        for (int j = 0; j<s; j++){
          quad_center(p4est_,ng1[i],     p1); p1 -= p0;
          quad_center(p4est_,ng2[c+j],   p2); p2 -= p0;
          quad_center(p4est_,ng2[c+j+1], p3); p3 -= p0;

          if(test_tetrahedron(p1, p2, p3, px, uvw)){
            qu123[0] = ng1[i];
            qu123[1] = ng2[c+j];
            qu123[2] = ng2[c+j+1];
            return true;
          }
        }
        c += s;
        ng2s.push_back(c);
      }
    }

    //backward
    if (ng1.size() > 1)
      for (size_t i=0; i<ng1.size() - 1; i++){
        quad_center(p4est_,ng1[i],       p1); p1 -= p0;
        quad_center(p4est_,ng1[i+1],     p2); p2 -= p0;
        quad_center(p4est_,ng2[ng2s[i]], p3); p3 -= p0;

        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = ng1[i];
          qu123[1] = ng1[i+1];
          qu123[2] = ng2[ng2s[i]];
          return true;
        }
      }
  }

  return false;
}

bool InterpolatingFunctionCellBase::find_tetrahedron_containing_point_m00(p4est_locidx_t qu, p4est_topidx_t tr, Point3& px, double *uvw, const quad_info_t **qu123) const
{
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = cnnn_->begin(q, i);
  cells[P4EST_FACES] = cnnn_->end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  Point3 p0, p1, p2, p3;
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    p0.x = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    p0.y = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    p0.z = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;
  }
  px -= p0;

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
  {
    quad_center(p4est_,it, p1); p1 -= p0;

    std::vector<const quad_info_t*> ng_0p0;
    std::vector<const quad_info_t*> ng_00p;
    int8_t l_0p0, l_00p;

    bool is_boundary_0p0 = it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_0p0)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x + P4EST_QUADRANT_LEN(it_->level) == it->quad->x + P4EST_QUADRANT_LEN(it->level) ){
          ng_0p0.push_back(it_);
          l_0p0 = it_->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x + P4EST_QUADRANT_LEN(it_->level) == it->quad->x + P4EST_QUADRANT_LEN(it->level)){
          ng_00p.push_back(it_);
          l_00p = it_->level;
        }
    }

    /* pp0 */
    if (!is_boundary_0p0 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_0p0 > l_00p)
        it_ = cnnn_->begin(ng_0p0[ng_0p0.size() - 1]->locidx, dir::f_00p);
      else
        it_ = cnnn_->begin(ng_00p[ng_00p.size() - 1]->locidx, dir::f_0p0);

      ng_0p0.push_back(it_);
      ng_00p.push_back(it_);
    }

    /* now that we have all cells, construct the elements */
    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        quad_center(p4est_,ng_0p0[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_0p0[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_0p0[i];
          qu123[2] = ng_0p0[i+1];
          return true;
        }
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        quad_center(p4est_,ng_00p[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_00p[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_00p[i];
          qu123[2] = ng_00p[i+1];
          return true;
        }
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_m00 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y <= quad->y){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 5 - across mm0 */
    if (n_0m0 != 0 && n_00m != 0)
    {
      quad_center(p4est_,begin_m00, p1); p1 -= p0;
      quad_center(p4est_,begin_0m0, p2); p2 -= p0;
      quad_center(p4est_,begin_00m, p3); p3 -= p0;

      if(test_tetrahedron(p1, p2, p3, px, uvw)){
        qu123[0] = begin_m00;
        qu123[1] = begin_0m0;
        qu123[2] = begin_00m;
        return true;
      }
    }

    /* 6 - across pm0 */
    if (n_0p0 != 0 && n_00m != 0)
    {
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      quad_center(p4est_,begin_0p0, p2); p2 -= p0; qu123[1] = begin_0p0;

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 7 - across mp0 */
    if (n_0m0 != 0 && n_00p != 0)
    {
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      quad_center(p4est_,begin_00p, p3); p3 -= p0; qu123[2] = begin_00p;
      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      quad_center(p4est_,end_m00-1, p1); p1 -= p0; qu123[0] = end_m00-1;

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
  }

  return false;
}

bool InterpolatingFunctionCellBase::find_tetrahedron_containing_point_p00(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const
{
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = cnnn_->begin(q, i);
  cells[P4EST_FACES] = cnnn_->end(q, P4EST_FACES - 1);

  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  Point3 p0, p1, p2, p3;
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    p0.x = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    p0.y = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    p0.z = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;
  }
  px -= p0;

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
  {
    quad_center(p4est_,it, p1); p1 -= p0;

    std::vector<const quad_info_t*> ng_0p0;
    std::vector<const quad_info_t*> ng_00p;
    int8_t l_0p0, l_00p;

    bool is_boundary_0p0 = it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_0p0)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x == it->quad->x){
          ng_0p0.push_back(it_);
          l_0p0 = it_->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x == it->quad->x){
          ng_00p.push_back(it_);
          l_00p = it_->level;
        }
    }

    /* pp0 */
    if (!is_boundary_0p0 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_0p0 > l_00p)
        it_ = cnnn_->begin(ng_0p0[ng_0p0.size() - 1]->locidx, dir::f_00p);
      else
        it_ = cnnn_->begin(ng_00p[ng_00p.size() - 1]->locidx, dir::f_0p0);

      ng_0p0.push_back(it_);
      ng_00p.push_back(it_);
    }

    /* now that we have all cells, construct the elements */
    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        quad_center(p4est_,ng_0p0[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_0p0[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_0p0[i];
          qu123[2] = ng_0p0[i+1];
          return true;
        }
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        quad_center(p4est_,ng_00p[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_00p[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_00p[i];
          qu123[2] = ng_00p[i+1];
          return true;
        }
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_p00 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y <= quad->y){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 5 - across mm0 */
    if (n_0m0 != 0 && n_00m != 0)
    {
      quad_center(p4est_, begin_p00, p1); p1 -= p0; qu123[0] = begin_p00;

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 6 - across pm0 */
    if (n_0p0 != 0 && n_00m != 0)
    {
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      quad_center(p4est_,end_00m-1, p3); p3 -= p0; qu123[2] = end_00m-1;

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 7 - across mp0 */
    if (n_0m0 != 0 && n_00p != 0)
    {

      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      quad_center(p4est_,end_0m0-1, p2); p2 -= p0; qu123[1] = end_0m0-1;

      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 8 - across pp0 */
    if (n_0p0 != 0 && n_0p0 != 0)
    {
      quad_center(p4est_,end_p00-1, p1); p1 -= p0; qu123[0] = end_p00-1;
      quad_center(p4est_,end_0p0-1, p2); p2 -= p0; qu123[1] = end_0p0-1;
      quad_center(p4est_,end_00p-1, p3); p3 -= p0; qu123[2] = end_00p-1;

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
  }

  return false;
}

bool InterpolatingFunctionCellBase::find_tetrahedron_containing_point_0m0(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const
{
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = cnnn_->begin(q, i);
  cells[P4EST_FACES] = cnnn_->end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  Point3 p0, p1, p2, p3;
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    p0.x = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    p0.y = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    p0.z = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;
  }
  px -= p0;

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
  {
    quad_center(p4est_,it, p1); p1 -= p0;

    std::vector<const quad_info_t*> ng_p00;
    std::vector<const quad_info_t*> ng_00p;
    int8_t l_p00, l_00p;

    bool is_boundary_p00 = it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_p00)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y + P4EST_QUADRANT_LEN(it_->level) == it->quad->y + P4EST_QUADRANT_LEN(it->level) ){
          ng_p00.push_back(it_);
          l_p00 = it_->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y + P4EST_QUADRANT_LEN(it_->level) == it->quad->y + P4EST_QUADRANT_LEN(it->level)){
          ng_00p.push_back(it_);
          l_00p = it_->level;
        }
    }

    /* pp0 */
    if (!is_boundary_p00 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_p00 > l_00p)
        it_ = cnnn_->begin(ng_p00[ng_p00.size() - 1]->locidx, dir::f_00p);
      else
        it_ = cnnn_->begin(ng_00p[ng_00p.size() - 1]->locidx, dir::f_p00);

      ng_p00.push_back(it_);
      ng_00p.push_back(it_);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        quad_center(p4est_,ng_p00[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_p00[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_p00[i];
          qu123[2] = ng_p00[i+1];
          return true;
        }
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        quad_center(p4est_,ng_00p[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_00p[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_00p[i];
          qu123[2] = ng_00p[i+1];
          return true;
        }
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_0m0 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x <= quad->x){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 5 - across mm0 */
    if (n_m00 != 0 && n_00m != 0)
    {
      quad_center(p4est_,begin_m00, p1); p1 -= p0; qu123[0] = begin_m00;
      quad_center(p4est_,begin_0m0, p2); p2 -= p0; qu123[1] = begin_0m0;
      quad_center(p4est_,begin_00m, p3); p3 -= p0; qu123[2] = begin_00m;

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_00m != 0)
    {
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      quad_center(p4est_,begin_p00, p2); p2 -= p0; qu123[1] = begin_p00;

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 7 - across mp0 */
    if (n_m00 != 0 && n_00p != 0)
    {
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      quad_center(p4est_,begin_00p, p3); p3 -= p0; qu123[2] = begin_00p;

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      quad_center(p4est_,end_0m0 - 1, p1); p1 -= p0; qu123[0] = end_0m0-1;
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
  }

  return false;
}

bool InterpolatingFunctionCellBase::find_tetrahedron_containing_point_0p0(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const
{
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = cnnn_->begin(q, i);
  cells[P4EST_FACES] = cnnn_->end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  Point3 p0, p1, p2, p3;
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    p0.x = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    p0.y = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    p0.z = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;
  }
  px -= p0;

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
  {
    quad_center(p4est_,it, p1); p1 -= p0;

    std::vector<const quad_info_t*> ng_p00;
    std::vector<const quad_info_t*> ng_00p;
    int8_t l_p00, l_00p;

    bool is_boundary_p00 = it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_p00)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y == it->quad->y ){
          ng_p00.push_back(it_);
          l_p00 = it_->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y == it->quad->y){
          ng_00p.push_back(it_);
          l_00p = it_->level;
        }
    }

    /* pp0 */
    if (!is_boundary_p00 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_p00 > l_00p)
        it_ = cnnn_->begin(ng_p00[ng_p00.size() - 1]->locidx, dir::f_00p);
      else
        it_ = cnnn_->begin(ng_00p[ng_00p.size() - 1]->locidx, dir::f_p00);

      ng_p00.push_back(it_);
      ng_00p.push_back(it_);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        quad_center(p4est_,ng_p00[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_p00[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_p00[i];
          qu123[2] = ng_p00[i+1];
          return true;
        }
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        quad_center(p4est_,ng_00p[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_00p[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_00p[i];
          qu123[2] = ng_00p[i+1];
          return true;
        }
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_0p0 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x <= quad->x){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 5 - across mm0 */
    if (n_m00 != 0 && n_00m != 0)
    {
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      quad_center(p4est_,begin_0p0, p2); p2 -= p0; qu123[1] = begin_0p0;

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_00m != 0)
    {
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      quad_center(p4est_,end_00m-1, p3); p3 -= p0; qu123[2] = end_00m-1;
      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 7 - across mp0 */
    if (n_m00 != 0 && n_00p != 0)
    {
      quad_center(p4est_,end_m00-1, p1); p1 -= p0; qu123[0] = end_m00-1;

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_00p != 0)
    {
      quad_center(p4est_,end_p00-1, p1); p1 -= p0; qu123[0] = end_p00-1;
      quad_center(p4est_,end_0p0-1, p2); p2 -= p0; qu123[1] = end_0p0-1;
      quad_center(p4est_,end_00p-1, p2); p3 -= p0; qu123[2] = end_00p-1;

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
  }

  return false;
}

bool InterpolatingFunctionCellBase::find_tetrahedron_containing_point_00m(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const
{
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = cnnn_->begin(q, i);
  cells[P4EST_FACES] = cnnn_->end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;

  Point3 p0, p1, p2, p3;
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    p0.x = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    p0.y = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    p0.z = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;
  }
  px -= p0;

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
  {
    quad_center(p4est_,it, p1); p1 -= p0;

    std::vector<const quad_info_t*> ng_p00;
    std::vector<const quad_info_t*> ng_0p0;
    int8_t l_p0, l_0p;

    bool is_p00_boundary = it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_0p0_boundary = it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_p00_boundary)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z + P4EST_QUADRANT_LEN(it_->level) == it->quad->z + P4EST_QUADRANT_LEN(it->level) ){
          ng_p00.push_back(it_);
          l_p0 = it_->level;
        }
    }

    /* 0p0 */
    if (!is_0p0_boundary)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z + P4EST_QUADRANT_LEN(it_->level) == it->quad->z + P4EST_QUADRANT_LEN(it->level)){
          ng_0p0.push_back(it_);
          l_0p = it_->level;
        }
    }

    /* pp0 */
    if (!is_p00_boundary && !is_0p0_boundary)
    {
      const quad_info_t *it_;
      if (l_p0 > l_0p)
        it_ = cnnn_->begin(ng_p00[ng_p00.size() - 1]->locidx, dir::f_0p0);
      else
        it_ = cnnn_->begin(ng_0p0[ng_0p0.size() - 1]->locidx, dir::f_p00);

      ng_p00.push_back(it_);
      ng_0p0.push_back(it_);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        quad_center(p4est_,ng_p00[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_p00[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_p00[i];
          qu123[2] = ng_p00[i+1];
          return true;
        }
      }

    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        quad_center(p4est_,ng_0p0[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_0p0[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_0p0[i];
          qu123[2] = ng_0p0[i+1];
          return true;
        }
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_00m != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x == quad->x){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) == quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y == quad->y){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) == quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 5 - across mm0 */
    if (n_m00 != 0 && n_0m0 != 0)
    {
      quad_center(p4est_,begin_00m, p1); p1 -= p0; qu123[0] = begin_00m;

      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z <= quad->z){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z <= quad->z){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_0m0 != 0)
    {
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      quad_center(p4est_,begin_p00, p2); p2 -= p0; qu123[1] = begin_p00;

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 7 - across mp0 */
    if (n_m00 != 0 && n_0p0 != 0)
    {
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      for (const quad_info_t* it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z <= quad->z){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      quad_center(p4est_,end_00m-1, p1); p1 -= p0; qu123[0] = end_00m-1;

      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if(test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
  }

  return false;
}

bool InterpolatingFunctionCellBase::find_tetrahedron_containing_point_00p(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const
{
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = cnnn_->begin(q, i);
  cells[P4EST_FACES] = cnnn_->end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  Point3 p0, p1, p2, p3;
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    p0.x = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    p0.y = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    p0.z = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;
  }
  px -= p0;

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
  {
    quad_center(p4est_,it, p1); p1 -= p0;

    std::vector<const quad_info_t*> ng_p00;
    std::vector<const quad_info_t*> ng_0p0;
    int8_t l_p0, l_0p;

    bool is_p00_boundary = it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_0p0_boundary = it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_p00_boundary)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z == it->quad->z){
          ng_p00.push_back(it_);
          l_p0 = it_->level;
        }
    }

    /* 0p0 */
    if (!is_0p0_boundary)
    {
      const quad_info_t *begin_ = cnnn_->begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = cnnn_->end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z == it->quad->z){
          ng_0p0.push_back(it_);
          l_0p = it_->level;
        }
    }

    /* pp0 */
    if (!is_p00_boundary && !is_0p0_boundary)
    {
      const quad_info_t *it_;
      if (l_p0 > l_0p)
        it_ = cnnn_->begin(ng_p00[ng_p00.size() - 1]->locidx, dir::f_0p0);
      else
        it_ = cnnn_->begin(ng_0p0[ng_0p0.size() - 1]->locidx, dir::f_p00);

      ng_p00.push_back(it_);
      ng_0p0.push_back(it_);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        quad_center(p4est_,ng_p00[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_p00[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_p00[i];
          qu123[2] = ng_p00[i+1];
          return true;
        }
      }

    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        quad_center(p4est_,ng_0p0[i]  , p2); p2 -= p0;
        quad_center(p4est_,ng_0p0[i+1], p3); p3 -= p0;
        if(test_tetrahedron(p1, p2, p3, px, uvw)){
          qu123[0] = it;
          qu123[1] = ng_0p0[i];
          qu123[2] = ng_0p0[i+1];
          return true;
        }
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_00p != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x == quad->x){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) == quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;

    }

    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y == quad->y){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<const quad_info_t*> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) == quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it);
        }

      std::vector<const quad_info_t*> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it);
        }

      if (test_edge_tetrahedrons(p4est_, p0, px, ng1, ng2, uvw, qu123))
        return true;
    }

    /* 5 - across mm0 */
    if (n_m00 != 0 && n_0m0 != 0)
    {
      quad_center(p4est_,begin_00p, p1); p1 -= p0; qu123[0] = begin_00p;

      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_0m0 != 0)
    {
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p2); p2 -= p0; qu123[1] = it;
          break;
        }

      quad_center(p4est_,end_0m0-1, p3); p3 -= p0; qu123[2] = end_0m0-1;

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 7 - across mp0 */
    if (n_m00 != 0 && n_0p0 != 0)
    {
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p1); p1 -= p0; qu123[0] = it;
          break;
        }

      quad_center(p4est_,end_m00-1, p2); p2 -= p0; qu123[1] = end_m00-1;

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          quad_center(p4est_,it, p3); p3 -= p0; qu123[2] = it;
          break;
        }

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      quad_center(p4est_,end_00p-1, p1); p1 -= p0; qu123[0] = end_00p-1;
      quad_center(p4est_,end_p00-1, p2); p2 -= p0; qu123[1] = end_p00-1;
      quad_center(p4est_,end_0p0-1, p3); p3 -= p0; qu123[2] = end_0p0-1;

      if (test_tetrahedron(p1, p2, p3, px, uvw))
        return true;
    }
  }

  return false;
}
