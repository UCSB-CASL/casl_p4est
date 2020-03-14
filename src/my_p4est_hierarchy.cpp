#ifdef P4_TO_P8
#include "my_p8est_hierarchy.h"
#include <p8est_communication.h>
#include <src/point3.h>
#else
#include "my_p4est_hierarchy.h"
#include <p4est_communication.h>
#include <src/point2.h>
#endif
#include "petsc_compatibility.h"

#include <src/types.h>
#include <src/casl_math.h>

#include <stdexcept>
#include <sstream>
#include <petsclog.h>

// logging variable -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_hierarchy_t;
#ifdef CASL_LOG_TINY_EVENTS
#warning "Use of 'CASL_LOG_TINY_EVENTS' macro is discouraged but supported. Logging tiny sections of the code may produce unreliable results due to overhead."
extern PetscLogEvent log_my_p4est_hierarchy_t_find_smallest_quad;
#endif
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

void my_p4est_hierarchy_t::split( int tree_idx, int ind )
{
  trees[tree_idx][ind].child = trees[tree_idx].size();

  p4est_qcoord_t size = P4EST_QUADRANT_LEN(trees[tree_idx][ind].level) / 2;
#ifdef P4_TO_P8
  for (unsigned char k = 0; k < 2; ++k)
#endif
    for (unsigned char j = 0; j < 2; ++j)
      for (unsigned char i = 0; i < 2; ++i) {
        HierarchyCell child =
        {
          CELL_LEAF, NOT_A_P4EST_QUADRANT,    /* child, quad */
          trees[tree_idx][ind].imin + i*size, /* imin */
          trees[tree_idx][ind].jmin + j*size, /* jmin */
  #ifdef P4_TO_P8
          trees[tree_idx][ind].kmin + k*size, /* kmin (3D) only */
  #endif
          (int8_t) (trees[tree_idx][ind].level + 1),       /* level */
          REMOTE_OWNER                        /* owner's rank */
        };
        trees[tree_idx].push_back(child);
      }
}

int my_p4est_hierarchy_t::update_tree( int tree_idx, const p4est_quadrant_t *quad )
{
  int ind = 0;
  while( trees[tree_idx][ind].level != quad->level )
  {
    if(trees[tree_idx][ind].child == CELL_LEAF)
      split(tree_idx, ind);

    /* now the intermediate cell is split, select the correct child */
    p4est_qcoord_t size = P4EST_QUADRANT_LEN(trees[tree_idx][ind].level) / 2;
    bool i = ( quad->x >= trees[tree_idx][ind].imin + size );
    bool j = ( quad->y >= trees[tree_idx][ind].jmin + size );
#ifdef P4_TO_P8
    bool k = ( quad->z >= trees[tree_idx][ind].kmin + size );
#endif
    ind = trees[tree_idx][ind].child + SUMD(i, 2*j, 4*k);
  }
  return ind;
}

p4est_locidx_t my_p4est_hierarchy_t::quad_idx_of_quad(const p4est_quadrant_t* quad, const p4est_topidx_t& tree_idx) const
{
  int ind = 0;
  while( trees[tree_idx][ind].level != quad->level )
  {
    p4est_qcoord_t size = P4EST_QUADRANT_LEN(trees[tree_idx][ind].level) / 2;
    bool i = (quad->x >= trees[tree_idx][ind].imin + size);
    bool j = (quad->y >= trees[tree_idx][ind].jmin + size);
#ifdef P4_TO_P8
    bool k = (quad->z >= trees[tree_idx][ind].kmin + size);
#endif
    ind = trees[tree_idx][ind].child + SUMD(i, 2*j, 4*k);
  }
  P4EST_ASSERT(trees[tree_idx][ind].child == CELL_LEAF);
  P4EST_ASSERT(ANDD((quad->x==trees[tree_idx][ind].imin), (quad->y==trees[tree_idx][ind].jmin), (quad->z==trees[tree_idx][ind].kmin)));
  return trees[tree_idx][ind].quad;
}

void my_p4est_hierarchy_t::construct_tree() {

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_hierarchy_t, 0, 0, 0, 0); CHKERRXX(ierr);

  local_inner_quadrant.resize(0);
  local_layer_quadrant.resize(0);

  size_t mirror_idx = 0;
  const p4est_quadrant_t* mirror = NULL;
  if(ghost != NULL && mirror_idx < ghost->mirrors.elem_count)
    mirror = p4est_quadrant_array_index(&ghost->mirrors, mirror_idx++);
  /* loop on the local quadrants */
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);

    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      // mirrors and quadrant are stored using the same convention, parse both simultaneously for efficiency,
      // but do not use p4est_quadrant_is_equal_piggy(), since the p.piggy3 member is not filled for regular quadrants but only for ghosts and mirrors
      if(ghost != NULL && mirror != NULL && p4est_quadrant_is_equal(quad, mirror) && mirror->p.piggy3.which_tree == tree_idx)
      {
        local_layer_quadrant.push_back(local_and_tree_indices(q + tree->quadrants_offset, tree_idx));
        if(mirror_idx < ghost->mirrors.elem_count)
          mirror = p4est_quadrant_array_index(&ghost->mirrors, mirror_idx++);
      }
      else
        local_inner_quadrant.push_back(local_and_tree_indices(q + tree->quadrants_offset, tree_idx));
      int ind = update_tree(tree_idx, quad);

      /* the cell corresponding to the quadrant has been found, associate it to the quadrant */
      trees[tree_idx][ind].quad = tree->quadrants_offset + q;

      /* this is a local quadrant */
      trees[tree_idx][ind].owner_rank = p4est->mpirank;
    }
  }

  P4EST_ASSERT(ghost == NULL || mirror_idx == ghost->mirrors.elem_count);
  P4EST_ASSERT(local_inner_quadrant.size() + local_layer_quadrant.size() == (size_t) p4est->local_num_quadrants);

  /* loop on the ghosts
   * We do this by looping over ghosts from each processor separately
   */

  if (ghost != NULL)
    for (int r = 0; r < p4est->mpisize; r++)
    {
      /* for each processor loop over the portion that is ghosted on this processor */
      for (p4est_locidx_t g = ghost->proc_offsets[r]; g < ghost->proc_offsets[r+1]; ++g)
      {
        p4est_quadrant_t *quad = p4est_quadrant_array_index(&ghost->ghosts, g);
        int ind = update_tree(quad->p.piggy3.which_tree, quad);

        /* the cell corresponding to the quadrant has been found, associate it to the quadrant */
        trees[quad->p.piggy3.which_tree][ind].quad = p4est->local_num_quadrants + g;

        /* set the owner rank */
        trees[quad->p.piggy3.which_tree][ind].owner_rank = r;
      }
    }

  ierr = PetscLogEventEnd(log_my_p4est_hierarchy_t, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_hierarchy_t::update(p4est_t *p4est_, p4est_ghost_t *ghost_)
{
  p4est = p4est_;
  ghost = ghost_;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    periodic[dir] = is_periodic(p4est, dir);

  trees.clear();
  trees.resize(p4est->connectivity->num_trees);

  for (size_t tr = 0; tr<trees.size(); tr++)
  {
    HierarchyCell root =
    {
      CELL_LEAF, NOT_A_P4EST_QUADRANT, /* child, quad */
      DIM(0, 0, 0),                    /* imin, jmin, kmin  */
      0,                               /* level */
      REMOTE_OWNER                     /* owner's rank */
    };
    trees[tr].push_back(root);
  }
  construct_tree();
}

void my_p4est_hierarchy_t::write_vtk(const char* filename) const
{
  p4est_connectivity_t* connectivity = p4est->connectivity;

  /* filename */
  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.vtk", filename, p4est->mpirank);

  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Quadtree Mesh \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  size_t num_quads = 0;
  for (size_t i = 0; i<trees.size(); ++i){
    for (size_t j = 0; j<trees[i].size(); j++){
      if (trees[i][j].child == CELL_LEAF)
        num_quads++;
    }
  }

  fprintf(vtk, "POINTS %ld double \n", P4EST_CHILDREN*num_quads);

  for (size_t i = 0; i<trees.size(); ++i){
    p4est_topidx_t v_m = connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];
    p4est_topidx_t v_p = connectivity->tree_to_vertex[P4EST_CHILDREN*i + P4EST_CHILDREN-1];

    double tree_xmin = connectivity->vertices[3*v_m + 0];
    double tree_xmax = connectivity->vertices[3*v_p + 0];
    double tree_ymin = connectivity->vertices[3*v_m + 1];
    double tree_ymax = connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_m + 2];
    double tree_zmax = connectivity->vertices[3*v_p + 2];
#endif

    for (size_t j = 0; j<trees[i].size(); j++){
      const HierarchyCell& cell = trees[i][j];
      if (cell.child == CELL_LEAF){
        double h = (double) P4EST_QUADRANT_LEN(cell.level) / (double)P4EST_ROOT_LEN;
#ifdef P4_TO_P8
        for (unsigned char xk = 0; xk < 2; xk++)
#endif
          for (unsigned char xj = 0; xj < 2; xj++)
            for (unsigned char xi = 0; xi < 2; xi++){
              double x = (tree_xmax - tree_xmin)*((double) cell.imin / (double)P4EST_ROOT_LEN + xi*h) + tree_xmin;
              double y = (tree_ymax - tree_ymin)*((double) cell.jmin / (double)P4EST_ROOT_LEN + xj*h) + tree_ymin;
#ifdef P4_TO_P8
              double z = (tree_zmax - tree_zmin)*((double) cell.kmin / (double)P4EST_ROOT_LEN + xk*h) + tree_zmin;
              fprintf(vtk, "%lf %lf %lf\n", x, y, z);
#else
              fprintf(vtk, "%lf %lf 0.0\n", x, y);
#endif
            }
      }
    }
  }

  fprintf(vtk, "CELLS %ld %ld \n", num_quads, (1+P4EST_CHILDREN)*num_quads);
  for (size_t i = 0; i < num_quads; ++i)
  {
    fprintf(vtk, "%d ", P4EST_CHILDREN);
    for (short j = 0; j < P4EST_CHILDREN; ++j)
      fprintf(vtk, "%ld ", P4EST_CHILDREN*i+j);
    fprintf(vtk,"\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_quads);
  for (size_t i = 0; i<num_quads; ++i)
    fprintf(vtk, "%d\n",P4EST_VTK_CELL_TYPE);
  fclose(vtk);
}

int my_p4est_hierarchy_t::find_smallest_quadrant_containing_point(const double *xyz, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_hierarchy_t_find_smallest_quad, 0, 0, 0, 0); CHKERRXX(ierr);
#endif

  /* rescale xyz to [0,nx] */
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1];

  double xmin = p4est->connectivity->vertices[3*v_m + 0];
  double xmax = p4est->connectivity->vertices[3*v_p + 0];
  double ymin = p4est->connectivity->vertices[3*v_m + 1];
  double ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*v_m + 2];
  double zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

  // we copy the given domain coordinates, first
  double xyz_[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    xyz_[dir] = xyz[dir];

  // we wrap them within the domain using periodicity if needed
  if ((xyz_[0] < xmin || xyz_[0] > xmax) && periodic[0]) xyz_[0] = xyz_[0] - floor((xyz_[0] - xmin)/(xmax - xmin))*(xmax - xmin);
  if ((xyz_[1] < ymin || xyz_[1] > ymax) && periodic[1]) xyz_[1] = xyz_[1] - floor((xyz_[1] - ymin)/(ymax - ymin))*(ymax - ymin);
#ifdef P4_TO_P8
  if ((xyz_[2] < zmin || xyz_[2] > zmax) && periodic[2]) xyz_[2] = xyz_[2] - floor((xyz_[2] - zmin)/(zmax - zmin))*(zmax - zmin);
#endif
  // at this point, xyz_ MUST be in the computational domain --> critical check in DEBUG
  P4EST_ASSERT(xmin <= xyz_[0] && xyz_[0] <= xmax);
  P4EST_ASSERT(ymin <= xyz_[1] && xyz_[1] <= ymax);
#ifdef P4_TO_P8
  P4EST_ASSERT(zmin <= xyz_[2] && xyz_[2] <= zmax);
#endif

  v_p   = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  xmax  = p4est->connectivity->vertices[3*v_p + 0];
  ymax  = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  zmax  = p4est->connectivity->vertices[3*v_p + 2];
#endif

  /* In order to use the standard vectors of HierarchyCell's, i.e. trees[tree_idx], as
   * constructed by this object, we need to rescale these coordinates to
   * [0, nx]x[0, ny]x[0xny] where nx, ny and nz are the numbers of trees in the brick,
   * i.e. the numbers of trees along the cartesian directions in the brick
   */
  xyz_[0] = (xyz_[0] - xmin)/(xmax - xmin); P4EST_ASSERT(0.0 <= xyz_[0] && xyz_[0] <= myb->nxyztrees[0]);
  xyz_[1] = (xyz_[1] - ymin)/(ymax - ymin); P4EST_ASSERT(0.0 <= xyz_[1] && xyz_[1] <= myb->nxyztrees[1]);
#ifdef P4_TO_P8
  xyz_[2] = (xyz_[2] - zmin)/(zmax - zmin); P4EST_ASSERT(0.0 <= xyz_[2] && xyz_[2] <= myb->nxyztrees[2]);
#endif

  int rank = -1; // initialize the return value --> this is what is returned if the quadrant of interest is remote
  P4EST_QUADRANT_INIT(&best_match);

  /*
   * At this stage, an integer value for xyz_[i], say "xyz_[i] == nn", theoretically means that the point of interest
   * lies exactly on the border between two trees of cartesian index (nn-1) and nn along cartesian direction i (if
   * both of these trees exist). Therefore, in such a case, one needs to perturb xyz_[i] by a small amount in the
   * positive and negative directions, and search then for both of these perturbed points in the respective trees.
   * However, such a test as "xyz_[i] == nn" will practically never return true since we are considering floating
   * point values for xyz_[i]. We need a finite, domain-independent, floating-point threshold value to determine
   * whether or not we consider that the point of coordinates xyz lies on the limit of a tree. Let us call that value
   * 'thresh' and replace the above test "xyz_[i] == nn" by "fabs(xyz[i]-nn) < thresh". Consistently, the small
   * amount by which we perturb xyz_[i] should be +/-thres, in that case.
   * -----------------------
   * Let's define 'threshold'
   * -----------------------
   * The most important constraint is to not miss the quadrant of interest, so we need to make sure that
   * 2*thresh < (logical) length of the smallest possible quadrant in a p4est grid, divided by P4EST_ROOT_LEN
   *
   * The (logical) length of the smallest quadrant ever possible is
   *      P4EST_QUADRANT_LEN(P4EST_QMAXLEVEL) = P4EST_QUADRANT_LEN(P4EST_MAXLEVEL - 1)
   * so let's define qeps as
   */
  const static double qeps = (double)P4EST_QUADRANT_LEN(P4EST_MAXLEVEL) / (double) P4EST_ROOT_LEN;
  /* so that qeps is half the logical length of the smallest possible quadrant as allowed by p4est (divided by P4EST_ROOT_LEN,
   * i.e., scaled down to a measure such that the scaled logical length of a root cell is 1.0)
   * Therefore, the smallest absolute difference between logical coordinate(s) of relevant grid-related data,
   * divided by P4EST_ROOT_LEN is qeps (e.g., difference between coordinates of a vertex and coordinates of
   * the center of the smallest possible quadrant).
   * --> qeps is a strict upper bound for thresh
   *
   * Given that we have 52 bits to represent the mantissa with 64-bit double values, as opposed to 32-bit integers
   * for the p4est_qcoord_t data type, we also have the following strict minimum bound for thresh
   * 2^(log2(max(number of trees along a cartesian direction))-20)*qeps
   * (any value smaller than that would possibly result in a comparison test equivalent to "xyz_[i] == nn").
   * Assuming that we can safely set log2(max(number of trees along a cartesian direction)) = 10,
   * this gives thresh > 0.001*qeps so I suggest to define thresh as
   */
  const static double  threshold  = 0.01*(double)P4EST_QUADRANT_LEN(P4EST_MAXLEVEL); // ==thresh*P4EST_ROOT_LEN

  /* In case of nonperiodic domain, we need to make sure that any point lying on the boundary of the domain is clearly
   * and unambiguously clipped inside, without changing the quadrant of interest, before we proceed further.
   * Otherwise, the routine will try to access a tree that does not exist...
   * Clearly and unambiguously clip the point inside the domain if not periodic
   */
  int tr_xyz_orig[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir){
    if (!periodic[dir])
      xyz_[dir] = MAX(qeps, MIN(xyz_[dir], myb->nxyztrees[dir] - qeps));
    tr_xyz_orig[dir] = (int)floor(xyz_[dir]);
  }
  double ii = (xyz_[0] - tr_xyz_orig[0]) * P4EST_ROOT_LEN;
  double jj = (xyz_[1] - tr_xyz_orig[1]) * P4EST_ROOT_LEN;
#ifdef P4_TO_P8
  double kk = (xyz_[2] - tr_xyz_orig[2]) * P4EST_ROOT_LEN;
#endif

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    if(periodic[dir])
      tr_xyz_orig[dir] = tr_xyz_orig[dir]%myb->nxyztrees[dir];

  const bool is_on_face_x = (fabs(ii - floor(ii)) < threshold || fabs(ceil(ii) - ii) < threshold);
  const bool is_on_face_y = (fabs(jj - floor(jj)) < threshold || fabs(ceil(jj) - jj) < threshold);
#ifdef P4_TO_P8
  const bool is_on_face_z = (fabs(kk - floor(kk)) < threshold || fabs(ceil(kk) - kk) < threshold);
#endif

  for (char i = (is_on_face_x ? -1 : 0); i < 2; i += 2)
    for (char j = (is_on_face_y ? -1 : 0); j < 2; j += 2)
#ifdef P4_TO_P8
      for (char k = (is_on_face_z ? -1 : 0);  k < 2; k += 2)
#endif
      {
        // perturb the point (note that i, j and/or k are 0 is no perturbation is required)
        PointDIM s(DIM((i == 0 ? ii : (ii + ((double) i)*threshold)), (j == 0 ? jj : (jj + ((double) j)*threshold)), (k == 0 ? kk : (kk + ((double) k)*threshold))));
        find_quadrant_containing_point(tr_xyz_orig, s, rank, best_match, remote_matches);
      }

#ifdef CASL_LOG_TINY_EVENTS
  ierr = PetscLogEventEnd(log_my_p4est_hierarchy_t_find_smallest_quad, 0, 0, 0, 0); CHKERRXX(ierr);
#endif

  return rank;
}

void my_p4est_hierarchy_t::find_quadrant_containing_point(const int* tr_xyz_orig, PointDIM& s, int& rank, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const
{
  const static p4est_qcoord_t qh = P4EST_QUADRANT_LEN(P4EST_QMAXLEVEL);

  int tr_xyz[P4EST_DIM] = {DIM(tr_xyz_orig[0], tr_xyz_orig[1], tr_xyz_orig[2])};

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if (s.xyz(dir) < 0 || s.xyz(dir)  > (double) P4EST_ROOT_LEN){
      const int ntree_to_slide = (int) ceil(s.xyz(dir)/((double) P4EST_ROOT_LEN)) - 1;
      P4EST_ASSERT((s.xyz(dir) < 0 && ntree_to_slide < 0) || (s.xyz(dir) > (double) P4EST_ROOT_LEN && ntree_to_slide > 0));
      s.xyz(dir) -= ((double) ntree_to_slide)*((double) P4EST_ROOT_LEN); // convert both terms to double BEFORE multiplying to avoid overflow in integer representation!
      tr_xyz[dir] = tr_xyz_orig[dir] + ntree_to_slide;
      if(periodic[dir])
        tr_xyz[dir] = mod(tr_xyz[dir], myb->nxyztrees[dir]);
    }
    P4EST_ASSERT(0 <= tr_xyz[dir] && tr_xyz[dir] < myb->nxyztrees[dir] && 0.0 <= s.xyz(dir) && s.xyz(dir) <= (double) P4EST_ROOT_LEN);
  }

  p4est_topidx_t tt = myb->nxyz_to_treeid[SUMD(tr_xyz[0], tr_xyz[1]*myb->nxyztrees[0], tr_xyz[2]*myb->nxyztrees[0]*myb->nxyztrees[1])];
  P4EST_ASSERT(0 <= tt && tt < p4est->connectivity->num_trees);

  const std::vector<HierarchyCell>& h_tr = trees[tt];
  const HierarchyCell *it, *begin; begin = it = &h_tr[0];
  while(CELL_LEAF != it->child){
    p4est_qcoord_t half_h = P4EST_QUADRANT_LEN(it->level) / 2;
    short cj = ((double)(it->jmin + half_h)) <= s.y;
    short ci = ((double)(it->imin + half_h)) <= s.x;
#ifdef P4_TO_P8
    short ck = ((double)(it->kmin + half_h)) <= s.z;
#endif
    it = begin + it->child + SUMD(ci, 2*cj, 4*ck);
  }

  if (it->owner_rank == p4est->mpirank) { // local quadrant
    p4est_tree_t *p4est_tr = p4est_tree_array_index(p4est->trees, tt);
    p4est_locidx_t pos = it->quad - p4est_tr->quadrants_offset;
    p4est_quadrant_t *tmp = p4est_quadrant_array_index(&p4est_tr->quadrants, pos);
    if (tmp->level >= best_match.level) { // note the >= --> ensures that we pick the local quadrant if we find one!
      best_match = *tmp;
      best_match.p.piggy3.which_tree = tt;
      best_match.p.piggy3.local_num  = pos;
      rank = it->owner_rank;
    }
  } else if (it->owner_rank != REMOTE_OWNER) { // ghost quadrant
    p4est_locidx_t pos = it->quad - p4est->local_num_quadrants;
    p4est_quadrant_t *tmp = p4est_quadrant_array_index(&ghost->ghosts, pos);
    if (tmp->level > best_match.level) {
      best_match = *tmp;
      best_match.p.piggy3.which_tree = tt;
      best_match.p.piggy3.local_num  = pos;
      rank = it->owner_rank;
    }
  } else { // remote quadrant
#ifdef CASL_THROWS
    if (it->quad != NOT_A_P4EST_QUADRANT)
      throw std::runtime_error("[ERROR]:my_p4est_hierarchy_t::find_quadrant_containing_point: a quadrant was marked both remote and not remote!");
#endif
    p4est_quadrant_t sq;
    P4EST_QUADRANT_INIT(&sq);
    sq.level = P4EST_QMAXLEVEL;

    sq.p.piggy1.which_tree = tt;

    /* need to find the owner
     * ensure that quadrant is a multiple of qh, otherwise p4est function will freak out!
     */
    sq.x = (p4est_qcoord_t)(s.x) & ~(qh - 1); // this operation nullifies the last bit and ensures in the p4est_qcoord_t value, hence ensures divisibility by qh
    sq.y = (p4est_qcoord_t)(s.y) & ~(qh - 1);
#ifdef P4_TO_P8
    sq.z = (p4est_qcoord_t)(s.z) & ~(qh - 1);
#endif
    sq.p.piggy1.owner_rank = p4est_comm_find_owner(p4est, tt, &sq, p4est->mpirank);

    remote_matches.push_back(sq);
  }
}
