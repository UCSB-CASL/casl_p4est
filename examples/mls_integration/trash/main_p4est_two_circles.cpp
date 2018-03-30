// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
//#include <src/cube3.h>
//#include <src/point3.h>
#include <src/my_p8est_mls_integration.h>
#include <src/my_p8est_mls_integration_old.h>
#include <src/integration_via_delta3.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
//#include <src/cube2.h>
//#include <src/point2.h>
#include <src/my_p4est_mls_integration.h>
//#include <src/my_p4est_mls_integration_old.h>
//#include <src/integration_via_delta2.h>
#endif

#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

/* discretization */
int lmin = 3;
int lmax = 3;
int nb_splits = 4;

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

bool save_vtk = true;

/* geometry */

double xmin = -1.0;
double xmax =  1.0;
double ymin = -1.0;
double ymax =  1.0;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

double r0 = 0.5;
double d = 0.2;

double theta = 0.123;
#ifdef P4_TO_P8
double phy = 0.321;
#endif

double cosT = cos(theta);
double sinT = sin(theta);
#ifdef P4_TO_P8
double cosP = cos(phy);
double sinP = sin(phy);
#endif

#ifdef P4_TO_P8
double xc_0 = -d*sinT*cosP; double yc_0 =  d*cosT*cosP; double zc_0 =  d*sinP;
double xc_1 =  d*sinT*cosP; double yc_1 = -d*cosT*cosP; double zc_1 = -d*sinP;
#else
double xc_0 = -d*sinT; double yc_0 =  d*cosT;
double xc_1 =  d*sinT; double yc_1 = -d*cosT;
#endif

#ifdef P4_TO_P8
class LS_CIRCLE_0: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)));
  }
} ls_circle_0;
#else
class LS_CIRCLE_0: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0)));
  }
} ls_circle_0;
#endif


#ifdef P4_TO_P8
class LS_CIRCLE_1: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r0 - sqrt(SQR(x-xc_1) + SQR(y-yc_1) + SQR(z-zc_1)));
  }
} ls_circle_1;
#else
class LS_CIRCLE_1: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r0 - sqrt(SQR(x-xc_1) + SQR(y-yc_1)));
  }
} ls_circle_1;
#endif

#ifdef P4_TO_P8
class FUNC: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return 1.0;
  }
} func;
#else
class FUNC: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 1.0;
  }
} func;
#endif

#ifdef P4_TO_P8
class FUNC_R2: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return x*x+y*y+z*z;
  }
} func_r2;
#else
class FUNC_R2: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return x*x+y*y;
  }
} func_r2;
#endif

class Exact {
public:
  double ID;
  double IB;
  double IDr2;
  double IBr2;
  vector<double> ISB, ISBr2;
  vector<double> IXr2;
  vector<int> IXc0, IXc1;
  double alpha;

  Exact()
  {
#ifdef P4_TO_P8
    ID = 4.0/3.0*PI*r0*r0*r0;
    IB = 4.0*PI*r0*r0;
    IDr2 = 4.0/3.0*PI*r0*r0*r0*(1.5*0.4*r0*r0+d*d);
    IBr2 = 4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d);
#else
    // Auxiliary values
    alpha = acos(d/r0);
    double r_bar_A = 2.0*r0*sin(alpha)/alpha/3.0;
    double r_bar_B = r0*sin(alpha)/alpha;

    /* the whole domain */
    ID = 2.0*(alpha*r0*r0-d*sqrt(r0*r0-d*d));
    IDr2 = alpha*r0*r0*r0*r0 +
           2.0*alpha*r0*r0*d*(d-2.0*r_bar_A) -
           d*r0*r0*sqrt(r0*r0-d*d)/3.0;
    /* the whole boundary */
    IB = 2.0*2.0*alpha*r0;
    IBr2 = 2.0*2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d);
    /* sub-boundaries */
    ISB.push_back(2.0*alpha*r0);
    ISBr2.push_back(2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    ISB.push_back(2.0*alpha*r0);
    ISBr2.push_back(2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    /* intersections */
    IXr2.push_back(2.0*(r0*r0-d*d));
    IXc0.push_back(0);
    IXc1.push_back(1);
#endif
  }
} exact;

double n_subs = 2;
double n_Xs = 1;

/* Vectors to store numerical results */
class Result
{
public:
  vector<double> ID, IB, IDr2, IBr2;
  vector< vector<double> > ISB, ISBr2, IXr2;
  Result()
  {
    for (int i = 0; i < n_subs; i++)
    {
      ISB.push_back(vector<double>());
      ISBr2.push_back(vector<double>());
    }
    for (int i = 0; i < n_Xs; i++)
    {
      IXr2.push_back(vector<double>());
    }
  }
};

Result res_one, res_mlt, res_dis;

class Geometry
{
public:
#ifdef P4_TO_P8
  vector<CF_3 *> LSF;
#else
  vector<CF_2 *> LSF;
#endif
  vector<Action> action;
  vector<int> color;
  Geometry()
  {
    LSF.push_back(&ls_circle_0); action.push_back(intersection); color.push_back(0);
    LSF.push_back(&ls_circle_1); action.push_back(intersection); color.push_back(1);
  }
} geometry;


vector<double> level, h;

void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi,
              int compt)
{
  PetscErrorCode ierr;
#ifdef STAMPEDE
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#else
  char out_dir[10000];
  sprintf(out_dir, "/home/dbochkov/Projects/build-p4est_integration-Desktop-Debug/out");
#endif

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/nodes_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz,
                                    xmin, xmax, ymin, ymax, zmin, zmax,
                                    &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny,
                                    xmin, xmax, ymin, ymax,
                                    &brick);
#endif

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &ls_circle_0, 1.2);
    p4est->user_pointer = (void*)(&data);

    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    /* functions to integrate */
    Vec f;
    ierr = VecCreateGhostNodes(p4est, nodes, &f); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, func, f);

    Vec f_r2;
    ierr = VecCreateGhostNodes(p4est, nodes, &f_r2); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, func_r2, f_r2);

    /* one level-set */
    my_p4est_level_set_t ls(&ngbd_n);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, ls_circle_0, phi);
    ls.perturb_level_set_function(phi, EPS);

    /* multi level-set */
    vector<Vec> phi_vec;

    for (int i = 0; i < geometry.LSF.size(); i++)
    {
      phi_vec.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_vec[i]); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *geometry.LSF[i], phi_vec[i]);
    }

    /* find dx and dy smallest */
    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
    double dx = (xmax-xmin) / pow(2.,(double) data.max_lvl);
    double dy = (ymax-ymin) / pow(2.,(double) data.max_lvl);

#ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
    double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
    double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double diag = sqrt(dx*dx + dy*dy);
#endif

    /* Calculate and store results */

    /* multi level-set */
#ifdef P4_TO_P8
    vector<MLS_Cube3_mlt> cubes;
#else
    vector<MLS_Cube2_mlt> cubes;
#endif

    construct_domain(p4est, nodes, phi_vec, geometry.action, geometry.color, cubes);

#ifdef P4_TO_P8
    vector<MLS_Simplex3 *> simplices;
#else
    vector<MLS_Simplex2 *> simplices;
#endif
    for (int k = 0; k < cubes.size(); k++)
    {
#ifdef P4_TO_P8
        for (int l = 0; l < 5; l++)
#else
        for (int l = 0; l < 2; l++)
#endif
          simplices.push_back(&cubes[k].simplex[l]);
    }

    write_simplex_geometry(simplices, OUTPUT_DIR, iter);

    level.push_back(lmax+iter);
    h.push_back(dx);

    res_one.ID.push_back(integrate_over_negative_domain(p4est, nodes, one, phi_vec, geometry.action, geometry.color, f));
    res_mlt.ID.push_back(integrate_over_negative_domain(p4est, nodes, mlt, phi_vec, geometry.action, geometry.color, f));
    res_dis.ID.push_back(integrate_over_negative_domain(p4est, nodes, dis, phi_vec, geometry.action, geometry.color, f));

    res_one.IDr2.push_back(integrate_over_negative_domain(p4est, nodes, one, phi_vec, geometry.action, geometry.color, f_r2));
    res_mlt.IDr2.push_back(integrate_over_negative_domain(p4est, nodes, mlt, phi_vec, geometry.action, geometry.color, f_r2));
    res_dis.IDr2.push_back(integrate_over_negative_domain(p4est, nodes, dis, phi_vec, geometry.action, geometry.color, f_r2));

    res_one.IB.push_back(integrate_over_interface(p4est, nodes, one, phi_vec, geometry.action, geometry.color, f, -1));
    res_mlt.IB.push_back(integrate_over_interface(p4est, nodes, mlt, phi_vec, geometry.action, geometry.color, f, -1));
    res_dis.IB.push_back(integrate_over_interface(p4est, nodes, dis, phi_vec, geometry.action, geometry.color, f, -1));

    res_one.IBr2.push_back(integrate_over_interface(p4est, nodes, one, phi_vec, geometry.action, geometry.color, f_r2, -1));
    res_mlt.IBr2.push_back(integrate_over_interface(p4est, nodes, mlt, phi_vec, geometry.action, geometry.color, f_r2, -1));
    res_dis.IBr2.push_back(integrate_over_interface(p4est, nodes, dis, phi_vec, geometry.action, geometry.color, f_r2, -1));

    for (int i = 0; i < n_subs; i++)
    {
      res_one.ISB[i].push_back(integrate_over_interface(p4est, nodes, one, phi_vec, geometry.action, geometry.color, f, geometry.color[i]));
      res_mlt.ISB[i].push_back(integrate_over_interface(p4est, nodes, mlt, phi_vec, geometry.action, geometry.color, f, geometry.color[i]));
      res_dis.ISB[i].push_back(integrate_over_interface(p4est, nodes, dis, phi_vec, geometry.action, geometry.color, f, geometry.color[i]));

      res_one.ISBr2[i].push_back(integrate_over_interface(p4est, nodes, one, phi_vec, geometry.action, geometry.color, f_r2, geometry.color[i]));
      res_mlt.ISBr2[i].push_back(integrate_over_interface(p4est, nodes, mlt, phi_vec, geometry.action, geometry.color, f_r2, geometry.color[i]));
      res_dis.ISBr2[i].push_back(integrate_over_interface(p4est, nodes, dis, phi_vec, geometry.action, geometry.color, f_r2, geometry.color[i]));
    }

    for (int i = 0; i < n_Xs; i++)
    {
      res_mlt.IXr2[i].push_back(integrate_over_intersection(p4est, nodes, mlt, phi_vec, geometry.action, geometry.color, f_r2, exact.IXc0[i], exact.IXc1[i]));
      res_dis.IXr2[i].push_back(integrate_over_intersection(p4est, nodes, dis, phi_vec, geometry.action, geometry.color, f_r2, exact.IXc0[i], exact.IXc1[i])*cos(0.5*PI-2.0*exact.alpha));
    }

//    if(save_vtk)
//    {
//      save_VTK(p4est, ghost, nodes, &brick, f_delta, iter);
//    }


    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(f); CHKERRXX(ierr);
    ierr = VecDestroy(f_r2); CHKERRXX(ierr);

    ierr = VecDestroy(phi_vec[0]); CHKERRXX(ierr);

    phi_vec.clear();

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  Gnuplot plot_ID;
  print_Table("Domain", exact.ID, level, h, "ONE", res_one.ID, 1, &plot_ID);
  print_Table("Domain", exact.ID, level, h, "MLT", res_mlt.ID, 2, &plot_ID);
  print_Table("Domain", exact.ID, level, h, "DIS", res_dis.ID, 3, &plot_ID);

  Gnuplot plot_IB;
  print_Table("Interface", exact.IB, level, h, "ONE", res_one.IB, 1, &plot_IB);
  print_Table("Interface", exact.IB, level, h, "MLT", res_mlt.IB, 2, &plot_IB);
  print_Table("Interface", exact.IB, level, h, "DIS", res_dis.IB, 3, &plot_IB);

  Gnuplot plot_IDr2;
  print_Table("2nd moment of domain", exact.IDr2, level, h, "ONE", res_one.IDr2, 1, &plot_IDr2);
  print_Table("2nd moment of domain", exact.IDr2, level, h, "MLT", res_mlt.IDr2, 2, &plot_IDr2);
  print_Table("2nd moment of domain", exact.IDr2, level, h, "DIS", res_dis.IDr2, 3, &plot_IDr2);

  Gnuplot plot_IBr2;
  print_Table("2nd moment of interface", exact.IBr2, level, h, "ONE", res_one.IBr2, 1, &plot_IBr2);
  print_Table("2nd moment of interface", exact.IBr2, level, h, "MLT", res_mlt.IBr2, 2, &plot_IBr2);
  print_Table("2nd moment of interface", exact.IBr2, level, h, "DIS", res_dis.IBr2, 3, &plot_IBr2);

  vector<Gnuplot *> plot_ISB;
  vector<Gnuplot *> plot_ISBr2;
  for (int i = 0; i < n_subs; i++)
  {
    plot_ISB.push_back(new Gnuplot());
    print_Table("Interface #"+to_string(i), exact.ISB[i], level, h, "ONE", res_one.ISB[i], 1, plot_ISB[i]);
    print_Table("Interface #"+to_string(i), exact.ISB[i], level, h, "MLT", res_mlt.ISB[i], 2, plot_ISB[i]);
    print_Table("Interface #"+to_string(i), exact.ISB[i], level, h, "DIS", res_dis.ISB[i], 3, plot_ISB[i]);

    plot_ISBr2.push_back(new Gnuplot());
    print_Table("2nd moment of interface #"+to_string(i), exact.ISBr2[i], level, h, "ONE", res_one.ISBr2[i], 1, plot_ISBr2[i]);
    print_Table("2nd moment of interface #"+to_string(i), exact.ISBr2[i], level, h, "MLT", res_mlt.ISBr2[i], 2, plot_ISBr2[i]);
    print_Table("2nd moment of interface #"+to_string(i), exact.ISBr2[i], level, h, "DIS", res_dis.ISBr2[i], 3, plot_ISBr2[i]);
  }

  vector<Gnuplot *> plot_IXr2;
  for (int i = 0; i < n_Xs; i++)
  {
    plot_IXr2.push_back(new Gnuplot());
    print_Table("Intersection of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), exact.IXr2[i], level, h, "MLT", res_mlt.IXr2[i], 2, plot_IXr2[i]);
    print_Table("Intersection of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), exact.IXr2[i], level, h, "DIS", res_dis.IXr2[i], 3, plot_IXr2[i]);
  }


  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  std::cin.get();

  for (int i = 0; i < n_subs; i++)
  {
    delete plot_ISB[i];
    delete plot_ISBr2[i];
  }

  for (int i = 0; i < n_Xs; i++)
  {
    delete plot_IXr2[i];
  }

  return 0;
}
