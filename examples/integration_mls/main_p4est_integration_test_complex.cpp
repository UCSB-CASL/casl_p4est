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
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex3_mls_q_vtk.h>
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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex2_mls_q_vtk.h>
#endif

#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#include "geometry_one_circle.h"
#include "geometry_two_circles_union.h"
#include "geometry_two_circles_intersection.h"
#include "geometry_rose.h"
#include "geometry_two_circles_coloration.h"
#include "geometry_four_flowers.h"

//#include "geometry_four_flowers.cpp"

#undef MIN
#undef MAX

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

using namespace std;

//#define TEST_OH2_ERRORS_HYPOTHESIS

/* grid and discretization */
#ifdef P4_TO_P8
int lmin = 3;
int lmax = 3;
int nb_splits = 7;
int nb_splits_per_split = 10;
int nx_shifts = 1;
int ny_shifts = 1;
int nz_shifts = 1;
int num_shifts = nx_shifts*ny_shifts*nz_shifts;
#else
int lmin = 3;
int lmax = 3;
int nb_splits = 8;
int nb_splits_per_split = 4;
int nx_shifts = 10;
int ny_shifts = 10;
int num_shifts = nx_shifts*ny_shifts;
#endif

int num_resolutions = (nb_splits-1)*nb_splits_per_split + 1;
int num_iter_tot = num_resolutions*num_shifts;

bool reinitialize_level_set = 0;

const int n_xyz[] = {1, 1, 1};
const int periodic[] = {0, 0, 0};

const double p_xyz_min[] = {-1.0, -1.0, -1.0};
const double p_xyz_max[] = { 1.0,  1.0,  1.0};

bool save_vtk = 0;

bool check_for_curvature = 1;
bool integrate_one_cell = 0;

// function to integrate
int func_num = 0;

#ifdef P4_TO_P8
class func_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(func_num) {
      case 0: return 1;
      case 1: return x*x+y*y+z*z;
      case 2: return sin(x)*cos(y)*exp(z);
      case 3: return log(3.0+x*x-z*z)*(y-x);
    }
  }
} func;
#else
class func_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(func_num) {
      case 0: return 1;
      case 1: return x*x+y*y;
      case 2: return sin(x)*cos(y);
      case 3: return log(3.0+x*x-y*y)*(y-x);
    }
  }
} func;
#endif

/* Geometry:
 * 0 - union of two circles
 * 1 - intersection of two circles
 * 2 - coloration with two circles
 * 3 - four flower-shaped domains
 * 4 - rose-like domain
 * 5 - one circle
 */
int geometry_num = 0;

geometry_two_circles_union_t        geometry_two_circles_union;
geometry_two_circles_intersection_t geometry_two_circles_intersection;
geometry_two_circles_coloration_t   geometry_two_circles_coloration;
geometry_four_flowers_t             geometry_four_flowers;
geometry_rose_t                     geometry_rose;
geometry_one_circle_t               geometry_one_circle;


int num_of_domains;
#ifdef P4_TO_P8
vector<CF_3 *> *LSF;
#else
vector<CF_2 *> *LSF;
#endif
vector<action_t> *action;
vector<int> *color;


// exact values
class exact_t {
public:
  double ID;
  double IB;
  vector<double> ISB;
  vector<double> IX;
  vector<double> IX3;

  double n_subs;

  double n_Xs;
  vector<int> IXc0, IXc1;

  double n_X3s;
  vector<int> IX3c0, IX3c1, IX3c2;

  bool provided;
} exact;

// structure to keep results of integration
class result_t
{
public:
  vector<double> ID;              // integral  over domain
  vector<double> IB;              // integral  over the whole boundary
  vector< vector<double> > ISB;   // integrals over smooth boundary parts
  vector< vector<double> > IX;    // integrals over intersection (junctures) of 2 smooth boundary parts
  vector< vector<double> > IX3;   // integrals over intersection (junctures) of 3 smooth boundary parts (only 3D)

  result_t(int n_subs, int n_Xs, int n_X3s, int length)
  {
    initialize(n_subs, n_Xs, n_X3s, length);
  }

  void initialize(int n_subs, int n_Xs, int n_X3s, int length)
  {
    ID .resize(length, 0);
    IB .resize(length, 0);
    ISB.resize(n_subs, std::vector<double>(length, 0));
    IX .resize(n_Xs,   std::vector<double>(length, 0));
    IX3.resize(n_X3s,  std::vector<double>(length, 0));
  }
};

void set_parameters()
{
  // set geometry
  switch (geometry_num)
  {
    case 0:
      {
        LSF     = &geometry_two_circles_union.LSF;
        action  = &geometry_two_circles_union.action;
        color   = &geometry_two_circles_union.color;

        num_of_domains  = geometry_two_circles_union.n_subs;
        exact.n_subs    = geometry_two_circles_union.n_subs;
        exact.n_Xs      = geometry_two_circles_union.n_Xs;
        exact.IXc0      = geometry_two_circles_union.IXc0;
        exact.IXc1      = geometry_two_circles_union.IXc1;

        exact.n_X3s      = geometry_two_circles_union.n_X3s;
        exact.IX3c0      = geometry_two_circles_union.IX3c0;
        exact.IX3c1      = geometry_two_circles_union.IX3c1;
        exact.IX3c1      = geometry_two_circles_union.IX3c2;
      } break;
    case 1:
      {
        LSF     = &geometry_two_circles_intersection.LSF;
        action  = &geometry_two_circles_intersection.action;
        color   = &geometry_two_circles_intersection.color;

        num_of_domains  = geometry_two_circles_intersection.n_subs;
        exact.n_subs    = geometry_two_circles_intersection.n_subs;
        exact.n_Xs      = geometry_two_circles_intersection.n_Xs;
        exact.IXc0      = geometry_two_circles_intersection.IXc0;
        exact.IXc1      = geometry_two_circles_intersection.IXc1;

        exact.n_X3s      = geometry_two_circles_coloration.n_X3s;
        exact.IX3c0      = geometry_two_circles_coloration.IX3c0;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c1;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c2;
      } break;
    case 2:
      {
        LSF     = &geometry_two_circles_coloration.LSF;
        action  = &geometry_two_circles_coloration.action;
        color   = &geometry_two_circles_coloration.color;

        num_of_domains  = geometry_two_circles_coloration.n_subs;
        exact.n_subs    = geometry_two_circles_coloration.n_subs;
        exact.n_Xs      = geometry_two_circles_coloration.n_Xs;
        exact.IXc0      = geometry_two_circles_coloration.IXc0;
        exact.IXc1      = geometry_two_circles_coloration.IXc1;

        exact.n_X3s      = geometry_two_circles_coloration.n_X3s;
        exact.IX3c0      = geometry_two_circles_coloration.IX3c0;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c1;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c2;
      } break;
    case 3:
      {
        LSF     = &geometry_four_flowers.LSF;
        action  = &geometry_four_flowers.action;
        color   = &geometry_four_flowers.color;

        num_of_domains  = geometry_four_flowers.n_subs;
        exact.n_subs    = geometry_four_flowers.n_subs;
        exact.n_Xs      = geometry_four_flowers.n_Xs;
        exact.IXc0      = geometry_four_flowers.IXc0;
        exact.IXc1      = geometry_four_flowers.IXc1;

        exact.n_X3s      = geometry_four_flowers.n_X3s;
        exact.IX3c0      = geometry_four_flowers.IX3c0;
        exact.IX3c1      = geometry_four_flowers.IX3c1;
        exact.IX3c2      = geometry_four_flowers.IX3c2;
      } break;
    case 4:
      {
        LSF     = &geometry_rose.LSF;
        action  = &geometry_rose.action;
        color   = &geometry_rose.color;

        num_of_domains  = geometry_rose.n_subs;
        exact.n_subs    = geometry_rose.n_subs;
        exact.n_Xs      = geometry_rose.n_Xs;
        exact.IXc0      = geometry_rose.IXc0;
        exact.IXc1      = geometry_rose.IXc1;

        exact.n_X3s      = geometry_rose.n_X3s;
        exact.IX3c0      = geometry_rose.IX3c0;
        exact.IX3c1      = geometry_rose.IX3c1;
        exact.IX3c2      = geometry_rose.IX3c2;
      } break;
    case 5:
      {
        LSF     = &geometry_one_circle.LSF;
        action  = &geometry_one_circle.action;
        color   = &geometry_one_circle.color;

        num_of_domains  = geometry_one_circle.n_subs;
        exact.n_subs    = geometry_one_circle.n_subs;
        exact.n_Xs      = geometry_one_circle.n_Xs;
        exact.IXc0      = geometry_one_circle.IXc0;
        exact.IXc1      = geometry_one_circle.IXc1;

        exact.n_X3s      = geometry_one_circle.n_X3s;
        exact.IX3c0      = geometry_one_circle.IX3c0;
        exact.IX3c1      = geometry_one_circle.IX3c1;
        exact.IX3c2      = geometry_one_circle.IX3c2;
      } break;
  }

  exact.provided = false;

  // get exact values if available
  if (func_num == 0)
    switch (geometry_num)
    {
      case 0: {
          exact.ID  = geometry_two_circles_union.exact0.ID;
          exact.ISB = geometry_two_circles_union.exact0.ISB;
          exact.IX  = geometry_two_circles_union.exact0.IX;
          exact.provided = true;
        } break;
      case 1: {
          exact.ID  = geometry_two_circles_intersection.exact0.ID;
          exact.ISB = geometry_two_circles_intersection.exact0.ISB;
          exact.IX  = geometry_two_circles_intersection.exact0.IX;
          exact.provided = true;
        } break;
      case 2: {
          exact.ID  = geometry_two_circles_coloration.exact0.ID;
          exact.ISB = geometry_two_circles_coloration.exact0.ISB;
          exact.IX  = geometry_two_circles_coloration.exact0.IX;
          exact.provided = true;
        } break;
      case 5: {
          exact.ID  = geometry_one_circle.exact0.ID;
          exact.ISB = geometry_one_circle.exact0.ISB;
          exact.IX  = geometry_one_circle.exact0.IX;
          exact.provided = true;
        } break;
    }
  else if (func_num == 1)
    switch (geometry_num)
    {
      case 0: {
          exact.ID  = geometry_two_circles_union.exact1.ID;
          exact.ISB = geometry_two_circles_union.exact1.ISB;
          exact.IX  = geometry_two_circles_union.exact1.IX;
          exact.provided = true;
        } break;
      case 1: {
          exact.ID  = geometry_two_circles_intersection.exact1.ID;
          exact.ISB = geometry_two_circles_intersection.exact1.ISB;
          exact.IX  = geometry_two_circles_intersection.exact1.IX;
          exact.provided = true;
        } break;
      case 2: {
          exact.ID  = geometry_two_circles_coloration.exact1.ID;
          exact.ISB = geometry_two_circles_coloration.exact1.ISB;
          exact.IX  = geometry_two_circles_coloration.exact1.IX;
          exact.provided = true;
        } break;
      case 5: {
          exact.ID  = geometry_one_circle.exact1.ID;
          exact.ISB = geometry_one_circle.exact1.ISB;
          exact.IX  = geometry_one_circle.exact1.IX;
          exact.provided = true;
        } break;
    }
}

#ifdef P4_TO_P8
class level_set_tot_t : public CF_3
#else
class level_set_tot_t : public CF_2
#endif
{
#ifdef P4_TO_P8
  std::vector<CF_3 *>   *phi_cf;
#else
  std::vector<CF_2 *>   *phi_cf;
#endif
  std::vector<action_t> *action;
  std::vector<int>      *color;

public:

#ifdef P4_TO_P8
  level_set_tot_t(std::vector<CF_3 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color) :
#else
  level_set_tot_t(std::vector<CF_2 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color) :
#endif
    phi_cf(phi_cf), action(action), color(color) {}

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
#else
  double operator()(double x, double y) const
#endif
  {
    double phi_total = -10;
    double phi_current = -10;
    for (short i = 0; i < color->size(); ++i)
    {
      if (action->at(i) == INTERSECTION)
      {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current > phi_total) phi_total = phi_current;
      } else if (action->at(i) == ADDITION) {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current < phi_total) phi_total = phi_current;
      }
    }
    return phi_total;
  }
};

vector<double> level, h;

// forward declaration
void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              std::vector<Vec> phi, Vec phi_tot,
              int compt);

double compute_convergence_order(std::vector<double> &x, std::vector<double> &y);
void save_vector(const char *filename, const std::vector<double> &data, ios_base::openmode mode = ios_base::out, char delim = ',');
void print_convergence_table(MPI_Comm mpi_comm,
                             std::vector<double> &level, std::vector<double> &h,
                             std::vector<double> &L_one, std::vector<double> &L_avg, std::vector<double> &L_dev, std::vector<double> &L_max,
                             std::vector<double> &Q_one, std::vector<double> &Q_avg, std::vector<double> &Q_dev, std::vector<double> &Q_max);

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  parStopWatch w;
  w.start("total time");

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("nb_splits_per_split", "");
  cmd.add_option("nx_shifts", "");
  cmd.add_option("ny_shifts", "");
#ifdef P4_TO_P8
  cmd.add_option("nz_shifts", "");
#endif

  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("reinit", "reinitialize level-set function");
  cmd.add_option("func_num", "test function");
  cmd.add_option("geometry_num", "geometry_num");

  cmd.add_option("check_for_curvature", "check_for_curvature");
  cmd.add_option("integrate_one_cell", "integrate_one_cell");

  cmd.parse(argc, argv);

  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);
  nb_splits_per_split = cmd.get("nb_splits_per_split", nb_splits_per_split);
  nx_shifts = cmd.get("nx_shifts", nx_shifts);
  ny_shifts = cmd.get("ny_shifts", ny_shifts);
#ifdef P4_TO_P8
  nz_shifts = cmd.get("nz_shifts", nz_shifts);
#endif

  reinitialize_level_set = cmd.get("reinit", reinitialize_level_set);
  geometry_num = cmd.get("geometry_num", geometry_num);
  func_num = cmd.get("func_num", func_num);
  save_vtk = cmd.get("save_vtk", save_vtk);

  check_for_curvature = cmd.get("check_for_curvature", check_for_curvature);
  integrate_one_cell = cmd.get("integrate_one_cell", integrate_one_cell);

  set_parameters();

#ifdef P4_TO_P8
  num_shifts = nx_shifts*ny_shifts*nz_shifts;
#else
  num_shifts = nx_shifts*ny_shifts;
#endif

  num_resolutions = (nb_splits-1)*nb_splits_per_split + 1;
  num_iter_tot = num_resolutions*num_shifts;

  level_set_tot_t ls_tot(LSF, action, color);


  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;


  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  int file_num = 0;

  result_t result_L_all(exact.n_subs, exact.n_Xs, exact.n_X3s, num_iter_tot);
  result_t result_Q_all(exact.n_subs, exact.n_Xs, exact.n_X3s, num_iter_tot);

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d.\n", lmin+iter, lmax+iter); CHKERRXX(ierr);

    int num_sub_iter = (iter == 0 ? 1 : nb_splits_per_split);

    for (int sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter)
    {

      double p_xyz_min_alt[3];
      double p_xyz_max_alt[3];

      double scale = (double) (num_sub_iter-1-sub_iter) / (double) num_sub_iter;
      p_xyz_min_alt[0] = p_xyz_min[0] - .5*(pow(2.,scale)-1)*(p_xyz_max[0]-p_xyz_min[0]); p_xyz_max_alt[0] = p_xyz_max[0] + .5*(pow(2.,scale)-1)*(p_xyz_max[0]-p_xyz_min[0]);
      p_xyz_min_alt[1] = p_xyz_min[1] - .5*(pow(2.,scale)-1)*(p_xyz_max[1]-p_xyz_min[1]); p_xyz_max_alt[1] = p_xyz_max[1] + .5*(pow(2.,scale)-1)*(p_xyz_max[1]-p_xyz_min[1]);
      p_xyz_min_alt[2] = p_xyz_min[2] - .5*(pow(2.,scale)-1)*(p_xyz_max[2]-p_xyz_min[2]); p_xyz_max_alt[2] = p_xyz_max[2] + .5*(pow(2.,scale)-1)*(p_xyz_max[2]-p_xyz_min[2]);


      double dxyz[3] = { (p_xyz_max_alt[0]-p_xyz_min_alt[0])/pow(2., (double) lmax+iter),
                         (p_xyz_max_alt[1]-p_xyz_min_alt[1])/pow(2., (double) lmax+iter),
                         (p_xyz_max_alt[2]-p_xyz_min_alt[2])/pow(2., (double) lmax+iter) };

      double p_xyz_min_shift[3];
      double p_xyz_max_shift[3];

#ifdef P4_TO_P8
      h.push_back(MIN(dxyz[0],dxyz[1],dxyz[2]));
#else
      h.push_back(MIN(dxyz[0],dxyz[1]));
#endif

      level.push_back(lmax+iter-scale);
      ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f).\n", lmin+iter, lmax+iter, sub_iter, lmin+iter-scale, lmax+iter-scale); CHKERRXX(ierr);

#ifdef P4_TO_P8
      for (int k_shift = 0; k_shift < nz_shifts; ++k_shift)
      {
        p_xyz_min_shift[2] = p_xyz_min_alt[2] + (double) (k_shift) / (double) (nz_shifts) * dxyz[2];
        p_xyz_max_shift[2] = p_xyz_max_alt[2] + (double) (k_shift) / (double) (nz_shifts) * dxyz[2];
#endif
        for (int j_shift = 0; j_shift < ny_shifts; ++j_shift)
        {
          p_xyz_min_shift[1] = p_xyz_min_alt[1] + (double) (j_shift) / (double) (ny_shifts) * dxyz[1];
          p_xyz_max_shift[1] = p_xyz_max_alt[1] + (double) (j_shift) / (double) (ny_shifts) * dxyz[1];

          for (int i_shift = 0; i_shift < nx_shifts; ++i_shift)
          {
            p_xyz_min_shift[0] = p_xyz_min_alt[0] + (double) (i_shift) / (double) (nx_shifts) * dxyz[0];
            p_xyz_max_shift[0] = p_xyz_max_alt[0] + (double) (i_shift) / (double) (nx_shifts) * dxyz[0];

            connectivity = my_p4est_brick_new(n_xyz, p_xyz_min_shift, p_xyz_max_shift, &brick, periodic);

            p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

            // testing O(h^2) errors hypothesis
            double xyz_cell[P4EST_DIM];
#ifdef P4_TO_P8
            if (geometry_num == 5)
            {
              // sphere's radius
              double R = geometry_one_circle.domain0.phi.r0;

              // relative cap's base radius
              double a = 0.5;

              // relative margin
              double b = 2;

              // shift sphere to domain boundary
              geometry_one_circle.domain0.set_params(R, p_xyz_max_shift[0] - b*h.back() - sqrt(SQR(R) - SQR(a*h.back())), 0, 0);

              // center of required cube
              xyz_cell[0] = p_xyz_max_shift[0]-b*dxyz[0]+.5*dxyz[0];
              xyz_cell[1] = .5*dxyz[1];
              xyz_cell[2] = .5*dxyz[2];
            }
#endif

            splitting_criteria_cf_t data(0, lmax+iter, &ls_tot, 1.2);
            if (func_num != 0) data.min_lvl = lmin+iter;

            p4est->user_pointer = (void*)(&data);

//            my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
//            my_p4est_partition(p4est, P4EST_FALSE, NULL);
            for (int lvl = 0; lvl < data.max_lvl; ++lvl)
            {
              my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
              my_p4est_partition(p4est, P4EST_TRUE, NULL);
            }
//            p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
//            my_p4est_partition(p4est, P4EST_FALSE, NULL);

//            ghost = NULL;
            ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
//            my_p4est_ghost_expand(p4est, ghost);
            nodes = my_p4est_nodes_new(p4est, ghost);

            my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
            my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

            /* function to integrate */
            Vec func_vec;
            ierr = VecCreateGhostNodes(p4est, nodes, &func_vec); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, func, func_vec);

            Vec fdd[P4EST_DIM];

            for (short dir = 0; dir < P4EST_DIM; ++dir)
            {
              ierr = VecCreateGhostNodes(p4est, nodes, &fdd[dir]); CHKERRXX(ierr);
            }

            ngbd_n.second_derivatives_central(func_vec, fdd);


            my_p4est_level_set_t ls(&ngbd_n);

            /* level-set functions */
            vector<Vec> phi_vec, phi_xx_vec, phi_yy_vec;
#ifdef P4_TO_P8
            vector<Vec> phi_zz_vec;
#endif

            for (int i = 0; i < num_of_domains; i++)
            {
              phi_vec.push_back(Vec());     ierr = VecCreateGhostNodes(p4est, nodes, &phi_vec[i]); CHKERRXX(ierr);
              phi_xx_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_vec[i]); CHKERRXX(ierr);
              phi_yy_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_vec[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
              phi_zz_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz_vec[i]); CHKERRXX(ierr);
#endif

              sample_cf_on_nodes(p4est, nodes, *LSF->at(i), phi_vec[i]);

              if (reinitialize_level_set)
                ls.reinitialize_1st_order_time_2nd_order_space(phi_vec.back());

#ifdef P4_TO_P8
              ngbd_n.second_derivatives_central(phi_vec[i], phi_xx_vec[i], phi_yy_vec[i], phi_zz_vec[i]);
#else
              ngbd_n.second_derivatives_central(phi_vec[i], phi_xx_vec[i], phi_yy_vec[i]);
#endif
            }

            Vec phi_tot;
            ierr = VecCreateGhostNodes(p4est, nodes, &phi_tot); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, ls_tot, phi_tot);

            my_p4est_integration_mls_t integration_L(p4est, nodes);

            //    integration.set_phi(phi_vec, geometry.action, geometry.color);
            //    integration.set_phi(geometry.LSF, geometry.action, geometry.color);
            //#ifdef P4_TO_P8
            //    integration.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, phi_zz_vec, *action, *color);
            //#else
            //    integration.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, *action, *color);
            //#endif
//#ifdef P4_TO_P8
//            integration_L.set_phi(phi_vec, *action, *color);
//#else
//            integration_L.set_phi(phi_vec, *action, *color);
//#endif
            //    integration.set_use_cube_refined(0);

            my_p4est_integration_mls_t integration_Q(p4est, nodes);

//#ifdef P4_TO_P8
//            integration_Q.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, phi_zz_vec, *action, *color);
//#else
//            integration_Q.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, *action, *color);
//#endif

            integration_L.set_phi(*LSF, *action, *color, 1);
            integration_Q.set_phi(*LSF, *action, *color, 0);

#ifdef P4_TO_P8
            integration_Q.check_for_curvature = check_for_curvature;
#endif

            char *out_dir;
            out_dir = getenv("OUT_DIR");
            std::ostringstream oss;

            oss << out_dir
                << "/geometry";

            std::ostringstream command;
            command << "mkdir -p " << out_dir << "/geometry";
            int ret_sys = system(command.str().c_str());
            if (ret_sys<0)
              throw std::invalid_argument("could not create directory");

//            if (0)
            if (save_vtk)
            {
              integration_L.initialize();
#ifdef P4_TO_P8
              vector<simplex3_mls_l_t *> simplices;
              int n_sps = NTETS;
#else
              vector<simplex2_mls_l_t *> simplices;
              int n_sps = 2;
#endif

              for (int k = 0; k < integration_L.cubes.size(); k++)
                for (int kk = 0; kk < integration_L.cubes[k].cubes_l_.size(); kk++)
                  if (integration_L.cubes[k].cubes_l_[kk]->loc == FCE)
                    for (int l = 0; l < n_sps; l++)
                      simplices.push_back(&integration_L.cubes[k].cubes_l_[kk]->simplex[l]);

#ifdef P4_TO_P8
              simplex3_mls_l_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_num));
#else
              simplex2_mls_l_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_num));
#endif
              save_VTK(p4est, ghost, nodes, &brick, phi_vec, phi_tot, file_num);
            }

            if (save_vtk)
            {
              integration_Q.initialize();
#ifdef P4_TO_P8
              vector<simplex3_mls_q_t *> simplices;
              int n_sps = NUM_TETS;
#else
              vector<simplex2_mls_q_t *> simplices;
              int n_sps = 2;
#endif

              for (int k = 0; k < integration_Q.cubes.size(); k++)
                for (int kk = 0; kk < integration_Q.cubes[k].cubes_q_.size(); kk++)
                  if (integration_Q.cubes[k].cubes_q_[kk]->loc == FCE)
                    for (int l = 0; l < n_sps; l++)
                      simplices.push_back(&integration_Q.cubes[k].cubes_q_[kk]->simplex[l]);

#ifdef P4_TO_P8
              simplex3_mls_q_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_num));
#else
              simplex2_mls_q_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_num));
#endif
              PetscPrintf(p4est->mpicomm, "VTK saved %d\n", file_num);

            }

            double *xyz_c = NULL;

#ifdef P4_TO_P8
            if (integrate_one_cell) xyz_c = xyz_cell;
#endif

            /* Calculate and store results */
            result_L_all.ID[file_num] = integration_L.integrate_over_domain(func_vec, NULL, xyz_c);
            result_Q_all.ID[file_num] = integration_Q.integrate_over_domain(func_vec, fdd, xyz_c);

            for (int i = 0; i < exact.n_subs; i++)
            {
              result_L_all.ISB[i][file_num] = integration_L.integrate_over_interface(color->at(i), func_vec, NULL, xyz_c);
              result_Q_all.ISB[i][file_num] = integration_Q.integrate_over_interface(color->at(i), func_vec, fdd, xyz_c);
            }

            for (int i = 0; i < exact.n_Xs; i++)
            {
              result_L_all.IX[i][file_num] = integration_L.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec, NULL, xyz_c);
              result_Q_all.IX[i][file_num] = integration_Q.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec, fdd, xyz_c);
            }

            ierr = VecDestroy(func_vec); CHKERRXX(ierr);
            ierr = VecDestroy(phi_tot); CHKERRXX(ierr);

            for (short dir = 0; dir < P4EST_DIM; ++dir)
            {
              ierr = VecDestroy(fdd[dir]); CHKERRXX(ierr);
            }

            for (int i = 0; i < phi_vec.size(); i++)
            {
              ierr = VecDestroy(phi_vec[i]); CHKERRXX(ierr);
              ierr = VecDestroy(phi_xx_vec[i]); CHKERRXX(ierr);
              ierr = VecDestroy(phi_yy_vec[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
              ierr = VecDestroy(phi_zz_vec[i]); CHKERRXX(ierr);
#endif
            }
            phi_vec.clear();

            p4est_nodes_destroy(nodes);
            p4est_ghost_destroy(ghost);
            p4est_destroy      (p4est);

            my_p4est_brick_destroy(connectivity, &brick);

            ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f). Case no. %6d/%d is finished.\n", lmin+iter, lmax+iter, sub_iter, lmin+iter-scale, lmax+iter-scale, file_num, num_iter_tot); CHKERRXX(ierr);
            ++file_num;

          }
        }
#ifdef P4_TO_P8
      }
#endif
    }
  }

  /* post-process results */

  // if exact values are not provided use the most refined grid for reference
  if (!exact.provided)
  {
    exact.ID = 0;

    for (int i = 0; i < exact.n_subs; i++)
      exact.ISB.push_back(0);

    for (int i = 0; i < exact.n_Xs; i++)
      exact.IX.push_back(0);

#ifdef P4_TO_P8
    for (int i = 0; i < exact.n_X3s; i++)
      exact.IX3.push_back(0);
#endif

    for (int s = 0; s < num_shifts; ++s)
    {
      int p = (num_resolutions-1)*num_shifts + s;

      exact.ID += result_Q_all.ID[p];

      for (int i = 0; i < exact.n_subs; i++)
        exact.ISB[i] += result_Q_all.ISB[i][p];

      for (int i = 0; i < exact.n_Xs; i++)
        exact.IX[i] += result_Q_all.IX[i][p];

#ifdef P4_TO_P8
      for (int i = 0; i < exact.n_X3s; i++)
        exact.IX3[i] += result_Q_all.IX3[i][p];
#endif
    }

    exact.ID /= num_shifts;

    for (int i = 0; i < exact.n_subs; i++)
      exact.ISB[i] /= num_shifts;

    for (int i = 0; i < exact.n_Xs; i++)
      exact.IX[i] /= num_shifts;

#ifdef P4_TO_P8
    for (int i = 0; i < exact.n_X3s; i++)
      exact.IX3[i] /= num_shifts;
#endif

    num_resolutions -= 1;
    num_iter_tot -= num_shifts;

    level.resize(num_resolutions);
    h.resize(num_resolutions);

    result_L_all.ID.resize(num_iter_tot);
    result_Q_all.ID.resize(num_iter_tot);

    for (int i = 0; i < exact.n_subs; i++)
    {
      result_L_all.ISB[i].resize(num_iter_tot);
      result_Q_all.ISB[i].resize(num_iter_tot);
    }

    for (int i = 0; i < exact.n_Xs; i++)
    {
      result_L_all.IX[i].resize(num_iter_tot);
      result_Q_all.IX[i].resize(num_iter_tot);
    }
  }

  // compute errors in all cases
//  for (int p = 0; p < num_iter_tot; ++p)
//  {
  for (int r = 0; r < num_resolutions; ++r)
  {

    // testing O(h^2) errors hypothesis
    if (integrate_one_cell)
    {
      // sphere's radius
      double R = geometry_one_circle.domain0.phi.r0;

      // relative cap's base radius
      double a = 0.5;

      double H = (R-sqrt(SQR(R) - SQR(a*h[r])));
      exact.ID = .25*PI*H*H*(R-H/3.);
      exact.ISB[0] = .5*PI*R*H;

//      exact.ID     *= 4.;
//      exact.ISB[0] *= 4.;
    }

    for (int s = 0; s < num_shifts; ++s)
    {
      int p = r*num_shifts+s;
      result_L_all.ID[p] = fabs(result_L_all.ID[p]-exact.ID);
      result_Q_all.ID[p] = fabs(result_Q_all.ID[p]-exact.ID);

      for (int i = 0; i < exact.n_subs; i++)
      {
        result_L_all.ISB[i][p] = fabs(result_L_all.ISB[i][p] - exact.ISB[i]);
        result_Q_all.ISB[i][p] = fabs(result_Q_all.ISB[i][p] - exact.ISB[i]);
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
        result_L_all.IX[i][p] = fabs(result_L_all.IX[i][p] - exact.IX[i]);
        result_Q_all.IX[i][p] = fabs(result_Q_all.IX[i][p] - exact.IX[i]);
      }
    }
  }

  result_t result_L_max(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);
  result_t result_L_avg(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);
  result_t result_L_dev(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);
  result_t result_L_one(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);

  result_t result_Q_max(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);
  result_t result_Q_avg(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);
  result_t result_Q_dev(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);
  result_t result_Q_one(exact.n_subs, exact.n_Xs, exact.n_X3s, num_resolutions);

  // for each resolution compute max, mean and deviation
  for (int p = 0; p < num_resolutions; ++p)
  {
    // one
    result_L_one.ID[p] = result_L_all.ID[p*num_shifts];
    result_Q_one.ID[p] = result_Q_all.ID[p*num_shifts];

    for (int i = 0; i < exact.n_subs; i++)
    {
      result_L_one.ISB[i][p] = result_L_all.ISB[i][p*num_shifts];
      result_Q_one.ISB[i][p] = result_Q_all.ISB[i][p*num_shifts];
    }

    for (int i = 0; i < exact.n_Xs; i++)
    {
      result_L_one.IX[i][p] = result_L_all.IX[i][p*num_shifts];
      result_Q_one.IX[i][p] = result_Q_all.IX[i][p*num_shifts];
    }

    // max
    for (int s = 0; s < num_shifts; ++s)
    {
      result_L_max.ID[p] = MAX(result_L_max.ID[p], result_L_all.ID[p*num_shifts + s]);
      result_Q_max.ID[p] = MAX(result_Q_max.ID[p], result_Q_all.ID[p*num_shifts + s]);

      for (int i = 0; i < exact.n_subs; i++)
      {
        result_L_max.ISB[i][p] = MAX(result_L_max.ISB[i][p], result_L_all.ISB[i][p*num_shifts + s]);
        result_Q_max.ISB[i][p] = MAX(result_Q_max.ISB[i][p], result_Q_all.ISB[i][p*num_shifts + s]);
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
        result_L_max.IX[i][p] = MAX(result_L_max.IX[i][p], result_L_all.IX[i][p*num_shifts + s]);
        result_Q_max.IX[i][p] = MAX(result_Q_max.IX[i][p], result_Q_all.IX[i][p*num_shifts + s]);
      }
    }

    // avg
    for (int s = 0; s < num_shifts; ++s)
    {
      result_L_avg.ID[p] += result_L_all.ID[p*num_shifts + s];
      result_Q_avg.ID[p] += result_Q_all.ID[p*num_shifts + s];

      for (int i = 0; i < exact.n_subs; i++)
      {
        result_L_avg.ISB[i][p] += result_L_all.ISB[i][p*num_shifts + s];
        result_Q_avg.ISB[i][p] += result_Q_all.ISB[i][p*num_shifts + s];
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
        result_L_avg.IX[i][p] += result_L_all.IX[i][p*num_shifts + s];
        result_Q_avg.IX[i][p] += result_Q_all.IX[i][p*num_shifts + s];
      }
    }

    result_L_avg.ID[p] /= num_shifts;
    result_Q_avg.ID[p] /= num_shifts;

    for (int i = 0; i < exact.n_subs; i++)
    {
      result_L_avg.ISB[i][p] /= num_shifts;
      result_Q_avg.ISB[i][p] /= num_shifts;
    }

    for (int i = 0; i < exact.n_Xs; i++)
    {
      result_L_avg.IX[i][p] /= num_shifts;
      result_Q_avg.IX[i][p] /= num_shifts;
    }

    // deviation
    if (num_shifts != 1)
    {
      for (int s = 0; s < num_shifts; ++s)
      {
        // avg
        result_L_dev.ID[p] += pow(result_L_all.ID[p*num_shifts + s] - result_L_avg.ID[p], 2.);
        result_Q_dev.ID[p] += pow(result_Q_all.ID[p*num_shifts + s] - result_Q_avg.ID[p], 2.);

        for (int i = 0; i < exact.n_subs; i++)
        {
          result_L_dev.ISB[i][p] += pow(result_L_all.ISB[i][p*num_shifts + s] - result_L_avg.ISB[i][p], 2.);
          result_Q_dev.ISB[i][p] += pow(result_Q_all.ISB[i][p*num_shifts + s] - result_Q_avg.ISB[i][p], 2.);
        }

        for (int i = 0; i < exact.n_Xs; i++)
        {
          result_L_dev.IX[i][p] += pow(result_L_all.IX[i][p*num_shifts + s] - result_L_avg.IX[i][p], 2.);
          result_Q_dev.IX[i][p] += pow(result_Q_all.IX[i][p*num_shifts + s] - result_Q_avg.IX[i][p], 2.);
        }
      }

      result_L_dev.ID[p] = sqrt(result_L_dev.ID[p]/(num_shifts-1));
      result_Q_dev.ID[p] = sqrt(result_Q_dev.ID[p]/(num_shifts-1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        result_L_dev.ISB[i][p] = sqrt(result_L_dev.ISB[i][p]/(num_shifts-1));
        result_Q_dev.ISB[i][p] = sqrt(result_Q_dev.ISB[i][p]/(num_shifts-1));
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
        result_L_dev.IX[i][p] = sqrt(result_L_dev.IX[i][p]/(num_shifts-1));
        result_Q_dev.IX[i][p] = sqrt(result_Q_dev.IX[i][p]/(num_shifts-1));
      }
    }
  }

  // print tables



  ierr = PetscPrintf(mpi.comm(), "\nDomain Integral\n"); CHKERRXX(ierr);
  print_convergence_table(mpi.comm(),
                          level, h,
                          result_L_one.ID, result_L_avg.ID, result_L_dev.ID, result_L_max.ID,
                          result_Q_one.ID, result_Q_avg.ID, result_Q_dev.ID, result_Q_max.ID);

  for (int i = 0; i < exact.n_subs; ++i)
  {
    ierr = PetscPrintf(mpi.comm(), "Boundary Integral no. %d\n", i); CHKERRXX(ierr);
    print_convergence_table(mpi.comm(),
                            level, h,
                            result_L_one.ISB[i], result_L_avg.ISB[i], result_L_dev.ISB[i], result_L_max.ISB[i],
                            result_Q_one.ISB[i], result_Q_avg.ISB[i], result_Q_dev.ISB[i], result_Q_max.ISB[i]);
  }

  for (int i = 0; i < exact.n_Xs; ++i)
  {
    ierr = PetscPrintf(mpi.comm(), "Intersection Integral no. %d\n", i); CHKERRXX(ierr);
    print_convergence_table(mpi.comm(),
                            level, h,
                            result_L_one.IX[i], result_L_avg.IX[i], result_L_dev.IX[i], result_L_max.IX[i],
                            result_Q_one.IX[i], result_Q_avg.IX[i], result_Q_dev.IX[i], result_Q_max.IX[i]);
  }

  if (mpi.rank() == 0)
  {
    char *out_dir;
    out_dir = getenv("OUT_DIR");
    std::ostringstream oss_dir;
    std::string filename;

    oss_dir << out_dir
            << "/convergence";

    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create directory");

    // save level and resolution
    filename = out_dir; filename += "/convergence/lvl.txt";      save_vector(filename.c_str(), level);
    filename = out_dir; filename += "/convergence/h.txt";        save_vector(filename.c_str(), h);

    // domain integral
    filename = out_dir; filename += "/convergence/l_all_id.txt"; save_vector(filename.c_str(), result_L_all.ID);
    filename = out_dir; filename += "/convergence/q_all_id.txt"; save_vector(filename.c_str(), result_Q_all.ID);
    filename = out_dir; filename += "/convergence/l_max_id.txt"; save_vector(filename.c_str(), result_L_max.ID);
    filename = out_dir; filename += "/convergence/q_max_id.txt"; save_vector(filename.c_str(), result_Q_max.ID);
    filename = out_dir; filename += "/convergence/l_avg_id.txt"; save_vector(filename.c_str(), result_L_avg.ID);
    filename = out_dir; filename += "/convergence/q_avg_id.txt"; save_vector(filename.c_str(), result_Q_avg.ID);
    filename = out_dir; filename += "/convergence/l_one_id.txt"; save_vector(filename.c_str(), result_L_one.ID);
    filename = out_dir; filename += "/convergence/q_one_id.txt"; save_vector(filename.c_str(), result_Q_one.ID);
    filename = out_dir; filename += "/convergence/l_dev_id.txt"; save_vector(filename.c_str(), result_L_dev.ID);
    filename = out_dir; filename += "/convergence/q_dev_id.txt"; save_vector(filename.c_str(), result_Q_dev.ID);

    // boundary integrals
    for (int i = 0; i < exact.n_subs; i++)
    {
      ios_base::openmode mode = (i == 0 ? ios_base::out : ios_base::app);
      filename = out_dir; filename += "/convergence/l_all_isb.txt"; save_vector(filename.c_str(), result_L_all.ISB[i], mode);
      filename = out_dir; filename += "/convergence/q_all_isb.txt"; save_vector(filename.c_str(), result_Q_all.ISB[i], mode);
      filename = out_dir; filename += "/convergence/l_max_isb.txt"; save_vector(filename.c_str(), result_L_max.ISB[i], mode);
      filename = out_dir; filename += "/convergence/q_max_isb.txt"; save_vector(filename.c_str(), result_Q_max.ISB[i], mode);
      filename = out_dir; filename += "/convergence/l_avg_isb.txt"; save_vector(filename.c_str(), result_L_avg.ISB[i], mode);
      filename = out_dir; filename += "/convergence/q_avg_isb.txt"; save_vector(filename.c_str(), result_Q_avg.ISB[i], mode);
      filename = out_dir; filename += "/convergence/l_one_isb.txt"; save_vector(filename.c_str(), result_L_one.ISB[i], mode);
      filename = out_dir; filename += "/convergence/q_one_isb.txt"; save_vector(filename.c_str(), result_Q_one.ISB[i], mode);
      filename = out_dir; filename += "/convergence/l_dev_isb.txt"; save_vector(filename.c_str(), result_L_dev.ISB[i], mode);
      filename = out_dir; filename += "/convergence/q_dev_isb.txt"; save_vector(filename.c_str(), result_Q_dev.ISB[i], mode);
    }

    // intersection integrals
    for (int i = 0; i < exact.n_Xs; i++)
    {
      ios_base::openmode mode = (i == 0 ? ios_base::out : ios_base::app);
      filename = out_dir; filename += "/convergence/l_all_ix.txt"; save_vector(filename.c_str(), result_L_all.IX[i], mode);
      filename = out_dir; filename += "/convergence/q_all_ix.txt"; save_vector(filename.c_str(), result_Q_all.IX[i], mode);
      filename = out_dir; filename += "/convergence/l_max_ix.txt"; save_vector(filename.c_str(), result_L_max.IX[i], mode);
      filename = out_dir; filename += "/convergence/q_max_ix.txt"; save_vector(filename.c_str(), result_Q_max.IX[i], mode);
      filename = out_dir; filename += "/convergence/l_avg_ix.txt"; save_vector(filename.c_str(), result_L_avg.IX[i], mode);
      filename = out_dir; filename += "/convergence/q_avg_ix.txt"; save_vector(filename.c_str(), result_Q_avg.IX[i], mode);
      filename = out_dir; filename += "/convergence/l_one_ix.txt"; save_vector(filename.c_str(), result_L_one.IX[i], mode);
      filename = out_dir; filename += "/convergence/q_one_ix.txt"; save_vector(filename.c_str(), result_Q_one.IX[i], mode);
      filename = out_dir; filename += "/convergence/l_dev_ix.txt"; save_vector(filename.c_str(), result_L_dev.IX[i], mode);
      filename = out_dir; filename += "/convergence/q_dev_ix.txt"; save_vector(filename.c_str(), result_Q_dev.IX[i], mode);
    }

  }



  /// boundary integrals
  /// integrals over intersections

  // save convergence data


  w.stop(); w.read_duration();
  // make a plot
  int plot_color = 1;
  if (mpi.rank() == -1)
  {
    // plot convergence results for a quick check
    Gnuplot plot;
    print_Table("Convergence", 0, level, h, "Domain", result_L_max.ID, 1, &plot);
    plot_color++;

    for (int i = 0; i < exact.n_subs; i++)
    {
      print_Table("Convergence", 0, level, h, "Sub-boundary #"+to_string(i), result_L_max.ISB[i], plot_color, &plot);
      plot_color++;
    }

    for (int i = 0; i < exact.n_Xs; i++)
    {
      print_Table("Convergence", 0, level, h, "X of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), result_L_max.IX[i], plot_color, &plot);
      plot_color++;
    }


#ifdef P4_TO_P8
    for (int i = 0; i < exact.n_X3s; i++)
    {
      print_Table("Convergence", 0, level, h, "X of #"+to_string(exact.IX3c0[i])+" and #"+to_string(exact.IX3c1[i])+" and #"+to_string(exact.IX3c2[i]), result_L_max.IX3[i], plot_color, &plot);
      plot_color++;
    }
#endif

//    plot_color = 1;
    print_Table("Convergence", 0, level, h, "Domain", result_Q_max.ID, plot_color, &plot);
    plot_color++;

    for (int i = 0; i < exact.n_subs; i++)
    {
      print_Table("Convergence", 0, level, h, "Sub-boundary #"+to_string(i), result_Q_max.ISB[i], plot_color, &plot);
      plot_color++;
    }

//    for (int i = 0; i < result_quadratic.ISB[0].size(); ++i)
//      result_quadratic.ISB[0][i] += result_quadratic.ISB[1][i];

//    print_Table("Convergence", exact.ISB[0]+exact.ISB[1], level, h, "Total Boundary", result_quadratic.ISB[0], plot_color, &plot);
//    plot_color++;

    for (int i = 0; i < exact.n_Xs; i++)
    {
      print_Table("Convergence", 0, level, h, "X of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), result_Q_max.IX[i], plot_color, &plot);
      plot_color++;
    }


#ifdef P4_TO_P8
    for (int i = 0; i < exact.n_X3s; i++)
    {
      print_Table("Convergence", 0, level, h, "X of #"+to_string(exact.IX3c0[i])+" and #"+to_string(exact.IX3c1[i])+" and #"+to_string(exact.IX3c2[i]), result_Q_max.IX3[i], plot_color, &plot);
      plot_color++;
    }
#endif
    std::cin.get();
  }

  return 0;
}

void print_convergence_table(MPI_Comm mpi_comm,
                             std::vector<double> &level, std::vector<double> &h,
                             std::vector<double> &L_one, std::vector<double> &L_avg, std::vector<double> &L_dev, std::vector<double> &L_max,
                             std::vector<double> &Q_one, std::vector<double> &Q_avg, std::vector<double> &Q_dev, std::vector<double> &Q_max)
{
  PetscErrorCode ierr;
  double order;

  ierr = PetscPrintf(mpi_comm, "\n"); CHKERRXX(ierr);

  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "                    |  Linear Integration                                                                     |  Quadratic Integration                                                                \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "                    |  Average                    |  One                        |  Max                        |  Average                    |  One                        |  Max                      \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "lvl  | Resolution   |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);


  for (int i = 0; i < num_resolutions; ++i)
  {
    // lvl and h
    ierr = PetscPrintf(mpi_comm, "%.2f | %.5e", level[i], h[i]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);


    /* linear integration */
    // avg
    if (i == 0) order = compute_convergence_order(h, L_avg);
    else        order = log(L_avg[i-1]/L_avg[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", L_avg[i], 100.*L_dev[i]/L_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // one
    if (i == 0) order = compute_convergence_order(h, L_one);
    else        order = log(L_one[i-1]/L_one[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", L_one[i], 100.*fabs(L_one[i]-L_avg[i])/L_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // max
    if (i == 0) order = compute_convergence_order(h, L_max);
    else        order = log(L_max[i-1]/L_max[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", L_max[i], 100.*fabs(L_max[i]-L_avg[i])/L_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    /* quadratic integration */
    //avg
    if (i == 0) order = compute_convergence_order(h, Q_avg);
    else        order = log(Q_avg[i-1]/Q_avg[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", Q_avg[i], 100.*Q_dev[i]/Q_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // one
    if (i == 0) order = compute_convergence_order(h, Q_one);
    else        order = log(Q_one[i-1]/Q_one[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", Q_one[i], 100.*fabs(Q_one[i]-Q_avg[i])/Q_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // max
    if (i == 0) order = compute_convergence_order(h, Q_max);
    else        order = log(Q_max[i-1]/Q_max[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", Q_max[i], 100.*fabs(Q_max[i]-Q_avg[i])/Q_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "\n"); CHKERRXX(ierr);
  }
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);

  ierr = PetscPrintf(mpi_comm, "\n"); CHKERRXX(ierr);
}

double compute_convergence_order(std::vector<double> &x, std::vector<double> &y)
{
  if (x.size() != y.size())
  {
    std::cout << "[ERROR]: sizes of arrays do not coincide\n";
    return 0;
  }

  int n = x.size();

  double sum_x  = 0;
  double sum_y  = 0;
  double sum_xy = 0;
  double sum_xx = 0;

  for (int i = 0; i < n; ++i)
  {
    double log_x = log(x[i]);
    double log_y = log(y[i]);

    sum_x  += log_x;
    sum_y  += log_y;
    sum_xy += log_x*log_y;
    sum_xx += log_x*log_x;
  }

  return (sum_xy - sum_x*sum_y/n)/(sum_xx - sum_x*sum_x/n);
}

void save_vector(const char *filename, const std::vector<double> &data, ios_base::openmode mode, char delim)
{
  ofstream ofs;
  ofs.open(filename, mode);

  for (int i = 0; i < data.size(); ++i)
  {
    if (i != 0) ofs << delim;
    ofs << data[i];
  }

  ofs << "\n";
}



void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              std::vector<Vec> phi, Vec phi_tot,
              int compt)
{
  PetscErrorCode ierr;
  char *out_dir;
  out_dir = getenv("OUT_DIR");

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

  std::ostringstream command;
  command << "mkdir -p " << out_dir << "/vtu";
  int ret_sys = system(command.str().c_str());
  if (ret_sys<0)
    throw std::invalid_argument("could not create directory");

  double *phi_p;
  std::vector<double *> point_data(phi.size(), NULL);
  std::vector<std::string> point_data_names;

  for (int i = 0; i < phi.size(); ++i)
  {
    ierr = VecGetArray(phi[i], &point_data[i]); CHKERRXX(ierr);
    point_data_names.push_back("phi"+to_string(i));
  }

  point_data.push_back(NULL);
  ierr = VecGetArray(phi_tot, &point_data.back()); CHKERRXX(ierr);
  point_data_names.push_back("phi_tot");

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

  std::vector<double *> cell_data;
  std::vector<std::string> cell_data_names;

  cell_data.push_back(l_p);
  cell_data_names.push_back("leaf_level");

  my_p4est_vtk_write_all_lists(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               oss.str().c_str(),
                               point_data, point_data_names,
                               cell_data,  cell_data_names);

//  my_p4est_vtk_write_all(p4est, nodes, ghost,
//                         P4EST_TRUE, P4EST_TRUE,
//                         1, 1, oss.str().c_str(),
//                         VTK_POINT_DATA, "phi", phi_p,
//                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  for (int i = 0; i < phi.size(); ++i)
  {
    ierr = VecRestoreArray(phi[i], &point_data[i]); CHKERRXX(ierr);
  }

  ierr = VecGetArray(phi_tot, &point_data.back()); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}
