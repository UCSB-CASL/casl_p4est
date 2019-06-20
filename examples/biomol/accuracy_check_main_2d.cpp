// System
#include <stdexcept>
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <boost/algorithm/string.hpp>

// p4est
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_jump_nodes_extended.h>
#include <src/my_p4est_level_set.h>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/math.h>

#include <src/my_p4est_biomolecules.h>

using namespace std;

template<typename T> inline T my_conversion(const string&);
template<> inline double my_conversion<double>(const string& st){return atof(st.c_str());}
template<> inline string my_conversion<string>(const string& st){return st;}

template <typename T>
vector<T>* read_from_list(const string& list_of_values, const int n_values = -1)
{
  int size          = list_of_values.size();
  int n_comma       = 0;
  for (int k = 0; k < size; ++k) {
    if(list_of_values.at(k) == ',')
      n_comma++;
  }
  vector<T>* values = new vector<T>;
  if(n_values >= 0)
    values->resize(n_values);
  else
    values->resize(n_comma + 1);
#ifdef CASL_THROWS
  if(n_comma != values->size()-1)
    throw invalid_argument("read_from_list::invalid list of values. " + to_string(values->size()) +" values required, separated by commas.");
#endif
  int m, j = m = 0;
  for (int k = 0; k < size; ++k) {
    if(list_of_values.at(k) == ',')
    {
      if(k-j >0)
        values->at(m++) = my_conversion<T>(list_of_values.substr(j, k-j));
#ifdef CASL_THROWS
      else
        throw invalid_argument("read_from_list::invalid syntax in list of values. Two consecutive commas found.");
#endif
      j = k+1;
    }
  }
#ifdef CASL_THROWS
  if(j == size)
    throw invalid_argument("read_from_list::invalid syntax in list of values. Trailing ',' found: a value is missing.");
#endif

  if(j < size)
  {
    if (list_of_values.at(size-1) == '.')
    {
      if (j < size -1)
        values->at(m++) = my_conversion<T>(list_of_values.substr(j, size-1-j));
#ifdef CASL_THROWS
      else
        throw invalid_argument("read_from_list::invalid syntax in list of values. Trailing ',.' found: a value is missing.");
#endif
    }
    else
      values->at(m++) = my_conversion<T>(list_of_values.substr(j, size-j));
  }
#ifdef CASL_THROWS
  if(m != values->size())
    throw invalid_argument("read_from_list::not enough values read when initializing the vector of values.");
#endif
  return values;
}

void get_norm_of_errors(double& error_on_phi_infty_norm, double& error_on_phi_one_norm,
                        p4est_t*& reference_p4est, p4est_nodes_t*& reference_nodes, const Vec& reference_phi,
                        const my_p4est_node_neighbors_t& ngbd, const Vec& approx_phi)
{
  my_p4est_interpolation_nodes_t interp(&ngbd);
  interp.set_input(approx_phi, linear);
  double xyz[P4EST_DIM];
  for (size_t k = 0; k < reference_nodes->indep_nodes.elem_count; ++k) {
    node_xyz_fr_n(k, reference_p4est, reference_nodes, xyz);
    interp.add_point(k, xyz);
  }
  Vec abs_approx_phi_on_reference_grid;
  PetscErrorCode ierr = VecDuplicate(reference_phi, &abs_approx_phi_on_reference_grid); CHKERRXX(ierr);
  interp.interpolate(abs_approx_phi_on_reference_grid);
  double *abs_approx_phi_on_reference_grid_p = NULL;

  ierr = VecGetArray(abs_approx_phi_on_reference_grid, &abs_approx_phi_on_reference_grid_p); CHKERRXX(ierr);
  for (size_t k = 0; k < reference_nodes->indep_nodes.elem_count; ++k)
    abs_approx_phi_on_reference_grid_p[k] = fabs(abs_approx_phi_on_reference_grid_p[k]);
  ierr = VecRestoreArray(abs_approx_phi_on_reference_grid, &abs_approx_phi_on_reference_grid_p); abs_approx_phi_on_reference_grid_p = NULL; CHKERRXX(ierr);

  error_on_phi_infty_norm = max_over_interface(reference_p4est, reference_nodes, reference_phi, abs_approx_phi_on_reference_grid);
  error_on_phi_one_norm   = integrate_over_interface(reference_p4est, reference_nodes, reference_phi, abs_approx_phi_on_reference_grid);

  ierr = VecDestroy(abs_approx_phi_on_reference_grid); abs_approx_phi_on_reference_grid = NULL; CHKERRXX(ierr);
}

int main(int argc, char *argv[]) {
  try{
    PetscErrorCode ierr;
    mpi_environment_t mpi;
    mpi.init(argc, argv);

    cmdParser cmd;
    // List of parameters
    // molecule(s)
    cmd.add_option("input-dir",     "folder in which the pqr files are located");
    cmd.add_option("pqr",           "name(s) of the pqr file(s) (syntax: '-pqr name0,name1,name2...)");
#ifdef P4_TO_P8
    cmd.add_option("centroid",      "centroid(s) coordinates of the molecule(s), domain [0, 1]X[0, 1]X[0, 1] (syntax: '-pqr x0,y0,z0,x1,y1,z1,x2,y2,z2,...)");
    cmd.add_option("rotangle",      "rotation angle(s) of the molecule(s) (syntax: '-pqr angle00,angle01,angle02,angle10,angle11,angle12,angle20,angle21,angle22,...)");
#else
    cmd.add_option("centroid",      "centroid(s) coordinates of the molecule(s), domain [0, 1]X[0, 1] (syntax: '-pqr x0,y0,x1,y1,x2,y2,...)");
    cmd.add_option("rotangle",      "rotation angle(s) of the molecule(s) (syntax: '-pqr angle0,angle1,angle2,...)");
#endif
    cmd.add_option("boxsize",       "relative side length of the biggest bounding box to the min domain dimension (0 < boxsize < 1)");
    // grid construction and problem accuracy
    cmd.add_option("ntree_dim",     "number of tree per dimension (ntree^3 trees in total, ntree >=1)");
    cmd.add_option("lmax",          "the max level of the tree for the accurate reference solution (> lmin+2)");
    cmd.add_option("lmin",          "the min level for the accuracy analaysis (< lmax -2)");
    cmd.add_option("lip",           "Lipchitz constant for the levelset (>= 1.0)");
    cmd.add_option("rp",            "probe radius (in Angstrom, >= 0.0)");
    // exportation of results and/or timing
    cmd.add_option("output-dir",    "folder to save the results in");
    cmd.add_option("logfile",       "log file");
    cmd.add_option("err-log",       "error log file");

    cmd.parse(argc, argv);

    // read the options and/or set default parameters
    /* default comparison strings */
    string nullstr = "NULL";
    string stdout_str = "stdout";
    string stderr_str = "stderr";
    // molecule(s)
    const string input_folder                 = cmd.get<string> ("input-dir", "/home/egan/workspace/projects/biomol/mols");
    const string pqr_input                    = cmd.get<string>("pqr", "1a2k.");
    const vector<string>* pqr = NULL;
    if(!boost::iequals(nullstr,pqr_input))
      pqr                                     = read_from_list<string>(pqr_input);
#ifdef P4_TO_P8
    const string list_of_centroids            = cmd.get<string>("centroid", "0.5,0.5,0.5");
    const string list_of_angles               = cmd.get<string>("rotangle", "0.0,0.0,0.0");
#else
    const string list_of_centroids            = cmd.get<string>("centroid", "0.5,0.5");
    const string list_of_angles               = cmd.get<string>("rotangle", "0.0");
#endif
    vector<double>* centroids = NULL;
    vector<double>* angles = NULL;
    if(!boost::iequals(nullstr, list_of_centroids))
      centroids                               = read_from_list<double>(list_of_centroids);
    if(!boost::iequals(nullstr, list_of_angles))
      angles                                  = read_from_list<double>(list_of_angles);

    const double rel_side_length_biggest_box  = cmd.get<double>("boxsize", 0.5);
    // grid construction
    const int ntree_per_dim                   = cmd.get<int>("ntree_dim", 1);
    const int lmin                            = cmd.get<int>("lmin", 8);
    const int lmax                            = cmd.get<int>("lmax", 8);
    const double lip                          = cmd.get<double>("lip", 1.2);
    const double probe_radius                 = cmd.get<double>("rp", 1.4);


    // exportation folder and files
    string output_folder                      = cmd.get<string>("output-dir", "/home/egan/workspace/projects/biomol/output/accuracy/two_spheres");
    /* create the exportation folder if it does not exist yet */
    mkdir(output_folder.c_str(), 0755);
    output_folder += ((output_folder[output_folder.size()-1] == '/')?"":"/");
    /* open/create the log file if needed */
    const string log_file                     = cmd.get<string>("logfile", "stdout");
    my_p4est_biomolecules_t::log_file = NULL;
    if(!boost::iequals(nullstr, log_file))
    {
      if(boost::iequals(stdout_str, log_file))
        my_p4est_biomolecules_t::log_file = stdout;
      else
      {
        string log_file_path = output_folder + log_file;
        my_p4est_biomolecules_t::log_file = fopen(log_file_path.c_str(), "w");
      }
    }
    my_p4est_biomolecules_t::timing_file = stdout /*NULL*/;
    const string errlog_file              = cmd.get<string>("err-log", "stderr");
    my_p4est_biomolecules_t::error_file   = NULL;
    if(!boost::iequals(nullstr, errlog_file))
    {
      if(boost::iequals(stderr_str, errlog_file))
        my_p4est_biomolecules_t::error_file = stderr;
      else
      {
        string error_log_path = output_folder + errlog_file;
        my_p4est_biomolecules_t::error_file = fopen(error_log_path.c_str(), "w");
      }
    }

    // sanity checks
#ifdef CASL_THROWS
    if (ntree_per_dim < 1)
      throw invalid_argument("Invalid number of tree per dimension (option ntree_dim), must be greater than or equal to 1");
#endif

    // create the macro mesh connectivity
    my_p4est_brick_t brick;
    int n_xyz []      = {ntree_per_dim, ntree_per_dim, ntree_per_dim};
    double xyz_min [] = {0, 0, 0};
    double xyz_max [] = {domain_side_length, domain_side_length, domain_side_length};
    int periodic []   = {0, 0, 0};
    p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

    vector<double> errors_1st_order_infty(lmax-lmin+1, 0.0);
    vector<double> errors_1st_order_one_norm(lmax-lmin+1, 0.0);
    vector<double> errors_2nd_order_infty(lmax-lmin+1, 0.0);
    vector<double> errors_2nd_order_one_norm(lmax-lmin+1, 0.0);
    for (int l = lmin; l < lmax+1; ++l) {
      /* create the reference p4est, nodes and ghosts */
      splitting_criteria_t sp(0, l, lip);
//      p4est_t *reference_p4est = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
//      reference_p4est->user_pointer = (void *) &sp;
//      my_p4est_biomolecules_t reference_biomol(reference_p4est, &mpi, pqr, &input_folder, angles, centroids, &rel_side_length_biggest_box);
//      reference_biomol.set_grid_and_surface_parameters(0, l, lip, probe_radius, 2);
//      reference_p4est                 = reference_biomol.construct_SES(list_reduction_with_exact_phi, false, false, "null");
//      Vec reference_phi               = reference_biomol.return_phi_vector();
//      p4est_nodes_t* reference_nodes  = reference_biomol.return_nodes();
//      p4est_ghost_t* reference_ghost  = reference_biomol.return_ghost();
//      {
//        double *reference_phi_p = NULL;
//        ierr = VecGetArray(reference_phi, &reference_phi_p); CHKERRXX(ierr);
//        string vtk_file = output_folder + "/reference_level_" +to_string(l);
//        my_p4est_vtk_write_all(reference_p4est, reference_nodes, reference_ghost,
//                               P4EST_TRUE, P4EST_TRUE,
//                               1, 0, vtk_file.c_str(),
//                               VTK_POINT_DATA, "phi", reference_phi_p);
//        ierr = VecRestoreArray(reference_phi, &reference_phi_p); reference_phi_p = NULL; CHKERRXX(ierr);
//      }

      /* create the 1st order method p4est, nodes and ghosts */
      p4est_t *p4est = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
      p4est->user_pointer = (void *) &sp;
      my_p4est_biomolecules_t biomol_ooa1(p4est, &mpi, pqr, &input_folder, angles, centroids, &rel_side_length_biggest_box);
      biomol_ooa1.set_grid_and_surface_parameters(0, l, lip, probe_radius, 1);
      p4est                 = biomol_ooa1.construct_SES(list_reduction, false, false, "null");
      Vec phi               = biomol_ooa1.return_phi_vector();
      p4est_nodes_t* nodes  = biomol_ooa1.return_nodes();
      p4est_ghost_t* ghost  = biomol_ooa1.return_ghost();
      my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
      my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);

      {
        double *phi_p = NULL;
        ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
        string vtk_file = output_folder + "/ooa_1_level_" +to_string(l);
        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               1, 0, vtk_file.c_str(),
                               VTK_POINT_DATA, "phi", phi_p);
        ierr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
      }

//      get_norm_of_errors(errors_1st_order_infty.at(l-lmin), errors_1st_order_one_norm.at(l-lmin),
//                               reference_p4est, reference_nodes, reference_phi,
//                               ngbd, phi);

      p4est_destroy(p4est);
      ierr = VecDestroy(phi); CHKERRXX(ierr);
      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);



      p4est = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
      p4est->user_pointer = (void *) &sp;
      my_p4est_biomolecules_t biomol_ooa2(p4est, &mpi, pqr, &input_folder, angles, centroids, &rel_side_length_biggest_box);
      biomol_ooa2.set_grid_and_surface_parameters(0, l, lip, probe_radius, 2);
      p4est             = biomol_ooa2.construct_SES(list_reduction, false, false, "null");
      phi               = biomol_ooa2.return_phi_vector();
      nodes  = biomol_ooa2.return_nodes();
      ghost  = biomol_ooa2.return_ghost();
      hierarchy.update(p4est, ghost);
      ngbd.update(&hierarchy, nodes);
      {
        double *phi_p = NULL;
        ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
        string vtk_file = output_folder + "/ooa_2_level_" +to_string(l);
        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               1, 0, vtk_file.c_str(),
                               VTK_POINT_DATA, "phi", phi_p);
        ierr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
      }

//      get_norm_of_errors(errors_2nd_order_infty.at(l-lmin), errors_2nd_order_one_norm.at(l-lmin),
//                               reference_p4est, reference_nodes, reference_phi,
//                               ngbd, phi);

      p4est_destroy(p4est);
      ierr = VecDestroy(phi); CHKERRXX(ierr);
      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);

//      p4est_destroy(reference_p4est);
//      ierr = VecDestroy(reference_phi); CHKERRXX(ierr);
//      p4est_nodes_destroy(reference_nodes);
//      p4est_ghost_destroy(reference_ghost);
    }

    ierr = PetscFPrintf(mpi.comm(), my_p4est_biomolecules_t::log_file, "\nThe errors for the 1st order method are:"); CHKERRXX(ierr);
    for (int l = lmin; l < lmax+1; ++l) {
      ierr = PetscFPrintf(mpi.comm(), my_p4est_biomolecules_t::log_file, "\nLevel %d: error_phi_infty = %g, log2(error_phi_infty) = %g, error_phi_one = %g, log2(error_phi_one) = %g", l,
                          errors_1st_order_infty.at(l-lmin), log2(errors_1st_order_infty.at(l-lmin)),
                          errors_1st_order_one_norm.at(l-lmin), log2(errors_1st_order_one_norm.at(l-lmin))); CHKERRXX(ierr);
    }
    ierr = PetscFPrintf(mpi.comm(), my_p4est_biomolecules_t::log_file, "\nThe errors for the 2nd order method are:"); CHKERRXX(ierr);
    for (int l = lmin; l < lmax+1; ++l) {
      ierr = PetscFPrintf(mpi.comm(), my_p4est_biomolecules_t::log_file, "\nLevel %d: error_phi_infty = %g, log2(error_phi_infty) = %g, error_phi_one = %g, log2(error_phi_one) = %g", l,
                          errors_2nd_order_infty.at(l-lmin), log2(errors_2nd_order_infty.at(l-lmin)),
                          errors_2nd_order_one_norm.at(l-lmin), log2(errors_2nd_order_one_norm.at(l-lmin))); CHKERRXX(ierr);
    }

    ierr = PetscFPrintf(mpi.comm(), my_p4est_biomolecules_t::log_file, "\n\n"); CHKERRXX(ierr);
    /* release memory */

    my_p4est_brick_destroy(connectivity, &brick);

    delete pqr;
    delete angles;
    delete centroids;
    if(mpi.rank() == 0)
    {
      if(!boost::iequals(nullstr, errlog_file) && !boost::iequals(stderr_str, errlog_file))
        fclose(my_p4est_biomolecules_t::error_file);
      if(!boost::iequals(nullstr, log_file) && !boost::iequals(stdout_str, log_file))
        fclose(my_p4est_biomolecules_t::log_file);
    }

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}
