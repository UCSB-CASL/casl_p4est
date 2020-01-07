// System
#include <stdexcept>
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <boost/algorithm/string.hpp>

// p4est
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
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_general_poisson_nodes_mls_solver.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex3_mls_q_vtk.h>
#include <src/my_p8est_biomolecules.h>
#include <src/my_p8est_poisson_jump_nodes_extended.h>
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
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_general_poisson_nodes_mls_solver.h>
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex2_mls_q_vtk.h>
#include <src/my_p4est_biomolecules.h>
#include <src/my_p4est_poisson_jump_nodes_extended.h>
#endif
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/casl_math.h>
using namespace std;


static const double domain_side_length = 1.0;

template<typename T> inline T my_conversion(const string&);
template<> inline double my_conversion<double>(const string& st){return atof(st.c_str());}
template<> inline string my_conversion<string>(const string& st){return st;}

template <typename T>
vector<T>* read_from_list(const string& list_of_values, const int n_values = -1)
{
  size_t size     = list_of_values.size();
  size_t n_comma  = 0;
  for (size_t k = 0; k < size; ++k) {
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
  size_t m, j = m = 0;
  for (size_t k = 0; k < size; ++k) {
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

int main(int argc, char *argv[]) {
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  // List of parameters
  // molecule(s)
  cmd.add_option("input-dir",     "folder in which the pqr files are located");
  cmd.add_option("pqr",           "name(s) of the pqr file(s) (syntax: '-pqr name0,name1,name2...)");
#ifdef P4_TO_P8
  cmd.add_option("centroid",      "centroid(s) coordinates of the molecule(s), domain [0, 1]X[0, 1]X[0, 1] (syntax: '-centroid x0,y0,z0,x1,y1,z1,x2,y2,z2,...)");
  cmd.add_option("rotangle",      "rotation angle(s) of the molecule(s) (syntax: '-rotangle angle00,angle01,angle02,angle10,angle11,angle12,angle20,angle21,angle22,...)");
#else
  cmd.add_option("centroid",      "centroid(s) coordinates of the molecule(s), domain [0, 1]X[0, 1] (syntax: '-centroid x0,y0,x1,y1,x2,y2,...)");
  cmd.add_option("rotangle",      "rotation angle(s) of the molecule(s) (syntax: '-rotangle angle0,angle1,angle2,...)");
#endif
  cmd.add_option("boxsize",       "relative side length of the biggest bounding box to the min domain dimension (0 < boxsize < 1)");
  // grid construction and problem accuracy
  cmd.add_option("ntree_dim",     "number of tree per dimension (ntree^3 trees in total, ntree >=1)");
  cmd.add_option("lmin",          "the min level of the tree (>=0)");
  cmd.add_option("lmax",          "the max level of the tree (>= lmin)");
  cmd.add_option("lip",           "Lipchitz constant for the levelset (>= 1.0)");
  cmd.add_option("rp",            "probe radius (in Angstrom, >= 0.0)");
  cmd.add_option("OOA",           "order of accuracy (1 or 2)");
  cmd.add_option("surfgen",       "Method for surface generation (0: brute force, 1: list reduction, 2: list reduction with exact calculation of distances)");
  // physical and solver parameters
  cmd.add_option("eps_mol",       "relative permittivity of the molecule (>=1.0, default is 2.0)");
  cmd.add_option("eps_elec",      "relative permittivity of the electrolyte (>=1.0, default is 80.0)");
  cmd.add_option("ion",           "ion charge in the symmetrical electrolyte (integer > 0, default is 1)");
  cmd.add_option("temperature",   "absolute temperature in K (>0.0, default is 300.0)");
  cmd.add_option("n0",            "far-field ion concentration in the electrolyte in mol/L (>0.0, default is 0.01)");
  cmd.add_option("rtol",          "tolerance on the 2-norm of the residual for the nonlinear solver (>0.0, default is 1e-8)");
  cmd.add_option("niter",         "number of Newton iterations for the nonlinear solver (>0, default is 1000)");
  cmd.add_option("linear",        "flag activating the linearization of the P-B problem (1 or 'yes' or 'true' to activate that feature, niter and rtol are irrelevant when this is activated)");
  //  exportation of results and/or timing
  cmd.add_option("output-dir",    "folder to save the results in");
  cmd.add_option("subvtk",        "flag activating the exportation of the grid after intermediary steps (1 or 'yes' or 'true' to deactivate that feature)");
  cmd.add_option("vtk",           "name of the vtk file(s) in the output-dir (no exportation of vtk file if 'null').");
  cmd.add_option("timing",        "timing exportation file");
  cmd.add_option("SAS-timing",    "flag activating the timer of the SAS constructor (1 or 'yes' or 'true' to activate that feature)");
  cmd.add_option("SAS-subtiming", "flag activating the subtimer of the SAS constructor (1 or 'yes' or 'true' to activate that feature, activated only if SAS_timing is activated too)");
  cmd.add_option("logfile",       "log file");
  cmd.add_option("err-log",       "error log file (default is stderr)");

  cmd.parse(argc, argv);
  // Now, read the options and/or set default parameters

  // which molecule(s)
  //  const string input_folder                 = cmd.get<string> ("input-dir", "/home/rochi/LabCode/casl_p4est/examples/biomol/mols");
  const string input_folder                 = cmd.get<string> ("input-dir", "/home/regan/Desktop/casl_p4est_develop/examples/biomol/mols");
  const string pqr_input                    = cmd.get<string>("pqr", "single_sphere.");
  //    const string pqr_input                    = cmd.get<string>("pqr", "3J6D."); // in 2D, for the illustrative planar molecule in the paper
  //    const string pqr_input                    = cmd.get<string>("pqr", "/3J3Q/pqr/3j3q-bundle."); // in 3D, for the graphical abstract of the paper
  const vector<string>* pqr = NULL;
  if(!boost::iequals(null_str, pqr_input))
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
  if(!boost::iequals(null_str, list_of_centroids))
    centroids                               = read_from_list<double>(list_of_centroids);
  if(!boost::iequals(null_str, list_of_angles))
    angles                                  = read_from_list<double>(list_of_angles);

  const double rel_side_length_biggest_box  = cmd.get<double>("boxsize", 0.3);
  // grid construction
  const int ntree_per_dim                   = cmd.get<int>("ntree_dim", 1);
  const int lmin                            = cmd.get<int>("lmin", 7);
  const int lmax                            = cmd.get<int>("lmax", 9);
  const double lip                          = cmd.get<double>("lip", 1.2);
  const int surf_gen                        = cmd.get<int>("surfgen", 2);
  const double probe_radius                 = cmd.get<double>("rp", 0.01);
  const int order_of_accuracy               = cmd.get<int>("OOA", 2);
  // physical and solver parameters
  const double eps_mol                      = cmd.get<double>("eps_mol", 1.0);
  const double eps_elec                     = cmd.get<double>("eps_elec", 78.54);
  const int ion_charge                      = cmd.get<int>("ion", 1);
  const double temperature                  = cmd.get<double>("temperature", 298.15);
  const double far_field_ion_concentration  = cmd.get<double>("n0", 0.000);
  const double rtol                         = cmd.get<double>("rtol", 1e-11);
  const int niter_max                       = cmd.get<int>("niter", 1000);
  const string linearization_string         = cmd.get<string>("linear", "yes");
  const bool linearization_flag             = (boost::iequals("1", linearization_string) || boost::iequals("yes", linearization_string) || boost::iequals("true", linearization_string));


  // exportation folder and files
  //  string output_folder                      = cmd.get<string>("output-dir", "/home/rochi/LabCode/results/biomol");
  string output_folder                      = cmd.get<string>("output-dir", "/home/regan/workspace/projects/biomol");
  mkdir(output_folder.c_str(), 0755);
  string subvtk                             = cmd.get<string>("subvtk", "yes");
  const bool subvtk_flag                    = (boost::iequals("1", subvtk) || boost::iequals("yes", subvtk) || boost::iequals("true", subvtk));
  const string vtk_name                     = cmd.get<string>("vtk", "illustration");
  //const string vtk_name = vtk + "_" + to_string(lmin) + "_" + to_string(lmax);
  /* create the exportation folder if it does not exist yet */
  output_folder += ((output_folder[output_folder.size()-1] == '/')?"":"/");
  /* open/create the log file if needed */
  const string log_file                     = cmd.get<string>("logfile", "stdout");
  my_p4est_biomolecules_t::log_file = NULL;
  if(!boost::iequals(null_str, log_file))
  {
    if(boost::iequals(stdout_str, log_file))
      my_p4est_biomolecules_t::log_file = stdout;
    else
    {
      string log_file_path = output_folder + log_file;
      my_p4est_biomolecules_t::log_file = fopen(log_file_path.c_str(), "w");
    }
  }
  const string timing_file                  = cmd.get<string>("timing", null_str);
  my_p4est_biomolecules_t::timing_file = NULL;
  if(!boost::iequals(null_str, timing_file))
  {
    if(boost::iequals(stdout_str, timing_file))
      my_p4est_biomolecules_t::timing_file = stdout;
    else
    {
      string timing_file_path = output_folder + timing_file;
      my_p4est_biomolecules_t::timing_file = fopen(timing_file_path.c_str(), "w");
    }
  }
  const string SAS_timing                   = cmd.get<string>("SAS-timing", "no");
  const bool SAS_timing_flag                = (boost::iequals("1", SAS_timing) || boost::iequals("yes", SAS_timing) || boost::iequals("true", SAS_timing));
  const string SAS_subtiming                = cmd.get<string>("SAS-subtiming", "no");
  const bool SAS_subtiming_flag             = (boost::iequals("1", SAS_subtiming) || boost::iequals("yes", SAS_subtiming) || boost::iequals("true", SAS_subtiming));
  const string errlog_file                  = cmd.get<string>("err-log", "stderr");
  my_p4est_biomolecules_t::error_file  = NULL;
  if(!boost::iequals(null_str, errlog_file))
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
  if (surf_gen != 0 && surf_gen != 1 && surf_gen != 2)
    throw invalid_argument("Invalid surface generation method (option surfgen), must be either 0 (brute force method) or 1 (recursive list reduction) or 2 (recursive list reduction with calculation of exact distances)");
#endif

  // create the macro mesh connectivity
  int n_xyz []      = {ntree_per_dim, ntree_per_dim, ntree_per_dim};
  double xyz_min [] = {0, 0, 0};
  double xyz_max [] = {domain_side_length, domain_side_length, domain_side_length};
  int periodic []   = {0, 0, 0};
  my_p4est_brick_t brick;
  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);


  /* create the p4est, nodes and ghosts */
  splitting_criteria_t sp(lmin, lmax, lip);
  p4est_t *p4est = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void *) &sp;


  // read the molecules and rotate them
  my_p4est_biomolecules_t my_biomol(&brick, p4est, rel_side_length_biggest_box, pqr, &input_folder, angles, centroids);

  my_biomol.set_grid_and_surface_parameters(lmin, lmax, lip, probe_radius, order_of_accuracy);
  p4est = my_biomol.construct_SES(((surf_gen == 0)? brute_force: ((surf_gen == 1)?list_reduction:list_reduction_with_exact_phi)), SAS_timing_flag, SAS_subtiming_flag, subvtk_flag?output_folder:"null");
  my_biomol.expand_ghost();


  my_p4est_biomolecules_solver_t solver(&my_biomol);
  solver.set_relative_permittivities(eps_mol, eps_elec);
  solver.set_ion_charge(ion_charge);
  solver.set_temperature_in_kelvin(temperature);
  solver.set_molar_concentration_of_electrolyte_in_mol_per_liter(far_field_ion_concentration);
  solver.solve_nonlinear(rtol, (linearization_flag)?1:niter_max);
#ifdef P4_TO_P8
  solver.get_solvation_free_energy(!linearization_flag);
#endif

  Vec psi_hat = NULL;
  solver.return_psi_hat(psi_hat);

  Vec phi               = my_biomol.return_phi_vector();
  p4est_nodes_t* nodes  = my_biomol.return_nodes();
  p4est_ghost_t* ghost  = my_biomol.return_ghost();
  if(!boost::iequals(null_str, vtk_name))
  {

    double *phi_p = NULL,*psi_hat_p = NULL;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(psi_hat, &psi_hat_p); CHKERRXX(ierr);
    string vtk_file = output_folder + vtk_name;


    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "psi_hat", psi_hat_p,
                           VTK_POINT_DATA, "phi", phi_p);


    ierr = VecRestoreArray(psi_hat, &psi_hat_p); psi_hat_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);

  }

  /* release memory */
  ierr = VecDestroy(phi); phi = NULL; CHKERRXX(ierr);
  ierr = VecDestroy(psi_hat); psi_hat = NULL; CHKERRXX(ierr);

  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  delete pqr;
  delete angles;
  delete centroids;
  if(mpi.rank() == 0)
  {
    if(!boost::iequals(null_str, timing_file) && !boost::iequals(stdout_str, timing_file))
      fclose(my_p4est_biomolecules_t::timing_file);
    if(!boost::iequals(null_str, errlog_file) && !boost::iequals(stderr_str, errlog_file))
      fclose(my_p4est_biomolecules_t::error_file);
    if(!boost::iequals(null_str, log_file) && !boost::iequals(stdout_str, log_file))
      fclose(my_p4est_biomolecules_t::log_file);
  }
  return 0;
}
