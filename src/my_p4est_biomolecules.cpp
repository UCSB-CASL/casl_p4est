
#ifdef P4_TO_P8
#include "my_p8est_biomolecules.h"
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include "my_p4est_biomolecules.h"
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <fstream>
#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <src/matrix.h>
#include <algorithm>
#include <sys/stat.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

using namespace std;

// initialize all static variables
const string Atom::ATOM = "ATOM  ";
const unsigned int my_p4est_biomolecules_t::nangle_per_mol = (P4EST_DIM == 3 ? 3 : 1);

FILE* my_p4est_biomolecules_t::log_file     = NULL;
FILE* my_p4est_biomolecules_t::timing_file  = NULL;
FILE* my_p4est_biomolecules_t::error_file   = stderr;

int reduced_list::nb_reduced_lists          = 0;

my_p4est_biomolecules_t::molecule::molecule(const my_p4est_biomolecules_t *owner, const string &pqr_, const double *angstrom_to_domain_, const double *xyz_c, double *angles, const int &overlap):
  environment(owner)
{
  read(pqr_, overlap);
#ifndef P4_TO_P8
  // remove possible duplicates (one of the coordinates has been disregarded for each atom when reading the list, there might be duplicates)
  sort(atoms.begin(), atoms.end());
  atoms.erase(unique(atoms.begin(), atoms.end()), atoms.end());
#endif
  index_of_charged_atom.clear();
  scale_rotate_and_translate(angstrom_to_domain_, xyz_c, angles);
}

void my_p4est_biomolecules_t::molecule::read(const string &pqr, const int &overlap)
{
#ifdef CASL_THROWS
  string              err_msg;
#endif
  int mpiret;
  if (atoms.size() != 0)
  {
    PetscPrintf(environment->p4est->mpicomm, "------------------------------------------------ \n");
    PetscPrintf(environment->p4est->mpicomm, "---  Reinitialization of the list of atoms  ---- \n");
    PetscPrintf(environment->p4est->mpicomm, "------------------------------------------------ \n");
    // reinitialize the list of atoms and related parameters
    atoms.clear();
  }
  P4EST_ASSERT(overlap>=1); // require an integer >= 1 for the overlap argument. overlap <= 0 does not make sense

  string bundle = "bundle";
  string extension = ".pqr";
  int pqr_length = pqr.size();
  bool is_a_bundle = (pqr_length >= 6 && !bundle.compare(pqr.substr(pqr_length-6, string::npos)));
  bool add_extension = !(pqr_length>4 && !extension.compare(pqr.substr(pqr_length-4, string::npos)));
  int file_idx = 1;
  string filename = pqr + ((is_a_bundle)? to_string(file_idx):"") + ((add_extension)? extension:"");
  MPI_File file_handle; // do not use a pointer, allocation needed...

  // initialize the centroid
  for (int k = 0; k < P4EST_DIM; ++k)
    molecule_centroid[k] = 0.0;
  // initialize the largest radius
  largest_radius = -DBL_MAX;
  // vector of atoms read by this proc
  vector<Atom> atoms_in_chunk; atoms_in_chunk.clear();
  // number of charged atoms read by this proc
  n_charged_atoms = 0;

  while ((is_a_bundle || file_idx == 1) && file_exists(filename)){
    // copy the path to the file (MPI_File_open requires a non-constant char*... :-B)
    char file_name[filename.size()+1];
    filename.copy(file_name, filename.size(), 0);
    file_name[filename.size()] = '\0';

    mpiret = MPI_File_open(environment->p4est->mpicomm, &file_name[0], MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
    P4EST_ASSERT(mpiret == sc_MPI_SUCCESS);

    /* read relevant chunk of file, which starts at location
     * globalstart in the file and has size mysize
     */

    /* figure out who reads what */
    MPI_Offset filesize;
    MPI_File_get_size(file_handle, &filesize);
    filesize--;  /* get rid of eof character */
    string chunk;
    int mysize = filesize/(environment->p4est->mpisize);
    int rank_max = environment->p4est->mpisize -1;
    if (mysize < overlap)
    {
      /* too many procs for the number of atoms in the file if this occurs.
     * It's possible that no relevant line is fully considered by any process.
     * This would result in garbage information for the atoms
     */
      if (error_file != NULL)
      {
        string warning_message = "bio_molecule::read(const string*, const int): !WARNING! the file is too small for the number of processes \nbio_molecule::read(const string*, const int): !WARNING! fewer procs will read chunks \n";
        PetscFPrintf(environment->p4est->mpicomm, error_file, warning_message.c_str());
      }
      /* We'll use less procs, the last ones will do nothing but wait for the final Allgather*/
      P4EST_ASSERT(filesize/overlap <= environment->p4est->mpisize); // something went wrong, a logic error occurred, inconsistent integer divisions...
      P4EST_ASSERT(filesize/overlap > 0); // the file is way too small, doesn't even contain one relevant line...
      rank_max = filesize/overlap;
      mysize = overlap;
      MPI_Offset globalstart;
      if(environment->p4est->mpirank <= rank_max)
        globalstart = mysize*environment->p4est->mpirank;
      else
        globalstart = filesize-1;
      MPI_Offset globalend    = (environment->p4est->mpirank < rank_max)?globalstart + mysize:filesize;
      /* add overlap to the end of everyone's chunk except last proc... */
      if (environment->p4est->mpirank < rank_max)
        globalend += overlap;
      mysize =  globalend - globalstart;
      /* allocate memory */
      chunk.resize(mysize + 1);
      /* everyone reads in their part */
      MPI_File_read_at_all(file_handle, globalstart, &chunk[0], mysize, MPI_CHAR, MPI_STATUS_IGNORE);
      chunk[mysize] = '\0';
    }
    else
    {
      MPI_Offset globalstart  = mysize*environment->p4est->mpirank;
      MPI_Offset globalend    = (environment->p4est->mpirank != rank_max)?globalstart + mysize:filesize;
      /* add overlap to the end of everyone's chunk except last proc... */
      if (environment->p4est->mpirank != rank_max)
        globalend += overlap;
      mysize =  globalend - globalstart;
      /* allocate memory */
      chunk.resize(mysize + 1);
      /* everyone reads in their part */
      MPI_File_read_at_all(file_handle, globalstart, &chunk[0], mysize, MPI_CHAR, MPI_STATUS_IGNORE);
      chunk[mysize] = '\0';
    }

    /*
     * everyone calculates what their start and end *really* are by going
     * from the first newline after start to the end of the overlapped line
     * (after end - overlap)
     */
    int locstart=0, locend=mysize;
    if (environment->p4est->mpirank != 0 && environment->p4est->mpirank <= rank_max) { /* second condition needed if using fewer procs than available, useless otherwise.*/
      while(chunk[locstart] != '\n' && locstart < locend)
        locstart++;
      locstart++; // skip the found '\n' character
    }
    if (environment->p4est->mpirank < rank_max) {
      locend-=overlap;
      while(chunk[locend] != '\n' && locend < mysize) locend++;
    }

    /* Process our chunk line by line */
    string line;
    int i = locstart;
    int line_size;
    while (i <= locend)
    {
      line_size = 0;
      while(chunk[i+line_size] != '\n' && (i+line_size) < locend){line_size++;}
      line = chunk.substr(i, line_size);

      Atom atom;
      if(line >> atom)
      {
        atoms_in_chunk.push_back(atom);
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
          molecule_centroid[dir] += atom.xyz_c[dir];
        largest_radius = MAX(largest_radius, atom.r_vdw);
        n_charged_atoms += (fabs(atom.q) > 0.00005)? 1:0; // the charge resolution is 0.0001 in regular pqr files
      }
      i += line_size+1; // +1 to skip the '\n' character and/or avoid unterminated loop because of trailing '\n'
    }
    MPI_File_close(&file_handle);

    file_idx++;
    filename = pqr + ((is_a_bundle)? to_string(file_idx):"") + ((add_extension)? extension:"");
  }

  vector<int> byte_offset_in_proc(environment->p4est->mpisize);
  vector<int> nb_of_bytes_in_proc(environment->p4est->mpisize);
  nb_of_bytes_in_proc[environment->p4est->mpirank] = atoms_in_chunk.size()*sizeof(Atom);
  mpiret = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &nb_of_bytes_in_proc[0], 1, MPI_INT, environment->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &molecule_centroid[0], P4EST_DIM, MPI_DOUBLE, MPI_SUM, environment->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &n_charged_atoms, 1, MPI_INT, MPI_SUM, environment->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &largest_radius, 1, MPI_DOUBLE, MPI_MAX, environment->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  int total_nb_of_atoms = nb_of_bytes_in_proc[0]/sizeof(Atom);
  byte_offset_in_proc[0] = 0;
  for (int k = 1; k < environment->p4est->mpisize; ++k) {
    total_nb_of_atoms       += nb_of_bytes_in_proc[k]/sizeof(Atom);
    byte_offset_in_proc[k]  = byte_offset_in_proc[k-1] + nb_of_bytes_in_proc[k-1];
  }
  for (int k = 0; k < P4EST_DIM; ++k) {
    molecule_centroid[k] /= total_nb_of_atoms;
  }

  atoms.resize(total_nb_of_atoms);
  mpiret = MPI_Allgatherv(&atoms_in_chunk[0],
      atoms_in_chunk.size()*sizeof(Atom),
      MPI_BYTE,
      &atoms[0],
      &nb_of_bytes_in_proc[0],
      &byte_offset_in_proc[0],
      MPI_BYTE,
      environment->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  return;
}

double my_p4est_biomolecules_t::molecule::calculate_scaling_factor(const double &box_max_side_length_to_min_domain_size) const
{
  P4EST_ASSERT((box_max_side_length_to_min_domain_size > 0.0) && (box_max_side_length_to_min_domain_size < 1.0));
  const double domain_dim_min = MIN(DIM(environment->parameters.domain_dim(0), environment->parameters.domain_dim(1), environment->parameters.domain_dim(2)));
  // calculate the conversion factor
  double domain_to_angstrom = (scaling.is_set)? 1.0/scaling.angstrom_to_domain : 1.0;
  double max_side_length_of_bounding_box_angstrom = MAX(DIM(side_length_of_bounding_box[0], side_length_of_bounding_box[1], side_length_of_bounding_box[2]))*domain_to_angstrom;
  return box_max_side_length_to_min_domain_size*domain_dim_min/max_side_length_of_bounding_box_angstrom; // new angstrom_to_domain factor
  // In real physical space, this value is in [1/ansgtrom] and is equal
  // to the inverse of the distance that becomes "1.0" in the domain
}
void my_p4est_biomolecules_t::molecule::scale_rotate_and_translate(const double* angstrom_to_domain_, const double* xyz_c, double* angles)
{
  P4EST_ASSERT(angstrom_to_domain_==NULL || (*angstrom_to_domain_ > 0.0));

  const bool scaling_required = (scaling.is_set && angstrom_to_domain_ != NULL && (fabs(scaling.angstrom_to_domain-*angstrom_to_domain_) > EPS*MAX(scaling.angstrom_to_domain, *angstrom_to_domain_))) || (!scaling.is_set && angstrom_to_domain_ != NULL);
  const double scaling_factor = scaling_required? ((scaling.is_set && angstrom_to_domain_ != NULL)? (*angstrom_to_domain_)/scaling.angstrom_to_domain : *angstrom_to_domain_) : 1.0;

  double xyz_tmp[P4EST_DIM];
  double rotated_xyz_tmp[P4EST_DIM];
  double **rotation_matrix = NULL;
  if(angles != NULL)
  {
    rotation_matrix = new double* [P4EST_DIM];
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      rotation_matrix[dim] = new double [P4EST_DIM];
    angles[0] = fmod(angles[0], 2*PI); if(angles[0] < 0) angles[0] += 2*PI;
#ifdef P4_TO_P8
    angles[2] = fmod(angles[2], 2*PI); if(angles[2] < 0) angles[2] += 2*PI;
    angles[1] = fmod(angles[1], PI);
    if(angles[1] < 0)
    {
      angles[2] = fmod(angles[2]+PI, 2*PI); // reverse the azimuthal angle
      angles[1] *= -1;
    }
    double rotation_axis[P4EST_DIM];
    rotation_axis[0] = sin(angles[1])*cos(angles[2]);
    rotation_axis[1] = sin(angles[1])*sin(angles[2]);
    rotation_axis[2] = cos(angles[1]);
    // Rodrigues' rotation formula
    rotation_matrix[0][0] = (1.0 - cos(angles[0]))*rotation_axis[0]*rotation_axis[0] + cos(angles[0]);
    rotation_matrix[0][1] = (1.0 - cos(angles[0]))*rotation_axis[0]*rotation_axis[1] - sin(angles[0])*rotation_axis[2];
    rotation_matrix[0][2] = (1.0 - cos(angles[0]))*rotation_axis[0]*rotation_axis[2] + sin(angles[0])*rotation_axis[1];
    rotation_matrix[1][0] = (1.0 - cos(angles[0]))*rotation_axis[0]*rotation_axis[1] + sin(angles[0])*rotation_axis[2];
    rotation_matrix[1][1] = (1.0 - cos(angles[0]))*rotation_axis[1]*rotation_axis[1] + cos(angles[0]);
    rotation_matrix[1][2] = (1.0 - cos(angles[0]))*rotation_axis[1]*rotation_axis[2] - sin(angles[0])*rotation_axis[0];
    rotation_matrix[2][0] = (1.0 - cos(angles[0]))*rotation_axis[0]*rotation_axis[2] - sin(angles[0])*rotation_axis[1];
    rotation_matrix[2][1] = (1.0 - cos(angles[0]))*rotation_axis[1]*rotation_axis[2] + sin(angles[0])*rotation_axis[0];
    rotation_matrix[2][2] = (1.0 - cos(angles[0]))*rotation_axis[2]*rotation_axis[2] + cos(angles[0]);
#else
    rotation_matrix[0][0] = cos(angles[0]);
    rotation_matrix[0][1] = -sin(angles[0]);
    rotation_matrix[1][0] = sin(angles[0]);
    rotation_matrix[1][1] = cos(angles[0]);
#endif
  }


  const bool create_list_of_charged_atoms = (index_of_charged_atom.size() == 0);
  if(xyz_c != NULL || scaling_required || create_list_of_charged_atoms || (angles != NULL))
  {
    double new_centroid[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      side_length_of_bounding_box[dir] = 0.0;
      new_centroid[dir] = (xyz_c == NULL)? scaling_factor*molecule_centroid[dir] : xyz_c[dir];
    }
    int charged_atoms_found = 0;
    if(create_list_of_charged_atoms)
      index_of_charged_atom.resize(n_charged_atoms);
    for (size_t k = 0; k < atoms.size(); ++k) {
      if(create_list_of_charged_atoms && fabs(atoms[k].q) > 0.00005)
        index_of_charged_atom[charged_atoms_found++] = k;
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        xyz_tmp[dir] = scaling_factor*(atoms[k].xyz_c[dir] - molecule_centroid[dir]);
      if(rotation_matrix != NULL)
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
          rotated_xyz_tmp[dir] = SUMD(rotation_matrix[dir][0]*xyz_tmp[0], rotation_matrix[dir][1]*xyz_tmp[1], rotation_matrix[dir][2]*xyz_tmp[2]);
      else
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
          rotated_xyz_tmp[dir] = xyz_tmp[dir];
      atoms[k].r_vdw               *= scaling_factor; // don't forget to scale the radius
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        atoms[k].xyz_c[dir]           = new_centroid[dir] + rotated_xyz_tmp[dir];
        side_length_of_bounding_box[dir] = MAX(side_length_of_bounding_box[dir], fabs(atoms[k].xyz_c[dir] - new_centroid[dir])+atoms[k].r_vdw);
      }
    }
    if(create_list_of_charged_atoms)
      P4EST_ASSERT(charged_atoms_found == n_charged_atoms);
    // it's the max distance from the centroid of the molecule to the cube faces so far, hence half the side length
    // --> multiply by 2
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      molecule_centroid[dir] = new_centroid[dir];
      side_length_of_bounding_box[dir] *= 2.;
    }
    largest_radius *= scaling_factor;
  }

  if(!scaling.is_set && angstrom_to_domain_!= NULL)
    scaling.is_set = true;
  if(scaling_required)
    scaling.angstrom_to_domain = *angstrom_to_domain_;
  if(rotation_matrix != NULL)
  {
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      delete [] rotation_matrix[dim];
    delete [] rotation_matrix;
  }
  P4EST_ASSERT(!scaling.is_set || is_bounding_box_in_domain());
}

void my_p4est_biomolecules_t::molecule::reduce_to_single_atom()
{
  atoms.resize(1);
  n_charged_atoms = (fabs(atoms[0].q) > 0.00005)? 1 : 0;
  index_of_charged_atom.resize(n_charged_atoms);
  if(n_charged_atoms)
    index_of_charged_atom[0] = 0;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    molecule_centroid[dir] = atoms[0].xyz_c[dir];
    side_length_of_bounding_box[dir] = 2*atoms[0].r_vdw;
  }
  largest_radius = atoms[0].r_vdw;
}
bool my_p4est_biomolecules_t::molecule::is_bounding_box_in_domain() const
{
  bool to_return = true;
  double *vertices_to_coordinates = environment->p4est->connectivity->vertices;
  p4est_topidx_t *tree_to_vertex  = environment->p4est->connectivity->tree_to_vertex;
  for (unsigned char dir = 0; to_return && (dir < P4EST_DIM); ++dir)
    if(!is_periodic(environment->p4est->connectivity, dir))
    {
      double coord_min = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + dir];
      double coord_max = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(environment->p4est->connectivity->num_trees-1) + P4EST_CHILDREN-1] + dir];
      to_return = to_return && (molecule_centroid[dir]+0.5*side_length_of_bounding_box[dir] <= coord_max) && (molecule_centroid[dir]-0.5*side_length_of_bounding_box[dir] >= coord_min);
    }
  return to_return;
}

p4est_bool_t my_p4est_biomolecules_t::SAS_creator::refine_for_exact_calculation_fn(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  // Let's not enforce the min level when we build the SAS grid because that might lead to irrelevant
  // handling and/or communications of kinda big reduced lists of atoms for the sole purpose of refining
  // the grid up to a desired minimum level --> does not scale well! (does not scale AT ALL, actually)
  int max_lvl = biomol->parameters.lmax();
  (void) which_tree;
  if (quad->level >= max_lvl)
    return P4EST_FALSE;
  else
  {
    P4EST_ASSERT(biomol->phi_read_only_p != NULL);
    const double cell_diag        = biomol->parameters.tree_diag()/(1<<quad->level);
    const double rp               = biomol->parameters.probe_radius();
    const double L                = biomol->parameters.lip();
    const p4est_nodes_t* nodes    = biomol->nodes;
    // one MUST use the 'long' user-defined integer since p4est_locidx_t is an alias for int32_t
    // only long format ensures 32 bits, int ensures 16 bits only
    p4est_locidx_t former_quad_idx = (forest->mpisize>1)?(quad->p.user_long & (biomol->max_quad_loc_idx - 1)) : quad->p.user_long; // bitwise filtering

    double f[P4EST_CHILDREN];
    for (unsigned char k = 0; k < P4EST_CHILDREN; ++k) {
      p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*former_quad_idx+k];
      P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
      f[k] = biomol->phi_read_only_p[node_idx] - rp;
      if(fabs(f[k]) < 0.5*L*cell_diag || f[k] < -biomol->parameters.tree_diag())
        return P4EST_TRUE;
    }
    if (f[0]*f[1] < 0.0 || f[0]*f[2] < 0.0 || f[1]*f[3] < 0.0 || f[2]*f[3] < 0.0
    #ifdef P4_TO_P8
        || f[0]*f[4] < 0.0 || f[1]*f[5] < 0.0 || f[2]*f[6] < 0.0 || f[3]*f[7] < 0.0
        || f[4]*f[5] < 0.0 || f[4]*f[6] < 0.0 || f[5]*f[7] < 0.0 || f[6]*f[7] < 0.0
    #endif
        )
      return P4EST_TRUE;
    return P4EST_FALSE;
  }
}

p4est_bool_t my_p4est_biomolecules_t::SAS_creator::refine_for_reinitialization_fn(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  // Let's not enforce the min level when we build the SAS grid because that might lead to irrelevant
  // handling and/or communications of kinda big reduced lists of atoms for the sole purpose of refining
  // the grid up to a desired minimum level --> does not scale well! (does not scale AT ALL, actually)
  int max_lvl = biomol->parameters.lmax();
  (void) which_tree;
  if (quad->level >= max_lvl)
    return P4EST_FALSE;
  else
  {
    P4EST_ASSERT(biomol->phi_read_only_p != NULL);
    const double cell_diag        = biomol->parameters.tree_diag()/(1<<quad->level);
    const double layer_thickness  = biomol->parameters.layer_thickness();
    const double rp               = biomol->parameters.probe_radius();
    const double L                = biomol->parameters.lip();
    const p4est_nodes_t* nodes    = biomol->nodes;
    // one MUST use the 'long' user-defined integer since p4est_locidx_t is an alias for int32_t
    // only long format ensures 32 bits, int ensures 16 bits only
    p4est_locidx_t former_quad_idx = (forest->mpisize>1)?(quad->p.user_long & (biomol->max_quad_loc_idx - 1)) : quad->p.user_long; // bitwise filtering

    double f[P4EST_CHILDREN];
    for (unsigned char k = 0; k < P4EST_CHILDREN; ++k) {
      p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*former_quad_idx+k];
      P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
      f[k] = biomol->phi_read_only_p[node_idx];
      if((-layer_thickness <= f[k]) && (f[k] <= rp))
        return P4EST_TRUE;
      if((0 <= f[k] - rp) && (f[k] - rp <= 0.5*L*cell_diag))
        return P4EST_TRUE;
      if(f[k] < -layer_thickness && ((rp-f[k]) <= 0.5*L*cell_diag))
        return P4EST_TRUE;
    }
    if (f[0]*f[1] < 0.0 || f[0]*f[2] < 0.0 || f[1]*f[3] < 0.0 || f[2]*f[3] < 0.0
    #ifdef P4_TO_P8
        || f[0]*f[4] < 0.0 || f[1]*f[5] < 0.0 || f[2]*f[6] < 0.0 || f[3]*f[7] < 0.0
        || f[4]*f[5] < 0.0 || f[4]*f[6] < 0.0 || f[5]*f[7] < 0.0 || f[6]*f[7] < 0.0
    #endif
        )
      return P4EST_TRUE;
    return P4EST_FALSE;
  }
}

void my_p4est_biomolecules_t::SAS_creator::determine_locally_known_values(p4est_t* &forest)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;

  // needed for point equality check comparison: boundary points are clamped inside
  // the domain and the Morton codes are compared, two points are equivalent if Morton
  // codes are identical after being clamped
  const int clamped = 1;
  P4EST_ASSERT((biomol->phi == NULL && biomol->nodes == NULL) || (biomol->phi != NULL && biomol->nodes != NULL));
  // ok let's work
  // initialization might be needed first
  if(biomol->phi == NULL && biomol->nodes == NULL)
  {
    parStopWatch* subsubtimer = NULL;
    if(sub_timer != NULL)
    {
      subsubtimer = new parStopWatch(parStopWatch::all_timings, biomol->timing_file, mpi_comm);
      subsubtimer->start("             SAS_creator::initialization routine");
    }
    initialization_routine(forest);
    PetscInt my_index_offset = 0;
    for (int proc_rank = 0; proc_rank < mpi_rank; ++proc_rank)
      my_index_offset += biomol->nodes->global_owned_indeps[proc_rank];
    global_indices_of_known_values.resize(biomol->nodes->num_owned_indeps);
    // scatter locally the values that are already known, count the number of points that need calculations
    // and keep track of the global indices of values to be scattered to the new layout afterwards
    for (p4est_locidx_t k = 0; k < biomol->nodes->num_owned_indeps; ++k)
      global_indices_of_known_values[k] = my_index_offset + k;

    // we don't need the coarse nodes and corresponding data any more
    p4est_nodes_destroy(biomol->nodes); biomol->nodes = NULL; // we no longer need the nodes
    if(subsubtimer != NULL)
    {
      subsubtimer->stop();subsubtimer->read_duration(true);
      delete subsubtimer; subsubtimer = NULL;
    }
    return;
  }

  p4est_nodes_t* local_refined_nodes = my_p4est_nodes_new(forest, NULL);
  p4est_locidx_t coarse_idx   = 0;
  p4est_indep_t *coarse_node = NULL;
  if(coarse_idx < biomol->nodes->num_owned_indeps)
    coarse_node = (p4est_indep_t*) sc_array_index(&biomol->nodes->indep_nodes, coarse_idx);
  PetscInt my_index_offset = 0;
  for (int proc_rank = 0; proc_rank < mpi_rank; ++proc_rank)
    my_index_offset += local_refined_nodes->global_owned_indeps[proc_rank];
  global_indices_of_known_values.resize(biomol->nodes->num_owned_indeps);
  // scatter locally the values that are already known, count the number of points that need calculations
  // and keep track of the global indices of values to be scattered to the new layout afterwards
  for (p4est_locidx_t k = 0; (k < local_refined_nodes->num_owned_indeps) && (coarse_idx < biomol->nodes->num_owned_indeps); ++k) {
    p4est_indep_t *fine_node  = (p4est_indep_t*) sc_array_index(&local_refined_nodes->indep_nodes,k);
    if(p4est_node_equal_piggy_fn (fine_node, coarse_node, &clamped))
    {
      global_indices_of_known_values[coarse_idx++] = my_index_offset + k;
      if(coarse_idx < biomol->nodes->num_owned_indeps)
        coarse_node = (p4est_indep_t*) sc_array_index(&biomol->nodes->indep_nodes,coarse_idx);
    }
  }
  // sanity check
  P4EST_ASSERT(coarse_idx == biomol->nodes->num_owned_indeps);
  p4est_nodes_destroy(biomol->nodes); biomol->nodes = NULL; // we no longer need the local (unpartitioned) nodes
  p4est_nodes_destroy(local_refined_nodes);                 // we no longer need the local_refined_nodes neither
}

void my_p4est_biomolecules_t::SAS_creator::scatter_to_new_layout(p4est_t* &forest, const bool &ghost_flag)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  // scatter relevant value before calculating the new phi_sas values
  if(ghost_flag)
  {
    P4EST_ASSERT(biomol->ghost == NULL);
    biomol->ghost = p4est_ghost_new(forest, P4EST_CONNECT_FULL);
  }
  P4EST_ASSERT(biomol->nodes==NULL);
  biomol->nodes = my_p4est_nodes_new(forest, biomol->ghost);
  // scatter the vector to the new layout
  // create the new (partioned) vector of phi_sas
  Vec partitioned_phi_sas, partitioned_phi_sas_local;
  ierr = VecCreateGhostNodes(forest, biomol->nodes, &partitioned_phi_sas);                                            CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(partitioned_phi_sas, &partitioned_phi_sas_local);                                       CHKERRXX(ierr);
  ierr = VecSet(partitioned_phi_sas_local, -1.5*fabs(phi_sas_lower_bound));                                           CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(partitioned_phi_sas, &partitioned_phi_sas_local);                                   CHKERRXX(ierr);
  // initialize it to values that are below the theoretical lower bound so that apporiate
  // values will be computed...

  // scatter known values:
  ierr = VecScatterAllToSome(mpi_comm, biomol->phi, partitioned_phi_sas, global_indices_of_known_values, ghost_flag); CHKERRXX(ierr);
  // destroy previous vector and update with new one:
  ierr    = VecDestroy(biomol->phi);                                                                                  CHKERRXX(ierr);
  biomol->phi = partitioned_phi_sas;
}

void my_p4est_biomolecules_t::SAS_creator::partition_forest_and_update_sas(p4est_t* &forest)
{
  if(sub_timer != NULL)
    sub_timer->start("        SAS_creator::scatter_locally");
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  biomol->update_max_level();
  determine_locally_known_values(forest);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration(true);
    sub_timer->start("        SAS_creator::weighted_partition");
  }
  weighted_partition(forest);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration(true);
    sub_timer->start("        SAS_creator::scatter_to_new_layout");
  }
  scatter_to_new_layout(forest);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration(true);
    sub_timer->start("        SAS_creator::update_phi_sas_and_quadrant_data");
  }
  update_phi_sas_and_quadrant_data(forest);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration(true);
  }
  if(biomol->global_max_level == biomol->parameters.lmax()) // final call
  {
    if(sub_timer != NULL)
      sub_timer->start("        SAS_creator:ghost creation");
    ghost_creation_and_final_partitioning(forest);
    if(sub_timer != NULL)
    {
      sub_timer->stop();sub_timer->read_duration(true);
      delete sub_timer; sub_timer = NULL;
    }
  }
}

int  my_p4est_biomolecules_t::SAS_creator::reinitialization_weight_fn(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  p4est_locidx_t former_quad_idx = (forest->mpisize>1)?(quadrant->p.user_long & (biomol->max_quad_loc_idx - 1)) : quadrant->p.user_long;
  for (unsigned char k = 0; k < P4EST_CHILDREN; ++k) {
    p4est_locidx_t node_idx = biomol->nodes->local_nodes[P4EST_CHILDREN*former_quad_idx+k];
    if (biomol->phi_read_only_p[node_idx] > -EPS)
      return 1;
  }
  return 0;
}

void my_p4est_biomolecules_t::SAS_creator::ghost_creation_and_final_partitioning(p4est_t *&forest)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  PetscInt my_index_offset = 0;
  for (int proc_rank = 0; proc_rank < mpi_rank; ++proc_rank)
    my_index_offset += biomol->nodes->global_owned_indeps[proc_rank];
  global_indices_of_known_values.resize(biomol->nodes->num_owned_indeps);
  for (p4est_locidx_t k = 0; k < biomol->nodes->num_owned_indeps; ++k)
    global_indices_of_known_values[k] = my_index_offset + k;
  P4EST_ASSERT(biomol->phi_read_only_p == NULL);
  ierr = VecGetArrayRead(biomol->phi, &biomol->phi_read_only_p);                                      CHKERRXX(ierr);
  my_p4est_partition(forest, P4EST_TRUE, reinitialization_weight_fn); // balance calculations for next reinitialization, and allow for coarsening afterwards
  ierr = VecRestoreArrayRead(biomol->phi, &biomol->phi_read_only_p); biomol->phi_read_only_p = NULL;  CHKERRXX(ierr);
  p4est_nodes_destroy(biomol->nodes); biomol->nodes = NULL;
  scatter_to_new_layout(forest, true);
}

void my_p4est_biomolecules_t::SAS_creator::refine_and_partition(p4est_t* &forest, const int& step_idx)
{
  if(sas_timer != NULL)
    sas_timer->start("    step " + to_string(step_idx) + ": refining the grid");
  refine_the_p4est(forest);
  if(sas_timer != NULL)
  {
    sas_timer->stop(); sas_timer->read_duration(true);
    sas_timer->start("    step " + to_string(step_idx) + ": updating phi_sas, and partitioning the grid");
  }
  partition_forest_and_update_sas(forest);
  if (sas_timer != NULL)
  {
    sas_timer->stop(); sas_timer->read_duration(true);
  }
}

void my_p4est_biomolecules_t::SAS_creator::refine_the_p4est(p4est_t* &forest)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  P4EST_ASSERT(biomol->phi_read_only_p == NULL);
  ierr = VecGetArrayRead(biomol->phi, &biomol->phi_read_only_p);                                      CHKERRXX(ierr);
  specific_refinement(forest);
  ierr = VecRestoreArrayRead(biomol->phi, &biomol->phi_read_only_p); biomol->phi_read_only_p = NULL;  CHKERRXX(ierr);
}

my_p4est_biomolecules_t::SAS_creator::SAS_creator(p4est_t* &forest, const bool timing_flag, const bool subtiming_flag)
  : mpi_rank(forest->mpirank),
    mpi_size(forest->mpisize),
    mpi_comm(forest->mpicomm),
    phi_sas_lower_bound(-1.5*((my_p4est_biomolecules_t*) (forest->user_pointer))->parameters.domain_diag()),
    sas_timer((my_p4est_biomolecules_t::timing_file != NULL && timing_flag)?new parStopWatch(parStopWatch::all_timings, my_p4est_biomolecules_t::timing_file, forest->mpicomm):NULL),
    sub_timer((my_p4est_biomolecules_t::timing_file != NULL && timing_flag && subtiming_flag)?new parStopWatch(parStopWatch::all_timings, my_p4est_biomolecules_t::timing_file, forest->mpicomm):NULL)
{}

void my_p4est_biomolecules_t::SAS_creator::construct_SAS(p4est_t* &forest)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  int step_idx=1;
  for (; biomol->global_max_level < biomol->parameters.lmax(); ++step_idx)
    refine_and_partition(forest, step_idx);
  if (sas_timer != NULL)
  {
    delete sas_timer; sas_timer = NULL;
  }
}

my_p4est_biomolecules_t::SAS_creator::~SAS_creator()
{
  if(sas_timer != NULL)
  {
    sas_timer->stop();sas_timer->read_duration(true);
    delete sas_timer; sas_timer = NULL;
  }
  if(sub_timer != NULL)
  {
    sub_timer->stop(); sub_timer->read_duration(true);
    delete sub_timer; sub_timer = NULL;
  }
}

void my_p4est_biomolecules_t::SAS_creator_brute_force::initialization_routine(p4est_t* &forest)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  biomol->nodes = my_p4est_nodes_new(forest, NULL); // we don't need ghost cells yet...
  ierr  = VecCreateGhostNodes(forest, biomol->nodes, &biomol->phi);         CHKERRXX(ierr);
  {
    double *phi_p;
    ierr = VecGetArray(biomol->phi, &phi_p);                                CHKERRXX(ierr);
    for (size_t i = 0; i < biomol->nodes->indep_nodes.elem_count; ++i) {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(i, biomol->p4est, biomol->nodes, xyz);
      phi_p[i] = (*biomol)(xyz);
    }
    ierr = VecRestoreArray(biomol->phi, &phi_p);                            CHKERRXX(ierr);
  }
}

int  my_p4est_biomolecules_t::SAS_creator_brute_force::weight_fn(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  const my_p4est_biomolecules_t* biomol = (const my_p4est_biomolecules_t*) forest->user_pointer;
  if(quadrant->level == biomol->global_max_level)
    return 1; // we'll have work to do only for newly added points, which are associated with newly created quadrants... workload is constant (loop through ALL atoms)
  else
    return 0;
}

void my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(p4est_t* &forest)
{
  my_p4est_biomolecules_t* biomol            = (my_p4est_biomolecules_t*) forest->user_pointer;

  double *phi_p;
  ierr = VecGetArray(biomol->phi, &phi_p); CHKERRXX(ierr);
  vector<int> nb_to_send_recv; nb_to_send_recv.resize(mpi_size, 0);
  // get the number of values to be calculated locally
  for (p4est_locidx_t k = 0; k < biomol->nodes->num_owned_indeps; ++k)
    if(phi_p[k] < phi_sas_lower_bound)
      nb_to_send_recv[mpi_rank]++;

  // determine first the BALANCED number of points that each proc has to calculate
  // so share yours with all procs
  mpiret = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nb_to_send_recv.data(), 1, MPI_INT, mpi_comm); SC_CHECK_MPI(mpiret);
  // split the number of calculations evenly through the processes (+-1 calculation difference betweeb procs)
  vector<int> nb_to_be_calculated_balanced; nb_to_be_calculated_balanced.resize(mpi_size, 0);
  int total_nb_to_be_calculated = 0;
  for (int k = 0; k < mpi_size; ++k)
    total_nb_to_be_calculated += nb_to_send_recv[k];
  for (int k = 0; k < mpi_size; ++k)
  {
    nb_to_be_calculated_balanced[k] = ((k < total_nb_to_be_calculated % mpi_size)?1:0) + total_nb_to_be_calculated/mpi_size;
    nb_to_send_recv[k] -= nb_to_be_calculated_balanced[k];
  }
  // --> if kth value n_k is
  //     - positive: proc k has  n_k points to send for calculation by (an)other proc(s)
  //     - negative: proc k has -n_k points to recv from (an)other proc(s) for calculation
  //     - zero: proc k can be left alone, it has right enough values to calculate;
#ifdef CASL_THROWS
  int sum_check = 0;
  for (int k = 0; k < mpi_size; ++k)
    sum_check += nb_to_send_recv[k];
  P4EST_ASSERT(sum_check==0);
#endif
  // coordinates buffer
  double xyz[P4EST_DIM];

  vector<MPI_Request> query_req; query_req.clear();
  vector<MPI_Request> reply_req; reply_req.clear();

  vector<receiver_data> receivers; receivers.clear();
  const int my_nb_to_send_recv = nb_to_send_recv[mpi_rank];
  if(my_nb_to_send_recv > 0)
  {
    int how_many_left_to_assign = my_nb_to_send_recv;
    // calculate the number of previous off-proc requests filling existing gaps before yourself
    int nb_sent_by_previous_procs = 0;
    for (int k = 0; k < mpi_rank; ++k)
      nb_sent_by_previous_procs += MAX(nb_to_send_recv[k], 0);
    // find the first proc to which you can send request(s)
    int recv_rank = 0;
    int nb_recv_by_proc_before_recv_rank = 0;
    while (nb_recv_by_proc_before_recv_rank + MAX(-nb_to_send_recv[recv_rank], 0) <= nb_sent_by_previous_procs)
      nb_recv_by_proc_before_recv_rank += MAX(-nb_to_send_recv[recv_rank++], 0);
    P4EST_ASSERT(nb_to_send_recv[recv_rank] < 0);
    P4EST_ASSERT(recv_rank != mpi_rank);
    int that_proc_can_recv = nb_recv_by_proc_before_recv_rank + MAX(-nb_to_send_recv[recv_rank], 0) - nb_sent_by_previous_procs;
    int nb_to_send_to_that_rank = MIN(that_proc_can_recv, how_many_left_to_assign);
    receivers.push_back({recv_rank, nb_to_send_to_that_rank});
    how_many_left_to_assign -= nb_to_send_to_that_rank;
    while (how_many_left_to_assign > 0) {
      recv_rank++;
      if(nb_to_send_recv[recv_rank] < 0)
      {
        P4EST_ASSERT(recv_rank != mpi_rank);
        nb_to_send_to_that_rank = MIN(-nb_to_send_recv[recv_rank], how_many_left_to_assign);
        receivers.push_back({recv_rank, nb_to_send_to_that_rank});
        how_many_left_to_assign -= nb_to_send_to_that_rank;
      }
    }
    P4EST_ASSERT(how_many_left_to_assign == 0);
  }
  int num_remaining_replies = receivers.size();
  int num_remaining_queries = 0;
  vector<int> is_a_receiver; is_a_receiver.resize(mpi_size, 0);
  for (int j = 0; j < num_remaining_replies; ++j) {
    receiver_data& r_data = receivers[j];
    is_a_receiver[r_data.recv_rank] = 1;
  }
  vector<int> nb_results_per_proc; nb_results_per_proc.resize(mpi_size, 1);
  mpiret = MPI_Reduce_scatter(is_a_receiver.data(), &num_remaining_queries, nb_results_per_proc.data(), MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);
  P4EST_ASSERT((my_nb_to_send_recv <= 0)    || (num_remaining_queries == 0)); // a proc wants to send messages but expects queries too...
  P4EST_ASSERT((num_remaining_queries <= 0) || (my_nb_to_send_recv < 0));     // a proc expects queries but has to send messages too
  P4EST_ASSERT((my_nb_to_send_recv != 0)    || (num_remaining_queries == 0));   // a proc expects queries but has just the right number of values to compute...
  P4EST_ASSERT((num_remaining_queries != 0) || (my_nb_to_send_recv >= 0));      // a proc does not expect queries but has too few values to compute...

  // pack the coordinates to send and send them; declare reply buffer(s)
  map<int, query_buffer> query_buffers; query_buffers.clear();
  map<int, vector<double> > reply_buffers; reply_buffers.clear();
  if(my_nb_to_send_recv > 0) // this proc has too many values to calculate
  {
    int value_to_be_calculated_counter = 0;
    p4est_locidx_t k = 0;
    // skip values to be calculated locally
    while(value_to_be_calculated_counter < nb_to_be_calculated_balanced[mpi_rank] && k < biomol->nodes->num_owned_indeps)
      if(phi_p[k++] <= phi_sas_lower_bound)
        value_to_be_calculated_counter++;
    for (int query_idx = 0; query_idx < num_remaining_replies; ++query_idx) {
      receiver_data& r_data = receivers[query_idx];
      vector<double>& coordinates = query_buffers[r_data.recv_rank].node_coordinates;
      coordinates.resize(P4EST_DIM*r_data.recv_count);
      vector<p4est_locidx_t>& node_indices = query_buffers[r_data.recv_rank].node_local_indices;
      node_indices.resize(r_data.recv_count);
      int off_proc_value_index = 0;
      while(off_proc_value_index < r_data.recv_count && k < biomol->nodes->num_owned_indeps)
      {
        if(phi_p[k] <= phi_sas_lower_bound)
        {
          node_xyz_fr_n(k, forest, biomol->nodes, xyz);
          for (int j = 0; j < P4EST_DIM; ++j)
            coordinates[P4EST_DIM*off_proc_value_index+j] = xyz[j];
          node_indices[off_proc_value_index] = k;
          off_proc_value_index++;
        }
        k++;
      }
    }
    for (int query_idx = 0; query_idx < num_remaining_replies; ++query_idx) {
      receiver_data& r_data = receivers[query_idx];
      const vector<double>& to_send = query_buffers[r_data.recv_rank].node_coordinates;
      MPI_Request req;
      mpiret = MPI_Isend((void*) to_send.data(), P4EST_DIM*r_data.recv_count, MPI_DOUBLE, r_data.recv_rank, query_tag, mpi_comm, &req);
      SC_CHECK_MPI(mpiret);
      query_req.push_back(req);
    }
  }
  size_t nb_local_calculated = 0, nb_local_calculated_end = nb_to_be_calculated_balanced[mpi_rank] + ((my_nb_to_send_recv>=0)?0:my_nb_to_send_recv);
  MPI_Status status;
  bool done = false;
  int k = -1;
  while (!done) {
    // calculate local values
    if (nb_local_calculated < nb_local_calculated_end)
    {
      while(phi_p[++k] > phi_sas_lower_bound && k < biomol->nodes->num_owned_indeps){}
      node_xyz_fr_n(k, forest, biomol->nodes, xyz);
      phi_p[k] = (*biomol)( xyz);
      nb_local_calculated++;
    }

    // probe for incoming queries
    if (num_remaining_queries > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {
        int nb_coordinates;
        mpiret = MPI_Get_count(&status, MPI_DOUBLE, &nb_coordinates); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(nb_coordinates % P4EST_DIM == 0);
        int nb_nodes = nb_coordinates/P4EST_DIM;
        vector<double> node_coordinates; node_coordinates.resize(nb_coordinates);
        mpiret = MPI_Recv(node_coordinates.data(), nb_coordinates, MPI_DOUBLE, status.MPI_SOURCE, query_tag, mpi_comm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);
        vector<double>& reply = reply_buffers[status.MPI_SOURCE];
        reply.resize(nb_nodes);
        for (int j = 0; j < nb_nodes; ++j)
          reply[j] = (*biomol)((node_coordinates.data()+P4EST_DIM*j));
        const vector<double>& to_reply = reply;
        MPI_Request req;
        mpiret = MPI_Isend(to_reply.data(), nb_nodes, MPI_DOUBLE, status.MPI_SOURCE, reply_tag, mpi_comm, &req); SC_CHECK_MPI(mpiret);
        reply_req.push_back(req);
        num_remaining_queries--;
      }
    }
    // probe for incoming replies
    if (num_remaining_replies > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) {
        int nb_nodes;
        mpiret = MPI_Get_count(&status, MPI_DOUBLE, &nb_nodes); SC_CHECK_MPI(mpiret);
        vector<double> reply; reply.resize(nb_nodes);
        mpiret = MPI_Recv(reply.data(), nb_nodes, MPI_DOUBLE, status.MPI_SOURCE, reply_tag, mpi_comm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(nb_nodes == (int) query_buffers[status.MPI_SOURCE].node_local_indices.size()); // check that we received all that was expected
        for (int j = 0; j < nb_nodes; ++j) {
          p4est_locidx_t node_index = query_buffers[status.MPI_SOURCE].node_local_indices[j];
          phi_p[node_index] = reply[j];
        }
        num_remaining_replies--;
      }
    }
    done = num_remaining_queries == 0 && num_remaining_replies == 0 && nb_local_calculated == nb_local_calculated_end;
  }
  ierr = VecRestoreArray(biomol->phi, & phi_p); CHKERRXX(ierr);
  // START GHOST UPDATE
  ierr = VecGhostUpdateBegin(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // START GHOST UPDATE
  mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);


  // update_local_quadrant_indices in the p.user_long quadrant data, as needed in refine_fn
  for (p4est_topidx_t tree_id = forest->first_local_tree; tree_id <= forest->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(forest->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      quad->p.user_long = q + tree_k->quadrants_offset;
    }
  }
  P4EST_ASSERT(mpi_size == 1 || ((int64_t) forest->local_num_quadrants <= biomol->max_quad_loc_idx - 1)); // the maximum number of local quadrants has been reached, the method needs to be redesigned with real quadrant data...
  // END GHOST UPDATE
  ierr = VecGhostUpdateEnd(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // END GHOST UPDATE
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::initialization_routine(p4est_t* &forest)
{
  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() == 0);
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;

  for (p4est_topidx_t tt = forest->first_local_tree; tt <= forest->last_local_tree; ++tt) {
    p4est_tree_t* tree = p4est_tree_array_index(forest->trees, tt);
    for (size_t qq = 0; qq < tree->quadrants.elem_count; ++qq) {
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, qq);
      reduced_list_ptr parent_list(new reduced_list(biomol->total_nb_atoms));
      biomol->add_reduced_list(tt, quad, parent_list, get_exact_phi);
    }
  }
  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() == forest->local_num_quadrants);
  P4EST_ASSERT(biomol->nodes == NULL);
  P4EST_ASSERT(biomol->phi == NULL);
  biomol->nodes = my_p4est_nodes_new(forest, NULL); // we don't need ghost cells here...
  ierr  = VecCreateGhostNodes(forest, biomol->nodes, &biomol->phi);   CHKERRXX(ierr);
  Vec phi_ghost_loc;
  ierr  = VecGhostGetLocalForm(biomol->phi, &phi_ghost_loc);          CHKERRXX(ierr);
  ierr  = VecSet(phi_ghost_loc, -1.5*fabs(phi_sas_lower_bound));      CHKERRXX(ierr);
  ierr  = VecGhostRestoreLocalForm(biomol->phi, &phi_ghost_loc);      CHKERRXX(ierr);
}

int my_p4est_biomolecules_t::SAS_creator_list_reduction::weight_fn(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  const my_p4est_biomolecules_t* biomol = (const my_p4est_biomolecules_t*) forest->user_pointer;
  const int8_t min_lvl_to_consider = biomol->update_last_current_level_only? biomol->global_max_level : 0;
  if(quadrant->level >= min_lvl_to_consider)
  {
    const reduced_list& r_list = *(biomol->reduced_lists[(quadrant->p.user_long & (biomol->max_quad_loc_idx -1))]);
    return r_list.size(); // we'll have work to do only for newly added points, which are associated with newly created quadrants... workload is constant (loop through ALL atoms)
  }
  else
    return 0;
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::update_phi_sas_and_quadrant_data(p4est_t* &forest)
{
  my_p4est_biomolecules_t* biomol            = (my_p4est_biomolecules_t*) forest->user_pointer;

  // -- BEGIN GHOST UPDATE --
  ierr  = VecGhostUpdateBegin(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // -- BEGIN GHOST UPDATE --

  bool last_stage = (biomol->global_max_level == biomol->parameters.lmax());
  int8_t min_lvl_to_consider = (biomol->update_last_current_level_only? biomol->global_max_level : 0);
  vector<const p4est_quadrant_t*> locally_known_quadrants; locally_known_quadrants.clear();
  vector<p4est_locidx_t> local_idx_of_locally_known_quadrants; local_idx_of_locally_known_quadrants.clear();
  map<int, query_buffer> query_buffers; query_buffers.clear();
  map<int, vector<int> > reply_buffers; reply_buffers.clear();

  for (p4est_topidx_t tt = forest->first_local_tree; tt <= forest->last_local_tree; ++tt) {
    p4est_tree_t* tree = p4est_tree_array_index(forest->trees, tt);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, q);
      if(quad->level < min_lvl_to_consider)
        continue;
      int quad_rank_owner = (mpi_size > 1)? (((int64_t) quad->p.user_long)) >> (8*sizeof(long) - biomol->rank_encoding) : 0;
      if(quad_rank_owner != mpi_rank)
      {
        P4EST_ASSERT(quad_rank_owner < mpi_size);
        query_buffer& q_buf = query_buffers[quad_rank_owner];
        p4est_locidx_t new_idx = biomol->reduced_lists.size();
        reduced_list_ptr temporary_list(new reduced_list);
        biomol->reduced_lists.push_back(temporary_list);
        q_buf.new_list_idx.push_back(new_idx);
        q_buf.off_proc_list_idx.push_back((quad->p.user_long & (biomol->max_quad_loc_idx -1)));
        q_buf.local_quad_idx.push_back(tree->quadrants_offset + q);
        quad->p.user_long = ((long) mpi_rank << (8*sizeof(long) - biomol->rank_encoding)) + new_idx;
      }
      else
      {
        locally_known_quadrants.push_back(quad);
        local_idx_of_locally_known_quadrants.push_back(tree->quadrants_offset + q);
      }
    }
  }

#ifdef DEBUG
  p4est_bool_t check = P4EST_TRUE;;
  for (map<int, query_buffer>::const_iterator it = query_buffers.begin(); check && (it != query_buffers.end()); ++it)
  {
    check = check && (it->first != mpi_rank);
    const query_buffer& q_buf = it->second;
    check = check && (q_buf.new_list_idx.size() == q_buf.off_proc_list_idx.size());
    check = check && (q_buf.new_list_idx.size() == q_buf.local_quad_idx.size());
  }
  P4EST_ASSERT(check); // something went wrong when figuring out the off-proc queries
#endif

  vector<int> is_a_receiver;  is_a_receiver.resize(mpi_size, 0);
  vector<int> data_count;     data_count.resize(mpi_size, 1);
  int num_remaining_replies = 0;
  for (map<int, query_buffer>::const_iterator it = query_buffers.begin(); it != query_buffers.end(); ++it)
  {
    is_a_receiver[it->first] = 1;
    num_remaining_replies++;
  }
  int num_remaining_queries;
  mpiret = MPI_Reduce_scatter(is_a_receiver.data(), &num_remaining_queries, data_count.data(), MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);

#ifdef DEBUG
  int total_num_queries = 0, total_num_replies = 0;
  mpiret = MPI_Allreduce(&num_remaining_queries, &total_num_queries, 1, MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(&num_remaining_replies, &total_num_replies, 1, MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);
  P4EST_ASSERT(total_num_queries == total_num_replies); // the total numbers of expected queries and replies do not match
#endif

  vector<MPI_Request> query_req; query_req.clear();
  vector<MPI_Request> reply_req; reply_req.clear();

  for (map<int, query_buffer>::iterator it = query_buffers.begin(); it != query_buffers.end(); ++it)
  {
    query_buffer& q_buf = it->second;
    MPI_Request req;
    mpiret = MPI_Isend(q_buf.off_proc_list_idx.data(), q_buf.off_proc_list_idx.size(), MPI_INT, it->first, query_tag, mpi_comm, &req); SC_CHECK_MPI(mpiret);
    query_req.push_back(req);
  }

  MPI_Status status;
  int locally_known_quadrants_treated = 0;
  int nb_locally_known_quadrants = locally_known_quadrants.size();
  // -- END GHOST UPDATE --
  ierr  = VecGhostUpdateEnd(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // -- END GHOST UPDATE --
  double *phi_p;
  ierr = VecGetArray(biomol->phi, &phi_p); CHKERRXX(ierr);
  p4est_locidx_t quad_idx;
  double xyz[P4EST_DIM];
  bool done = false;

  while (!done)
  {
    if (locally_known_quadrants_treated < nb_locally_known_quadrants)
    {
      const p4est_quadrant_t* quad = locally_known_quadrants[locally_known_quadrants_treated];
      quad_idx = local_idx_of_locally_known_quadrants[locally_known_quadrants_treated];
      int reduced_list_idx = (mpi_size > 1)?(quad->p.user_long & (biomol->max_quad_loc_idx -1)):quad->p.user_long;
      for (unsigned char i = 0; i < P4EST_CHILDREN; i++) {
        p4est_locidx_t node_idx = biomol->nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
        node_xyz_fr_n(node_idx, forest, biomol->nodes, xyz);
        phi_p[node_idx] = MAX(phi_p[node_idx], biomol->reduced_operator(xyz, reduced_list_idx, get_exact_phi, last_stage));
        // NOTE THE 'MAX': it looks like we might compute several time the value of the function at the grid nodes,
        // but the max makes it a little more complicated...
      }
      locally_known_quadrants_treated++;
    }
    // probe for incoming queries
    if (num_remaining_queries > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {
        int nb_queried_reduced_lists;
        mpiret = MPI_Get_count(&status, MPI_INT, &nb_queried_reduced_lists); SC_CHECK_MPI(mpiret);
        vector<int> indices_of_queried_lists; indices_of_queried_lists.resize(nb_queried_reduced_lists);
        mpiret = MPI_Recv(indices_of_queried_lists.data(), nb_queried_reduced_lists, MPI_INT, status.MPI_SOURCE, query_tag, mpi_comm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);
        vector<int>& serialized_reply = reply_buffers[status.MPI_SOURCE];
        int size_serialized_reply = 0;
        for (int kk = 0; kk < nb_queried_reduced_lists; ++kk)
        {
          const reduced_list& queried_reduced_list = *(biomol->reduced_lists[indices_of_queried_lists[kk]]);
          size_serialized_reply += (1 + queried_reduced_list.size());
        }
        serialized_reply.resize(size_serialized_reply);
        int idx = 0;
        for (int kk = 0; kk < nb_queried_reduced_lists; ++kk)
        {
          const reduced_list& queried_reduced_list = *(biomol->reduced_lists[indices_of_queried_lists[kk]]);
          serialized_reply[idx++] = queried_reduced_list.size();
          for (size_t jj = 0; jj < queried_reduced_list.size(); ++jj)
            serialized_reply[idx++] = queried_reduced_list.atom_global_idx[jj];
        }
        P4EST_ASSERT(idx == size_serialized_reply);
        MPI_Request req;
        mpiret = MPI_Isend(serialized_reply.data(), size_serialized_reply, MPI_INT, status.MPI_SOURCE, reply_tag, mpi_comm, &req); SC_CHECK_MPI(mpiret);
        reply_req.push_back(req);
        num_remaining_queries--;
      }
    }

    // probe for incoming replies
    if (num_remaining_replies > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) {
        int size_serialized_reply;
        mpiret = MPI_Get_count(&status, MPI_INT, &size_serialized_reply); SC_CHECK_MPI(mpiret);
        const query_buffer& q_buf = query_buffers[status.MPI_SOURCE];
        const int nb_queried_lists = (int) q_buf.off_proc_list_idx.size();
        P4EST_ASSERT(size_serialized_reply >= 2*nb_queried_lists); // check that the reply is at least of the minimal size
        vector<int> serialized_reply; serialized_reply.resize(size_serialized_reply);
        mpiret = MPI_Recv(serialized_reply.data(), size_serialized_reply, MPI_INT, status.MPI_SOURCE, reply_tag, mpi_comm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        int idx = 0;
        for (int kk = 0; kk < nb_queried_lists; ++kk) {
          reduced_list_ptr new_list(new reduced_list(serialized_reply[idx++], -1));
          reduced_list& list_to_add = *new_list;
          for (size_t jj = 0; jj < list_to_add.size() ; ++jj)
            list_to_add.atom_global_idx[jj] = serialized_reply[idx++];
          biomol->reduced_lists[q_buf.new_list_idx[kk]] = new_list;

          quad_idx = q_buf.local_quad_idx[kk];
          for (unsigned char i = 0; i < P4EST_CHILDREN; i++) {
            p4est_locidx_t node_idx = biomol->nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
            node_xyz_fr_n(node_idx, forest, biomol->nodes, xyz);
            phi_p[node_idx] = MAX(phi_p[node_idx], biomol->reduced_operator(xyz, q_buf.new_list_idx[kk], get_exact_phi, last_stage));
          }
        }
        P4EST_ASSERT(idx == size_serialized_reply);
        num_remaining_replies--;
      }
    }
    done = num_remaining_queries == 0 && num_remaining_replies == 0 && locally_known_quadrants_treated == nb_locally_known_quadrants;
  }
  ierr = VecRestoreArray(biomol->phi, & phi_p); CHKERRXX(ierr);

  mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  // this is the reason we have to stick to our silly sign convention: no MIN_VALUES in PETSC...
  // -- START MAX GHOST UPDATE SCATTER REVERSE --
  ierr = VecGhostUpdateBegin(biomol->phi, MAX_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
  // -- START MAX GHOST UPDATE SCATTER REVERSE --

  P4EST_ASSERT(biomol->old_reduced_lists.size() == 0);

  if (last_stage)
  {
    // -- END MAX GHOST UPDATE SCATTER REVERSE here if last stage --
    ierr = VecGhostUpdateEnd(biomol->phi, MAX_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
    // -- END MAX GHOST UPDATE SCATTER REVERSE here if last stage --
    if(get_exact_phi)
    {
      ierr = VecGetArray(biomol->phi, & phi_p); CHKERRXX(ierr);
      double kink_point[P4EST_DIM];
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        kink_point[dim] = 2.0*biomol->parameters.tree_diag();
      set<p4est_locidx_t> done_nodes; done_nodes.clear();

      for (p4est_topidx_t tt = forest->first_local_tree; tt <= forest->last_local_tree; ++tt) {
        p4est_tree_t* tree = p4est_tree_array_index(forest->trees, tt);
        for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
          p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, q);
          if(quad->level < biomol->global_max_level)
            continue;
          p4est_locidx_t quad_idx = tree->quadrants_offset + q;
          int reduced_list_idx = (mpi_size > 1)?(quad->p.user_long & (biomol->max_quad_loc_idx -1)):quad->p.user_long;
          for (unsigned char i = 0; i < P4EST_CHILDREN; i++) {
            p4est_locidx_t node_idx = biomol->nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
            if(done_nodes.find(node_idx) == done_nodes.end() && 0 < phi_p[node_idx] && phi_p[node_idx] < biomol->parameters.probe_radius() + biomol->parameters.layer_thickness())
            {
              node_xyz_fr_n(node_idx, forest, biomol->nodes, xyz);
              double distance_to_sas = biomol->better_distance(xyz, reduced_list_idx, &kink_point[0]);
              phi_p[node_idx] = MAX(phi_p[node_idx], distance_to_sas);
              done_nodes.insert(node_idx);
            }
          }
        }
      }
      ierr = VecRestoreArray(biomol->phi, &phi_p); CHKERRXX(ierr);
    }
    ierr  = VecGhostUpdateBegin(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  // update_local_quadrant_indices in the p.user_long quadrant data, as needed in refine_fn
  // --> store the shared pointers to reduced lists in the biomol->old_reduced_lists map
  for (p4est_topidx_t tt = forest->first_local_tree; tt <= forest->last_local_tree; ++tt) {
    p4est_tree_t* tree = p4est_tree_array_index(forest->trees, tt);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_loc_idx = tree->quadrants_offset + q;
      if(quad->level >= min_lvl_to_consider && !last_stage)
      {
        int reduced_list_idx = ((mpi_size > 1)? quad->p.user_long & (biomol->max_quad_loc_idx - 1) : quad->p.user_long);
        biomol->old_reduced_lists[quad_loc_idx] = biomol->reduced_lists[reduced_list_idx];
      }
      quad->p.user_long = (mpi_size> 1)? ((long) mpi_rank << (8*sizeof(long) - biomol->rank_encoding)) + quad_loc_idx : quad_loc_idx;
    }
  }
  if(last_stage)
  {
    ierr  = VecGhostUpdateEnd(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  else
  {
    // -- END MAX GHOST UPDATE SCATTER REVERSE here if not the last stage --
    ierr = VecGhostUpdateEnd(biomol->phi, MAX_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
    // -- END MAX GHOST UPDATE SCATTER REVERSE here if not the last stage --
  }
  biomol->reduced_lists.clear();
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::replace_fn(p4est_t *forest, p4est_topidx_t which_tree, int num_outgoing, p4est_quadrant_t *outgoing[], int num_incoming, p4est_quadrant_t *incoming[])
{
  (void) num_incoming;
  P4EST_ASSERT(num_outgoing <= 1);
  (void) num_outgoing; // to avoid compiler warning in release;
  /* this is refinement */
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  SAS_creator_list_reduction* this_creator = dynamic_cast<SAS_creator_list_reduction*>(biomol->sas_creator);
  int parent_list_idx = (forest->mpisize > 1)?(outgoing[0]->p.user_long & (biomol->max_quad_loc_idx -1 )) : outgoing[0]->p.user_long;
  for (unsigned char i = 0; i < P4EST_CHILDREN; i++)
    biomol->add_reduced_list(which_tree, incoming[i], biomol->old_reduced_lists[parent_list_idx], this_creator->get_exact_phi);
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::specific_refinement(p4est_t* &forest)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  P4EST_ASSERT(biomol->reduced_lists.size() == 0);
  if(!get_exact_phi)
    p4est_refine_ext(forest, P4EST_FALSE, -1, refine_for_reinitialization_fn, NULL, replace_fn);
  else
    p4est_refine_ext(forest, P4EST_FALSE, -1, refine_for_exact_calculation_fn, NULL, replace_fn);
  biomol->old_reduced_lists.clear();
  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() <= forest->local_num_quadrants);
#ifdef CASL_THROWS
  p4est_gloidx_t nb_new_fine_quad = 0;
  for (p4est_topidx_t tt = forest->first_local_tree; tt <= forest->last_local_tree; ++tt) {
    p4est_tree_t* tree = p4est_tree_array_index(forest->trees, tt);
    nb_new_fine_quad += tree->quadrants_per_level[biomol->global_max_level+1];
  }
  P4EST_ASSERT((nb_new_fine_quad == (p4est_gloidx_t) biomol->reduced_lists.size())); // something went wrong when associating reduced lists to new quadrants
#endif
}

void my_p4est_biomolecules_t::check_validity_of_vector_of_mol() const
{
  P4EST_ASSERT(nmol()>0);
  P4EST_ASSERT(atom_index_offset.size() == nmol());
  P4EST_ASSERT(atom_index_offset[0] == 0);
  for (size_t k = 1; k < nmol(); ++k)
    P4EST_ASSERT(atom_index_offset[k] == atom_index_offset[k-1]+bio_molecules[k-1].get_number_of_atoms());
  // check box_size_of_biggest_mol and index_of_biggest_mol
  P4EST_ASSERT(fabs(box_size_of_biggest_mol - bio_molecules[index_of_biggest_mol].get_side_length_of_bounding_cube()) <= EPS*box_size_of_biggest_mol);
  P4EST_ASSERT(all_molecules_are_scaled_consistently());
  // check the molecules
  for (size_t k = 0; k < nmol(); ++k) {
    P4EST_ASSERT(bio_molecules[k].is_bounding_box_in_domain());
    P4EST_ASSERT(bio_molecules[k].get_side_length_of_bounding_cube() <= box_size_of_biggest_mol);
  }
  return;
}

bool my_p4est_biomolecules_t::all_molecules_are_scaled_consistently() const
{
  bool result = nmol() > 0;
  for (size_t k = 0; result && (k < nmol()); ++k)
    result &= (bio_molecules[k].is_scaled() && fabs(angstrom_to_domain - bio_molecules[k].get_scaling_factor()) < EPS*angstrom_to_domain);
  return result;
}
bool my_p4est_biomolecules_t::no_molecule_is_scaled() const
{
  bool result = true;
  for (size_t k = 0; result && (k < nmol()); ++k)
    result = result && !bio_molecules[k].is_scaled();
  return result;
}
void my_p4est_biomolecules_t::get_vector_of_current_centroids(vector<double>& current_centroids)
{
  current_centroids.resize(P4EST_DIM*nmol());
  for (size_t k = 0; k < nmol(); ++k)
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      current_centroids[k*P4EST_DIM+dim] = bio_molecules[k].get_centroid()[dim];
}
void my_p4est_biomolecules_t::rescale_all_molecules()
{
  vector<double> current_centroids;
  get_vector_of_current_centroids(current_centroids);
  rescale_all_molecules(current_centroids.data());
}
void my_p4est_biomolecules_t::rescale_all_molecules(const double* new_centroids)
{
  box_size_of_biggest_mol = -DBL_MAX;
  const double new_scale = angstrom_to_domain;
  for (size_t k = 0; k < nmol(); ++k) {
    bio_molecules[k].scale_and_translate(&new_scale, (new_centroids != NULL)?(new_centroids+P4EST_DIM*k):NULL);
    if(bio_molecules[k].get_side_length_of_bounding_cube() > box_size_of_biggest_mol)
    {
      box_size_of_biggest_mol = bio_molecules[k].get_side_length_of_bounding_cube();
      index_of_biggest_mol    = k;
    }
  }
}

void my_p4est_biomolecules_t::add_single_molecule(const string& file_path, const double* centroid, double* angles, const double* angstrom_to_domain_)
{
  //  double new_scaling_factor; // angstrom_to_domain scaling factor to be applied when reading the pqr file, on-the-fly
  if(angstrom_to_domain_ == NULL) // no specific scaling factor given for the new molecule
  {
    if (no_molecule_is_scaled())
    {
      add_single_molecule(molecule(this, file_path, NULL, centroid, angles)); // i.e. no scaling given, no scaling used so far, so do not scale when reading
      return;
    }
    if (!all_molecules_are_scaled_consistently()) // should never happen, this is a logic error...
    {
      P4EST_ASSERT(false); // in debug, we abort, otherwise, we'll fix it first!
      PetscFPrintf(p4est->mpicomm, (error_file != NULL?error_file : stderr), "my_p4est_biomolecules_t::add_single_molecule(...): the current vector of molecules has inconsistent scaling factors, this shouldn't happen...");
      PetscFPrintf(p4est->mpicomm, (error_file != NULL?error_file : stderr), "my_p4est_biomolecules_t::add_single_molecule(...): ... rescaling the molecules (fixing a logic error)...");
      rescale_all_molecules();
    }
  }
  else
  {
    // a specific scaling factor is given for the new molecule, scale all others consistently if needed before reading the new one
    // angstrom_to_domain is set to the new provided value after this
    if(fabs(angstrom_to_domain - *angstrom_to_domain_) > EPS*MAX(EPS, MAX(angstrom_to_domain, *angstrom_to_domain_)) || no_molecule_is_scaled())
      rescale_all_molecules(*angstrom_to_domain_); // local value set to the new one
    else if (!all_molecules_are_scaled_consistently())// should never happen, this is a logic error...
    {
      P4EST_ASSERT(false); // in debug, we abort, otherwise, we'll fix it first!
      PetscFPrintf(p4est->mpicomm, (error_file != NULL?error_file : stderr), "my_p4est_biomolecules_t::add_single_molecule(...): the current vector of molecules has inconsistent scaling factors, this shouldn't happen...");
      PetscFPrintf(p4est->mpicomm, (error_file != NULL?error_file : stderr), "my_p4est_biomolecules_t::add_single_molecule(...): ... rescaling the molecules (fixing a logic error)...");
      rescale_all_molecules(*angstrom_to_domain_);
    }
  }
  add_single_molecule(molecule(this, file_path, &angstrom_to_domain, centroid, angles));
  return;
}

int my_p4est_biomolecules_t::find_mol_index(const int& global_atom_index, const size_t& guess) const
{
  P4EST_ASSERT(global_atom_index >= 0 && global_atom_index < total_nb_atoms);
  // check the guess first
  if(guess < nmol() && atom_index_offset[guess] <= global_atom_index && global_atom_index < atom_index_offset[guess] + bio_molecules[guess].get_number_of_atoms())
    return guess;
  int L = 0;
  if(guess < nmol() && atom_index_offset[guess] <= global_atom_index)
    L = guess;
  int R = nmol();
  if(guess < nmol()-1 && global_atom_index < atom_index_offset[guess] + bio_molecules[guess].get_number_of_atoms())
    R = guess +1;
  int two = 2;
  int mol_index;
  while (R-L > 1)
  {
    mol_index = (L+R)/two;
    if(atom_index_offset[mol_index] <= global_atom_index)
      L = mol_index;
    else
      R = mol_index;
    P4EST_ASSERT(R>L);
  }
  return L;
}

const Atom* my_p4est_biomolecules_t::get_atom(const int& global_atom_index, size_t& guess) const
{
  guess = find_mol_index(global_atom_index, guess);
  int atom_index = global_atom_index - atom_index_offset[guess];
  P4EST_ASSERT(0<= atom_index && atom_index < bio_molecules[guess].get_number_of_atoms());
  return bio_molecules[guess].get_atom(atom_index);
}
my_p4est_biomolecules_t::my_p4est_biomolecules_t(my_p4est_brick_t *brick_, p4est_t* p4est_, const double& rel_side_length_biggest_box,
                                                 const vector<string>* pqr_names, const string* input_folder,
                                                 vector<double>* angles, const vector<double>* centroids) :
  parameters(p4est_),
  brick(brick_),
  p4est(p4est_),
  rank_encoding((int) (ceil(log2(p4est_->mpisize)))),
  max_quad_loc_idx(((int64_t) 1)<<(8*sizeof(long) - (int) (ceil(log2(p4est_->mpisize)))))
{
#ifdef DEBUG
  string err_msg;
  // sanity checks
  if(pqr_names == NULL && (angles != NULL || centroids != NULL))
    PetscFPrintf(p4est_->mpicomm, error_file, "my_p4est_biomolecules_t::my_p4est_biomolecules_t(...): angle(s) and/or centroid(s) set, but pqr file(s) undefined: no molecule will be read so no rotation/translation/scaling will be applied.");
  size_t nmolecules = (pqr_names != NULL)? pqr_names->size() : 0;
  P4EST_ASSERT((angles == NULL) || (angles->size() == nangle_per_mol*nmolecules) || (angles->size() == (size_t) nangle_per_mol));
  P4EST_ASSERT(centroids == NULL || centroids->size() == P4EST_DIM*nmolecules);
#endif
  if(timing_file != NULL)
    timer = new parStopWatch(parStopWatch::all_timings, timing_file, p4est->mpicomm);

  if(timer != NULL)
    timer->start("Construct the my_p4est_biomolecules_t object (reading the molecules, scaling them, ...)");

  // basic initialization of relevant parameters
  bio_molecules.clear();
  atom_index_offset.clear();
  total_nb_atoms          = 0;
  index_of_biggest_mol    = -1;       // absurd value, there is no molecule yet
  box_size_of_biggest_mol = -DBL_MAX; // absurd value, there is no molecule yet
  angstrom_to_domain      = 1.0;      // no molecule yet, so no scaling yet

  reduced_lists.clear(); // make sure no reduced_list exists beforehand
  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() == 0);
  nodes                     = NULL;     // we will build the nodes internally, NULL initialization
  ghost                     = NULL;     // we will build the ghost internally, NULL initialization
  phi                       = NULL;     // we will build that one too, NULL initialization
  phi_read_only_p           = NULL;
  inner_domain              = NULL;
  sas_creator               = NULL;     // NULL initialization
  hierarchy                 = NULL;     // we will build that one too, NULL initialization
  neighbors                 = NULL;     // we will build that one too, NULL initialization
  ls                        = NULL;     // we will build that one too, NULL initialization

  if(pqr_names != NULL)
  {
    // read and rotate them if desired
    string slash = "/";
    bool add_slash = false;
    if(input_folder != NULL)
    {
      if(!is_folder(*input_folder))
        throw std::invalid_argument("my_p4est_biomolecules_t::check_if_directory(const string*): invalid directory path: " + *input_folder + " does not exist or is not a directory."); // if it is not a directory, it is not for ALL proc --> send an exception here is alright
      add_slash = (*input_folder)[input_folder->size()-1] != '/';
    }
    for (size_t k = 0; k < pqr_names->size(); ++k) {
      string file_path= ((input_folder != NULL)?(*input_folder + ((add_slash)?slash:"")):"") + (*pqr_names)[k];
      add_single_molecule(file_path,
                          (centroids != NULL)? (centroids->data()+P4EST_DIM*k):NULL,
                          (angles != NULL)? (angles->data()+((nangle_per_mol*k < angles->size())?nangle_per_mol*k:0)):NULL); // no scaling yet, needs to know about the biggest molecule first, scale afterwards
    }
    // scale the molecules and locate them
    set_biggest_bounding_box(rel_side_length_biggest_box);
#ifdef DEBUG
    check_validity_of_vector_of_mol();
#endif
    print_summary();
  }

  if(timer != NULL)
  {
    timer->stop(); timer->read_duration(true);
  }
}
Vec my_p4est_biomolecules_t::return_phi_vector()
{
  P4EST_ASSERT(phi != NULL);
  Vec phi_to_return = phi; phi = NULL;
  return phi_to_return;
}
p4est_nodes_t* my_p4est_biomolecules_t::return_nodes()
{
  P4EST_ASSERT(nodes != NULL);
  p4est_nodes_t* nodes_to_return = nodes; nodes = NULL;
  return nodes_to_return;
}
p4est_ghost_t* my_p4est_biomolecules_t::return_ghost()
{
  P4EST_ASSERT(ghost != NULL);
  p4est_ghost_t* ghost_to_return = ghost; ghost = NULL;
  return ghost_to_return;
}
void my_p4est_biomolecules_t::return_phi_vector_nodes_and_ghost(Vec& phi_out, p4est_nodes_t* &nodes_out, p4est_ghost_t* &ghost_out)
{
  phi_out           = return_phi_vector();
  nodes_out         = return_nodes();
  ghost_out         = return_ghost();
}
my_p4est_biomolecules_t::~my_p4est_biomolecules_t()
{
  PetscErrorCode ierr;
  if(nodes != NULL)         { p4est_nodes_destroy(nodes);      nodes = NULL; }
  if(ghost != NULL)         { p4est_ghost_destroy(ghost);      ghost = NULL; }
  if(phi != NULL)           { ierr = VecDestroy(phi);          phi = NULL;          CHKERRXX(ierr);}
  if(inner_domain != NULL)  { ierr = VecDestroy(inner_domain); inner_domain = NULL; CHKERRXX(ierr);  }
  if(hierarchy != NULL)     { delete hierarchy;                hierarchy = NULL; }
  if(neighbors != NULL)     { delete neighbors;                neighbors = NULL; }
  if(ls != NULL)            { delete ls;                       ls = NULL; }
  if(timer != NULL)
  {
    timer->stop(); timer->read_duration(true);
    delete timer;
    timer = NULL;
  }
}
void my_p4est_biomolecules_t::add_single_molecule(const string& file_path, const vector<double> *centroid, vector<double> *angles, const double *angstrom_to_domain)
{
  P4EST_ASSERT((centroid == NULL) || (centroid->size()==P4EST_DIM));
  P4EST_ASSERT((angles ==NULL) || (angles->size() == nangle_per_mol));
  add_single_molecule(file_path, (centroid != NULL)? centroid->data(): NULL, (angles != NULL)? angles->data(): NULL, angstrom_to_domain);
}
void my_p4est_biomolecules_t::rescale_all_molecules(const double& new_scaling_factor)
{
  vector<double> current_centroids;
  get_vector_of_current_centroids(current_centroids);
  rescale_all_molecules(new_scaling_factor, &current_centroids);
}
void my_p4est_biomolecules_t::rescale_all_molecules(const double& new_scaling_factor, const vector<double>* centroids)
{
  P4EST_ASSERT(new_scaling_factor >= 0.0);
  P4EST_ASSERT(centroids == NULL || centroids->size() == P4EST_DIM*nmol());
  angstrom_to_domain = new_scaling_factor;
  rescale_all_molecules((centroids != NULL)?centroids->data():NULL);
}
void my_p4est_biomolecules_t::set_biggest_bounding_box(const double& biggest_cube_side_length_to_min_domain_size)
{
  P4EST_ASSERT(nmol()!=0);
  rescale_all_molecules(bio_molecules[index_of_biggest_mol].calculate_scaling_factor(biggest_cube_side_length_to_min_domain_size));
}
void my_p4est_biomolecules_t::print_summary() const
{
  if(log_file == NULL)
    return;
  PetscErrorCode ierr;
  for (size_t k = 0; k < nmol(); ++k) {
    const molecule& mol = bio_molecules[k];
    string message = "Molecule %d is located at x = " + to_string(*mol.get_centroid()) + ", y = " + to_string(*(mol.get_centroid()+1)) +
    #ifdef P4_TO_P8
        ", z = " + to_string(*(mol.get_centroid()+2)) +
    #endif
        ", is bounded by a cube of side length %.5f, and contains %d atoms, %d of which being charged. \n" ;
    ierr = PetscFPrintf(p4est->mpicomm, log_file, message.c_str(), k, mol.get_side_length_of_bounding_cube(), mol.get_number_of_atoms(), mol.get_number_of_charged_atoms()); CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(p4est->mpicomm, log_file, " \n \n "); CHKERRXX(ierr);
}

void my_p4est_biomolecules_t::set_grid_and_surface_parameters(const int &lmin, const int &lmax, const double &lip_, const double &rp_, const int& ooa_)
{
  // splitting criterion first
  int need_to_reset_p4est = (int) parameters.set_splitting_criterion(lmin, lmax, lip_);
  //probe_radius
  P4EST_ASSERT(all_molecules_are_scaled_consistently()); // the probe radius cannot be (re)set if molecules are not scaled consistently

  need_to_reset_p4est = need_to_reset_p4est || parameters.set_probe_radius(angstrom_to_domain*rp_);
  // order of accuracy (for thickess of accuracy layer)
  need_to_reset_p4est = need_to_reset_p4est || parameters.set_OOA(ooa_);
  if (need_to_reset_p4est)
    reset_p4est();
}

void my_p4est_biomolecules_t::set_splitting_criterion(const int& lmin, const int& lmax, const double& lip_)
{
  set_grid_and_surface_parameters(lmin, lmax, lip_, parameters.probe_radius(), parameters.order_of_accuracy());
}
void my_p4est_biomolecules_t::set_probe_radius(const double& rp)
{
  set_grid_and_surface_parameters(parameters.lmin(), parameters.lmax(), parameters.lip(), rp, parameters.order_of_accuracy());
}
void my_p4est_biomolecules_t::set_order_of_accuracy(const int& ooa)
{
  set_grid_and_surface_parameters(parameters.lmin(), parameters.lmax(), parameters.lip(), parameters.probe_radius(), ooa);
}
double my_p4est_biomolecules_t::get_largest_radius_of_all() const
{
#ifdef DEBUG
  check_validity_of_vector_of_mol();
#endif
  double largest_largest_radius = -DBL_MAX;
  for (size_t k = 0; k < nmol(); ++k)
    largest_largest_radius = MAX(largest_largest_radius, bio_molecules[k].get_largest_radius());
  return largest_largest_radius;
}
void my_p4est_biomolecules_t::reset_p4est()
{
  p4est_t* new_p4est = my_p4est_new(p4est->mpicomm, p4est->connectivity, 0, NULL, NULL);
  p4est_destroy(p4est);
  p4est = new_p4est;
  return;
}
void my_p4est_biomolecules_t::update_max_level()
{
  global_max_level = 0;
  for (p4est_topidx_t k = p4est->first_local_tree; k <= p4est->last_local_tree; ++k) {
    p4est_tree_t* tree_k = p4est_tree_array_index(p4est->trees, k);
    global_max_level = MAX(global_max_level, tree_k->maxlevel);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &global_max_level, 1, MPI_INT8_T, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_biomolecules_t::add_reduced_list(p4est_topidx_t which_tree, p4est_quadrant_t *quad, reduced_list_ptr& parent_list_ptr, const bool& need_exact_phi)
{
  quad->p.user_long = (p4est->mpisize>1)? ((((long) p4est->mpirank) << (8*sizeof(long) - rank_encoding)) + reduced_lists.size()) : reduced_lists.size();

  if(parent_list_ptr != NULL && (*parent_list_ptr).size() ==1)
  {
    reduced_lists.push_back(parent_list_ptr);
    return;
  }

  const double *xyz_min_tree    = (p4est->connectivity->vertices + (3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree]));
  const double quad_rel_size    = ((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN);
  const double dxyz[P4EST_DIM]  = { DIM(parameters.tree_dim(0)*quad_rel_size, parameters.tree_dim(1)*quad_rel_size, parameters.tree_dim(2)*quad_rel_size) };
  const double xyz_c[P4EST_DIM] = { DIM(xyz_min_tree[0] + parameters.tree_dim(0)*(((double) (quad->x))/((double)P4EST_ROOT_LEN)) + 0.5*dxyz[0],
                                    xyz_min_tree[1] + parameters.tree_dim(1)*(((double) (quad->y))/((double)P4EST_ROOT_LEN)) + 0.5*dxyz[1],
                                    xyz_min_tree[2] + parameters.tree_dim(2)*(((double) (quad->z))/((double)P4EST_ROOT_LEN)) + 0.5*dxyz[2]) };
  if(parent_list_ptr == NULL)
  {
    parent_list_ptr = reduced_list_ptr(new reduced_list);
    reduced_list& par_list = *parent_list_ptr;
    for (size_t mol_idx = 0; mol_idx < nmol(); ++mol_idx) {
      const molecule& mol = bio_molecules[mol_idx];
      const double* mol_centroid = mol.get_centroid();
      const double* mol_bounding_box = mol.get_side_length_of_bounding_box();
      const double xyz_box[P4EST_DIM]  = { DIM(
                                           ((xyz_c[0] > mol_centroid[0]+0.5*mol_bounding_box[0])? (mol_centroid[0]+0.5*mol_bounding_box[0]) : ((xyz_c[0] < mol_centroid[0]-0.5*mol_bounding_box[0])? (mol_centroid[0]-0.5*mol_bounding_box[0]) : xyz_c[0])),
                                           ((xyz_c[1] > mol_centroid[1]+0.5*mol_bounding_box[1])? (mol_centroid[1]+0.5*mol_bounding_box[1]) : ((xyz_c[1] < mol_centroid[1]-0.5*mol_bounding_box[1])? (mol_centroid[1]-0.5*mol_bounding_box[1]) : xyz_c[1])),
                                           ((xyz_c[2] > mol_centroid[2]+0.5*mol_bounding_box[2])? (mol_centroid[2]+0.5*mol_bounding_box[2]) : ((xyz_c[2] < mol_centroid[2]-0.5*mol_bounding_box[2])? (mol_centroid[2]-0.5*mol_bounding_box[2]) : xyz_c[2]))) };
      const double xyz_quad[P4EST_DIM] = { DIM(
                                           ((xyz_box[0] > xyz_c[0]+0.5*dxyz[0])? (xyz_c[0]+0.5*dxyz[0]) : ((xyz_box[0] < xyz_c[0]-0.5*dxyz[0])? (xyz_c[0]-0.5*dxyz[0]) : xyz_box[0])),
                                           ((xyz_box[1] > xyz_c[1]+0.5*dxyz[1])? (xyz_c[1]+0.5*dxyz[1]) : ((xyz_box[1] < xyz_c[1]-0.5*dxyz[1])? (xyz_c[1]-0.5*dxyz[1]) : xyz_box[1])),
                                           ((xyz_box[2] > xyz_c[2]+0.5*dxyz[2])? (xyz_c[2]+0.5*dxyz[2]) : ((xyz_box[2] < xyz_c[2]-0.5*dxyz[2])? (xyz_c[2]-0.5*dxyz[2]) : xyz_box[2]))) };
      if(sqrt(SUMD(SQR(xyz_box[0]-xyz_quad[0]), SQR(xyz_box[1]-xyz_quad[1]), SQR(xyz_box[2]-xyz_quad[2]))) <= MAX(parameters.layer_thickness() + (need_exact_phi? parameters.probe_radius(): 0.0), 0.5*parameters.lip()*parameters.tree_diag()*quad_rel_size - parameters.probe_radius())) // there might be relevant atoms in the molecule to consider
      {
        size_t former_nb_atoms = par_list.size();
        par_list.atom_global_idx.resize(former_nb_atoms+mol.get_number_of_atoms());
        for (int jj = 0; jj < mol.get_number_of_atoms(); ++jj)
          par_list.atom_global_idx[former_nb_atoms+jj] = atom_index_offset[mol_idx] + jj;
      }
    }
    if(par_list.size() == 0)
    {
      par_list.atom_global_idx.push_back(0);
      reduced_lists.push_back(parent_list_ptr);
      return;
    }
  }

  double  threshold_dist = MAX(parameters.layer_thickness() + (need_exact_phi? parameters.probe_radius(): 0.0), 0.25*parameters.lip()*parameters.tree_diag()*quad_rel_size - parameters.probe_radius());
  const reduced_list& parent_list = (*parent_list_ptr);

  int     nb_added_global_idx = 0;
  int     global_idx_maximizing_atom = -1; // absurd initialization
  double  max_phi_sas_to_vertices = -DBL_MAX;
  size_t  mol_index = 0;

  reduced_list_ptr child_list_ptr(new reduced_list);
  reduced_list& child_list = *child_list_ptr;

  for (size_t k = 0; k < parent_list.size(); ++k) {
    int global_atom_idx = parent_list.atom_global_idx[k];
    const Atom* a = get_atom(global_atom_idx, mol_index);
    double d = parameters.probe_radius() + a->max_phi_vdW_in_quad(xyz_c, dxyz);
    if(d >= -threshold_dist - EPS*parameters.probe_radius())
    {
      child_list.atom_global_idx.push_back(global_atom_idx);
      nb_added_global_idx++;
    }
    if(nb_added_global_idx == 0 && d >= MAX(parameters.probe_radius()-0.5*parameters.lip()*parameters.tree_diag()*quad_rel_size, max_phi_sas_to_vertices))
    {
      double phi_sas_vertex;
      for (unsigned char ii = 0; ii < 2; ++ii)
        for (unsigned char jj = 0; jj < 2; ++jj)
#ifdef P4_TO_P8
          for (unsigned char kk = 0; kk < 2; ++kk)
#endif
          {
            phi_sas_vertex = parameters.probe_radius() + a->dist_to_vdW_surface(DIM(xyz_c[0] + ((double) ii - 0.5)*dxyz[0], xyz_c[1] + ((double) jj - 0.5)*dxyz[1], xyz_c[2] + ((double) kk - 0.5)*dxyz[2]));
            if(phi_sas_vertex > max_phi_sas_to_vertices)
            {
              max_phi_sas_to_vertices = phi_sas_vertex;
              global_idx_maximizing_atom = global_atom_idx;
            }
          }
    }
  }

  if(nb_added_global_idx == 0)
    child_list.atom_global_idx.push_back((global_idx_maximizing_atom>=0)?global_idx_maximizing_atom:parent_list.atom_global_idx[0]);

  reduced_lists.push_back(child_list_ptr);

  return;
}

double my_p4est_biomolecules_t::operator ()(DIM(const double &x, const double &y, const double &z)) const
{
  double phi = -DBL_MAX;
  for (size_t k = 0; k < nmol(); ++k)
    phi = MAX(phi, bio_molecules[k](DIM(x, y, z)));

  phi += parameters.probe_radius();
  return phi;
}


double my_p4est_biomolecules_t::reduced_operator(const double *xyz, const int& reduced_list_idx, const bool need_exact_value, const bool last_stage) const
{
  bool get_better_phi = false;
  double phi = -DBL_MAX, tmp;
  const double zero_threshold = parameters.layer_thickness()*((parameters.order_of_accuracy()==1)? 0.01:(1.0/(1<<parameters.lmax())));
  size_t mol_idx = 0;
  const Atom *atom_i = NULL;
  size_t reduced_index_of_atom_i = SIZE_MAX;

  const reduced_list& r_list = *(reduced_lists[reduced_list_idx]);
  for (size_t k = 0; k < r_list.size(); ++k) {
    const Atom* a = get_atom(r_list.atom_global_idx[k], mol_idx);
    tmp = a->dist_to_vdW_surface(xyz);
    if(tmp > phi)
    {
      phi     = tmp;
      if(need_exact_value && (global_max_level >= parameters.threshold_level()))
      {
        atom_i  = a;
        reduced_index_of_atom_i = k;
      }
    }
  }
  phi += parameters.probe_radius();
  if(need_exact_value && global_max_level >= parameters.threshold_level() && 0.0 <= phi && phi <= parameters.probe_radius() + parameters.layer_thickness() + zero_threshold)
  {
    double dist_xyz_to_c_i = 0.0;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      dist_xyz_to_c_i += SQR(xyz[dir] - atom_i->xyz_c[dir]);
    dist_xyz_to_c_i = sqrt(dist_xyz_to_c_i);

    if(dist_xyz_to_c_i > zero_threshold)
    {
      double xyz_proj_i[P4EST_DIM]; // projection on \delta B(c_i, r_p + ri)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        xyz_proj_i[dir] = xyz[dir] + (xyz[dir] - atom_i->xyz_c[dir])*phi/dist_xyz_to_c_i;
      tmp = -DBL_MAX;
      mol_idx = 0;
      get_better_phi = false;
      for (size_t k = 0; k < r_list.size(); ++k) {
        if(k == reduced_index_of_atom_i)
          continue;
        const Atom* a = get_atom(r_list.atom_global_idx[k], mol_idx);
        tmp = MAX(tmp, a->dist_to_vdW_surface(xyz_proj_i) + parameters.probe_radius());
        if(tmp > zero_threshold)
        {
          get_better_phi = true;
          break;
        }
      }
    }
  }
  if(get_better_phi && !last_stage)
    phi = -1.5*parameters.tree_diag();

  return phi;
}

double my_p4est_biomolecules_t::better_distance(const double *xyz, const int& reduced_list_idx, double* kink_point) const
{
  double phi  = -DBL_MAX, tmp;
  size_t mol_idx = 0;

  const double zero_threshold = parameters.layer_thickness()*((parameters.order_of_accuracy()==1)? 0.01:(1.0/(1<<parameters.lmax())));
  vector<sorted_atom> sorted_atoms;
  const reduced_list& r_list = *(reduced_lists[reduced_list_idx]);
  sorted_atoms.resize(r_list.size());

  for (size_t k = 0; k < r_list.size(); ++k) {
    const Atom* a = get_atom(r_list.atom_global_idx[k], mol_idx);
    tmp = a->dist_to_vdW_surface(xyz) + parameters.probe_radius();
    phi = MAX(tmp, phi);
    sorted_atoms[k].global_atom_idx       = r_list.atom_global_idx[k];
    sorted_atoms[k].mol_idx               = mol_idx;
    sorted_atoms[k].distance_from_xyz     = tmp;
    sorted_atoms[k].distance_from_xyz_i   = 0.0;
    sorted_atoms[k].distance_from_graal   = 0.0;
  }

  if(r_list.size() == 1)
    return phi;

  const Atom *atom_i = NULL;
  const Atom* atom_j = NULL;
#ifdef P4_TO_P8
  const Atom* atom_k = NULL;
#endif
  double distance_to_kink = 0.0;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    distance_to_kink += SQR(xyz[dim] - kink_point[dim]);
  distance_to_kink = sqrt(distance_to_kink);
  double graal_point[P4EST_DIM];
  double closest_graal[P4EST_DIM];
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    closest_graal[dim] = DBL_MAX;
  double distance_to_closest_graal = parameters.tree_diag(); //1.5*(get_largest_radius_of_all()+parameters.probe_radius());
  bool graal_point_is_valid = false;

  for (size_t ii = 0; ii < sorted_atoms.size(); ++ii) {
    sort(sorted_atoms.begin(), sorted_atoms.end(), [](sorted_atom atom_a, sorted_atom atom_b){
      return (atom_a.distance_from_xyz > atom_b.distance_from_xyz);
    });
    atom_i = get_atom(sorted_atoms[ii].global_atom_idx, sorted_atoms[ii].mol_idx);
    if(sorted_atoms[ii].distance_from_xyz < ((parameters.probe_radius() + atom_i->r_vdw)*(1.0-sqrt(1.0 + SQR((parameters.probe_radius() + parameters.layer_thickness())/(parameters.probe_radius() + atom_i->r_vdw))))))
      continue; // might be the right atom but its irrelevant since the point is farther than parameters.layer_thickness() away from \Gamma_{\SES} in this case
    if(fabs(atom_i->dist_to_vdW_surface(xyz) + parameters.probe_radius()) > MIN(distance_to_kink, distance_to_closest_graal))
      continue;

    double dist_xyz_to_c_i = 0.0;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      dist_xyz_to_c_i += SQR(xyz[dir] - atom_i->xyz_c[dir]);
    dist_xyz_to_c_i = sqrt(dist_xyz_to_c_i);
    double xyz_proj_i[P4EST_DIM]; // projection on \delta B(c_i, r_p + ri)
    if(dist_xyz_to_c_i > zero_threshold)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        xyz_proj_i[dir] = xyz[dir] + (xyz[dir] - atom_i->xyz_c[dir])*sorted_atoms[ii].distance_from_xyz/dist_xyz_to_c_i;
    else
    {
      double xyz_to_kink[P4EST_DIM];
      double xyz_to_kink_norm = 0.0;
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      {
        xyz_to_kink[dim] = kink_point[dim] - xyz[dim];
        xyz_to_kink_norm += SQR(xyz_to_kink[dim]);
      }
      xyz_to_kink_norm = sqrt(xyz_to_kink_norm);
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        xyz_proj_i[dim] = xyz[dim] + (atom_i->r_vdw + parameters.probe_radius())*xyz_to_kink[dim]/xyz_to_kink_norm;
    }
    double distance_to_projected_point_i = sorted_atoms[ii].distance_from_xyz;

    graal_point_is_valid = true;
    double phi_sas_analytical_at_xyz_i = -DBL_MAX;
    for (size_t jj = 0; jj < sorted_atoms.size(); ++jj){
      const Atom* a = get_atom(sorted_atoms[jj].global_atom_idx, sorted_atoms[jj].mol_idx);
      sorted_atoms[jj].distance_from_xyz_i   = a->dist_to_vdW_surface(xyz_proj_i) + parameters.probe_radius();
      phi_sas_analytical_at_xyz_i = MAX(phi_sas_analytical_at_xyz_i, sorted_atoms[jj].distance_from_xyz_i);
      graal_point_is_valid = graal_point_is_valid && (sorted_atoms[jj].distance_from_xyz_i < zero_threshold);
      sorted_atoms[jj].distance_from_graal   = 0.0;
    }
    graal_point_is_valid = graal_point_is_valid && (fabs(phi_sas_analytical_at_xyz_i) < zero_threshold);
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      graal_point[dim] = xyz_proj_i[dim];
    double distance_xyz_to_graal = distance_to_projected_point_i;

    if(!graal_point_is_valid)
    {
      for (size_t jj = 0; jj < sorted_atoms.size(); ++jj){
        sort(sorted_atoms.begin(), sorted_atoms.end(), [](sorted_atom atom_a, sorted_atom atom_b){
          return (atom_a.distance_from_xyz_i > atom_b.distance_from_xyz_i);
        });

        atom_j = get_atom(sorted_atoms[jj].global_atom_idx, sorted_atoms[jj].mol_idx);
        if(fabs(atom_j->dist_to_vdW_surface(xyz) + parameters.probe_radius()) > MIN(distance_to_kink, distance_to_closest_graal))
          continue;
        const double alpha = sqrt(SUMD(SQR(atom_i->xyz_c[0] - atom_j->xyz_c[0]), SQR(atom_i->xyz_c[1] - atom_j->xyz_c[1]), SQR(atom_i->xyz_c[2] - atom_j->xyz_c[2])));
        const double lambda = 0.5 + 0.5*(SQR(parameters.probe_radius() + atom_j->r_vdw) - SQR(parameters.probe_radius() + atom_i->r_vdw))/SQR(alpha);
        double circle_center[P4EST_DIM];
        double normal_vector[P4EST_DIM];
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
          normal_vector[dir]    = (atom_j->xyz_c[dir] - atom_i->xyz_c[dir])/alpha;
          circle_center[dir]    = lambda*atom_i->xyz_c[dir] + (1.0-lambda)*atom_j->xyz_c[dir];
        }

        const double circle_radius  = sqrt(0.5*(SQR(parameters.probe_radius() + atom_i->r_vdw) + SQR(parameters.probe_radius() + atom_j->r_vdw))
                                           - 0.25*SQR(alpha)
                                           - 0.25*(SQR(SQR(parameters.probe_radius() + atom_j->r_vdw) - SQR(parameters.probe_radius() + atom_i->r_vdw)))/SQR(alpha));

        double mu_vector[P4EST_DIM];
        double projection_along_normal = 0.0;
        for (unsigned char k = 0; k < P4EST_DIM; ++k) {
          mu_vector[k] = xyz[k] - circle_center[k];
          projection_along_normal += normal_vector[k]*mu_vector[k];
        }
        double norm_of_mu = 0.0;
        for (unsigned char k = 0; k < P4EST_DIM; ++k)
        {
          mu_vector[k] = mu_vector[k] - projection_along_normal*normal_vector[k];
          norm_of_mu += SQR(mu_vector[k]);
        }
        norm_of_mu = sqrt(norm_of_mu);
        if(norm_of_mu > zero_threshold)
        {
          for (unsigned char k = 0; k < P4EST_DIM; ++k)
            mu_vector[k] /= norm_of_mu;
        }
        else
        {
          // choose arbitrarily
#ifndef P4_TO_P8
          mu_vector[0] =  normal_vector[1];
          mu_vector[1] = -normal_vector[0];
#else
          double nnn;
          if(fabs(normal_vector[1]) >= fabs(normal_vector[0]))
          {
            nnn = sqrt(SQR(normal_vector[1]) + SQR(normal_vector[2]));
            mu_vector[0] = 0.0;
            mu_vector[1] = -normal_vector[2]/nnn;
            mu_vector[2] = normal_vector[1]/nnn;
          }
          else
          {
            nnn = sqrt(SQR(normal_vector[0]) + SQR(normal_vector[2]));
            mu_vector[0] = normal_vector[2]/nnn;
            mu_vector[1] = 0.0;
            mu_vector[2] = -normal_vector[0]/nnn;
          }
#endif
        }
        norm_of_mu = 1.0;
#ifdef P4_TO_P8
        double nu_vector[P4EST_DIM];
        nu_vector[0] = normal_vector[1]*mu_vector[2] - normal_vector[2]*mu_vector[1];
        nu_vector[1] = normal_vector[2]*mu_vector[0] - normal_vector[0]*mu_vector[2];
        nu_vector[2] = normal_vector[0]*mu_vector[1] - normal_vector[1]*mu_vector[0];
#endif
        //      distance_xyz_to_graal = 0.0;
        double theta_angles[P4EST_DIM-1];
        double graal_theta;
        theta_angles[0] = 0.0; // positive
#ifdef P4_TO_P8
        bool take_first_angle = true;
        theta_angles[1] = 0.0; // negative
        double cc[P4EST_DIM]; // vector pointing from circle_center to the center of the intersection
        // between the other ball and the plane normal to the circle under investigation
        double cc_dot_n = 0.0, norm_of_cc = 0.0;
        double cc_dot_mu = 0.0, cc_dot_nu = 0.0;
        double radius_of_other_circle = 0.0;
        double angle_beta, angle_alpha;
#endif
        bool set_of_candidates_is_empty = false;

        while(!graal_point_is_valid && !set_of_candidates_is_empty)
        {
#ifdef P4_TO_P8
          set_of_candidates_is_empty = (fabs(theta_angles[0]-theta_angles[1] - 2.0*M_PI) <= EPS*M_PI);
          take_first_angle = (fabs(theta_angles[0]) <= fabs(theta_angles[1]));
          graal_theta = theta_angles[(take_first_angle?0:1)];
#else
          set_of_candidates_is_empty = (fabs(theta_angles[0] - M_PI) <= EPS*M_PI);
          graal_theta = theta_angles[0];
#endif
          distance_xyz_to_graal = 0.0;
          for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
            graal_point[dim] = circle_center[dim] + cos(graal_theta)*circle_radius*mu_vector[dim] ONLY3D(+ sin(graal_theta)*circle_radius*nu_vector[dim]);
            distance_xyz_to_graal += SQR(xyz[dim] - graal_point[dim]);
          }
          distance_xyz_to_graal = sqrt(distance_xyz_to_graal);
          if( distance_xyz_to_graal > MIN(distance_to_closest_graal, distance_to_kink))
            break;
          for (size_t kk = 0; kk < sorted_atoms.size(); ++kk){
            const Atom* a = get_atom(sorted_atoms[kk].global_atom_idx, sorted_atoms[kk].mol_idx);
            sorted_atoms[kk].distance_from_graal  = a->dist_to_vdW_surface(graal_point) + parameters.probe_radius();
          }
          sort(sorted_atoms.begin(), sorted_atoms.end(), [](sorted_atom atom_a, sorted_atom atom_b){
            return (atom_a.distance_from_graal > atom_b.distance_from_graal);
          });
          graal_point_is_valid = (fabs(sorted_atoms[0].distance_from_graal) < zero_threshold);
          if(!graal_point_is_valid)
          {
#ifndef P4_TO_P8
            theta_angles[0] = M_PI;
#else
            atom_k = get_atom(sorted_atoms[0].global_atom_idx, sorted_atoms[0].mol_idx);
            cc_dot_n = 0.0;
            for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
              cc[dir] = atom_k->xyz_c[dir] - circle_center[dir];
              cc_dot_n += cc[dir]*normal_vector[dir];
            }
            radius_of_other_circle = sqrt(SQR(parameters.probe_radius() + atom_k->r_vdw) - SQR(cc_dot_n));
            norm_of_cc = 0.0;
            cc_dot_mu = 0.0;
            cc_dot_nu = 0.0;
            for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
            {
              cc[dim] = cc[dim] - cc_dot_n*normal_vector[dim];
              norm_of_cc += SQR(cc[dim]);
              cc_dot_mu += cc[dim]*mu_vector[dim];
              cc_dot_nu += cc[dim]*nu_vector[dim];
            }
            norm_of_cc = sqrt(norm_of_cc);
            if(radius_of_other_circle >= norm_of_cc + circle_radius) // no intersection, investigated circle entirely contained
            {
              theta_angles[0] = M_PI;
              theta_angles[1] = -M_PI;
              set_of_candidates_is_empty = true;
            }
            else
            {
              angle_beta  = acos(cc_dot_mu/norm_of_cc)*((cc_dot_nu >= 0)? +1.0:-1.0);
              angle_alpha = acos((SQR(circle_radius) + SQR(norm_of_cc) - SQR(radius_of_other_circle))/(2.0*circle_radius*norm_of_cc));
              theta_angles[0] = MAX(theta_angles[0], MIN(angle_beta + angle_alpha, 2.0*M_PI + theta_angles[1]));
              theta_angles[1] = MIN(theta_angles[1], MAX(angle_beta - angle_alpha, - (2.0*M_PI - theta_angles[0])));
            }
#endif
          }
        }
        if(graal_point_is_valid && distance_xyz_to_graal < distance_to_closest_graal)
        {
          distance_to_closest_graal = distance_xyz_to_graal;
          for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
            closest_graal[dim] = graal_point[dim];
        }
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          graal_point[dim] = xyz_proj_i[dim];
        graal_point_is_valid = false;
        distance_xyz_to_graal = distance_to_projected_point_i;
      }
    }
    else
    {
      if(distance_xyz_to_graal < distance_to_closest_graal)
      {
        distance_to_closest_graal = distance_xyz_to_graal;
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          closest_graal[dim] = graal_point[dim];
      }
    }
  }
  if(distance_to_closest_graal < distance_to_kink && distance_to_closest_graal < parameters.probe_radius())
  {
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      kink_point[dim] = closest_graal[dim];
  }
  return MIN(distance_to_closest_graal, distance_to_kink);
}

void my_p4est_biomolecules_t::partition_uniformly(const bool export_cavities, const bool build_ghost)
{
  P4EST_ASSERT(p4est != NULL);
  P4EST_ASSERT(phi==NULL || nodes != NULL); // this function requires valid nodes if the levelset function is defined
#ifdef DEBUG
  if(phi != NULL)
  {
    p4est_gloidx_t total_nb_nodes = 0;
    for (int rank = 0; rank < p4est->mpisize; ++rank)
      total_nb_nodes += nodes->global_owned_indeps[rank];
    PetscInt size;
    PetscErrorCode ierr = VecGetSize(phi, &size); CHKERRXX(ierr);
    P4EST_ASSERT(size == total_nb_nodes); // the size of the levelset vector is not equal to the total number of nodes
  }
#endif
  my_p4est_partition(p4est, P4EST_FALSE, NULL); // do not allow for coarsening, it's the last stage...

  if(ghost != NULL)
  {
    p4est_ghost_destroy(ghost); ghost = NULL;
  }
  if(build_ghost)
    ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  if(hierarchy != NULL)
    hierarchy->update(p4est, ghost);
  if(nodes != NULL)
  {
    if(phi == NULL)
    {
      p4est_nodes_destroy(nodes);
      nodes = my_p4est_nodes_new(p4est, ghost);
    }
    else
    {
      PetscInt global_idx_offset = 0;
      for (int rank = 0; rank < p4est->mpirank; ++rank)
        global_idx_offset += nodes->global_owned_indeps[rank];
      vector<PetscInt> global_indices_of_known_values; global_indices_of_known_values.resize(nodes->num_owned_indeps);
      for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k)
        global_indices_of_known_values[k] = global_idx_offset + k;

      p4est_nodes_destroy(nodes);
      nodes = my_p4est_nodes_new(p4est, ghost);

      Vec old_phi = phi; // no cost, those are pointers...
      PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
      ierr = VecScatterAllToSome(p4est->mpicomm, old_phi, phi, global_indices_of_known_values, true); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL)
      {
        Vec old_inner_domain = inner_domain;
        ierr = VecCreateGhostNodes(p4est, nodes, &inner_domain); CHKERRXX(ierr);
        ierr = VecScatterAllToSome(p4est->mpicomm, old_inner_domain, inner_domain, global_indices_of_known_values, true); CHKERRXX(ierr);
      }
    }
    if(build_ghost && neighbors != NULL)
    {
      neighbors->update(hierarchy, nodes);
      ls->update(neighbors);
    }
  }
}

int my_p4est_biomolecules_t::partition_weight_for_enforcing_min_level(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  (void) which_tree;
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  if(quadrant->level < biomol->parameters.lmin())
    return (1<<(P4EST_DIM*(biomol->parameters.lmin()-quadrant->level)));
  else
    return 1;
}

void my_p4est_biomolecules_t::enforce_min_level(bool export_cavities)
{
  P4EST_ASSERT(p4est != NULL);
  P4EST_ASSERT(phi != NULL && nodes  != NULL); // this function requires valid nodes and a valid node-sampled levelset function
#ifdef DEBUG
  if(phi != NULL)
  {
    p4est_gloidx_t total_nb_nodes = 0;
    for (int rank = 0; rank < p4est->mpisize; ++rank)
      total_nb_nodes += nodes->global_owned_indeps[rank];
    PetscInt size;
    PetscErrorCode ierr = VecGetSize(phi, &size); CHKERRXX(ierr);
    P4EST_ASSERT(size == total_nb_nodes);
  }
#endif
  P4EST_ASSERT((export_cavities && inner_domain != NULL) || (!export_cavities && inner_domain == NULL));
  if(ghost != NULL){
    p4est_ghost_destroy(ghost); ghost = NULL;}
  vector<PetscInt> global_indices;
  global_indices.resize(nodes->num_owned_indeps);
  PetscInt global_idx_offset = 0;
  for (int rank = 0; rank < p4est->mpirank; ++rank)
    global_idx_offset += nodes->global_owned_indeps[rank];
  for (int k = 0; k < nodes->num_owned_indeps; ++k)
    global_indices[k] = global_idx_offset + k;
  p4est_nodes_destroy(nodes); nodes = NULL;
  my_p4est_partition(p4est, P4EST_FALSE, my_p4est_biomolecules_t::partition_weight_for_enforcing_min_level);
  // no need to rebuild the ghost right NOW, this is a very local process,
  // build them afterwards if needed
  nodes = my_p4est_nodes_new(p4est, ghost);

  // scatter the vector(s) to the new layout
  // store old vectors of phi
  PetscErrorCode ierr;
  VecScatter ctx_phi, ctx_inner_domain;
  Vec former_phi = phi, former_phi_loc;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi);                                                                       CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(former_phi, &former_phi_loc);                                                             CHKERRXX(ierr);
  ierr = VecScatterAllToSomeCreate(p4est->mpicomm, former_phi_loc, phi, global_indices, &ctx_phi);                      CHKERRXX(ierr);
  ierr = VecScatterAllToSomeBegin(ctx_phi, former_phi_loc, phi);                                                        CHKERRXX(ierr);
  Vec former_inner_domain, former_inner_domain_loc;
  if(export_cavities)
  {
    former_inner_domain = inner_domain;
    ierr = VecCreateGhostNodes(p4est, nodes, &inner_domain);                                                            CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(former_inner_domain, &former_inner_domain_loc);                                         CHKERRXX(ierr);
    ierr = VecScatterAllToSomeCreate(p4est->mpicomm, former_inner_domain_loc, phi, global_indices, &ctx_inner_domain);  CHKERRXX(ierr);
    ierr = VecScatterAllToSomeBegin(ctx_inner_domain, former_inner_domain_loc, inner_domain);                           CHKERRXX(ierr);
  }
  // assign current local index to quads'p.user_long to enable easy linear interpolation hereafter
  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad  = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      quad->p.user_long = q + tree_k->quadrants_offset;
    }
  }
  ierr = VecScatterAllToSomeEnd(ctx_phi, former_phi_loc, phi);                                                          CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);                                                              CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(former_phi, &former_phi_loc);                                                         CHKERRXX(ierr);
  ierr = VecScatterDestroy(ctx_phi);                                                                                    CHKERRXX(ierr);
  ierr = VecDestroy(former_phi);                                                                                        CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);                                                                CHKERRXX(ierr);
  // do the same with cavities if needed

  if(export_cavities){
    ierr = VecScatterAllToSomeEnd(ctx_inner_domain, former_inner_domain_loc, inner_domain);                             CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                                   CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(former_inner_domain, &former_inner_domain_loc);                                     CHKERRXX(ierr);
    ierr = VecScatterDestroy(ctx_inner_domain);                                                                         CHKERRXX(ierr);
    ierr = VecDestroy(former_inner_domain);                                                                             CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                                     CHKERRXX(ierr);
  }

  // impose the minimum level
  p4est_nodes_t* coarse_nodes = nodes;
  Vec coarse_phi              = phi;
  const double* coarse_phi_read_only_p = NULL, * coarse_inner_domain_read_only_p = NULL;
  ierr    = VecGetArrayRead(coarse_phi, &coarse_phi_read_only_p);                                                       CHKERRXX(ierr);
  Vec coarse_inner_domain     = NULL;
  if(export_cavities)
  {
    coarse_inner_domain       = inner_domain;
    ierr  = VecGetArrayRead(coarse_inner_domain, &coarse_inner_domain_read_only_p);                                     CHKERRXX(ierr);
  }
  // enforce the minimum level
  p4est_refine_ext(p4est, P4EST_TRUE, -1, my_p4est_biomolecules_t::refine_fn_min_level, NULL, my_p4est_biomolecules_t::replace_fn_min_level);
  // Fill missing data by linear interpolation:
  // create the nodes
  nodes   = my_p4est_nodes_new(p4est, ghost);
  // create the vector(s)
  double *phi_p;
  ierr    = VecCreateGhostNodes(p4est, nodes, &phi);                                                                    CHKERRXX(ierr);
  ierr    = VecGetArray(phi, &phi_p);                                                                                   CHKERRXX(ierr);
  double * inner_domain_p;
  if(export_cavities)
  {
    ierr  = VecCreateGhostNodes(p4est, nodes, &inner_domain);                                                           CHKERRXX(ierr);
    ierr  = VecGetArray(inner_domain, &inner_domain_p);                                                                 CHKERRXX(ierr);
  }

  // set the new layout of phi
  set<p4est_locidx_t> known_fine_indices; known_fine_indices.clear();
  // firt known nodes
  int clamped = 1;
  p4est_locidx_t  fine_idx;
  p4est_locidx_t  coarse_idx = 0;
  p4est_indep_t*  coarse_node = NULL;
  if(coarse_idx < coarse_nodes->num_owned_indeps)
    coarse_node = (p4est_indep_t*) sc_array_index(&coarse_nodes->indep_nodes, coarse_idx);
  for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k)
  {
    p4est_indep_t *node  = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes,k);
    if(coarse_node!= NULL && p4est_node_equal_piggy_fn (node, coarse_node, &clamped))
    {
      if(export_cavities)
        inner_domain_p[k] = coarse_inner_domain_read_only_p[coarse_idx];
      phi_p[k] = coarse_phi_read_only_p[coarse_idx++];
      known_fine_indices.insert(k);
      if(coarse_idx < coarse_nodes->num_owned_indeps)
        coarse_node = (p4est_indep_t*) sc_array_index(&coarse_nodes->indep_nodes, coarse_idx);
      else
        break;
    }
  }

  p4est_locidx_t  coarse_quad_idx = -1; // absurd initialization
  double          coarse_quad_corners[2][P4EST_DIM], fine_xyz[P4EST_DIM]; // back-lower-left, front-upper-right corners of the interpolation cube, point of interest, respectively
  double          coarse_quad_volume = 1.0; // initialization is irrelevant but the compiler complains otherwise

  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad  = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      if(quad->level == parameters.lmin() && coarse_quad_idx != (p4est_locidx_t) quad->p.user_long)
      {
        coarse_quad_idx = (p4est_locidx_t) quad->p.user_long;
        node_xyz_fr_n(coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx + 0], p4est, coarse_nodes, &coarse_quad_corners[0][0]);
        node_xyz_fr_n(coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx + P4EST_CHILDREN-1], p4est, coarse_nodes, &coarse_quad_corners[1][0]);
        coarse_quad_volume = MULTD((coarse_quad_corners[1][0]-coarse_quad_corners[0][0]), (coarse_quad_corners[1][1]-coarse_quad_corners[0][1]), (coarse_quad_corners[1][2]-coarse_quad_corners[0][2]));
      }
      for (unsigned char nn = 0; nn < P4EST_CHILDREN; ++nn) {
        fine_idx = nodes->local_nodes[P4EST_CHILDREN*(tree_k->quadrants_offset+q) + nn];
        if(known_fine_indices.find(fine_idx) == known_fine_indices.end())
        {
          if((int) quad->level > parameters.lmin())
          {
            if(export_cavities)
              inner_domain_p[fine_idx] = coarse_inner_domain_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*quad->p.user_long + nn]];
            phi_p[fine_idx] = coarse_phi_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*quad->p.user_long + nn]];
          }
          else
          {
            phi_p[fine_idx] = 0.0;
            if(export_cavities)
              inner_domain_p[fine_idx] = 0.0;
            node_xyz_fr_n(fine_idx, p4est, nodes, fine_xyz);
            for (unsigned char ii = 0; ii < 2; ++ii)
              for (unsigned char jj = 0; jj < 2; ++jj)
#ifdef P4_TO_P8
                for (unsigned char kk = 0; kk < 2; ++kk)
#endif
                {
                  phi_p[fine_idx] += coarse_phi_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx+ SUMD(ii, 2*jj, 4*kk)]]*fabs(MULTD((fine_xyz[0]-coarse_quad_corners[1-ii][0]), (fine_xyz[1]-coarse_quad_corners[1-jj][1]), (fine_xyz[2]-coarse_quad_corners[1-kk][2])));
                  if(export_cavities)
                    inner_domain_p[fine_idx] += coarse_inner_domain_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx+SUMD(ii, 2*jj, 4*kk)]]*fabs(MULTD((fine_xyz[0]-coarse_quad_corners[1-ii][0]), (fine_xyz[1]-coarse_quad_corners[1-jj][1]), (fine_xyz[2]-coarse_quad_corners[1-kk][2])));
                }
            phi_p[fine_idx] /= coarse_quad_volume;
            if(export_cavities)
              inner_domain_p[fine_idx] /= coarse_quad_volume;
          }
          known_fine_indices.insert(fine_idx);
        }
      }
    }
  }

  P4EST_ASSERT(known_fine_indices.size() == nodes->indep_nodes.elem_count);


  ierr    = VecRestoreArrayRead(coarse_phi, &coarse_phi_read_only_p);                   coarse_phi_read_only_p = NULL;          CHKERRXX(ierr);
  ierr    = VecDestroy(coarse_phi);                                                     coarse_phi = NULL;                      CHKERRXX(ierr);
  ierr    = VecRestoreArray(phi, &phi_p);                                               phi_p = NULL;                           CHKERRXX(ierr);
  // at this stage, all points have been updated, EXCEPT the points that are located exactly at a
  // processor boundary and are T-junctions. Those points have been treated (i.e. points that belong to the side where the biggest quadrants)
  ierr    = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);                                                           CHKERRXX(ierr);
  p4est_nodes_destroy(coarse_nodes);
  if(export_cavities)
  {
    ierr  = VecRestoreArrayRead(coarse_inner_domain, &coarse_inner_domain_read_only_p); coarse_inner_domain_read_only_p = NULL; CHKERRXX(ierr);
    ierr  = VecDestroy(coarse_inner_domain);                                            coarse_inner_domain = NULL;             CHKERRXX(ierr);
    ierr  = VecRestoreArray(inner_domain, &inner_domain_p);                             inner_domain_p = NULL;                  CHKERRXX(ierr);
  }
  ierr    = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);                                                             CHKERRXX(ierr);
}

void my_p4est_biomolecules_t::replace_fn_min_level(p4est_t *forest, p4est_topidx_t which_tree, int num_outgoing, p4est_quadrant_t *outgoing[], int num_incoming, p4est_quadrant_t *incoming[])
{
  (void) forest;
  (void) num_incoming;
  (void) which_tree;
  P4EST_ASSERT(num_outgoing <= 1); // this is coarsening, it should NEVER happend
  (void) num_outgoing;
  /* this is refinement */
  for (unsigned char i = 0; i < P4EST_CHILDREN; ++i)
    incoming[i]->p.user_long = outgoing[0]->p.user_long;
  // copy the local quadrant index of the original parent cell for further linear interpolation
}

p4est_bool_t my_p4est_biomolecules_t::refine_fn_min_level(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  (void) which_tree;
  return ((p4est_bool_t) (quad->level < biomol->parameters.lmin()));
}

double my_p4est_biomolecules_t::inner_box_identifier::operator ()(DIM(const double &x, const double &y, const double &z))const
{
  bool is_in_a_box = false;
  for (size_t mol_idx = 0; !is_in_a_box && (mol_idx < biomol_pointer->nmol()); ++mol_idx)
  {
    const double* mol_centroid    = biomol_pointer->bio_molecules[mol_idx].get_centroid();
    const double box_side_length  = biomol_pointer->bio_molecules[mol_idx].get_side_length_of_bounding_cube();
    is_in_a_box = ANDD((mol_centroid[0] - 0.5*box_side_length <= x && x <= mol_centroid[0] + 0.5*box_side_length),
        (mol_centroid[1] - 0.5*box_side_length <= y && y <= mol_centroid[1] + 0.5*box_side_length),
        (mol_centroid[2] - 0.5*box_side_length <= z && z <= mol_centroid[2] + 0.5*box_side_length));
  }
  return is_in_a_box? 1.0: 0.0;
}

bool my_p4est_biomolecules_t::is_point_in_outer_domain_and_updated(p4est_locidx_t k, quad_neighbor_nodes_of_node_t& qnnn, const my_p4est_node_neighbors_t* ngbd, double* inner_domain_p, const double* phi_read_p) const
{
  // (inner_domain_p[k] < 0.5) is equivalent to "grid node k was already tagged tagged as member of outer domain"
  if(inner_domain_p[k] < 0.5 || phi_read_p[k] >= 0.0)
    return false; // nothing to be done
  ngbd->get_neighbors(k, qnnn);
  inner_domain_p[k] =(
        (inner_domain_p[qnnn.node_m00_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_m00_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_pm] < 0.5)
    #ifdef P4_TO_P8
      ||(inner_domain_p[qnnn.node_m00_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_m00_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_pp] < 0.5)
    #endif
      )? 0.0 : 1.0;
  return inner_domain_p[k] < 0.5;
}

void my_p4est_biomolecules_t::remove_internal_cavities(const bool export_cavities)
{
  PetscErrorCode ierr;
  P4EST_ASSERT(phi != NULL);
  P4EST_ASSERT(nodes != NULL);
#ifdef DEBUG
  PetscInt size;
  p4est_gloidx_t nb_total_nodes = 0;
  for (int proc_rank = 0; proc_rank < p4est->mpisize; ++proc_rank)
    nb_total_nodes += nodes->global_owned_indeps[proc_rank];
  ierr = VecGetSize(phi, &size);                                                                CHKERRXX(ierr);
  P4EST_ASSERT(size == nb_total_nodes);
  P4EST_ASSERT(neighbors->neighbors_are_initialized()); // the neighbors must be initialized
#endif

  if(inner_domain != NULL){
    ierr = VecDestroy(inner_domain);                                                            CHKERRXX(ierr);}
  ierr = VecCreateGhostNodes(neighbors->get_p4est(), neighbors->get_nodes(), &inner_domain);    CHKERRXX(ierr);

  is_point_in_a_bounding_box.biomol_pointer = this;
  double *inner_domain_p;
  ierr = VecGetArray(inner_domain, &inner_domain_p);                                            CHKERRXX(ierr);
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i) {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(i, p4est, nodes, xyz);
    inner_domain_p[i] = is_point_in_a_bounding_box(xyz);
  }

  const double *phi_read_p;
  ierr = VecGetArrayRead(phi, &phi_read_p);                                                     CHKERRXX(ierr);
  size_t layer_size = neighbors->get_layer_size();
  size_t local_size = neighbors->get_local_size();
  quad_neighbor_nodes_of_node_t qnnn;
  int not_converged = 1;
  while (not_converged)
  {
    not_converged = 0;

    // forward
    for (size_t layer_node_idx = 0; layer_node_idx < layer_size; ++layer_node_idx)
    {
      p4est_locidx_t k = neighbors->get_layer_node(layer_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p, phi_read_p) || not_converged;
    }
    ierr = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                 CHKERRXX(ierr);
    for (size_t local_node_idx = 0; local_node_idx < local_size; ++local_node_idx)
    {
      p4est_locidx_t k = neighbors->get_local_node(local_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p, phi_read_p) || not_converged;
    }
    ierr = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                   CHKERRXX(ierr);
    // backward
    for (size_t layer_node_idx = 0; layer_node_idx < layer_size; ++layer_node_idx)
    {
      p4est_locidx_t k = neighbors->get_layer_node(layer_size-1-layer_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p, phi_read_p) || not_converged;
    }
    ierr = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                 CHKERRXX(ierr);
    for (size_t local_node_idx = 0; local_node_idx < local_size; ++local_node_idx)
    {
      p4est_locidx_t k = neighbors->get_local_node(local_size-1-local_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p, phi_read_p) || not_converged;
    }
    ierr = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                   CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &not_converged, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  }
  ierr = VecRestoreArrayRead(phi, &phi_read_p);                                               CHKERRXX(ierr);

  // remove the cavities
  P4EST_ASSERT(inner_domain_p != NULL);
  double *phi_p;
  ierr = VecGetArray(phi, &phi_p);                                                            CHKERRXX(ierr);
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
  {
    inner_domain_p[i] = (phi_p[i] <= 0 && inner_domain_p[i] > 0.5)? 1.0: 0.0;
    if (inner_domain_p[i] > 0.5)// internal cavity
      phi_p[i] = -phi_p[i];
  }

  ierr = VecRestoreArray(inner_domain, &inner_domain_p);                                      CHKERRXX(ierr);
  if(!export_cavities){
    ierr = VecDestroy(inner_domain);                        inner_domain = NULL;              CHKERRXX(ierr); }
  ierr = VecRestoreArray(phi, &phi_p);                      phi_p = NULL;                     CHKERRXX(ierr);
}

p4est_t* my_p4est_biomolecules_t::construct_SES(const sas_generation_method& method_to_use, const bool SAS_timing_flag, const bool SAS_subtiming_flag, string vtk_folder)
{
  PetscErrorCode ierr;
  // sanity checks
#ifdef DEBUG
  check_validity_of_vector_of_mol();
#endif
  if(nodes != NULL){
    p4est_nodes_destroy(nodes); nodes = NULL; }
  if(ghost != NULL){
    p4est_ghost_destroy(ghost); ghost = NULL; }
  if(phi != NULL){
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    phi = NULL; phi_read_only_p = NULL;
  }
  if(inner_domain != NULL){
    ierr = VecDestroy(inner_domain); CHKERRXX(ierr); inner_domain = NULL;}
  update_max_level();
  if(global_max_level > parameters.lmin()) // the p4est is already refined, the method assumes a pristine, coarse p4est when invoked
    reset_p4est();
  if(p4est->data_size != 0)
    p4est_reset_data(p4est, 0, NULL, p4est->user_pointer);

  parStopWatch* log_timer = NULL;
  if(log_file != NULL)
  {
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Construction of the grid with %d proc(s) \n", p4est->mpisize);                             CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "------------------------------------------- \n");                                          CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of trees: %d \n", p4est->connectivity->num_trees);                                  CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Min level: %d \n", parameters.lmin());                                                     CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Max level: %d \n", parameters.lmax());                                                     CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Proportionality constant L: %lf \n", parameters.lip());                                    CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Probe radius (in A): %lf \n", parameters.probe_radius()/angstrom_to_domain);               CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Probe radius (in domain): %lf \n", parameters.probe_radius());                             CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Finest cell diagonal (in domain): %lf \n", parameters.tree_diag()/(1<<parameters.lmax())); CHKERRXX(ierr);
    if(timer != NULL && log_timer == NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, log_file, p4est->mpicomm);
      log_timer->start(" CONSTRUCTION OF THE SES grid ");
    }
  }

  if(timer != NULL)
  {
    if(method_to_use != list_reduction_with_exact_phi)
      timer->start("Constructing the SAS and the initial grid");
    else
      timer->start("Constructing the exact distance to the SAS and the initial grid");
  }

  if(sas_creator != NULL)
  {
    bool need_deletion = false;
    switch (method_to_use) {
    case brute_force:
      need_deletion = (dynamic_cast<SAS_creator_brute_force*>(sas_creator) == nullptr);
      break;
    case list_reduction:
      need_deletion = (dynamic_cast<SAS_creator_list_reduction*>(sas_creator) == nullptr);
      break;
    case list_reduction_with_exact_phi:
      need_deletion = (dynamic_cast<SAS_creator_list_reduction*>(sas_creator) == nullptr);
      break;
    default:
      throw std::invalid_argument("my_p4est_biomolecules_t::construct_SES(const sas_generation_method& ): unknown sas generation method...");
      break;
    }
    if(need_deletion)
    {
      delete sas_creator; sas_creator = NULL;
    }
  }
  // the p4est must point to a my_p4est_biomolecules_t object for the construction of the SAS
  void* user_pointer_saved = p4est->user_pointer;
  p4est->user_pointer = (void*) this;

  if(sas_creator == NULL)
  {
    switch (method_to_use) {
    case brute_force:
      sas_creator = new SAS_creator_brute_force(p4est, SAS_timing_flag, SAS_subtiming_flag);
      break;
    case list_reduction:
      sas_creator = new SAS_creator_list_reduction(p4est, false, SAS_timing_flag, SAS_subtiming_flag);
      break;
    case list_reduction_with_exact_phi:
      sas_creator = new SAS_creator_list_reduction(p4est, true, SAS_timing_flag, SAS_subtiming_flag);
      break;
    default:
      throw std::invalid_argument("my_p4est_biomolecules_t::construct_SES(const sas_generation_method& ): unknown SAS generation method...");
      break;
    }
  }

  // construct the sas grid and surface
  sas_creator->construct_SAS(p4est);
  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() == 0); // some reduced lists have not been deleted after the creation of the SAS surface, memory leak has happened..."
  delete sas_creator; sas_creator = NULL;

  if(vtk_folder[vtk_folder.size()-1] == '/')
    vtk_folder = vtk_folder.substr(0, vtk_folder.size() - 1);
  bool export_intermediary_results = false; //!case_sensitive_string_compare(vtk_folder, no_vtk);

  if(export_intermediary_results)
  {
    if(timer != NULL)
    {
      timer->stop();timer->read_duration(true);
      timer->start("Exporting the SAS results");
    }
    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    string vtk_file = vtk_folder + "/SAS_grid";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SAS", phi_p);
    ierr = VecRestoreArrayRead(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
  }

  if(timer != NULL && method_to_use != list_reduction_with_exact_phi)
  {
    timer->stop();timer->read_duration(true);
    timer->start("Reinitializing the levelset");
  }

  // make sure the ghost have been constructed in the last step of the SAS construction
  P4EST_ASSERT(ghost != NULL);
  // create hirerachy, nodes neighbors and a levelset
  if(hierarchy != NULL)
    hierarchy->update(p4est, ghost);
  else
    hierarchy = new my_p4est_hierarchy_t(p4est, ghost, brick);
  if(neighbors != NULL)
    neighbors->update(hierarchy, nodes);
  else
    neighbors = new my_p4est_node_neighbors_t(hierarchy, nodes);
  neighbors->init_neighbors();
  if(ls != NULL)
    ls->update(neighbors);
  else
    ls = new my_p4est_level_set_t(neighbors);

  if(method_to_use != list_reduction_with_exact_phi)
  {
    double smallest_grid_size = parameters.tree_dim(0)/(1<<parameters.lmax());
    for (unsigned char dim = 1; dim < P4EST_DIM; ++dim)
      smallest_grid_size = MIN(smallest_grid_size, parameters.tree_dim(dim)/(1<<parameters.lmax()));
    double pseudo_time_step;
    int nb_it;

    // Reinitialization ONLY IN THE POSITIVE DOMAIN for computational efficiency
    // (load is balanced in the last partitioning step of the SAS construction)
    switch (parameters.order_of_accuracy()) {
    case 1:
      pseudo_time_step = 0.5*smallest_grid_size; // as it is in the first order reinitialization method
      nb_it = ceil(3.0*(parameters.probe_radius() + parameters.layer_thickness())/pseudo_time_step); // '3.0' == to ensure convergence
      ls->reinitialize_1st_order_above_threshold(phi, 0.0, MAX(nb_it, 10));
      break;
    case 2:
      pseudo_time_step = smallest_grid_size/((double) P4EST_DIM); // as it is in the second order reinitialization method
      nb_it = ceil(3.0*(parameters.probe_radius() + parameters.layer_thickness())/pseudo_time_step); // '3.0' == to ensure convergence
      ls->reinitialize_2nd_order_above_threshold(phi, 0.0, MAX(nb_it, 10));
      // VERY IMPORTANT NOTE: NEVER EVER use the following for this application, this scheme is not TVD,
      // it will result in an oscillatory behavior and no convergence can be expected...
      //    ls->reinitialize_1st_order_time_2nd_order_space_in_positive_domain(phi, diag_finest_cell, MAX(nb_it, 10));
      // END OF VERY IMPORTANT NOTE
      break;
    default:
      throw  std::invalid_argument("my_p4est_biomolecules_t::construct_SES(): the order of accuracy should be either 1 or 2!");
      break;
    }
  }

  // subtract probe radius
  Vec phi_l;
  ierr = VecGhostGetLocalForm(phi, &phi_l); CHKERRXX(ierr);
  ierr = VecShift(phi_l, -parameters.probe_radius()); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_l); CHKERRXX(ierr);

  if(export_intermediary_results)
  {
    if(timer != NULL)
    {
      timer->stop();timer->read_duration(true);
      timer->start("Exporting the calculated SES");
    }
    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    string vtk_file = vtk_folder + "/SES_not_cavity_free";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SES", phi_p);
    ierr = VecRestoreArrayRead(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
  }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration(true);
    timer->start("Removing cavities");
  }
  remove_internal_cavities(export_intermediary_results);

  if(export_intermediary_results)
  {
    if(timer != NULL)
    {
      timer->stop();timer->read_duration(true);
      timer->start("Exporting the SES and the cavities identification");
    }
    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    double* inner_domain_p = NULL;
    P4EST_ASSERT(inner_domain != NULL);
    ierr = VecGetArray(inner_domain, &inner_domain_p); CHKERRXX(ierr);
    string vtk_file = vtk_folder + "/SES_and_cavities";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SES", phi_p,
                           VTK_POINT_DATA, "cavities", inner_domain_p);
    ierr = VecRestoreArray(inner_domain, &inner_domain_p); inner_domain_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
  }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration(true);
    timer->start("Coarsening steps");
  }

  int nb_coarsening_steps = 0;
  while (coarsening_step(nb_coarsening_steps, export_intermediary_results))
    if(export_intermediary_results)
    {
      const double *phi_p;
      ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
      const double* inner_domain_p;
      P4EST_ASSERT(inner_domain != NULL);
      ierr = VecGetArrayRead(inner_domain, &inner_domain_p); CHKERRXX(ierr);
      string vtk_file = vtk_folder + "/coarsening_step_" + to_string(nb_coarsening_steps);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             2, 0, vtk_file.c_str(),
                             VTK_POINT_DATA, "phi_SES", phi_p,
                             VTK_POINT_DATA, "acceleration", inner_domain_p);
      ierr = VecRestoreArrayRead(inner_domain, &inner_domain_p); inner_domain_p = NULL; CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
    }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration(true);
    timer->start("Enforcing the min level");
  }

  enforce_min_level(export_intermediary_results);
  if(export_intermediary_results)
  {
    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    P4EST_ASSERT(inner_domain != NULL);
    const double* inner_domain_p = NULL;
    ierr = VecGetArrayRead(inner_domain, &inner_domain_p); CHKERRXX(ierr);
    string vtk_file = vtk_folder + "/enforcing_min_level";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SES", phi_p,
                           VTK_POINT_DATA, "acceleration", inner_domain_p);
    ierr = VecRestoreArrayRead(inner_domain, &inner_domain_p); inner_domain_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
  }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration(true);
    timer->start("Final uniform partitioning");
  }

  partition_uniformly(export_intermediary_results);

  Vec phi_local;
  ierr = VecGhostGetLocalForm(phi, &phi_local); CHKERRXX(ierr);
  ierr = VecScale(phi_local, -1.0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_local); CHKERRXX(ierr);

  if(timer != NULL)
  {
    timer->stop(); timer->read_duration(true);
    delete timer; timer = NULL;
  }

  if(log_file != NULL)
  {
    if(log_timer != NULL)
    {
      log_timer->stop(); log_timer->read_duration(true);
      delete log_timer; log_timer = NULL;
    }
    p4est_gloidx_t total_nb_nodes = 0;
    for (int rank = 0; rank < p4est->mpisize; ++rank)
      total_nb_nodes += nodes->global_owned_indeps[rank];
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of coarsening steps: %d \n", nb_coarsening_steps); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of nodes: %ld \n", total_nb_nodes); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of quadrants: %ld \n", p4est->global_num_quadrants); CHKERRXX(ierr);
    Vec ones = NULL, ones_ghost_local = NULL;
    ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(ones, &ones_ghost_local); CHKERRXX(ierr);
    ierr = VecSet(ones_ghost_local, 1.0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(ones, &ones_ghost_local); ones_ghost_local = NULL; CHKERRXX(ierr);
    double molecule_area = integrate_over_interface(p4est, nodes, phi, ones);
    ierr = VecDestroy(ones); ones = NULL; CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Surface of the molecule: %g (in domain dimensions), %g A^2\n", molecule_area, molecule_area*pow(angstrom_to_domain, -(P4EST_DIM-1)), P4EST_DIM-1); CHKERRXX(ierr);
#else
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Surface of the molecule: %g (in domain dimensions), %g A\n", molecule_area, molecule_area*pow(angstrom_to_domain, -(P4EST_DIM-1))); CHKERRXX(ierr);
#endif
  }
  // return the user pointer of the p4est as it was and return the p4est
  p4est->user_pointer = user_pointer_saved;
  return p4est;
}

void my_p4est_biomolecules_t::expand_ghost()
{
  P4EST_ASSERT(p4est != NULL && nodes != NULL && ghost != NULL &&
      hierarchy != NULL && neighbors != NULL && ls != NULL &&
      phi != NULL);
  // create the list of known global indices
  vector<PetscInt> global_indices(nodes->num_owned_indeps, 0);
  p4est_gloidx_t node_offset = 0;
  for (int r = 0; r < p4est->mpirank; ++r)
    node_offset += nodes->global_owned_indeps[r];
  for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k)
    global_indices[k] = k + node_offset;
  // expand ghost, and create new nodes
  p4est_ghost_expand(p4est, ghost);
  p4est_nodes_destroy(nodes);
  nodes   = my_p4est_nodes_new(p4est, ghost);

  PetscErrorCode ierr;
  // save known fields and create new ones
  Vec old_phi = phi, old_inner_domain = inner_domain;
  ierr    = VecCreateGhostNodes(p4est, nodes, &phi);                                                                          CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr  = VecCreateGhostNodes(p4est, nodes, &inner_domain);                                                                 CHKERRXX(ierr); }
  P4EST_ASSERT(phi != NULL && ((old_inner_domain == NULL) || (old_inner_domain != NULL && inner_domain != NULL)));
  // so now, we have to rescatter the vector(s), update the hierarchy, the node neighbors, the levelset object
  VecScatter ctx_phi, ctx_inner_domain;
  Vec old_phi_loc, old_inner_domain_loc;
  ierr = VecGhostGetLocalForm(old_phi, &old_phi_loc);                                                                         CHKERRXX(ierr);
  ierr = VecScatterAllToSomeCreate(p4est->mpicomm, old_phi_loc, phi, global_indices, &ctx_phi);                               CHKERRXX(ierr);
  ierr = VecScatterAllToSomeBegin(ctx_phi, old_phi_loc, phi);                                                                 CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr = VecGhostGetLocalForm(old_inner_domain, &old_inner_domain_loc);                                                     CHKERRXX(ierr);
    ierr = VecScatterAllToSomeCreate(p4est->mpicomm, old_inner_domain_loc, inner_domain, global_indices, &ctx_inner_domain);  CHKERRXX(ierr);
    ierr = VecScatterAllToSomeBegin(ctx_inner_domain, old_inner_domain_loc, inner_domain);                                    CHKERRXX(ierr);
  }
  // place those operations here for optimizing the execution
  hierarchy->update(p4est, ghost);
  neighbors->update(hierarchy, nodes);
  ls->update(neighbors);
  // back to scattering vector values
  ierr = VecScatterAllToSomeEnd(ctx_phi, old_phi_loc, phi);                                                                   CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);                                                            CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(old_phi, &old_phi_loc);                                                                     CHKERRXX(ierr);
  ierr = VecScatterDestroy(ctx_phi);                                                                                          CHKERRXX(ierr);
  ierr = VecDestroy(old_phi);                                                                                                 CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);                                                              CHKERRXX(ierr);
  if(old_inner_domain != NULL)
  {
    ierr = VecScatterAllToSomeEnd(ctx_inner_domain, old_inner_domain_loc, inner_domain);                                      CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                                 CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(old_inner_domain, &old_inner_domain_loc);                                                 CHKERRXX(ierr);
    ierr = VecScatterDestroy(ctx_inner_domain);                                                                               CHKERRXX(ierr);
    ierr = VecDestroy(old_inner_domain);                                                                                      CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                                   CHKERRXX(ierr);
  }
}

void my_p4est_biomolecules_t::set_quad_weight(p4est_quadrant_t* &quad, const p4est_nodes_t* & nodes, const double* const& phi_fct, const double& lower_bound)
{
  p4est_locidx_t quad_idx = quad->p.user_long;
  quad->p.user_long       = 0;
  for (unsigned char n = 0; n < P4EST_CHILDREN; ++n)
  {
    p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx+n];
    P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
    if(phi_fct[node_idx] > lower_bound)
    {
      quad->p.user_long = 1;
      return;
    }
  }
}

p4est_bool_t my_p4est_biomolecules_t::coarsen_fn(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t *quad[])
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) forest->user_pointer;
  const int min_lvl                     = biomol->parameters.lmin();
  const int max_lvl                     = biomol->parameters.lmax();
  const p4est_nodes_t* nodes            = biomol->nodes;
  (void) which_tree;

  p4est_bool_t result;
  if (quad[0]->level <= min_lvl)
    result = P4EST_FALSE;
  else if (quad[0]->level > max_lvl)
    result = P4EST_TRUE;
  else
  {
    const double parent_cell_diag = 2.0*biomol->parameters.tree_diag()/(1<<quad[0]->level);
    const double lip              = biomol->parameters.lip();

    double f[P4EST_CHILDREN];
    for (unsigned char k = 0; k < P4EST_CHILDREN; ++k) { // not exactly the same as in the paper, but equivalent!
      p4est_locidx_t quad_idx = quad[k]->p.user_long;
      p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx+k];
      P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
      f[k] = biomol->phi_read_only_p[node_idx];
      if(fabs(f[k]) <= 0.5*lip*parent_cell_diag)
      {
        result = P4EST_FALSE;
        goto function_end;
      }
    }
    // no need to check for interface crossing, this is prevented by
    // (phi = signed distance) + (lip >= 1)...
    result = P4EST_TRUE;
  }

function_end:
  for (unsigned char q = 0; q < P4EST_CHILDREN; ++q)
    set_quad_weight(quad[q], nodes, biomol->phi_read_only_p, 0.0);
  // if I'm not mistaken when reading p4est_coarsen source file, the weight of the first child is important in case of coarsening...
  // --> should work as well

  return result;
}

int my_p4est_biomolecules_t::weight_for_coarsening(p4est_t *forest, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  (void) forest;
  return quadrant->p.user_long;
}

bool my_p4est_biomolecules_t::coarsening_step(int& step_idx, bool export_acceleration)
{
  P4EST_ASSERT(nodes != NULL);
  P4EST_ASSERT(ghost != NULL);
  P4EST_ASSERT(phi_read_only_p == NULL);
  PetscErrorCode ierr;
  int mpiret;

  if(log_file != NULL && step_idx > log2(box_size_of_biggest_mol*sqrt(P4EST_DIM)*(1<<(parameters.lmax()-1))/(parameters.lip()*parameters.tree_diag()))){
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "More coarsening steps than expected... This is weird! \n");                                CHKERRXX(ierr);
  }
  // Explanation:
  // At each iteration, we reinitialize to capture a new layer of inner cells (away from the interface) that are potential candidates for coarsening.
  // When step_idx = k, the width of that candidate layer is
  //   dist_k = parameters.lip()*(2*diag of cell of level k) = parameters.lip()*(diag of root cell)*(2^(k+1-parameters.max_lvl()))
  // When dist_k > diagonal of the biggest bounding box, it makes no sense to keep doing it (and it should NEVER happen actually)...

  Vec phi_ghost_loc;
  // Given the number of iterations used for capturing the SES accurately,
  // WE assumed that the first layer is captured too, without reinitialization...
  // [Wrong only if L is absurdly large...]
  double already_captured_layer = parameters.lip()*parameters.tree_diag()/(1<<(parameters.lmax()-step_idx++));
  // "remove internal cavities" == trick to accelerate the reinitialization steps
  // and avoid more than one local coarsening operation
  ierr = VecGhostGetLocalForm(phi, &phi_ghost_loc);                                                                                           CHKERRXX(ierr);
  ierr = VecShift(phi_ghost_loc, -already_captured_layer);                                                                                    CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_ghost_loc);                                                                                       CHKERRXX(ierr);
  remove_internal_cavities(export_acceleration);
  ierr = VecGhostGetLocalForm(phi, &phi_ghost_loc);                                                                                           CHKERRXX(ierr);
  ierr = VecShift(phi_ghost_loc, +already_captured_layer);                                                                                    CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_ghost_loc);                                                                                       CHKERRXX(ierr);
  P4EST_ASSERT((export_acceleration && inner_domain != NULL) || (!export_acceleration && inner_domain == NULL));

  // needed for coarsening
  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad  = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      quad->p.user_long = q + tree_k->quadrants_offset;
    }
  }

  p4est_locidx_t former_quad_count = p4est->local_num_quadrants;
  ierr  = VecGetArrayRead(phi, &phi_read_only_p);                                                                                             CHKERRXX(ierr);
  my_p4est_coarsen(p4est, P4EST_FALSE, my_p4est_biomolecules_t::coarsen_fn, NULL);
  ierr = VecRestoreArrayRead(phi, &phi_read_only_p); phi_read_only_p = NULL;                                                                  CHKERRXX(ierr);
  P4EST_ASSERT(former_quad_count >= p4est->local_num_quadrants);
  int grid_has_changed = (former_quad_count != p4est->local_num_quadrants);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &grid_has_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  if(!grid_has_changed)
    return grid_has_changed;

  // else, repartition the tree and the levelset function for further coarsening
  // - repartition for balanced further reinitialization (see set_quad_weight in coarsen_fn)
  // - allowing new families of quadrants to be grouped together for further coarsening steps

  // store the current (fine) nodes, and get the new ones
  p4est_nodes_t* fine_nodes = nodes;
  nodes = my_p4est_nodes_new(p4est, NULL);

  int clamped = 1;
  vector<PetscInt> global_idx_to_scatter_from(nodes->num_owned_indeps);
  vector<PetscInt> global_idx_to_scatter_to(nodes->num_owned_indeps);

  p4est_locidx_t idx   = 0;
  p4est_indep_t *node = NULL;
  if(idx < nodes->num_owned_indeps)
    node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx);
  PetscInt my_index_offset_from = 0;
  PetscInt my_index_offset_to = 0;
  for (int proc_rank = 0; proc_rank < p4est->mpirank; ++proc_rank)
  {
    my_index_offset_from += fine_nodes->global_owned_indeps[proc_rank];
    my_index_offset_to   += nodes->global_owned_indeps[proc_rank];
  }
  for (p4est_locidx_t k = 0; k < fine_nodes->num_owned_indeps; ++k)
  {
    p4est_indep_t *fine_node  = (p4est_indep_t*) sc_array_index(&fine_nodes->indep_nodes,k);
    if(node!= NULL && p4est_node_equal_piggy_fn (fine_node, node, &clamped))
    {
      global_idx_to_scatter_to[idx]     = my_index_offset_to + idx;
      global_idx_to_scatter_from[idx++] = my_index_offset_from + k;
      if(idx < nodes->num_owned_indeps)
        node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes,idx);
      else
        break;
    }
  }
  // sanity check
  P4EST_ASSERT(idx == nodes->num_owned_indeps);

  ierr  = VecGetArrayRead(phi, &phi_read_only_p);                                                                                             CHKERRXX(ierr);
  my_p4est_partition(p4est, P4EST_TRUE, my_p4est_biomolecules_t::weight_for_coarsening);
  ierr  = VecRestoreArrayRead(phi, &phi_read_only_p); phi_read_only_p = NULL;                                                                 CHKERRXX(ierr);
  // we don't need the fine ghosts, nor the fine and locally coarse nodes (before the partition)
  p4est_ghost_destroy(ghost); ghost = NULL;
  p4est_nodes_destroy(fine_nodes); fine_nodes = NULL;
  p4est_nodes_destroy(nodes); nodes = NULL;
  // create the new ones
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  nodes = my_p4est_nodes_new(p4est, ghost);

  IS is_from;
  IS is_to;
  ierr = ISCreateGeneral(p4est->mpicomm, global_idx_to_scatter_from.size(), global_idx_to_scatter_from.data(), PETSC_USE_POINTER, &is_from);  CHKERRXX(ierr);
  ierr = ISCreateGeneral(p4est->mpicomm, global_idx_to_scatter_to.size(), global_idx_to_scatter_to.data(), PETSC_USE_POINTER, &is_to);        CHKERRXX(ierr);
  Vec fine_phi = phi;
  VecScatter ctx_phi;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi);                                                                                             CHKERRXX(ierr);
  ierr = VecScatterCreate(fine_phi, is_from, phi, is_to, &ctx_phi);                                                                           CHKERRXX(ierr);
  ierr = VecScatterBegin(ctx_phi, fine_phi, phi, INSERT_VALUES, SCATTER_FORWARD);                                                             CHKERRXX(ierr);
  Vec fine_inner_domain = NULL;
  VecScatter ctx_inner_domain;
  if(export_acceleration)
  {
    fine_inner_domain = inner_domain;
    ierr = VecCreateGhostNodes(p4est, nodes, &inner_domain);                                                                                  CHKERRXX(ierr);
    ierr = VecScatterCreate(fine_inner_domain, is_from, inner_domain, is_to, &ctx_inner_domain);                                              CHKERRXX(ierr);
    ierr = VecScatterBegin(ctx_inner_domain, fine_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                CHKERRXX(ierr);
  }
  hierarchy->update(p4est, ghost);
  neighbors->update(hierarchy, nodes);
  ls->update(neighbors);
  ierr = VecScatterEnd(ctx_phi, fine_phi, phi, INSERT_VALUES, SCATTER_FORWARD);                                                               CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);                                                                            CHKERRXX(ierr);
  ierr = VecDestroy(fine_phi);                                                                                                                CHKERRXX(ierr);
  ierr = VecScatterDestroy(ctx_phi);                                                                                                          CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);                                                                              CHKERRXX(ierr);
  if(export_acceleration)
  {
    ierr = VecScatterEnd(ctx_inner_domain, fine_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                  CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                                                 CHKERRXX(ierr);
    ierr = VecDestroy(fine_inner_domain);                                                                                                     CHKERRXX(ierr);
    ierr = VecScatterDestroy(ctx_inner_domain);                                                                                               CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD);                                                                   CHKERRXX(ierr);
  }
  ierr = ISDestroy(is_from);                                                                                                                  CHKERRXX(ierr);
  ierr = ISDestroy(is_to);                                                                                                                    CHKERRXX(ierr);

  const double min_root_cell_dim = MIN(DIM(parameters.tree_dim(0), parameters.tree_dim(1), parameters.tree_dim(2)));
  const int n_iter = ceil(1.5*parameters.lip()*parameters.tree_diag()/(0.5*min_root_cell_dim));
  // Explanation:
  // 1) parameters.lip()*diag of current cell level = theoretical step of pseudo-time until which
  // the reinitialization equation must be further solved to capture the next layer of coarsened cells
  // (assuming the current level set function is accurate enough for all other points that are closer to the surface)
  // --> call that pseudo-time tau_end
  // 2) tau_end/(0.5*min dim of current cell level) = corresponding number of iterations in the following
  // first-order reinitialization algorithm
  // 3) (diag of current cell level)/(min dim of current cell level) is level-invariant, so evaluate that ratio for root cell...
  // 4) 1.5 == 'safety factor' for better convergence
  // (might not be 100% exact at the end of the day, but what else can we do?)

  // note the "_above_threshold" and the value of the threshold --> the actual SES does NOT MOVE
  // the order of accuracy is irrelevant for this application, really --> choose the fastest (1st order) method
  ls->reinitialize_1st_order_above_threshold(phi, parameters.layer_thickness(), n_iter);

  return grid_has_changed;
}

my_p4est_biomolecules_solver_t::my_p4est_biomolecules_solver_t(const my_p4est_biomolecules_t *biomolecules_)
  : biomolecules(biomolecules_)
{
  temperature           = -1.0; // absurd initialization
  far_field_ion_density = -1.0; // absurd initialization
  ion_charge            = 0;    // absurd initialization
  mol_rel_permittivity  = 0.0;  // absurd initialization
  elec_rel_permittivity = 0.0;  // absurd initialization

  // initialize psi_star, psi_naught, psi_bar, and psi, will be created when needed
  psi_star              = NULL;
  psi_hat               = NULL;
  psi_bar               = NULL;
  psi_naught            = NULL;
  psi_hat_is_set = psi_star_is_set = false;
  nb_iterations_for_setting_psi_hat = -1; // absurd initialization

  // create the solvers
  jump_solver     = new my_p4est_general_poisson_nodes_mls_solver_t(biomolecules->neighbors);
  jump_solver_v2  = new my_p4est_poisson_nodes_mls_t(biomolecules->neighbors);
  node_solver     = new my_p4est_poisson_nodes_t(biomolecules->neighbors);
}
void    my_p4est_biomolecules_solver_t::set_molecular_relative_permittivity(double epsilon_molecule)
{
  P4EST_ASSERT(epsilon_molecule >= 1.0-EPS);
  // if the value is changed, psi_star, psi_bar and psi are no longer valid
  psi_star_is_set = psi_hat_is_set = !(fabs(mol_rel_permittivity - epsilon_molecule) > EPS*fabs(mol_rel_permittivity));
  if(!psi_hat_is_set)
    nb_iterations_for_setting_psi_hat = -1;
  mol_rel_permittivity  = epsilon_molecule;
}
void    my_p4est_biomolecules_solver_t::set_electrolyte_relative_permittivity(double epsilon_electrolyte)
{
  P4EST_ASSERT(epsilon_electrolyte > 1.0);
#ifdef CASL_THROWS
  // print a warning
  if(elec_rel_permittivity > 1.0-EPS && fabs(elec_rel_permittivity - epsilon_electrolyte) > EPS*elec_rel_permittivity) {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::set_electrolyte_relative_permittivity(...): the electrolyte permittivity was already set, it will be reset...\n"); CHKERRXX(ierr); }
#endif
  // if the value is changed, psi is no longer valid
  psi_hat_is_set = !(fabs(elec_rel_permittivity - epsilon_electrolyte) > EPS*fabs(elec_rel_permittivity));
  if(!psi_hat_is_set)
    nb_iterations_for_setting_psi_hat = -1;
  elec_rel_permittivity   = epsilon_electrolyte;
}
void    my_p4est_biomolecules_solver_t::set_relative_permittivities(double epsilon_molecule, double epsilon_electrolyte)
{
  set_molecular_relative_permittivity(epsilon_molecule);
  set_electrolyte_relative_permittivity(epsilon_electrolyte);
}
void    my_p4est_biomolecules_solver_t::set_temperature_in_kelvin(double temperature_in_K)
{
  P4EST_ASSERT(temperature_in_K > EPS); //the value of the temperature must be strictly positive
#ifdef CASL_THROWS
  // print a warning
  if(temperature_is_set() && temperature_in_K > EPS && fabs(temperature - temperature_in_K) > EPS*temperature) {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::set_temperature_in_kelvin(...): the temperature was already set, it will be reset...\n"); CHKERRXX(ierr); }
#endif
  // if the value is changed, psi_star, psi_bar and psi are no longer valid
  psi_star_is_set = psi_hat_is_set = !(fabs(temperature - temperature_in_K) > EPS*fabs(temperature));
  if(!psi_hat_is_set)
    nb_iterations_for_setting_psi_hat = -1;
  temperature = temperature_in_K;
}
void    my_p4est_biomolecules_solver_t::set_far_field_ion_density(double n_0)
{
#ifdef CASL_THROWS
  // print a warning
  if(far_field_ion_density_is_set()){
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::set_far_field_ion_density(...): the far-field ion density was already set, it will be reset...\n"); CHKERRXX(ierr); }
#endif
  // if the value is changed, psi is no longer valid
  psi_hat_is_set = !(fabs(far_field_ion_density - n_0) > EPS*fabs(far_field_ion_density));
  if(!psi_hat_is_set)
    nb_iterations_for_setting_psi_hat = -1;
  far_field_ion_density = n_0;
}
void    my_p4est_biomolecules_solver_t::set_ion_charge(int z)
{
  P4EST_ASSERT(z>0); // the ion charge must be a strictly positive integer
#ifdef CASL_THROWS
  // print a warning
  if(ion_charge_is_set() && z > 0 && ion_charge != z) {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::set_ion_charge(...): the ion charge was already set, it will be reset...\n"); CHKERRXX(ierr); }
#endif
  // if the value is changed, psi_star and psi_hat are no longer valid
  psi_star_is_set = psi_hat_is_set = !(ion_charge != z);
  if(!psi_hat_is_set)
    nb_iterations_for_setting_psi_hat = -1;
  ion_charge = z;
}
void    my_p4est_biomolecules_solver_t::set_inverse_debye_length_in_meters_inverse(double inverse_debye_length_in_m_inverse)
{
  P4EST_ASSERT(inverse_debye_length_in_m_inverse > 0.0);
  // if all three parameters are set, it's either consistent or not. If not, abort
  if(all_debye_parameters_are_set())
  {
    // Either it's the correct value,
    if(fabs(get_inverse_debye_length_in_meters_inverse() - inverse_debye_length_in_m_inverse) < EPS*get_inverse_debye_length_in_meters_inverse())
      return; // nothing to be done
    // or the user is dumb and we should teach him
    P4EST_ASSERT(false); // the debye length is already set by the relevant parameters, it can't be set to the new value: abort;
    MPI_Abort(biomolecules->p4est->mpicomm, 759957);
  }
  PetscErrorCode ierr;
  // if only 2 of the three parameters are set, it's the "easy" case, the third one might be calculated
  if(ion_charge_is_set() && temperature_is_set() && !far_field_ion_density_is_set())
  {
    // temperature and ion charge are set, the far-field ion density is not, let's calculate it
    set_far_field_ion_density(eps_0*kB*temperature*inverse_debye_length_in_m_inverse/(2.0*SQR(((double) ion_charge)*electron)));
    if(biomolecules->log_file != NULL)
    {
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "setting far field ion density based on other values \n"); CHKERRXX(ierr);
    }
    return;
  }
  if(ion_charge_is_set() && !temperature_is_set() && far_field_ion_density_is_set())
  {
    // far-field ion density and ion charge are set, temperature is not, let's calculate it
    set_temperature_in_kelvin(2.0*far_field_ion_density*SQR((1/inverse_debye_length_in_m_inverse)*((double) ion_charge)*electron)/(eps_0*kB));
    if(biomolecules->log_file != NULL)
    {
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "setting temperature based on other values \n"); CHKERRXX(ierr);
    }
    return;
  }
  if(!ion_charge_is_set() && temperature_is_set() && far_field_ion_density_is_set())
  {
    // ion charge might be freely set
    double my_z = sqrt(eps_0*kB*temperature/(2.0*far_field_ion_density*SQR((1/inverse_debye_length_in_m_inverse)*electron)));
    // but it's supposed to be an integer, so let's check that
    P4EST_ASSERT(fabs(my_z - round(my_z)) < EPS*my_z);
    set_ion_charge((int) round(my_z));
    if(biomolecules->log_file != NULL)
    {
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "setting ion charge based on other values \n"); CHKERRXX(ierr);
    }
    return;
  }
  // if none or only one of the three parameters is set, we have one more degree of freedom, and we can set arbitrary values
  if(!ion_charge_is_set())
  {
    // set to default (1:1 electrolyte)
    set_ion_charge();
    // one or two of the parameters are set, now
    // if only one (the ion charge), the next pass will set the temperature and the far-field ion density will be calculated
    // if two (the ion charge and either the temperature or the far-field density), the next pass will set the remaining one
    set_inverse_debye_length_in_meters_inverse(inverse_debye_length_in_m_inverse);
    if(biomolecules->log_file != NULL)
    {
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "setting ion charge and then inverse_debye_length \n"); CHKERRXX(ierr);
    }
  }
  else
  {
    // the ion charge only is set, set the temperature to 300 K (default value) and that's it
    set_temperature_in_kelvin();
    set_inverse_debye_length_in_meters_inverse(inverse_debye_length_in_m_inverse);
    if(biomolecules->log_file != NULL)
    {
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "setting temperature and then inverse_debye_length \n"); CHKERRXX(ierr);
    }
  }
  return;
}
double my_p4est_biomolecules_solver_t::get_inverse_debye_length_in_meters_inverse() const
{
  P4EST_ASSERT(all_debye_parameters_are_set());
  return sqrt((2.0*far_field_ion_density*SQR(((double) ion_charge)*electron))/(eps_0*kB*temperature));
}

void my_p4est_biomolecules_solver_t::return_psi_star(Vec &psi_star_out)
{
  P4EST_ASSERT(psi_star != NULL);
  psi_star_out    = psi_star; psi_star = NULL;
  psi_star_is_set = false;
}

void my_p4est_biomolecules_solver_t::return_psi_hat(Vec &psi_hat_out)
{
  P4EST_ASSERT(psi_hat != NULL);
  psi_hat_out     = psi_hat; psi_hat=NULL;
  psi_hat_is_set  = false;
  nb_iterations_for_setting_psi_hat = -1;
}

void my_p4est_biomolecules_solver_t::return_psi_bar(Vec &psi_bar_out)
{
  P4EST_ASSERT(psi_bar != NULL);
  psi_bar_out     = psi_bar; psi_bar=NULL;
}

void my_p4est_biomolecules_solver_t::return_all_psi_vectors(Vec &psi_star_out, Vec &psi_hat_out)
{
  return_psi_star(psi_star_out);
  return_psi_hat(psi_hat_out);
}

void my_p4est_biomolecules_solver_t::make_sure_is_node_sampled(Vec &vector)
{
  if(vector != NULL)
  {
    PetscInt size_local, size_local_ghost, size_global;
    Vec vector_local_ghost = NULL;
    ierr = VecGetSize(vector, &size_global);                                                  CHKERRXX(ierr);
    ierr = VecGetLocalSize(vector, &size_local);                                              CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(vector, &vector_local_ghost);                                 CHKERRXX(ierr);
    ierr = VecGetSize(vector_local_ghost, &size_local_ghost);                                 CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(vector, &vector_local_ghost);                             CHKERRXX(ierr);
    p4est_gloidx_t nb_nodes_total = 0;
    for (int rr = 0; rr < biomolecules->p4est->mpisize; ++rr)
      nb_nodes_total += biomolecules->nodes->global_owned_indeps[rr];
    int logic_error = ((size_global != nb_nodes_total) ||
                       (size_local_ghost != (PetscInt) biomolecules->nodes->indep_nodes.elem_count) ||
                       (size_local != biomolecules->nodes->num_owned_indeps));
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &logic_error, 1, MPI_INT, MPI_LOR, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
    if(logic_error)
    {
      ierr = VecDestroy(vector);                                                              CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(biomolecules->p4est, biomolecules->nodes, &vector);          CHKERRXX(ierr);
    }
  }
  else{
    ierr = VecCreateGhostNodes(biomolecules->p4est, biomolecules->nodes, &vector);            CHKERRXX(ierr); }
}

void my_p4est_biomolecules_solver_t::calculate_jumps_in_normal_gradient(Vec &eps_grad_n_psi_hat_jump)
{
  const double *phi_read_only_p = NULL , *psi_star_read_only_p = NULL;
  double *eps_grad_n_psi_hat_jump_p = NULL;
  ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p);              CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_star, &psi_star_read_only_p);                  CHKERRXX(ierr);
  ierr = VecGetArray(eps_grad_n_psi_hat_jump, &eps_grad_n_psi_hat_jump_p);  CHKERRXX(ierr);
  quad_neighbor_nodes_of_node_t qnnn;
  p4est_locidx_t node_idx;
  double n_xyz[P4EST_DIM];
  double grad_psi_star[P4EST_DIM];
  double norm_of_gradient;
  for (size_t k = 0; k < biomolecules->neighbors->get_layer_size(); ++k)
  {
    node_idx = biomolecules->neighbors->get_layer_node(k);
    if(fabs(phi_read_only_p[node_idx]) <= (1.5*biomolecules->parameters.layer_thickness())) // 1.5 == safety factor
    {
      biomolecules->neighbors->get_neighbors(node_idx, qnnn);
      qnnn.gradient(phi_read_only_p, n_xyz);
      qnnn.gradient(psi_star_read_only_p, grad_psi_star);
      norm_of_gradient = MAX(sqrt(SUMD(SQR(n_xyz[0]), SQR(n_xyz[1]), SQR(n_xyz[2]))), EPS);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        n_xyz[dir] /= norm_of_gradient;
      eps_grad_n_psi_hat_jump_p[node_idx]  = mol_rel_permittivity*SUMD(n_xyz[0]*grad_psi_star[0], n_xyz[1]*grad_psi_star[1], n_xyz[2]*grad_psi_star[2]);
    }
    else
      eps_grad_n_psi_hat_jump_p[node_idx]  = 0.0; // irrelevant far away from the interface but let's set it to 0.0
  }
  ierr = VecGhostUpdateBegin(eps_grad_n_psi_hat_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < biomolecules->neighbors->get_local_size(); ++k)
  {
    node_idx = biomolecules->neighbors->get_local_node(k);
    if(fabs(phi_read_only_p[node_idx]) <= (1.5*biomolecules->parameters.layer_thickness())) // 1.5 == safety factor
    {
      biomolecules->neighbors->get_neighbors(node_idx, qnnn);
      norm_of_gradient  = 0.0;
      qnnn.gradient(phi_read_only_p, n_xyz);
      qnnn.gradient(psi_star_read_only_p, grad_psi_star);
      norm_of_gradient = MAX(sqrt(SUMD(SQR(n_xyz[0]), SQR(n_xyz[1]), SQR(n_xyz[2]))), EPS);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        n_xyz[dir] /= norm_of_gradient;
      eps_grad_n_psi_hat_jump_p[node_idx]  = mol_rel_permittivity*SUMD(n_xyz[0]*grad_psi_star[0], n_xyz[1]*grad_psi_star[1], n_xyz[2]*grad_psi_star[2]);
    }
    else
      eps_grad_n_psi_hat_jump_p[node_idx]  = 0.0; // irrelevant far away from the interface but let's set it to 0.0
  }
  ierr = VecGhostUpdateEnd(eps_grad_n_psi_hat_jump, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);

  ierr = VecRestoreArray(eps_grad_n_psi_hat_jump, &eps_grad_n_psi_hat_jump_p);        CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p);                    CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_star, &psi_star_read_only_p);                        CHKERRXX(ierr);
}
void my_p4est_biomolecules_solver_t::calculate_psi_and_grad_psi(Vec &psi, Vec &grad_psi)
{
  const double *psi_star_read_only_p = NULL, *psi_hat_read_only_p=NULL;
  double *psi_p = NULL;

  ierr = VecGetArray(psi, &psi_p);
;
  ierr = VecGetArrayRead(psi_star, &psi_star_read_only_p); CHKERRXX(ierr);

  ierr = VecGetArrayRead(psi_hat, &psi_hat_read_only_p);   CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  p4est_locidx_t node_idx;
  for (size_t k = 0; k < biomolecules->neighbors->get_layer_size(); ++k)
  {
    node_idx = biomolecules->neighbors->get_layer_node(k);
    psi_p[node_idx]=psi_star_read_only_p[node_idx]+psi_hat_read_only_p[node_idx];
  }

  ierr = VecGhostUpdateBegin(psi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < biomolecules->neighbors->get_local_size(); ++k)
  {
    node_idx = biomolecules->neighbors->get_local_node(k);
    psi_p[node_idx]=psi_star_read_only_p[node_idx]+psi_hat_read_only_p[node_idx];
  }

  ierr = VecGhostUpdateEnd(psi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi, &psi_p);        CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_star, &psi_star_read_only_p);                    CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_hat, &psi_hat_read_only_p);

  biomolecules->neighbors->first_derivatives_central(psi,grad_psi);
}
void my_p4est_biomolecules_solver_t::calculate_jumps_in_normal_gradient_with_psi_bar(Vec &eps_grad_n_psi_hat_jump)
{
  const double *phi_read_only_p = NULL , *psi_bar_read_only_p = NULL;
  double *eps_grad_n_psi_hat_jump_p = NULL;
  ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p);              CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_bar, &psi_bar_read_only_p);                  CHKERRXX(ierr);
  ierr = VecGetArray(eps_grad_n_psi_hat_jump, &eps_grad_n_psi_hat_jump_p);  CHKERRXX(ierr);
  quad_neighbor_nodes_of_node_t qnnn;
  p4est_locidx_t node_idx;
  double n_xyz[P4EST_DIM];
  double grad_psi_bar[P4EST_DIM];
  double norm_of_gradient;
  for (size_t k = 0; k < biomolecules->neighbors->get_layer_size(); ++k)
  {
    node_idx = biomolecules->neighbors->get_layer_node(k);
    if(fabs(phi_read_only_p[node_idx]) <= (1.5*biomolecules->parameters.layer_thickness())) // 1.5 == safety factor
    {
      biomolecules->neighbors->get_neighbors(node_idx, qnnn);
      qnnn.gradient(phi_read_only_p, n_xyz);
      qnnn.gradient(psi_bar_read_only_p, grad_psi_bar);
      norm_of_gradient = MAX(sqrt(SUMD(SQR(n_xyz[0]), SQR(n_xyz[1]), SQR(n_xyz[2]))), EPS);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        n_xyz[dir] /= norm_of_gradient;
      eps_grad_n_psi_hat_jump_p[node_idx]  = mol_rel_permittivity*SUMD(n_xyz[0]*grad_psi_bar[0], n_xyz[1]*grad_psi_bar[1], n_xyz[2]*grad_psi_bar[2]);
    }
    else
      eps_grad_n_psi_hat_jump_p[node_idx]  = 0.0; // irrelevant far away from the interface but let's set it to 0.0
  }
  ierr = VecGhostUpdateBegin(eps_grad_n_psi_hat_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < biomolecules->neighbors->get_local_size(); ++k)
  {
    node_idx = biomolecules->neighbors->get_local_node(k);
    if(fabs(phi_read_only_p[node_idx]) <= (1.5*biomolecules->parameters.layer_thickness())) // 1.5 == safety factor
    {
      biomolecules->neighbors->get_neighbors(node_idx, qnnn);
      norm_of_gradient  = 0.0;
      qnnn.gradient(phi_read_only_p, n_xyz);
      qnnn.gradient(psi_bar_read_only_p, grad_psi_bar);
      norm_of_gradient = MAX(sqrt(SUMD(SQR(n_xyz[0]), SQR(n_xyz[1]), SQR(n_xyz[2]))), EPS);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        n_xyz[dir] /= norm_of_gradient;
      eps_grad_n_psi_hat_jump_p[node_idx]  = mol_rel_permittivity*SUMD(n_xyz[0]*grad_psi_bar[0], n_xyz[1]*grad_psi_bar[1], n_xyz[2]*grad_psi_bar[2]);
    }
    else
      eps_grad_n_psi_hat_jump_p[node_idx]  = 0.0; // irrelevant far away from the interface but let's set it to 0.0
  }
  ierr = VecGhostUpdateEnd(eps_grad_n_psi_hat_jump, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);

  ierr = VecRestoreArray(eps_grad_n_psi_hat_jump, &eps_grad_n_psi_hat_jump_p);        CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p);                    CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_bar, &psi_bar_read_only_p);                        CHKERRXX(ierr);
}
int my_p4est_biomolecules_solver_t::solve_nonlinear(double upper_bound_residual, int it_max)
{
  int iter = 0;
  if (psi_hat_is_set)
    return iter;
  P4EST_ASSERT(all_parameters_are_set());
  P4EST_ASSERT(it_max >= 1);
  P4EST_ASSERT((it_max == 1) || (upper_bound_residual > 0.0));
  parStopWatch *log_timer = NULL, *solve_subtimer = NULL;
  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " \n");                                                                                 CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Solving the Poisson-Boltzmann equation on a %d/%d grid with %d proc(s) \n",
                        (int) biomolecules->parameters.lmin(), (int) biomolecules->parameters.lmax(), biomolecules->p4est->mpisize);                                  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The ionic charge is %d \n", ion_charge);                                               CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The far-field electrolyte density is %g m^{-3} \n", far_field_ion_density);            CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The temperature is %g K \n", temperature);                                             CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The inverse debye length %g A^{-1}, %g m^{-1}, or %g in domain units\n",
                        (get_inverse_debye_length_in_angstrom_inverse()), (get_inverse_debye_length_in_meters_inverse()), get_inverse_debye_length_in_domain());      CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "\n");                                                                                  CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, biomolecules->log_file, biomolecules->p4est->mpicomm);
      log_timer->start("Resolution of the nonlinear Poisson-Boltzmann Equation");
    }
  }

  if(biomolecules->timing_file != NULL)
  {
    P4EST_ASSERT(solve_subtimer == NULL);
    solve_subtimer = new parStopWatch(parStopWatch::root_timings, biomolecules->timing_file, biomolecules->p4est->mpicomm);
  }
  if(!psi_star_is_set)
  {
    if(solve_subtimer != NULL)
      solve_subtimer->start("Evaluating singular parts");
    make_sure_is_node_sampled(psi_star);
    double *psi_star_p = NULL;
    ierr = VecGetArray(psi_star, &psi_star_p);                        CHKERRXX(ierr);
    // sample the contribution of singular charges at grid nodes, but only in the inner domain(s)
    double xyz[P4EST_DIM];
    const double* phi_read_only_p = NULL;
    ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p);      CHKERRXX(ierr);
    for (size_t i = 0; i<biomolecules->nodes->indep_nodes.elem_count; ++i) {
      if(phi_read_only_p[i] <= 5.0*biomolecules->parameters.layer_thickness()) //1.5 == safety factor
      {
        node_xyz_fr_n(i, biomolecules->p4est, biomolecules->nodes, xyz);
        psi_star_p[i] = non_dimensional_coulomb_in_mol(DIM(xyz[0], xyz[1], xyz[2]));
      }
      else
        psi_star_p[i] = 0.0;
    }
    ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(psi_star, &psi_star_p);                    CHKERRXX(ierr);
    //biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, psi_star, 20, biomolecules->parameters.order_of_accuracy()); // Technically not needed but looks like this makes things more accurate... Why? No clue!

    if(solve_subtimer != NULL){
      solve_subtimer->stop(); solve_subtimer->read_duration(); }

    psi_star_is_set = true;
  }

  if(solve_subtimer != NULL)
    solve_subtimer->start("Initializing the solver");
  // Create a node-sampled zero vector (will be useful)
  Vec node_sampled_zero = NULL;
  make_sure_is_node_sampled(node_sampled_zero);
  // Create vector for the jump condition in normal gradient
  // Vec eps_grad_n_psi_hat_jump = NULL;

  Vec eps_grad_n_psi_hat_jump         = NULL;
  Vec eps_grad_n_psi_hat_jump_xxyyzz  = NULL;
  Vec psi_star_xxyyzz                 = NULL;
  make_sure_is_node_sampled(eps_grad_n_psi_hat_jump);
  ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &eps_grad_n_psi_hat_jump_xxyyzz);  CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &psi_star_xxyyzz);                 CHKERRXX(ierr);

  // Create vectors for the diagonal term in the outer domain
  Vec add_plus = NULL;
  make_sure_is_node_sampled(add_plus);

  // define rhs's (nonzero only for validation purposes)
  Vec rhs_minus = NULL, rhs_plus = NULL;

  calculate_jumps_in_normal_gradient(eps_grad_n_psi_hat_jump);
  biomolecules->neighbors->second_derivatives_central(eps_grad_n_psi_hat_jump, eps_grad_n_psi_hat_jump_xxyyzz);
  biomolecules->neighbors->second_derivatives_central(psi_star, psi_star_xxyyzz);
  my_p4est_interpolation_nodes_t eps_grad_n_psi_hat_jump_interp_(biomolecules->neighbors);
  eps_grad_n_psi_hat_jump_interp_.set_input(eps_grad_n_psi_hat_jump, eps_grad_n_psi_hat_jump_xxyyzz, quadratic_non_oscillatory_continuous_v2);

  my_p4est_interpolation_nodes_t psi_hat_jump_interp_(biomolecules->neighbors);
  psi_hat_jump_interp_.set_input(psi_star, psi_star_xxyyzz, quadratic_non_oscillatory_continuous_v2);

  double *node_sampled_zero_p = NULL, *add_plus_p = NULL;
  ierr = VecGetArray(node_sampled_zero, &node_sampled_zero_p);      CHKERRXX(ierr);
  ierr = VecGetArray(add_plus, &add_plus_p);                        CHKERRXX(ierr);

  const double inverse_square_debye_length_in_domain = SQR(get_inverse_debye_length_in_domain());
  for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
    node_sampled_zero_p[k]  = 0.0;
    add_plus_p[k]           = inverse_square_debye_length_in_domain;
  }

  ierr = VecRestoreArray(add_plus, &add_plus_p);                    CHKERRXX(ierr);
  ierr = VecRestoreArray(node_sampled_zero, &node_sampled_zero_p);  CHKERRXX(ierr);
  rhs_minus = node_sampled_zero;
  rhs_plus  = node_sampled_zero;

  jump_solver->set_use_centroid_always(true);
  jump_solver->set_store_finite_volumes(true);
  jump_solver->set_jump_scheme(0);
  jump_solver->set_use_sc_scheme(false);
  jump_solver->set_integration_order(2);
  jump_solver->set_lip(biomolecules->parameters.lip());
  jump_solver->add_interface(MLS_INTERSECTION, biomolecules->phi, DIM(NULL, NULL, NULL), psi_hat_jump_interp_, eps_grad_n_psi_hat_jump_interp_);


  jump_solver->set_mu(mol_rel_permittivity, elec_rel_permittivity);
  class bc_wall_type_t : public WallBCDIM
  {
  public:
    BoundaryConditionType operator()(DIM(double, double, double)) const { return DIRICHLET; }
  } bc_wall_type;
  far_field_boundary_cond far_bc(this);
  jump_solver->set_wc(bc_wall_type, zero_cf);
  jump_solver->set_diag(node_sampled_zero,add_plus);
  jump_solver->set_use_taylor_correction(true);
  jump_solver->set_kink_treatment(true);
  jump_solver->set_rhs(rhs_minus, rhs_plus);

  make_sure_is_node_sampled(psi_hat);
  if(solve_subtimer != NULL)
  {
    solve_subtimer->stop(); solve_subtimer->read_duration();
    string timer_msg = "Solving nonlinear iterations ";
    solve_subtimer->start(timer_msg);
  }

  nb_iterations_for_setting_psi_hat = jump_solver->solve_nonlinear(psi_hat, upper_bound_residual, it_max, true);
  string timer_msg = "End of nonlinear iterations ";
  if(solve_subtimer != NULL)
  {
    solve_subtimer->stop(); solve_subtimer->read_duration();
    string timer_msg = "End of nonlinear iterations ";
    solve_subtimer->stop();
  }
  ierr = VecDestroy(add_plus);                        CHKERRXX(ierr);
  ierr = VecDestroy(eps_grad_n_psi_hat_jump);         CHKERRXX(ierr);
  ierr = VecDestroy(eps_grad_n_psi_hat_jump_xxyyzz);  CHKERRXX(ierr);
  ierr = VecDestroy(node_sampled_zero);               CHKERRXX(ierr);
  ierr = VecDestroy(psi_star_xxyyzz);                 CHKERRXX(ierr);

  if(solve_subtimer != NULL){
    delete solve_subtimer; solve_subtimer = NULL; }

  if(log_timer != NULL)
  {
    log_timer->stop(); log_timer->read_duration();
    delete log_timer;
  }

  psi_hat_is_set = true;
  return iter;
}

int my_p4est_biomolecules_solver_t::solve_nonlinear_v2(double upper_bound_residual, int it_max)
{
  int iter = 0;
  if (psi_hat_is_set)
    return iter;
  P4EST_ASSERT(all_parameters_are_set());
  P4EST_ASSERT(it_max >= 1);
  P4EST_ASSERT((it_max == 1) || (upper_bound_residual > 0.0));
  parStopWatch *log_timer = NULL, *solve_subtimer = NULL;
  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " \n");                                                                                 CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Solving the Poisson-Boltzmann equation on a %d/%d grid with %d proc(s) \n",
                        (int) biomolecules->parameters.lmin(), (int) biomolecules->parameters.lmax(), biomolecules->p4est->mpisize);                                  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The ionic charge is %d \n", ion_charge);                                               CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The far-field electrolyte density is %g m^{-3} \n", far_field_ion_density);            CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The temperature is %g K \n", temperature);                                             CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The inverse debye length %g A^{-1}, %g m^{-1}, or %g in domain units\n",
                        (get_inverse_debye_length_in_angstrom_inverse()), (get_inverse_debye_length_in_meters_inverse()), get_inverse_debye_length_in_domain());      CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "\n");                                                                                  CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, biomolecules->log_file, biomolecules->p4est->mpicomm);
      log_timer->start("Resolution of the nonlinear Poisson-Boltzmann Equation");
    }
  }
  if(biomolecules->timing_file != NULL)
  {
    P4EST_ASSERT(solve_subtimer == NULL);
    solve_subtimer = new parStopWatch(parStopWatch::root_timings, biomolecules->timing_file, biomolecules->p4est->mpicomm);
  }

  //parameter for nonlinear term
  param_list_t pl;

  param_t<int>    nonlinear_term_m       (pl, 0, "nonlinear_term_m",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
  param_t<int>    nonlinear_term_m_coeff (pl, 0, "nonlinear_term_m_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
  param_t<double> nonlinear_term_m_mag   (pl, 1, "nonlinear_term_m_mag",   "Scaling of nonlinear term in negative domain");

  param_t<int>    nonlinear_term_p       (pl, 0, "nonlinear_term_p",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
  param_t<int>    nonlinear_term_p_coeff (pl, 0, "nonlinear_term_p_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
  param_t<double> nonlinear_term_p_mag   (pl, 1, "nonlinear_term_p_mag",   "Scaling of nonlinear term in negative domain");

  param_t<int>    nonlinear_method (pl, 1, "nonlinear_method", "Method to solve nonlinear eqautions: 0 - solving for solution itself, 1 - solving for change in the solution");
  param_t<int>    nonlinear_itmax  (pl, 1000, "nonlinear_itmax", "Maximum iteration for solving nonlinear equations");
  param_t<double> nonlinear_tol    (pl, 1.e-10, "nonlinear_tol", "Tolerance for solving nonlinear equations");

  if(it_max ==1)  {
      nonlinear_term_m.val = 0;
      nonlinear_term_p.val = 0;
  }
  else {
      nonlinear_term_m.val = 2;
      nonlinear_term_p.val = 2;

      nonlinear_term_m_coeff.val = 0;
      nonlinear_term_m_mag.val = 0;

      const double inverse_square_debye_length_in_domain = SQR(get_inverse_debye_length_in_domain());
      nonlinear_term_p_coeff.val = 0;
      nonlinear_term_p_mag.val = inverse_square_debye_length_in_domain;
  }

  class nonlinear_term_cf_t: public CF_1
  {
    int *n;
    cf_value_type_t what;
  public:
    nonlinear_term_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
    double operator()(double u) const {
      switch (*n) {
        case 0:
          switch (what) {
            case VAL: return 0.;
            case DDX: return 0.;
            default: throw;
          }
        case 1:
          switch (what) {
            case VAL: return u;
            case DDX: return 1.;
            default: throw;
          }
        case 2:
          switch (what) {
            case VAL: return sinh(MIN(MAX(u,-500.),500.));
            case DDX: return cosh(MIN(MAX(u,-500.),500.));
            default: throw;
          }
        case 3:
          switch (what) {
            case VAL: return u/(1.+u);
            case DDX: return 1./SQR(1.+u);
            default: throw;
          }
        default:
          throw;
      }
    }
  };

  nonlinear_term_cf_t nonlinear_term_m_cf(VAL, nonlinear_term_m.val), nonlinear_term_m_prime_cf(DDX, nonlinear_term_m.val);
  nonlinear_term_cf_t nonlinear_term_p_cf(VAL, nonlinear_term_p.val), nonlinear_term_p_prime_cf(DDX, nonlinear_term_p.val);

  class nonlinear_term_coeff_cf_t: public CF_DIM
  {
    int *n;
    double *mag;
  public:
    nonlinear_term_coeff_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const {
      switch (*n) {
        case 0: return (*mag)*1.;
        case 1:
  #ifdef P4_TO_P8
          return (*mag)*cos(x+z)*exp(y);
  #else
          return (*mag)*cos(x)*exp(y);
  #endif
      }
    }
  };

  nonlinear_term_coeff_cf_t nonlinear_term_m_coeff_cf(nonlinear_term_m_coeff.val, nonlinear_term_m_mag.val);
  nonlinear_term_coeff_cf_t nonlinear_term_p_coeff_cf(nonlinear_term_p_coeff.val, nonlinear_term_p_mag.val);



  if(!psi_star_is_set)
  {
    if(solve_subtimer != NULL)
      solve_subtimer->start("Evaluating singular parts");
    make_sure_is_node_sampled(psi_star);
    double *psi_star_p = NULL;
    ierr = VecGetArray(psi_star, &psi_star_p);                        CHKERRXX(ierr);
    // sample the contribution of singular charges at grid nodes, but only in the inner domain(s)
    double xyz[P4EST_DIM];
    const double* phi_read_only_p = NULL;
    ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p);      CHKERRXX(ierr);
    for (size_t i = 0; i<biomolecules->nodes->indep_nodes.elem_count; ++i) {
//     if(phi_read_only_p[i] <= 5.0*biomolecules->parameters.layer_thickness()) //1.5 == safety factor
//      {
        node_xyz_fr_n(i, biomolecules->p4est, biomolecules->nodes, xyz);
        psi_star_p[i] = non_dimensional_coulomb_in_mol(DIM(xyz[0], xyz[1], xyz[2]));
//        if(isinf(psi_star_p[i])|| isnan(psi_star_p[i])) throw;
//      }
//     else
//        psi_star_p[i] = 0.0;
    }
    ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(psi_star, &psi_star_p);                    CHKERRXX(ierr);
//    biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, psi_star, 20, biomolecules->parameters.order_of_accuracy()); // Technically not needed but looks like this makes things more accurate... Why? No clue!
//    double psi_star_max;
//    double psi_star_min;
//    VecMax(psi_star,NULL,&psi_star_max);
//    VecMin(psi_star,NULL,&psi_star_min);
//    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_star  max is  %g and min is %g  \n", psi_star_max, psi_star_min ); CHKERRXX(ierr);

    if(solve_subtimer != NULL){
      solve_subtimer->stop(); solve_subtimer->read_duration(); }

    psi_star_is_set = true;
  }

  ierr = VecGhostUpdateBegin(psi_star, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (psi_star, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  double psi_star_max;
//  VecMax(psi_star, NULL, &psi_star_max);
//  std::cout << psi_star_max << std::endl;

  if(solve_subtimer != NULL)
    solve_subtimer->start("Initializing the solver");
  // Create a node-sampled zero vector (will be useful)
  Vec node_sampled_zero = NULL;
  make_sure_is_node_sampled(node_sampled_zero);
  // Create vector for the jump condition in normal gradient
  // Vec eps_grad_n_psi_hat_jump = NULL;

  Vec eps_grad_n_psi_hat_jump         = NULL;
  Vec eps_grad_n_psi_hat_jump_xxyyzz  = NULL;
  Vec psi_star_xxyyzz                 = NULL;
  make_sure_is_node_sampled(eps_grad_n_psi_hat_jump);
  ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &eps_grad_n_psi_hat_jump_xxyyzz);  CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &psi_star_xxyyzz);                 CHKERRXX(ierr);

  // Create vectors for the diagonal term in the outer domain
  Vec add_plus = NULL;
  make_sure_is_node_sampled(add_plus);

  // define rhs's (nonzero only for validation purposes)
  Vec rhs_minus = NULL, rhs_plus = NULL;

  calculate_jumps_in_normal_gradient(eps_grad_n_psi_hat_jump);
  biomolecules->neighbors->second_derivatives_central(eps_grad_n_psi_hat_jump, eps_grad_n_psi_hat_jump_xxyyzz);
  biomolecules->neighbors->second_derivatives_central(psi_star, psi_star_xxyyzz);
  my_p4est_interpolation_nodes_t eps_grad_n_psi_hat_jump_interp_(biomolecules->neighbors);
  eps_grad_n_psi_hat_jump_interp_.set_input(eps_grad_n_psi_hat_jump, eps_grad_n_psi_hat_jump_xxyyzz, quadratic_non_oscillatory_continuous_v2);
  //eps_grad_n_psi_hat_jump_interp_.set_input(eps_grad_n_psi_hat_jump, linear);

  my_p4est_interpolation_nodes_t psi_hat_jump_interp_(biomolecules->neighbors);
  psi_hat_jump_interp_.set_input(psi_star, psi_star_xxyyzz, quadratic_non_oscillatory_continuous_v2);
  //psi_hat_jump_interp_.set_input(psi_star, linear);



  double *node_sampled_zero_p = NULL, *add_plus_p = NULL;
  ierr = VecGetArray(node_sampled_zero, &node_sampled_zero_p);      CHKERRXX(ierr);
  ierr = VecGetArray(add_plus, &add_plus_p);                        CHKERRXX(ierr);
  const double inverse_square_debye_length_in_domain = SQR(get_inverse_debye_length_in_domain());
 for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
    node_sampled_zero_p[k]  = 0.0;
    if(it_max ==1)  {
      add_plus_p[k]           = inverse_square_debye_length_in_domain;
    }
    }

  ierr = VecRestoreArray(add_plus, &add_plus_p);                    CHKERRXX(ierr);
  ierr = VecRestoreArray(node_sampled_zero, &node_sampled_zero_p);  CHKERRXX(ierr);
  rhs_minus = node_sampled_zero;
  rhs_plus  = node_sampled_zero;
  jump_solver_v2->set_use_centroid_always(true);
  jump_solver_v2->set_store_finite_volumes(true);
  jump_solver_v2->set_jump_scheme(0);
  jump_solver_v2->set_use_sc_scheme(false);
  jump_solver_v2->set_integration_order(1);
  jump_solver_v2->set_lip(biomolecules->parameters.lip());

  jump_solver_v2->add_interface(MLS_INTERSECTION, biomolecules->phi, DIM(NULL, NULL, NULL), psi_hat_jump_interp_, eps_grad_n_psi_hat_jump_interp_);

  jump_solver_v2->set_mu(mol_rel_permittivity, elec_rel_permittivity);
  class bc_wall_type_t : public WallBCDIM
  {
  public:
    BoundaryConditionType operator()(DIM(double, double, double)) const { return DIRICHLET; }
  } bc_wall_type;
  far_field_boundary_cond far_bc(this);
  jump_solver_v2->set_wc(bc_wall_type, zero_cf);
  if(it_max==1){
    jump_solver_v2->set_diag(node_sampled_zero, add_plus);
  }
  else {
    jump_solver_v2->set_diag(node_sampled_zero, node_sampled_zero);
  }
  jump_solver_v2->set_use_taylor_correction(true);
  jump_solver_v2->set_kink_treatment(true);
  jump_solver_v2->set_rhs(rhs_minus, rhs_plus);

  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " Entering nonlinear solver \n"); CHKERRXX(ierr);

  make_sure_is_node_sampled(psi_hat);
  if(solve_subtimer != NULL)
  {
    solve_subtimer->stop(); solve_subtimer->read_duration();
    string timer_msg = "Solving nonlinear iterations ";
    solve_subtimer->start(timer_msg);
  }
  if (nonlinear_term_m() ==0 && nonlinear_term_p() ==0)
  {
    jump_solver_v2->solve(psi_hat, 0);
  }
  else
  {
    Vec nonlinear_term_m_coeff_sampled;
    Vec nonlinear_term_p_coeff_sampled;
    //ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " f u 3802 \n"); CHKERRXX(ierr);
    ierr = VecDuplicate(node_sampled_zero, &nonlinear_term_m_coeff_sampled); CHKERRXX(ierr);
    ierr = VecDuplicate(node_sampled_zero, &nonlinear_term_p_coeff_sampled); CHKERRXX(ierr);

    sample_cf_on_nodes(biomolecules->p4est, biomolecules->nodes, nonlinear_term_m_coeff_cf, nonlinear_term_m_coeff_sampled);
    sample_cf_on_nodes(biomolecules->p4est, biomolecules->nodes, nonlinear_term_p_coeff_cf, nonlinear_term_p_coeff_sampled);
    //ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " f u 3808 \n"); CHKERRXX(ierr);
    jump_solver_v2->set_nonlinear_term(nonlinear_term_m_coeff_sampled, nonlinear_term_m_cf, nonlinear_term_m_prime_cf,
                              nonlinear_term_p_coeff_sampled, nonlinear_term_p_cf, nonlinear_term_p_prime_cf);
    //ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " f u 3811 \n"); CHKERRXX(ierr);
    jump_solver_v2->set_solve_nonlinear_parameters(nonlinear_method.val, nonlinear_itmax.val, nonlinear_tol.val, 0);
    //ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " f u 3813 \n"); CHKERRXX(ierr);
    VecSetGhost(psi_hat,0);
    nb_iterations_for_setting_psi_hat = jump_solver_v2->solve_nonlinear(psi_hat, 0);
    //ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " f u 3816 \n"); CHKERRXX(ierr);
    ierr = VecDestroy(nonlinear_term_m_coeff_sampled); CHKERRXX(ierr);
    ierr = VecDestroy(nonlinear_term_p_coeff_sampled); CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " End of nonlinear solver \n"); CHKERRXX(ierr);

  // nb_iterations_for_setting_psi_hat = jump_solver->solve_nonlinear(psi_hat, upper_bound_residual, it_max, true);
  string timer_msg = "End of nonlinear iterations ";
  if(solve_subtimer != NULL)
  {
    solve_subtimer->stop(); solve_subtimer->read_duration();
    string timer_msg = "End of nonlinear iterations ";
    solve_subtimer->stop();
  }
  ierr = VecDestroy(add_plus);                        CHKERRXX(ierr);
  ierr = VecDestroy(eps_grad_n_psi_hat_jump);         CHKERRXX(ierr);
  ierr = VecDestroy(eps_grad_n_psi_hat_jump_xxyyzz);  CHKERRXX(ierr);
  ierr = VecDestroy(node_sampled_zero);               CHKERRXX(ierr);
  ierr = VecDestroy(psi_star_xxyyzz);                 CHKERRXX(ierr);

  if(solve_subtimer != NULL){
    delete solve_subtimer; solve_subtimer = NULL; }

  if(log_timer != NULL)
  {
    log_timer->stop(); log_timer->read_duration();
    delete log_timer;
  }
  double psi_hat_max;
  double psi_hat_min;
//  VecMax(psi_hat,NULL,&psi_hat_max);
//  VecMin(psi_hat,NULL,&psi_hat_min);
//  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_hat  max is  %g and min is %g  \n", psi_hat_max, psi_hat_min ); CHKERRXX(ierr);

  psi_hat_is_set = true;
  return iter;
}
int my_p4est_biomolecules_solver_t::solve_nonlinear_first_approach(double upper_bound_residual, int it_max)
{
  int iter = 0;
  if (psi_hat_is_set)
    return iter;
  P4EST_ASSERT(all_parameters_are_set());
  P4EST_ASSERT(it_max >= 1);
  P4EST_ASSERT((it_max == 1) || (upper_bound_residual > 0.0));
  parStopWatch *log_timer = NULL, *solve_subtimer = NULL;
  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " \n");                                                                                 CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Solving the Poisson-Boltzmann equation on a %d/%d grid with %d proc(s) \n",
                        (int) biomolecules->parameters.lmin(), (int) biomolecules->parameters.lmax(), biomolecules->p4est->mpisize);                                  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The ionic charge is %d \n", ion_charge);                                               CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The far-field electrolyte density is %g m^{-3} \n", far_field_ion_density);            CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The temperature is %g K \n", temperature);                                             CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The inverse debye length %g A^{-1}, %g m^{-1}, or %g in domain units\n",
                        (get_inverse_debye_length_in_angstrom_inverse()), (get_inverse_debye_length_in_meters_inverse()), get_inverse_debye_length_in_domain());      CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "\n");                                                                                  CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, biomolecules->log_file, biomolecules->p4est->mpicomm);
      log_timer->start("Resolution of the nonlinear Poisson-Boltzmann Equation");
    }
  }
  if(biomolecules->timing_file != NULL)
  {
    P4EST_ASSERT(solve_subtimer == NULL);
    solve_subtimer = new parStopWatch(parStopWatch::root_timings, biomolecules->timing_file, biomolecules->p4est->mpicomm);
  }

  //parameter for nonlinear term
  param_list_t pl;

  param_t<int>    nonlinear_term_m       (pl, 0, "nonlinear_term_m",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
  param_t<int>    nonlinear_term_m_coeff (pl, 0, "nonlinear_term_m_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
  param_t<double> nonlinear_term_m_mag   (pl, 1, "nonlinear_term_m_mag",   "Scaling of nonlinear term in negative domain");

  param_t<int>    nonlinear_term_p       (pl, 0, "nonlinear_term_p",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
  param_t<int>    nonlinear_term_p_coeff (pl, 0, "nonlinear_term_p_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
  param_t<double> nonlinear_term_p_mag   (pl, 1, "nonlinear_term_p_mag",   "Scaling of nonlinear term in negative domain");

  param_t<int>    nonlinear_method (pl, 1, "nonlinear_method", "Method to solve nonlinear eqautions: 0 - solving for solution itself, 1 - solving for change in the solution");
  param_t<int>    nonlinear_itmax  (pl, 1000, "nonlinear_itmax", "Maximum iteration for solving nonlinear equations");
  param_t<double> nonlinear_tol    (pl, 1.e-10, "nonlinear_tol", "Tolerance for solving nonlinear equations");

  if(it_max ==1)  {
      nonlinear_term_m.val = 0;
      nonlinear_term_p.val = 0;
  }
  else {
      nonlinear_term_m.val = 0;
      nonlinear_term_p.val = 0;

      nonlinear_term_m_coeff.val = 0;
      nonlinear_term_m_mag.val = 0;

      const double inverse_square_debye_length_in_domain = SQR(get_inverse_debye_length_in_domain());
      nonlinear_term_p_coeff.val = 0;
      nonlinear_term_p_mag.val = inverse_square_debye_length_in_domain;
  }

  class nonlinear_term_cf_t: public CF_1
  {
    int *n;
    cf_value_type_t what;
  public:
    nonlinear_term_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
    double operator()(double u) const {
      switch (*n) {
        case 0:
          switch (what) {
            case VAL: return 0.;
            case DDX: return 0.;
            default: throw;
          }
        case 1:
          switch (what) {
            case VAL: return u;
            case DDX: return 1.;
            default: throw;
          }
        case 2:
          switch (what) {
            case VAL: return sinh(MIN(MAX(u,-500.),500.));
            case DDX: return cosh(MIN(MAX(u,-500.),500.));
            default: throw;
          }
        case 3:
          switch (what) {
            case VAL: return u/(1.+u);
            case DDX: return 1./SQR(1.+u);
            default: throw;
          }
        default:
          throw;
      }
    }
  };

  nonlinear_term_cf_t nonlinear_term_m_cf(VAL, nonlinear_term_m.val), nonlinear_term_m_prime_cf(DDX, nonlinear_term_m.val);
  nonlinear_term_cf_t nonlinear_term_p_cf(VAL, nonlinear_term_p.val), nonlinear_term_p_prime_cf(DDX, nonlinear_term_p.val);

  class nonlinear_term_coeff_cf_t: public CF_DIM
  {
    int *n;
    double *mag;
  public:
    nonlinear_term_coeff_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const {
      switch (*n) {
        case 0: return (*mag)*1.;
        case 1:
  #ifdef P4_TO_P8
          return (*mag)*cos(x+z)*exp(y);
  #else
          return (*mag)*cos(x)*exp(y);
  #endif
      }
    }
  };

  nonlinear_term_coeff_cf_t nonlinear_term_m_coeff_cf(nonlinear_term_m_coeff.val, nonlinear_term_m_mag.val);
  nonlinear_term_coeff_cf_t nonlinear_term_p_coeff_cf(nonlinear_term_p_coeff.val, nonlinear_term_p_mag.val);



  if(!psi_star_is_set)
  {
    if(solve_subtimer != NULL)
      solve_subtimer->start("Evaluating singular parts");
    make_sure_is_node_sampled(psi_star);
    double *psi_star_p = NULL;
    ierr = VecGetArray(psi_star, &psi_star_p);                        CHKERRXX(ierr);
    // sample the contribution of singular charges at grid nodes, but only in the inner domain(s)
    double xyz[P4EST_DIM];
    const double* phi_read_only_p = NULL;
    ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p);      CHKERRXX(ierr);
    for (size_t i = 0; i<biomolecules->nodes->indep_nodes.elem_count; ++i) {
//     if(phi_read_only_p[i] <= 5.0*biomolecules->parameters.layer_thickness()) //1.5 == safety factor
//      {
        node_xyz_fr_n(i, biomolecules->p4est, biomolecules->nodes, xyz);
        psi_star_p[i] = -non_dimensional_coulomb_in_mol(DIM(xyz[0], xyz[1], xyz[2])); // to be used for the jump condition for psi_naught
//        if(isinf(psi_star_p[i])|| isnan(psi_star_p[i])) throw;
//      }
//     else
//        psi_star_p[i] = 0.0;
    }
    ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(psi_star, &psi_star_p);   CHKERRXX(ierr);

//    biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, psi_star, 20, biomolecules->parameters.order_of_accuracy()); // Technically not needed but looks like this makes things more accurate... Why? No clue!

    if(solve_subtimer != NULL){
      solve_subtimer->stop(); solve_subtimer->read_duration(); }

    psi_star_is_set = true;
  }

  ierr = VecGhostUpdateBegin(psi_star, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (psi_star, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  make_sure_is_node_sampled(psi_naught);
  make_sure_is_node_sampled(psi_bar);
  double *psi_bar_p= NULL, *psi_naught_p=NULL;
  ierr =VecGetArray(psi_naught, &psi_naught_p); CHKERRXX(ierr);
  ierr =VecGetArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);
  for (size_t i = 0; i<biomolecules->nodes->indep_nodes.elem_count; ++i){
    psi_bar_p[i]=0.0;
    psi_naught_p[i]=0.0;
  }
  ierr =VecRestoreArray(psi_naught, &psi_naught_p); CHKERRXX(ierr);
  ierr =VecRestoreArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);

  // solve poisson equation to obtain the smooth part of the solution in the inner domain
  Vec psi_star_xxyyzz=NULL;
  ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &psi_star_xxyyzz);  CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t bc_interface_value(biomolecules->neighbors);
  switch(biomolecules-> parameters.order_of_accuracy()){
  case 1:
    bc_interface_value.set_input(psi_star,linear);
    break;
  case 2:
    biomolecules->neighbors->second_derivatives_central(psi_star, psi_star_xxyyzz);
    bc_interface_value.set_input(psi_star, psi_star_xxyyzz, quadratic_non_oscillatory_continuous_v2);
  break;
  }
#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif

  struct:
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
    double operator()(double, double
                  #ifdef P4_TO_P8
                      ,double
                  #endif
                      ) const {return 0.0; }
  }bc_wall_value;

  struct:
    #ifdef P4_TO_P8
      WallBC3D
    #else
      WallBC2D
    #endif
  {
    BoundaryConditionType operator()(double, double
                                 #ifdef P4_TO_P8
                                     ,double
                                 #endif
                                     ) const {return DIRICHLET; }
  }bc_wall_type;
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_value);
  node_solver->set_bc(bc);
  node_solver->set_phi(biomolecules->phi);
  node_solver->set_rhs(psi_naught);
  node_solver->solve(psi_naught);
  double psi_naught_max;
  double psi_naught_min;
  VecMax(psi_naught,NULL,&psi_naught_max);
  VecMin(psi_naught,NULL,&psi_naught_min);
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_naught  max is  %g and min is %g  \n", psi_naught_max, psi_naught_min ); CHKERRXX(ierr);
  biomolecules->ls->extend_Over_Interface_TVD_Full(biomolecules->phi, psi_naught,20, biomolecules->parameters.order_of_accuracy());
  Vec psi_naught_values_inside_only=NULL;
  ierr = VecDuplicate(psi_naught,&psi_naught_values_inside_only);
  double *psi_naught_values_inside_only_p = NULL;
  const double *psi_naught_read_only_p = NULL;
  double *psi_star_p = NULL;
  const double *phi_read_only_p= NULL;
  ierr = VecGetArray(psi_star, &psi_star_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_naught, &psi_naught_read_only_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_bar,&psi_bar_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p);
  ierr = VecGetArray(psi_naught_values_inside_only,&psi_naught_values_inside_only_p); CHKERRXX(ierr);
  for(size_t i =0; i< biomolecules->nodes->indep_nodes.elem_count; ++i)
  {
    psi_star_p[i]*= -1.0;
    psi_bar_p[i]= psi_star_p[i] +psi_naught_read_only_p[i];
//    if(phi_read_only_p[i] <= 1.5*biomolecules->parameters.layer_thickness()|| (fabs(psi_naught_read_only_p[i]>EPS)))
//      psi_bar_p[i]= psi_star_p[i] +psi_naught_read_only_p[i];
//    else {
//      psi_bar_p[i]=0.0;
//    }
  }
  for(size_t i =0; i< biomolecules->nodes->indep_nodes.elem_count; ++i)
  {
    if(phi_read_only_p[i] <= 1.5*biomolecules->parameters.layer_thickness())
      psi_naught_values_inside_only_p[i]= psi_naught_read_only_p[i];
    else {
      psi_naught_values_inside_only_p[i]=0.0;
    }
  }
  ierr = VecRestoreArray(psi_naught_values_inside_only,&psi_naught_values_inside_only_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p); phi_read_only_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_bar, &psi_bar_p); psi_bar_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_naught, &psi_naught_read_only_p); psi_naught_read_only_p= NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_star, &psi_star_p); psi_star_p= NULL; CHKERRXX(ierr);
  double psi_star_max;
  double psi_star_min;
  VecMax(psi_star,NULL,&psi_star_max);
  VecMin(psi_star,NULL,&psi_star_min);
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_star  max is  %g and min is %g  \n", psi_star_max, psi_star_min ); CHKERRXX(ierr);
  double psi_naught_values_inside_only_max;
  double psi_naught_values_inside_only_min;
  VecMax(psi_naught_values_inside_only,NULL,&psi_naught_values_inside_only_max);
  VecMin(psi_naught_values_inside_only,NULL,&psi_naught_values_inside_only_min);
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_naught_values_inside_only  max is  %g and min is %g  \n", psi_naught_values_inside_only_max, psi_naught_values_inside_only_min ); CHKERRXX(ierr);
  ierr= VecDestroy(psi_naught_values_inside_only); CHKERRXX(ierr);
    // Once psi_bar is calculated, it needs to be extended over the interface so that its normal gradient can
    // be correctly calculated at the interface for imposing the jump condition on the normal gradient of
    // psi_hat afterwards.
    // If the normal gradient of psi_bar needs to be evaluated with order of accuracy 1 (resp. 2),
    // psi bar must be extended over the interface with a 2nd (resp. 3rd) order accurate method.
    // Therefore, the first (resp. and second) derivative(s) of psi_bar needs to be extended as well
    // NOTE: it is known that the Shortley-Weller method actually leads to 3rd order accurate results close
    // to the interface (leading in turn the superconvergence for the gradient). If psi_star is evaluated at
    // the interface with 3rd order accuracy (second order non-oscillatory interpolation method), the gradient
    // will be second-order accurate close to the interface and, hopefully, still second order accurate after
    // extension of psi_bar over the interface.
    // biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, psi_bar,20, biomolecules->parameters.order_of_accuracy());

  if(psi_star_xxyyzz != NULL){
      ierr = VecDestroy(psi_star_xxyyzz); psi_star_xxyyzz = NULL;
  }


    if(solve_subtimer != NULL)
      solve_subtimer->start("Initializing the solver");
    // Create a node-sampled zero vector (will be useful)
    Vec node_sampled_zero = NULL;
    make_sure_is_node_sampled(node_sampled_zero);
    // Create vector for the jump condition in normal gradient
    // Vec eps_grad_n_psi_hat_jump = NULL;

    Vec eps_grad_n_psi_hat_jump         = NULL;
    Vec eps_grad_n_psi_hat_jump_xxyyzz  = NULL;
    make_sure_is_node_sampled(eps_grad_n_psi_hat_jump);
    ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &eps_grad_n_psi_hat_jump_xxyyzz);  CHKERRXX(ierr);

    // Create vectors for the diagonal term in the outer domain
    Vec add_plus = NULL;
    make_sure_is_node_sampled(add_plus);

    // define rhs's (nonzero only for validation purposes)
    Vec rhs_minus = NULL, rhs_plus = NULL;

    calculate_jumps_in_normal_gradient_with_psi_bar(eps_grad_n_psi_hat_jump);
    biomolecules->neighbors->second_derivatives_central(eps_grad_n_psi_hat_jump, eps_grad_n_psi_hat_jump_xxyyzz);
    my_p4est_interpolation_nodes_t eps_grad_n_psi_hat_jump_interp_(biomolecules->neighbors);
    eps_grad_n_psi_hat_jump_interp_.set_input(eps_grad_n_psi_hat_jump, eps_grad_n_psi_hat_jump_xxyyzz, quadratic_non_oscillatory_continuous_v2);
    //eps_grad_n_psi_hat_jump_interp_.set_input(eps_grad_n_psi_hat_jump, linear);

    Vec psi_bar_xxyyzz                 = NULL;
    ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &psi_bar_xxyyzz);                 CHKERRXX(ierr);

    biomolecules->neighbors->second_derivatives_central(psi_bar, psi_bar_xxyyzz);
     my_p4est_interpolation_nodes_t psi_hat_jump_interp_(biomolecules->neighbors);
    psi_hat_jump_interp_.set_input(psi_bar, psi_bar_xxyyzz, quadratic_non_oscillatory_continuous_v2);
    //psi_hat_jump_interp_.set_input(psi_star, linear);


    double *node_sampled_zero_p = NULL, *add_plus_p = NULL;
    ierr = VecGetArray(node_sampled_zero, &node_sampled_zero_p);      CHKERRXX(ierr);
    ierr = VecGetArray(add_plus, &add_plus_p);                        CHKERRXX(ierr);
    const double inverse_square_debye_length_in_domain = SQR(get_inverse_debye_length_in_domain());
   for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
      node_sampled_zero_p[k]  = 0.0;
      if(it_max ==1)  {
        add_plus_p[k]           = inverse_square_debye_length_in_domain;
      }
    }

    ierr = VecRestoreArray(add_plus, &add_plus_p);                    CHKERRXX(ierr);
    ierr = VecRestoreArray(node_sampled_zero, &node_sampled_zero_p);  CHKERRXX(ierr);
    rhs_minus = node_sampled_zero;
    rhs_plus  = node_sampled_zero;
    jump_solver_v2->set_use_centroid_always(true);
    jump_solver_v2->set_store_finite_volumes(true);
    jump_solver_v2->set_jump_scheme(0);
    jump_solver_v2->set_use_sc_scheme(false);
    jump_solver_v2->set_integration_order(1);
    jump_solver_v2->set_lip(biomolecules->parameters.lip());

    jump_solver_v2->add_interface(MLS_INTERSECTION, biomolecules->phi, DIM(NULL, NULL, NULL), psi_hat_jump_interp_, eps_grad_n_psi_hat_jump_interp_);

    jump_solver_v2->set_mu(mol_rel_permittivity, elec_rel_permittivity);
    class bc_wall_type_t : public WallBCDIM
    {
    public:
      BoundaryConditionType operator()(DIM(double, double, double)) const { return DIRICHLET; }
    } bc_wall_type_nl;
    far_field_boundary_cond far_bc(this);
    jump_solver_v2->set_wc(bc_wall_type_nl, zero_cf);
    if(it_max==1){
      jump_solver_v2->set_diag(node_sampled_zero, add_plus);
    }
    else {
      jump_solver_v2->set_diag(node_sampled_zero, node_sampled_zero);
    }
    jump_solver_v2->set_use_taylor_correction(true);
    jump_solver_v2->set_kink_treatment(true);
    jump_solver_v2->set_rhs(rhs_minus, rhs_plus);


    make_sure_is_node_sampled(psi_hat);
    if(solve_subtimer != NULL)
    {
      solve_subtimer->stop(); solve_subtimer->read_duration();
      string timer_msg = "Solving nonlinear iterations ";
      solve_subtimer->start(timer_msg);
    }
    if (nonlinear_term_m() ==0 && nonlinear_term_p() ==0)
    {
      jump_solver_v2->solve(psi_hat, 0);
    }
    else
    {
      Vec nonlinear_term_m_coeff_sampled;
      Vec nonlinear_term_p_coeff_sampled;

      ierr = VecDuplicate(node_sampled_zero, &nonlinear_term_m_coeff_sampled); CHKERRXX(ierr);
      ierr = VecDuplicate(node_sampled_zero, &nonlinear_term_p_coeff_sampled); CHKERRXX(ierr);

      sample_cf_on_nodes(biomolecules->p4est, biomolecules->nodes, nonlinear_term_m_coeff_cf, nonlinear_term_m_coeff_sampled);
      sample_cf_on_nodes(biomolecules->p4est, biomolecules->nodes, nonlinear_term_p_coeff_cf, nonlinear_term_p_coeff_sampled);

      jump_solver_v2->set_nonlinear_term(nonlinear_term_m_coeff_sampled, nonlinear_term_m_cf, nonlinear_term_m_prime_cf,
                                nonlinear_term_p_coeff_sampled, nonlinear_term_p_cf, nonlinear_term_p_prime_cf);

      jump_solver_v2->set_solve_nonlinear_parameters(nonlinear_method.val, nonlinear_itmax.val, nonlinear_tol.val, 0);

      VecSetGhost(psi_hat,0);
      nb_iterations_for_setting_psi_hat = jump_solver_v2->solve_nonlinear(psi_hat, 0);

      ierr = VecDestroy(nonlinear_term_m_coeff_sampled); CHKERRXX(ierr);
      ierr = VecDestroy(nonlinear_term_p_coeff_sampled); CHKERRXX(ierr);
    }

    // nb_iterations_for_setting_psi_hat = jump_solver->solve_nonlinear(psi_hat, upper_bound_residual, it_max, true);
    string timer_msg = "End of nonlinear iterations ";
    if(solve_subtimer != NULL)
    {
      solve_subtimer->stop(); solve_subtimer->read_duration();
      string timer_msg = "End of nonlinear iterations ";
      solve_subtimer->stop();
    }
    double *psi_hat_p= NULL;


    ierr = VecGetArray(psi_bar,&psi_bar_p); CHKERRXX(ierr);
    ierr = VecGetArray(psi_hat,&psi_hat_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p);
    for(size_t i =0; i< biomolecules->nodes->indep_nodes.elem_count; ++i)
    {
      //ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "psi_bar is %g , psi_hat is %g  \n", psi_bar_p[i],psi_hat_p[i] ); CHKERRXX(ierr);
      psi_hat_p[i]+=psi_bar_p[i];
      //ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_hat is %g  \n", psi_hat_p[i] ); CHKERRXX(ierr);

    }
    ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p); phi_read_only_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArray(psi_bar, &psi_bar_p); psi_bar_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArray(psi_hat,&psi_hat_p); CHKERRXX(ierr);

    double psi_hat_max;
    double psi_hat_min;
    VecMax(psi_hat,NULL,&psi_hat_max);
    VecMin(psi_hat,NULL,&psi_hat_min);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_hat  max is  %g and min is %g  \n", psi_hat_max, psi_hat_min ); CHKERRXX(ierr);

    ierr = VecDestroy(add_plus);                        CHKERRXX(ierr);
    ierr = VecDestroy(eps_grad_n_psi_hat_jump);         CHKERRXX(ierr);
    ierr = VecDestroy(eps_grad_n_psi_hat_jump_xxyyzz);  CHKERRXX(ierr);
    ierr = VecDestroy(node_sampled_zero);               CHKERRXX(ierr);

    if(solve_subtimer != NULL){
      delete solve_subtimer; solve_subtimer = NULL; }

    if(log_timer != NULL)
    {
      log_timer->stop(); log_timer->read_duration();
      delete log_timer;
    }
    psi_hat_is_set = true;
    return iter;
}

//void my_p4est_biomolecules_solver_t::get_solvation_free_energy(const bool &nonlinear_flag)
double my_p4est_biomolecules_solver_t::get_solvation_free_energy(const bool &nonlinear_flag)
{
#ifndef P4_TO_P8
  // this makes sense only in 3D
#ifdef CASL_THROWS
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): the solvation free energy is not properly defined in 2D, forget it! \n    Returning... \n"); CHKERRXX(ierr);
#endif
  return 0;// return ;
#else
  P4EST_ASSERT(all_parameters_are_set());
//  if(!psi_hat_is_set || (psi_hat_set_for_linear_pb() && nonlinear_flag) || (psi_hat_set_for_nonlinear_pb() && !nonlinear_flag))
//  {
//#ifdef CASL_THROWS
//    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): the solution of the general nonlinear Poisson-Boltzmann equation is not known or not consistent, it will be calculated...\n"); CHKERRXX(ierr);
//#endif
//    reset_psi_hat();
//    if(nonlinear_flag)
//      solve_nonlinear_v2();
//    else
//      solve_linear();
//  }

  P4EST_ASSERT((psi_star != NULL) && (psi_hat != NULL));

  parStopWatch* log_timer = NULL;
  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " \n");                                                                                 CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Calculating the solvation free energy a %d/%d grid with %d proc(s) \n",
                        (int) biomolecules->parameters.lmin(), (int) biomolecules->parameters.lmax(), biomolecules->p4est->mpisize);                                  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "\n");                                                                                  CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, biomolecules->log_file, biomolecules->p4est->mpicomm);
      log_timer->start("Calculating the solvation free energy");
    }
  }

  // contribution from the electrolyte
  Vec integrand = NULL, psi_hat_xxyyzz = NULL;
  make_sure_is_node_sampled(integrand);
  double *integrand_p = NULL, *phi_p = NULL;
  const double *psi_hat_read_only_p = NULL;
  ierr = VecGetArray(integrand, &integrand_p);            CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_hat, &psi_hat_read_only_p);  CHKERRXX(ierr);
  ierr = VecGetArray(biomolecules->phi, &phi_p);          CHKERRXX(ierr);
  for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
    if(phi_p[k] > 0.0)
    {
      if(nonlinear_flag)
        integrand_p[k] = kB*temperature*far_field_ion_density*(psi_hat_read_only_p[k]*sinh(psi_hat_read_only_p[k])-2.0*(cosh(psi_hat_read_only_p[k]) - 1.0)); // relevant value
      else
        integrand_p[k] = kB*temperature*far_field_ion_density*(psi_hat_read_only_p[k]*psi_hat_read_only_p[k]*2); // relevant value
    }
    else
      integrand_p[k] = 0.0;               // needs to be extrapolated (bc of jump on the normal derivative)
    phi_p[k] *= -1.0;                     // we need to integrate over the exterior domain --> reverse the levelset
  }
  ierr = VecRestoreArray(biomolecules->phi, &phi_p);                  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_hat, &psi_hat_read_only_p);          CHKERRXX(ierr);
  ierr = VecRestoreArray(integrand, &integrand_p);                    CHKERRXX(ierr);
  biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, integrand, 20, 2); // 20 for the number of iterations, default parameter
  solvation_free_energy = integrate_over_negative_domain(biomolecules->p4est, biomolecules->nodes, biomolecules->phi, integrand)*(pow(length_scale_in_meter(), 3.0));
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Integration over negative domain is %g J, that is %g kJ/mol ,that is %g kcal/mol \n", solvation_free_energy, solvation_free_energy*avogadro_number*0.001, solvation_free_energy*avogadro_number*0.000239006); CHKERRXX(ierr);

  Vec phi_ghost_loc = NULL;
  ierr = VecGhostGetLocalForm(biomolecules->phi, &phi_ghost_loc);     CHKERRXX(ierr);
  ierr = VecScale(phi_ghost_loc, -1.0);                               CHKERRXX(ierr); // reverse the levelset function to get back to original state
  ierr = VecGhostRestoreLocalForm(biomolecules->phi, &phi_ghost_loc); CHKERRXX(ierr);

  // contributions from singular point charges
  double integral_contribution_from_singular_charges = 0.0;

  my_p4est_interpolation_nodes_t interpolate_psi_hat(biomolecules->neighbors);
  switch (biomolecules->parameters.order_of_accuracy()) {
  case 1:
    interpolate_psi_hat.set_input(psi_hat, linear);
    break;
  case 2:
    ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &psi_hat_xxyyzz); CHKERRXX(ierr);
    biomolecules->neighbors->second_derivatives_central(psi_hat, psi_hat_xxyyzz);
    interpolate_psi_hat.set_input(psi_hat, psi_hat_xxyyzz, quadratic_non_oscillatory_continuous_v2);
    break;
  default:
    throw  std::invalid_argument("my_p4est_biomolecules_solver_t::get_solvation_energy(), the order of accuracy should be either 1 or 2!");
    break;
  }
  int total_nb_charged_atoms = 0.0;
  for (size_t mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
  {
    const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
    total_nb_charged_atoms += mol.get_number_of_charged_atoms();
  }
  int proc_has_atom_if_rank_below = MIN(total_nb_charged_atoms, biomolecules->p4est->mpisize);

  int first_charged_atom_idx          = MIN(biomolecules->p4est->mpirank*total_nb_charged_atoms/proc_has_atom_if_rank_below, total_nb_charged_atoms);
  int idx_of_charged_atom_after_last  = MIN((biomolecules->p4est->mpirank+1)*total_nb_charged_atoms/proc_has_atom_if_rank_below, total_nb_charged_atoms);
  int nb_atoms_for_me                 = idx_of_charged_atom_after_last - first_charged_atom_idx;

  vector<double> point_values_of_psi_hat(nb_atoms_for_me, 0.0);
  int charged_atom_idx_offset         = 0;
  int global_charged_atom_idx;
  p4est_locidx_t local_idx = 0;
  if(first_charged_atom_idx < total_nb_charged_atoms)
  {
    for (size_t mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      if((charged_atom_idx_offset + mol.get_number_of_charged_atoms() >= first_charged_atom_idx) && (charged_atom_idx_offset < idx_of_charged_atom_after_last))
        for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
        {
          global_charged_atom_idx = charged_atom_idx_offset + charged_atom_idx;
          if((first_charged_atom_idx <= global_charged_atom_idx) && (global_charged_atom_idx < idx_of_charged_atom_after_last))
          {
            const Atom* a = mol.get_charged_atom(charged_atom_idx);
            interpolate_psi_hat.add_point(local_idx++, a->xyz_c);
            P4EST_ASSERT(local_idx <= nb_atoms_for_me);
          }
        }
      charged_atom_idx_offset += mol.get_number_of_charged_atoms();
    }
  }
  interpolate_psi_hat.interpolate(point_values_of_psi_hat.data());
  local_idx = 0;
  charged_atom_idx_offset         = 0;
  if(first_charged_atom_idx < total_nb_charged_atoms)
  {
    for (size_t mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      if((charged_atom_idx_offset + mol.get_number_of_charged_atoms() >= first_charged_atom_idx) && (charged_atom_idx_offset < idx_of_charged_atom_after_last))
      {
        for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
        {
          global_charged_atom_idx = charged_atom_idx_offset + charged_atom_idx;

          if((first_charged_atom_idx <= global_charged_atom_idx) && (global_charged_atom_idx < idx_of_charged_atom_after_last))
          {
            const Atom* a = mol.get_charged_atom(charged_atom_idx);
            integral_contribution_from_singular_charges += (0.5*a->q*kB*temperature/((double) ion_charge))*(point_values_of_psi_hat.at(local_idx++));
            P4EST_ASSERT(local_idx <= nb_atoms_for_me);
          }
        }
      }
      charged_atom_idx_offset += mol.get_number_of_charged_atoms();
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &integral_contribution_from_singular_charges, 1, MPI_DOUBLE, MPI_SUM, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  solvation_free_energy += integral_contribution_from_singular_charges;
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Contribution of singular charges is %g J, that is %g kJ/mol ,that is %g kcal/mol \n", integral_contribution_from_singular_charges, integral_contribution_from_singular_charges*avogadro_number*0.001, integral_contribution_from_singular_charges*avogadro_number*0.000239006); CHKERRXX(ierr);

  ierr = VecDestroy(integrand);       CHKERRXX(ierr);
  ierr = VecDestroy(psi_hat_xxyyzz);  CHKERRXX(ierr);

  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The value of the solvation free energy is %g J, that is %g kJ/mol ,that is %g kcal/mol \n", solvation_free_energy, solvation_free_energy*avogadro_number*0.001, solvation_free_energy*avogadro_number*0.000239006); CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer->stop(); log_timer->read_duration();
      delete log_timer; log_timer = NULL;
    }
  }
  return (solvation_free_energy*avogadro_number*0.001);
#endif
}
double my_p4est_biomolecules_solver_t::get_solvation_free_energy_first_approach(const bool &nonlinear_flag)
{
#ifndef P4_TO_P8
  // this makes sense only in 3D
#ifdef CASL_THROWS
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): the solvation free energy is not properly defined in 2D, forget it! \n    Returning... \n"); CHKERRXX(ierr);
#endif
  return 0;// return ;
#else
  P4EST_ASSERT(all_parameters_are_set());
  if(!psi_hat_is_set || (psi_hat_set_for_linear_pb() && nonlinear_flag) || (psi_hat_set_for_nonlinear_pb() && !nonlinear_flag))
  {
#ifdef CASL_THROWS
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): the solution of the general nonlinear Poisson-Boltzmann equation is not known or not consistent, it will be calculated...\n"); CHKERRXX(ierr);
#endif
    reset_psi_hat();
    if(nonlinear_flag)
      solve_nonlinear_first_approach();
    else
      solve_linear();
  }

  P4EST_ASSERT((psi_star != NULL) && (psi_hat != NULL));

  parStopWatch* log_timer = NULL;
  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " \n");                                                                                 CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Calculating the solvation free energy a %d/%d grid with %d proc(s) \n",
                        (int) biomolecules->parameters.lmin(), (int) biomolecules->parameters.lmax(), biomolecules->p4est->mpisize);                                  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "\n");                                                                                  CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, biomolecules->log_file, biomolecules->p4est->mpicomm);
      log_timer->start("Calculating the solvation free energy");
    }
  }

  // contribution from the electrolyte
  Vec integrand = NULL, psi_hat_plus_psi_naught_xxyyzz = NULL;
  make_sure_is_node_sampled(integrand);
  double *integrand_p = NULL, *phi_p = NULL;
  const double *psi_hat_read_only_p = NULL;
  ierr = VecGetArray(integrand, &integrand_p);            CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_hat, &psi_hat_read_only_p);  CHKERRXX(ierr);
  ierr = VecGetArray(biomolecules->phi, &phi_p);          CHKERRXX(ierr);
  for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
    if(phi_p[k] > 0.0)
    {
      if(nonlinear_flag)
        integrand_p[k] = kB*temperature*far_field_ion_density*(psi_hat_read_only_p[k]*sinh(psi_hat_read_only_p[k])-2.0*(cosh(psi_hat_read_only_p[k]) - 1.0)); // relevant value
      else
        integrand_p[k] = kB*temperature*far_field_ion_density*(psi_hat_read_only_p[k]*psi_hat_read_only_p[k]*2); // relevant value
    }
    else
      integrand_p[k] = 0.0;               // needs to be extrapolated (bc of jump on the normal derivative)
    phi_p[k] *= -1.0;                     // we need to integrate over the exterior domain --> reverse the levelset
  }
  ierr = VecRestoreArray(biomolecules->phi, &phi_p);                  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_hat, &psi_hat_read_only_p);          CHKERRXX(ierr);
  ierr = VecRestoreArray(integrand, &integrand_p);                    CHKERRXX(ierr);
  biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, integrand, 20, 2); // 20 for the number of iterations, default parameter
  solvation_free_energy = integrate_over_negative_domain(biomolecules->p4est, biomolecules->nodes, biomolecules->phi, integrand)*(pow(length_scale_in_meter(), 3.0));
  Vec phi_ghost_loc = NULL;
  ierr = VecGhostGetLocalForm(biomolecules->phi, &phi_ghost_loc);     CHKERRXX(ierr);
  ierr = VecScale(phi_ghost_loc, -1.0);                               CHKERRXX(ierr); // reverse the levelset function to get back to original state
  ierr = VecGhostRestoreLocalForm(biomolecules->phi, &phi_ghost_loc); CHKERRXX(ierr);

  // contributions from singular point charges
  double integral_contribution_from_singular_charges = 0.0;

  Vec psi_hat_plus_psi_naught=NULL;
  ierr = VecDuplicate(psi_hat, &psi_hat_plus_psi_naught); CHKERRXX(ierr);
  double *psi_hat_plus_psi_naught_p,*psi_naught_p, *psi_hat_p;

  ierr = VecGetArray(psi_hat_plus_psi_naught, &psi_hat_plus_psi_naught_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_naught, &psi_naught_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_hat,&psi_hat_p); CHKERRXX(ierr);
  for(size_t i =0; i< biomolecules->nodes->indep_nodes.elem_count; ++i)
  {
    psi_hat_plus_psi_naught_p[i]=psi_hat_p[i]+psi_naught_p[i];
  }
  ierr = VecRestoreArray(psi_hat, &psi_hat_p); psi_hat_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_naught, &psi_naught_p); psi_naught_p= NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_hat_plus_psi_naught, &psi_hat_plus_psi_naught_p); psi_hat_plus_psi_naught_p= NULL; CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interpolate_psi_hat_plus_psi_naught(biomolecules->neighbors);
  switch (biomolecules->parameters.order_of_accuracy()) {
  case 1:
    interpolate_psi_hat_plus_psi_naught.set_input(psi_hat_plus_psi_naught, linear);
    break;
  case 2:
    ierr = VecCreateGhostNodesBlock(biomolecules->p4est, biomolecules->nodes, P4EST_DIM, &psi_hat_plus_psi_naught_xxyyzz); CHKERRXX(ierr);
    biomolecules->neighbors->second_derivatives_central(psi_hat_plus_psi_naught, psi_hat_plus_psi_naught_xxyyzz);
    interpolate_psi_hat_plus_psi_naught.set_input(psi_hat_plus_psi_naught, psi_hat_plus_psi_naught_xxyyzz, quadratic_non_oscillatory_continuous_v2);
    break;
  default:
    throw  std::invalid_argument("my_p4est_biomolecules_solver_t::get_solvation_energy(), the order of accuracy should be either 1 or 2!");
    break;
  }
  int total_nb_charged_atoms = 0.0;
  for (size_t mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
  {
    const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
    total_nb_charged_atoms += mol.get_number_of_charged_atoms();
  }
  int proc_has_atom_if_rank_below = MIN(total_nb_charged_atoms, biomolecules->p4est->mpisize);

  int first_charged_atom_idx          = MIN(biomolecules->p4est->mpirank*total_nb_charged_atoms/proc_has_atom_if_rank_below, total_nb_charged_atoms);
  int idx_of_charged_atom_after_last  = MIN((biomolecules->p4est->mpirank+1)*total_nb_charged_atoms/proc_has_atom_if_rank_below, total_nb_charged_atoms);
  int nb_atoms_for_me                 = idx_of_charged_atom_after_last - first_charged_atom_idx;

  vector<double> point_values_of_psi_hat_plus_psi_naught(nb_atoms_for_me, 0.0);
  int charged_atom_idx_offset         = 0;
  int global_charged_atom_idx;
  p4est_locidx_t local_idx = 0;
  if(first_charged_atom_idx < total_nb_charged_atoms)
  {
    for (size_t mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      if((charged_atom_idx_offset + mol.get_number_of_charged_atoms() >= first_charged_atom_idx) && (charged_atom_idx_offset < idx_of_charged_atom_after_last))
        for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
        {
          global_charged_atom_idx = charged_atom_idx_offset + charged_atom_idx;
          if((first_charged_atom_idx <= global_charged_atom_idx) && (global_charged_atom_idx < idx_of_charged_atom_after_last))
          {
            const Atom* a = mol.get_charged_atom(charged_atom_idx);
            interpolate_psi_hat_plus_psi_naught.add_point(local_idx++, a->xyz_c);
            P4EST_ASSERT(local_idx <= nb_atoms_for_me);
          }
        }
      charged_atom_idx_offset += mol.get_number_of_charged_atoms();
    }
  }
  interpolate_psi_hat_plus_psi_naught.interpolate(point_values_of_psi_hat_plus_psi_naught.data());
  local_idx = 0;
  charged_atom_idx_offset         = 0;
  if(first_charged_atom_idx < total_nb_charged_atoms)
  {
    for (size_t mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      if((charged_atom_idx_offset + mol.get_number_of_charged_atoms() >= first_charged_atom_idx) && (charged_atom_idx_offset < idx_of_charged_atom_after_last))
      {
        for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
        {
          global_charged_atom_idx = charged_atom_idx_offset + charged_atom_idx;

          if((first_charged_atom_idx <= global_charged_atom_idx) && (global_charged_atom_idx < idx_of_charged_atom_after_last))
          {
            const Atom* a = mol.get_charged_atom(charged_atom_idx);
            integral_contribution_from_singular_charges += (0.5*a->q*kB*temperature/((double) ion_charge))*(point_values_of_psi_hat_plus_psi_naught.at(local_idx++));
            P4EST_ASSERT(local_idx <= nb_atoms_for_me);
          }
        }
      }
      charged_atom_idx_offset += mol.get_number_of_charged_atoms();
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &integral_contribution_from_singular_charges, 1, MPI_DOUBLE, MPI_SUM, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  solvation_free_energy += integral_contribution_from_singular_charges;
  double psi_hat_plus_psi_naught_max;
  double psi_hat_plus_psi_naught_min;
  VecMax(psi_hat_plus_psi_naught,NULL,&psi_hat_plus_psi_naught_max);
  VecMin(psi_hat_plus_psi_naught,NULL,&psi_hat_plus_psi_naught_min);
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " psi_hat_plus_psi_naught  max is  %g and min is %g  \n", psi_hat_plus_psi_naught_max, psi_hat_plus_psi_naught_min ); CHKERRXX(ierr);
  ierr = VecDestroy(integrand);       CHKERRXX(ierr);
  ierr = VecDestroy(psi_hat_plus_psi_naught_xxyyzz);  CHKERRXX(ierr);

  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The value of the solvation free energy is %g J, that is %g kJ/mol ,that is %g kcal/mol \n", solvation_free_energy, solvation_free_energy*avogadro_number*0.001, solvation_free_energy*avogadro_number*0.000239006); CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer->stop(); log_timer->read_duration();
      delete log_timer; log_timer = NULL;
    }
  }
  ierr= VecDestroy(psi_hat_plus_psi_naught);CHKERRXX(ierr);
  return (solvation_free_energy*avogadro_number*0.001);
#endif
}
my_p4est_biomolecules_solver_t::~my_p4est_biomolecules_solver_t()
{
  if(node_solver != NULL){
    delete node_solver;           node_solver = NULL;}
  if(jump_solver != NULL){
    delete jump_solver;           jump_solver = NULL;}
  if(psi_star != NULL){
    ierr = VecDestroy(psi_star);  psi_star = NULL;    CHKERRXX(ierr); }
  if(psi_hat != NULL){
    ierr = VecDestroy(psi_hat);   psi_hat = NULL;     CHKERRXX(ierr); }
}

