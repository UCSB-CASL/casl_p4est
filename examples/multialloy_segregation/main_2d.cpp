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
#include <src/my_p8est_save_load.h>
#include <src/my_p8est_macros.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_save_load.h>
#include <src/my_p4est_macros.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

using namespace std;

param_list_t pl;

param_t<int> num_comp (pl, 2, "num_comp", "");
param_t<double> ymin (pl, 1.5, "ymin", "");
param_t<double> ymax (pl, 2.0, "ymax", "");
param_t<string> folder (pl, "/media/dbochkov/Data/Output/d0.00002/g250/n_48.N_1.5997575/p4est/", "folder", "");
param_t<string> grid_file (pl, "grid_lvl_5_10.00026", "grid_name", "");
param_t<string> vecs_file (pl, "vecs_lvl_5_10.00026", "vecs_name", "");
param_t<string> out_file (pl, "segregation", "vecs_name", "");


int main (int argc, char* argv[])
{
  PetscErrorCode ierr;

  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.set_from_cmd_all(cmd);

  p4est_t              *p4est = NULL;
  p4est_nodes_t        *nodes = NULL;
  p4est_ghost_t        *ghost = NULL;
  p4est_connectivity_t *conn  = NULL;

  int num_vecs = 7 + 2*num_comp.val;
  std::vector<Vec> vecs_to_load(num_vecs);

  my_p4est_load_forest_and_data(mpi.comm(), folder.val.c_str(),
                                p4est, conn, 0, ghost, nodes, grid_file.val.c_str(),
                                1, vecs_file.val.c_str(), NODE_DATA, num_vecs, vecs_to_load.data());

  vec_and_ptr_t contr_phi; contr_phi.set(vecs_to_load[0]);
  vec_and_ptr_t front_phi; front_phi.set(vecs_to_load[1]);
  vec_and_ptr_t front_curvature; front_curvature.set(vecs_to_load[2]);
  vec_and_ptr_t front_velo_norm; front_velo_norm.set(vecs_to_load[3]);
  vec_and_ptr_t front_seed; front_seed.set(vecs_to_load[4]);
  vec_and_ptr_t tf; tf.set(vecs_to_load[5]);
  vec_and_ptr_t time; time.set(vecs_to_load[6]);

  vec_and_ptr_array_t cl(num_comp.val);
  vec_and_ptr_array_t kp(num_comp.val);

  for (int i = 0; i < num_comp.val; ++i) {
    cl.vec[i] = vecs_to_load[7+2*i];
    kp.vec[i] = vecs_to_load[7+2*i+1];
  }

  front_phi.get_array();
  cl.get_array();
  kp.get_array();
  time.get_array();
  front_velo_norm.get_array();

  FILE *fich;
  char file_out[1024];
  sprintf(file_out, "%s/%s", folder.val.c_str(), out_file.val.c_str());

  ierr = PetscFOpen(mpi.comm(), file_out, "w", &fich); CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fich, "x y time velo"); CHKERRXX(ierr);
  for (int i = 0; i < num_comp.val; ++i) {
    ierr = PetscFPrintf(mpi.comm(), fich, " cs%d", i); CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fich, "\n"); CHKERRXX(ierr);

  foreach_node(n, nodes) {
    if (front_phi.ptr[n] > 0) {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      if (xyz[1] > ymin.val && xyz[1] < ymax.val) {
        ierr = PetscFPrintf(mpi.comm(), fich, "%e %e %e %e", xyz[0], xyz[1], time.ptr[n], front_velo_norm.ptr[n]); CHKERRXX(ierr);
        for (int i = 0; i < num_comp.val; ++i) {
          ierr = PetscFPrintf(mpi.comm(), fich, " %e", kp.ptr[i][n]*cl.ptr[i][n]); CHKERRXX(ierr);
        }
        ierr = PetscFPrintf(mpi.comm(), fich, "\n"); CHKERRXX(ierr);
      }
    }
  }

  ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);



  p4est_destroy(p4est);
  p4est_connectivity_destroy(conn);
  p4est_ghost_destroy(ghost);
  p4est_nodes_destroy(nodes);

  for (int i = 0; i < num_vecs; ++i) {
    ierr = VecDestroy(vecs_to_load[i]);
  }

  return 0;
}
