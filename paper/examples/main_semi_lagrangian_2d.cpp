// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

using namespace std;

#ifdef P4_TO_P8
static class: public CF_3
{
	public:
		double operator()(double x, double y, double z) const {
			return 2.0*SQR(sin(M_PI*x))*sin(2*M_PI*y)*sin(2*M_PI*z);
		}
} vx_vortex;

static class: public CF_3
{
	public:
		double operator()(double x, double y, double z) const {
			return  -SQR(sin(M_PI*y))*sin(2*M_PI*x)*sin(2*M_PI*z);
		}
} vy_vortex;

static class: public CF_3
{
	public:
		double operator()(double x, double y, double z) const {
			return  -SQR(sin(M_PI*z))*sin(2*M_PI*x)*sin(2*M_PI*y);
		}
} vz_vortex;

struct circle:CF_3{
	circle(double x0_, double y0_, double z0_, double r_)
		: x0(x0_), y0(y0_), z0(z0_), r(r_)
	{}
	void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
	double operator()(double x, double y, double z) const {
		return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
	}
	private:
	double  x0, y0, z0, r;
};

struct square:CF_3{
	square(double x0_, double y0_, double z0_, double h_)
		: x0(x0_), y0(y0_), z0(z0_), h(h_)
	{}
	void update (double x0_, double y0_, double z0_, double h_) {x0 = x0_; y0 = y0_; z0 = z0_; h = h_; }
	double operator()(double x, double y, double z) const {
		return h - MIN(ABS(x-x0) , ABS(y-y0), ABS(z-z0));
	}
	private:
	double  x0, y0, z0, h;
};

#else
static class: public CF_2
{
  double f;
	public:
void update (double t) { f = t <= 1 ? 1:-1; /*f = sin(M_PI*t); f = f>0 ? ceil(f):floor(f);*/}
		double operator()(double x, double y) const {
      return (-SQR(sin(M_PI*x))*sin(2*M_PI*y));
		}
} vx_vortex;

static class: public CF_2
{
  double f;
	public:
    void update (double t) { f = t <= 1 ? 1:-1;/*f = sin(M_PI*t); f = f>0 ? ceil(f):floor(f);*/}
		double operator()(double x, double y) const {
      return  (SQR(sin(M_PI*y))*sin(2*M_PI*x));
		}
} vy_vortex;

struct circle:CF_2{
	circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
	void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
	double operator()(double x, double y) const {
		return r - sqrt(SQR(x-x0) + SQR(y-y0));
	}
	private:
	double  x0, y0, r;
};

struct square:CF_2{
	square(double x0_, double y0_, double h_): x0(x0_), y0(y0_), h(h_) {}
	void update (double x0_, double y0_, double h_) {x0 = x0_; y0 = y0_; h = h_; }
	double operator()(double x, double y) const {
		return h - MIN(ABS(x-x0) , ABS(y-y0));
	}
	private:
	double  x0, y0, h;
};
#endif

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);
  try {

    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    p4est_ghost_t      *ghost;
    PetscErrorCode ierr;
    cmdParser cmd;
    cmd.add_option("lmin", "min level");
    cmd.add_option("lmax", "max level");
    cmd.add_option("tf", "t final");
    cmd.add_option("write-vtk", "pass this flag if interested in the vtk files");
    cmd.add_option("output-dir", "parent folder to save everythiong in");
    cmd.add_option("lip", "lip constant for refinement");
    cmd.add_option("cfl", "cfl number for the SL method");
    cmd.add_option("cfl-condition", "decide whether to use the cfl condition advection");
    cmd.add_option("write-stats", "set this flag if interested in writing the stats");
    cmd.add_option("dt-max", "maximum dt to be taken");
		cmd.add_option("it-max" ,"maximum iterations before termination");
    cmd.parse(argc, argv);
    cmd.print();

    const std::string foldername = cmd.get<std::string>("output-dir");
    const int lmin = cmd.get("lmin", 0);
    const int lmax = cmd.get("lmax", 7);
    const double lip = cmd.get("lip", 1.2);
    const bool write_vtk   = cmd.contains("write-vtk");
    const bool write_stats = cmd.contains("write-stats");
    mkdir(foldername.c_str(), 0777);

    PetscPrintf(mpi->mpicomm, "git commit hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG);

    double radius = 0.15;
#ifdef P4_TO_P8
    circle circ(0.35, 0.35, 0.35, radius);
#else
    circle circ(0.50, 0.75, radius);
#endif
    splitting_criteria_cf_t data(lmin, lmax, &circ, lip);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(1, 1, 1, &brick);
#else
    connectivity = my_p4est_brick_new(1, 1, &brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    // Note that we do non-recursive refine + partitioning to ensure that :
    // 1. the work is load-balanced (although this should not be a big deal here ...)
    // 2. and more importantly, we have enough memory for the global grid. If we do not do it this way, the code usually break around level 13 or so.
    w2.start("refine");
    p4est->user_pointer = (void*)(&data);
    for (int l=0; l<lmax; l++){
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    w2.stop(); w2.read_duration();

    // create the ghost layer
    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    // generate the node data structure
    nodes = my_p4est_nodes_new(p4est, ghost);

    // Initialize the level-set function
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, circ, phi);

    // loop over time
    double tf = cmd.get<double>("tf");
		int itmax = cmd.get<double>("it-max");
    int tc = 0;
    int ts_vtk = 0, ts_stats = 0;
    double save_vtk   = cmd.get("write-vtk", 0.1);
    int save_stats = cmd.get("write-stats", 1);

    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);

    // SemiLagrangian object
    SemiLagrangian sl(&p4est, &nodes, &ghost, &brick, &node_neighbors);
    double cfl = cmd.get<double>("cfl");
    sl.set_CFL(cfl);
		bool cfl_condition = cmd.contains("cfl-condition") && cfl <= 1.0;

#ifdef P4_TO_P8
    double dt_cfl = cfl * sl.compute_dt(vx_vortex, vy_vortex, vz_vortex);
#else
    double dt_cfl = cfl * sl.compute_dt(vx_vortex, vy_vortex);
#endif

    double dt = 0.05;
    double dt_max = MIN(save_vtk, cmd.get("dt-max", dt_cfl));

    // prepare to calculate mass loss
#ifdef P4_TO_P8
    double mass_exact = 4.0/3.0 * M_PI * pow(radius,3.0);
#else
    double mass_exact = M_PI*pow(radius,2.0);
#endif
    Vec ones;
    ostringstream oss;
#ifdef P4_TO_P8
    oss << foldername + "/" + "mass_";
    if (cfl_condition)
      oss << "CFL_" << cfl << "_";
    else
      oss << "dt-max_" << dt_max << "_";
    oss << p4est->mpisize << "p_"
        << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "x" << brick.nxyztrees[2] << "." << ts_vtk << ".dat";
#else
    oss << foldername + "/" + "mass_";
    if (cfl_condition)
      oss << "CFL_" << cfl << "_";
    else
      oss << "dt-max_" << dt_max << "_";
    oss << p4est->mpisize << "p_"
        << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "." << ts_vtk << ".dat";
#endif

    FILE *err_file;
    ierr = PetscFOpen(p4est->mpicomm, oss.str().c_str(), "w", &err_file); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, err_file, "%% Error measured as the mass loss of the level-set at different times\n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, err_file, "%% time | rel. err\n"); CHKERRXX(ierr);

    if (write_vtk){
      w2.start("saving vtk file");
      // Save stuff
      std::ostringstream oss; oss << foldername << "/semi_lagrangian_";
      if (cfl_condition)
        oss << "CFL_";

      oss << p4est->mpisize << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1]
   #ifdef P4_TO_P8
          << "x" << brick.nxyztrees[2]
   #endif
          << "." << 0;

      double *phi_ptr;
      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             1, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr);

      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
      w2.stop(); w2.read_duration();
    }


    for (double t=0; t<tf && tc<itmax; t+=dt, tc++){
      if (write_stats && tc % save_stats == 0){
				w2.start("writing stats");
				std::ostringstream partition_name, topology_name, neighbors_name;
        std::ostringstream sl_partition_name, sl_topology_name;
#ifdef P4_TO_P8
				partition_name << foldername + "/" + "partition_CFL_" << cfl << "_" << p4est->mpisize << "p_"
                       << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "x" << brick.nxyztrees[2] << "." << ts_stats << ".dat";
				topology_name  << foldername + "/" + "topology_CFL_"  << cfl << "_" << p4est->mpisize << "p_"
                       << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "x" << brick.nxyztrees[2] << "." << ts_stats << ".dat";
        neighbors_name << foldername + "/" + "neighbors_CFL_" << cfl << "_" << p4est->mpisize << "p_"
                       << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "x" << brick.nxyztrees[2] << "." << ts_stats << ".dat";

        sl_partition_name << foldername + "/" + "SL_partition_CFL_" << cfl << "_" << p4est->mpisize << "p_"
                          << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "x" << brick.nxyztrees[2] << "." << ts_stats << ".dat";
        sl_topology_name  << foldername + "/" + "SL_topology_CFL_"  << cfl << "_" << p4est->mpisize << "p_"
                          << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "x" << brick.nxyztrees[2] << "." << ts_stats << ".dat";
#else
				partition_name << foldername + "/" + "partition_CFL_" << cfl << "_" << p4est->mpisize << "p_"
                       << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "." << ts_stats << ".dat";
				topology_name  << foldername + "/" + "topology_CFL_"  << cfl << "_" << p4est->mpisize << "p_"
                       << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "." << ts_stats << ".dat";
				neighbors_name << foldername + "/" + "neighbors_CFL_" << cfl << "_" << p4est->mpisize << "p_"
                       << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "." << ts_stats << ".dat";

        sl_partition_name << foldername + "/" + "SL_partition_CFL_" << cfl << "_" << p4est->mpisize << "p_"
                          << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "." << ts_stats << ".dat";
        sl_topology_name  << foldername + "/" + "SL_topology_CFL_"  << cfl << "_" << p4est->mpisize << "p_"
                          << brick.nxyztrees[0] << "x" << brick.nxyztrees[1] << "." << ts_stats << ".dat";
#endif
        write_comm_stats(p4est, ghost, nodes, partition_name.str().c_str(), topology_name.str().c_str(), neighbors_name.str().c_str());
        sl.set_comm_topology_filenames(sl_partition_name.str(), sl_topology_name.str());
        ts_stats++;
        w2.stop(); w2.read_duration();
			}
 
      if (write_vtk && t+dt >= (ts_vtk+1)*save_vtk){
        // advect to (ts+1)*save time
        if (((ts_vtk+1)*save_vtk - t)/save_vtk > 1e-6){
          dt = (ts_vtk+1)*save_vtk - t;
          w2.start("advecting for save");
#ifdef P4_TO_P8
          if (cfl_condition)
            dt = sl.update_p4est_second_order_CFL(vx_vortex, vy_vortex, vz_vortex, dt, phi);
          else
            sl.update_p4est_second_order(vx_vortex, vy_vortex, vz_vortex, dt, phi);
#else
          if (cfl_condition)
            dt = sl.update_p4est_second_order_CFL(vx_vortex, vy_vortex, dt, phi);
          else
            sl.update_p4est_second_order(vx_vortex, vy_vortex, dt, phi);
#endif
          hierarchy.update(p4est, ghost);
          node_neighbors.update(&hierarchy, nodes);
          PetscPrintf(p4est->mpicomm, "t = %f, dt = %f, tc = %d\n", t+dt, dt, tc+1);
          w2.stop(); w2.read_duration();

          w2.start("Reinit");
          node_neighbors.init_neighbors();
          my_p4est_level_set level_set(&node_neighbors);
          level_set.reinitialize_1st_order_time_2nd_order_space(phi, 10);
          w2.stop(); w2.read_duration();
        }

				w2.start("saving vtk file");
				// Save stuff
				std::ostringstream oss; oss << foldername << "/semi_lagrangian_";
				if (cfl_condition)
					oss << "CFL_";

				oss << p4est->mpisize << "_"
						<< brick.nxyztrees[0] << "x"
						<< brick.nxyztrees[1]
		 #ifdef P4_TO_P8
						<< "x" << brick.nxyztrees[2]
		 #endif
            << "." << ts_vtk;

				double *phi_ptr;
				ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
				my_p4est_vtk_write_all(p4est, nodes, ghost,
															 P4EST_TRUE, P4EST_TRUE,
															 1, 0, oss.str().c_str(),
															 VTK_POINT_DATA, "phi", phi_ptr);

				ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
				w2.stop(); w2.read_duration();
	
        ts_vtk++;
        continue;
      }

      // advect the function in time and get the computed time-step
      w2.start("advecting");
#ifdef P4_TO_P8
      if (cfl_condition)
        dt = sl.update_p4est_second_order_CFL(vx_vortex, vy_vortex, vz_vortex, dt_max, phi);
      else {
        sl.update_p4est_second_order(vx_vortex, vy_vortex, vz_vortex, dt_max, phi);
        dt = dt_max;
      }
#else
      if (cfl_condition)
        dt = sl.update_p4est_second_order_CFL(vx_vortex, vy_vortex, dt_max, phi);
      else {
        sl.update_p4est_second_order(vx_vortex, vy_vortex, dt_max, phi);
        dt = dt_max;
      }
#endif
      PetscPrintf(p4est->mpicomm, "t = %f, dt = %f, tc = %d\n", t+dt, dt, tc+1);
      w2.stop(); w2.read_duration();

      w2.start("Reinit");
      node_neighbors.init_neighbors();
      my_p4est_level_set level_set(&node_neighbors);
      level_set.reinitialize_1st_order_time_2nd_order_space(phi, 10);
      w2.stop(); w2.read_duration();

      // calculate mass loss error
      ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
      ierr = VecSet(ones, 1.0); CHKERRXX(ierr);
      double err = fabs(mass_exact - integrate_over_negative_domain(p4est, nodes, phi, ones))/mass_exact;
      ierr = PetscFPrintf(p4est->mpicomm, err_file, "%1.5e %1.5e\n", t, err); CHKERRXX(ierr);
      ierr = VecDestroy(ones); CHKERRXX(ierr);
    }

    ierr = PetscFClose(p4est->mpicomm, err_file); CHKERRXX(ierr);
    ierr = VecDestroy(phi); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    p4est_ghost_destroy(ghost);
    p4est_nodes_destroy(nodes);
    p4est_destroy(p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();
  } catch (const std::exception& e) {
    PetscFPrintf(MPI_COMM_SELF, stderr, "[%d] %s\n", mpi->mpirank, e.what());
  }

  return 0;
}
