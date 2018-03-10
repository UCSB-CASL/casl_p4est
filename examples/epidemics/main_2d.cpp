#include <src/Parser.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_epidemics.h>


int main (int argc, char** argv)
{
    PetscErrorCode ierr;
    mpi_environment_t mpi;
    mpi.init(argc, argv);

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.add_option("nx", "number of blox in x-dimension");
    cmd.add_option("ny", "number of blox in y-dimension");
    cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
    cmd.add_option("save_every_n", "save vtk every n iteration");
    cmd.add_option("tf", "final time");
    cmd.add_option("box_size", "set box_size");
    cmd.add_option("alpha_A", "set infection rate for A");
    cmd.add_option("alpha_B", "set infection rate for B");
    cmd.add_option("Xi_A", "set cooperativity for A");
    cmd.add_option("Xi_B", "set cooperativity for B");
    cmd.add_option("beta", "set recovery rate beta");
    cmd.add_option("D_A", "set diffusion coefficient for A");
    cmd.add_option("D_B", "set diffusion coefficient for B");
    cmd.add_option("D_AB", "set diffusion coefficient for AB");
    cmd.parse(argc, argv);

    int lmin = cmd.get("lmin", 7);
    int lmax = cmd.get("lmax", 10);
    int nx = cmd.get("nx", 2);
    int ny = cmd.get("ny", 2);
    int save_vtk = cmd.get("save_vtk", 1);
    int save_every_n = cmd.get("save_every_n", 1);

    double beta = cmd.get("beta", 1.);
    double t_final = cmd.get("tf", 5);
    double alpha_A = cmd.get("alpha_A", 2.0);
    double alpha_B = cmd.get("alpha_B", 2.0);
    double Xi_A = cmd.get("Xi_A", 5);
    double Xi_B = cmd.get("Xi_B", 5);
    double D_A = cmd.get("D_A", 1e-6);
    double D_B = cmd.get("D_B", 1e-6);
    double D_AB = cmd.get("D_AB", 1e-6);

    double box_size = cmd.get("box_size", 1);


    double R_A = alpha_A/beta;
    double R_B = alpha_B/beta;

    t_final /= beta;
    parStopWatch w1;
    w1.start("total time");

    /* create the p4est */
    my_p4est_brick_t brick;
    int n_xyz[] = {nx, ny};
    double xyz_min[] = {0, 0};
    double xyz_max[] = {box_size, box_size};
    int periodic[] = {1, 1};
    p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

    splitting_criteria_t data(lmin, lmax);
    p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, (void*)(&data));
    for(int lvl=0; lvl<lmin; ++lvl)
    {
        my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
        my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
    my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
    my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);



    /* initialize the solver */
    my_p4est_epidemics_t epidemics(ngbd);
    epidemics.read("US_census.dat");
    epidemics.set_parameters(R_A, R_B, Xi_A, Xi_B);
    epidemics.set_D(D_A, D_B, D_AB);

    // loop over time
    double tn = 0;

    FILE *fp = NULL;
    char name[1000];

    const char *out_dir = getenv("OUT_DIR");
    if (!out_dir) out_dir = ".";

    sprintf(name, "%s/epidemics_%dx%d_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], box_size, lmin, lmax);





    int iter = 0;
    while(tn<t_final)
    {
        p4est = epidemics.get_p4est();
        ierr = PetscPrintf(p4est->mpicomm, "###########################################\n"); CHKERRXX(ierr);
        ierr = PetscPrintf(p4est->mpicomm, "Iteration #%d, tn = %e\n", iter, tn); CHKERRXX(ierr);
        if(iter!=0)
        {
            epidemics.compute_velocity();
            epidemics.compute_dt();
            epidemics.update_grid();
        } else {
            epidemics.initialize_infections();
        }

        epidemics.solve(iter);

        if(save_vtk==true && iter%save_every_n==0)
        {
            epidemics.save_vtk(iter/save_every_n);
        }


        p4est = epidemics.get_p4est();
        nodes = epidemics.get_nodes();
        if(p4est->mpirank==0)
        {
            p4est_locidx_t nb_nodes_global = 0;
            for(int r=0; r<p4est->mpisize; ++r)
                nb_nodes_global += nodes->global_owned_indeps[r];

            printf("The p4est has %d nodes.\n", nb_nodes_global);

            fp = fopen(name, "a");
            if(fp==NULL)
            {
                throw std::invalid_argument("could not open file for coverage vs. time output");
            }
            fprintf(fp, "%.15e %.15e %d\n", tn, epidemics.get_dt(), nb_nodes_global);
            fclose(fp);
        }

        iter++;
        tn += epidemics.get_dt();

    }

    w1.stop(); w1.read_duration();

    return 0;
}
