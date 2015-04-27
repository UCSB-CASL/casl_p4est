#ifdef P4_TO_P8
#include "diffusion2.h"
#include <src/stresstensor2.h>
#else
#include "diffusion.h"
#include <src/stresstensor.h>
#endif

//#include "diffusion.h"

int Diffusion::scatter_petsc_vector(Vec *v2scatter)
{
    // std::cout<<this->i_mean_field<<" "<<this->mpi->mpirank<<" start to scatter "<<std::endl;

    this->ierr=VecGhostUpdateBegin(*v2scatter,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2scatter,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    // std::cout<<this->i_mean_field<<" "<<this->mpi->mpirank<<" finish to scatter "<<std::endl;

}

int Diffusion::Diffusion_initialyze_petsc(int argc, char *argv[])
{

    this->mpi = &mpi_context;
    this->mpi->mpicomm  = MPI_COMM_WORLD;
    this->mpi_session=new Session();
    this->mpi_session->init(argc, argv, this->mpi->mpicomm);
    std::cout<<this->mpi->mpirank<<"--------------------initialyzed petsc---------------------------"<<std::endl;

    return 0;
}

int Diffusion::Diffusion_finalyze_petsc()
{
    std::cout<<this->mpi->mpirank<<"--------------------finalyze petsc---------------------------"<<std::endl;
    delete this->mpi_session;
    return 0;
}



int Diffusion::clean_mean_field_step(bool remesh_next_iteration)
{
    std::cout<<" start to clean mean field step from diffusion "<<std::endl;
    delete this->rhoA_local;
    delete this->rhoB_local;
    //    delete this->q_forward;
    //    delete this->q_backward;

    VecDestroy(this->rhoA);
    VecDestroy(this->rhoB);
    VecDestroy(this->fp);
    VecDestroy(this->fm);
    VecDestroy( this->w_t);
    //VecDestroy(this->energy_integrand);
    if(this->remesh && this->remeshed && !this->clean && remesh_next_iteration)
    {
        // Check what does this destructor do exactly
        if(this->setup_finite_difference_solver)
            delete this->solver;

        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
                ||this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
                || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson       )
            delete this->myNeumanPoissonSolverNodeBase;

        //------------Petsc Object Destruction-----------------//
        this->ierr=VecDestroy(this->phi);       CHKERRXX(this->ierr);
        //this->ierr=VecDestroy(this->sol_0);     CHKERRXX(this->ierr);

        //this->ierr=VecDestroy(this->sol_0);      CHKERRXX(this->ierr);

        this->ierr=VecDestroy(this->sol_t);      CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->sol_tp1);     CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->b_t);         CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->r_t); CHKERRXX(this->ierr);

        if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
                ||this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson
                ||
                (this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson
                 & this->setup_finite_difference_solver))
        {
            this->ierr=MatDestroy(this->M1);        CHKERRXX(this->ierr);
            this->ierr=MatDestroy(this->M2);        CHKERRXX(this->ierr);
            this->ierr=KSPDestroy(this->myKsp);        CHKERRXX(this->ierr);
        }
        if(this->my_casl_diffusion_method!=Diffusion::periodic_crank_nicholson)
        {
            this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive); CHKERRXX(this->ierr);
            this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->is_crossed_neumann); CHKERRXX(this->ierr);

        }

        if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
        {
            this->ierr=VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_is_below_epsilon); CHKERRXX(this->ierr);
        }

        if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive && this->mapping_fftw_created && !this->mapping_fftw_destroyed)
        {
            this->ierr=ISDestroy(this->from_fftw); CHKERRXX(this->ierr);
            this->ierr=ISDestroy(this->to_p4est); CHKERRXX(this->ierr);
            this->ierr=VecScatterDestroy(&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
            delete this->idx_from_fftw_to_p4est_fftw_ix;
            delete this->idx_from_fftw_to_p4est_p4est_ix;

            this->mapping_fftw_created=false;
            this->mapping_fftw_destroyed=true;

        }

        delete this->ix_fromLocal2Global;
        this->remeshed=PETSC_FALSE;
        this->clean=PETSC_TRUE;

    }

    //    if(this->compute_stress_tensor_bool)
    //        delete this->my_stress_tensor;

    std::cout<<" finished to clean mean field step from diffusion"<<std::endl;
}


void Diffusion::setupEnergyAndForces()
{
    if(!this->advance_fields_scft_advance_mask_level_set)
    {
        if(!this->myMeanFieldPlan->neuman_with_mask && !this->myMeanFieldPlan->dirichlet_with_mask)
        {
            this->my_force_calculator=&Diffusion::compute_forces;
            this->my_energy_calculator=&Diffusion::compute_energy;
        }
        if(this->myMeanFieldPlan->neuman_with_mask && !this->spatial_xhi_w)
        {
            this->my_force_calculator=&Diffusion::compute_wall_forces_neuman;
            this->my_energy_calculator=&Diffusion::compute_energy_wall_neuman;
        }
        if(this->myMeanFieldPlan->dirichlet_with_mask && !this->spatial_xhi_w)
        {
            this->my_force_calculator=&Diffusion::compute_wall_forces_dirichlet;
            this->my_energy_calculator=&Diffusion::compute_energy_wall_dirichlet;
        }
        if(this->myMeanFieldPlan->neuman_with_mask && this->spatial_xhi_w)
        {
            this->my_force_calculator=&Diffusion::compute_wall_forces_with_spatial_selectivity_neuman;
            this->my_energy_calculator=&Diffusion::compute_energy_wall_neuman_with_spatial_selectivity;
        }


    }
    if(this->advance_fields_scft_advance_mask_level_set)
    {

        if(!this->spatial_xhi_w)
        {
            this->my_force_calculator=&Diffusion::compute_wall_forces;
            this->my_energy_calculator=&Diffusion::compute_energy_wall;
        }
        if(this->spatial_xhi_w)
        {
            this->my_force_calculator=&Diffusion::compute_wall_forces_with_spatial_selectivity;
            this->my_energy_calculator=&Diffusion::compute_energy_wall_with_spatial_selectivity;
        }
    }
}

void Diffusion::evolve_mean_field_step()
{

    this->evolve_probability_equation((int) (this->f*this->N_iterations),(int)((1-this->f)*this->N_iterations));

    this->compute_densities();
    this->computeVolume();
    if(!this->minimum_IO)
        this->printDensities();
    //this->printDensities_by_cells();
    if(this->compute_stress_tensor_bool && mod(this->i_mean_field+1,this->stress_tensor_computation_period)==0 )
    {

        this->computeStressTensor();

    }

    delete this->q_forward;
    delete this->q_backward;

    (this->*my_force_calculator)();
    if(!this->minimum_IO)
        this->printForces();
    this->compute_energy_difference_analyticaly();
    (this->*my_energy_calculator)();
    this->scatterForwardMyParallelVectors();
    this->computeAverageProperties();
}

void Diffusion::setupNeumanSolver(my_p4est_node_neighbors_t *node_neighbors)
{


    this->bc.setInterfaceType(NEUMANN);
    this->bc.setInterfaceValue(bc_interface_neumann_value2);
    this->bc.setWallTypes(bc_wall_neumann_type2);
    this->bc.setWallValues(bc_wall_neumann_value2);

    this->myNeumanPoissonSolverNodeBase=new PoissonSolverNodeBase(node_neighbors);
    this->myNeumanPoissonSolverNodeBase->set_phi(this->phi_polymer_shape);
    this->myNeumanPoissonSolverNodeBase->set_mu(1.00);
    if(!this->fast_algo)
    {
        this->myNeumanPoissonSolverNodeBase->set_mu(this->dt);
        this->myNeumanPoissonSolverNodeBase->set_diagonal(1.00);
    }
    this->myNeumanPoissonSolverNodeBase->set_bc(this->bc);
}


void Diffusion::setupDirichletSolver(my_p4est_node_neighbors_t *node_neighbors)
{


    this->bc.setInterfaceType(DIRICHLET);
    this->bc.setInterfaceValue(bc_interface_dirichlet_value0);
    this->bc.setWallTypes(bc_wall_dirichlet_type2);
    this->bc.setWallValues(bc_wall_dirichlet_value0);

    this->myNeumanPoissonSolverNodeBase=new PoissonSolverNodeBase(node_neighbors);
    this->myNeumanPoissonSolverNodeBase->set_phi(this->phi_polymer_shape);
    this->myNeumanPoissonSolverNodeBase->set_mu(1.00);
    this->myNeumanPoissonSolverNodeBase->set_bc(this->bc);
}



int Diffusion::initialyzeDiffusionFromMeanField(int min_level, int max_level,  Vec *phi, Session *mpi_session,
                                                mpi_context_t *mpi_context, mpi_context_t *mpi, p4est_t            *p4est, p4est_nodes_t      *nodes,
                                                p4est_ghost_t* ghost, p4est_connectivity_t *connectivity,
                                                my_p4est_brick_t *brick, my_p4est_hierarchy_t *hierarchy, my_p4est_node_neighbors_t *node_neighbors, double Lx, double Lx_physics,
                                                double f, double Xab, int N_iterations, Diffusion::phase myPhase, casl_diffusion_method my_casl_diffusion_method, numerical_scheme my_numerical_scheme,
                                                std::string my_io_path, int stress_tensor_computation_period, int nx_trees, double lambda,
                                                PetscBool setup_finite_difference_solver,
                                                bool periodic_xyz, string text_file_fields, string text_file_field_wp, string text_file_field_wm,MeanFieldPlan *myMeanFieldPlan)
{

    try
    {

        std::cout<<" start initialyzeDiffusionFromMeanField "<<std::endl;
        this->myMeanFieldPlan=myMeanFieldPlan;
        this->input_potential_path_w_p=text_file_field_wp;
        this->input_potential_path_w_m=text_file_field_wm;
        this->field_text_file=text_file_fields;
        this->setup_finite_difference_solver=setup_finite_difference_solver;
        this->lambda_minus=lambda;
        this->lambda_plus=lambda;
        this->nx_trees=nx_trees;
        this->stress_tensor_computation_period=stress_tensor_computation_period;
        this->my_casl_diffusion_method=my_casl_diffusion_method;
        if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
            this->periodic_xyz=PETSC_TRUE;
        else
            this->periodic_xyz=PETSC_FALSE;

        //this->periodic_xyz=periodic_xyz;
        this->N_iterations=N_iterations;
        this->N_t=this->N_iterations+1;
        this->dt=1.00/this->N_iterations;
        this->IO_path=my_io_path;
        std::cout<<(  (splitting_criteria_t*)(p4est->user_pointer))->max_lvl<<std::endl;
        this->myNumericalScheme=my_numerical_scheme;

        // get the data from mean field:
        this->min_level = min_level;
        this->max_level = max_level;
        //        this->phi = phi;
        //        this->ierr = VecDuplicate(*phi, &this->phi); CHKERRXX(this->ierr);
        //        this->ierr = VecCopy(*phi, this->phi); CHKERRXX(this->ierr);
        //        this->ierr = VecDuplicate(*phi, &this->uex); CHKERRXX(this->ierr);
        this->mpi_session=mpi_session;
        this->mpi_context=*mpi_context;
        this->mpi=mpi;
        std::cout<<this->mpi->mpirank<<"initialyzed in diffusion"<<std::endl;

        this->p4est = p4est;
        this->nodes=nodes;
        this->ghost=ghost;
        this->connectivity=connectivity;
        this->brick=*brick;
        this->nodes_neighbours=node_neighbors;

        if(this->setup_finite_difference_solver)
            this->solver=new PoissonSolverNodeBase(node_neighbors);
        //        this->solver->set_phi(this->phi);
        //        this->solver->set_mu(1.00);
        this->Lx=Lx;
        this->Lx_physics=Lx_physics;
        this->D=(this->Lx/this->Lx_physics)*(this->Lx/this->Lx_physics);
        this->f=f;
        this->X_ab=Xab;
        this->myPhase=myPhase;
        /* solve the system: used just when we benchmark the system */
        //this->solver->solve(this->sol);

        ierr = VecCreateGhostNodes(this->p4est, this->nodes, &this->phi); CHKERRXX(ierr);


        if(this->setup_finite_difference_solver)
        {
            this->solver->set_phi(this->phi);
            this->solver->set_mu(1.00);
        }
        std::cout<<"Print Data "<<std::endl;
        //        this->printForestNodes2TextFile();
        //        this->printForestOctants2TextFile();
        //        this->printForestNodesNeighborsCells2TextFile();
        //        this->printGhostCells();
        //        this->printMacroMesh();


        switch(this->my_casl_diffusion_method)
        {
        case Diffusion::periodic_crank_nicholson:
        {



            if(this->setup_finite_difference_solver)
            {
                std::cout<<this->mpi->mpirank<<"--------------------setup M---------------------------"<<std::endl;
                this->solver->setupM2();
                MatScale(this->solver->A,this->D);


                // this->solver->print_quad_neighbor_nodes_of_node_t();

                std::cout<<this->mpi->mpirank<<"------------------finished to setup M-----------------"<<std::endl;
                std::cout<<this->mpi->mpirank<<"--------------------start to print M---------------------------"<<std::endl;
                if(this->print_diffusion_matrices)
                    this->printDiffusionMatrices(&this->solver->A,"solver->A",0);
                std::cout<<this->mpi->mpirank<<"--------------------finish to print M---------------------------"<<std::endl;

                std::cout<<this->mpi->mpirank<<"--------------------start to print Neighbor Information---------------------------"<<std::endl;
                if(this->print_diffusion_matrices)
                    this->solver->print_quad_neighbor_nodes_of_node_t();
                std::cout<<this->mpi->mpirank<<"--------------------finish to print Neighbor Information---------------------------"<<std::endl;



                std::cout<<this->mpi->mpirank<<"--------------------start creating diffusion variables---------------------------"<<std::endl;
                this->createDiffusionMatrices();
                this->createKSPContext();
            }
            break;
        }
        case Diffusion::neuman_backward_euler:
        {
            this->setupNeumanSolver(node_neighbors);
            if(this->fast_algo)
            {
                this->myNeumanPoissonSolverNodeBase->setupM2_Neuman();
                this->createDiffusionMatricesNeuman();
                this->createKSPContext();

            }
            break;
        }
        case Diffusion::neuman_crank_nicholson:
        {
            this->setupNeumanSolver(node_neighbors);
            if(this->fast_algo)
            {
                this->myNeumanPoissonSolverNodeBase->setupM2_Neuman();
                this->createDiffusionMatricesNeuman();
                this->createKSPContext();

            }
            break;
        }
        case Diffusion::dirichlet_crank_nicholson:
        {
            this->setupDirichletSolver(node_neighbors);
            if(this->fast_algo)
            {
                this->myNeumanPoissonSolverNodeBase->setupM2_Dirichlet();
                this->createDiffusionMatricesDirichlet();
                this->createKSPContext();

            }
            break;
        }

        }

        this->createDiffusionSolution();
        std::cout<<this->mpi->mpirank<<" w benchmark "<<this->w_benchmark<<std::endl;

        PetscBool minus_one_iteration=PETSC_TRUE;

        this->allocateMemory2History(minus_one_iteration);
        //if(this->myPhase!=Diffusion::from_text_file)
        this->create_statistical_field();
        //        else
        //        {
        //            VecDuplicate(this->phi,&this->wp);
        //            VecDuplicate(this->phi,&this->wm);
        //            this->generate_disordered_phase();
        //        }

        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
            this->filter_irregular_potential();

        if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
        {
            this->filter_petsc_vector_dirichlet(&this->wp);
            this->filter_petsc_vector_dirichlet(&this->wm);
        }


        std::cout<<(  (splitting_criteria_t*)(p4est->user_pointer))->max_lvl<<std::endl;
        if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive)
        {
#ifdef P4_TO_P8
            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));
            int n_fftw=N0*N1*N2;
            int n_fftw_local=n_fftw/this->mpi->mpisize;

            this->ierr=VecCreate(MPI_COMM_WORLD,&this->w3dGlobal); CHKERRXX(this->ierr);
            this->ierr=VecSetSizes(this->w3dGlobal,n_fftw_local,n_fftw); CHKERRXX(this->ierr);
            this->ierr=VecSetFromOptions(this->w3dGlobal); CHKERRXX(this->ierr);

            this->ierr=VecCreate(MPI_COMM_WORLD,&this->fftw_global_petsc_output); CHKERRXX(this->ierr);
            this->ierr=VecSetSizes(this->fftw_global_petsc_output,n_fftw_local,n_fftw); CHKERRXX(this->ierr);
            this->ierr=VecSetFromOptions(this->fftw_global_petsc_output); CHKERRXX(this->ierr);
#else
            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));
            int n_fftw=N0*N1;
            int n_fftw_local=n_fftw/this->mpi->mpisize;

            this->ierr=VecCreate(MPI_COMM_WORLD,&this->w2dGlobal); CHKERRXX(this->ierr);
            this->ierr=VecSetSizes(this->w2dGlobal,n_fftw_local,n_fftw); CHKERRXX(this->ierr);
            this->ierr=VecSetFromOptions(this->w2dGlobal); CHKERRXX(this->ierr);

            this->ierr=VecCreate(MPI_COMM_WORLD,&this->fftw_global_petsc_output); CHKERRXX(this->ierr);
            this->ierr=VecSetSizes(this->fftw_global_petsc_output,n_fftw_local,n_fftw); CHKERRXX(this->ierr);
            this->ierr=VecSetFromOptions(this->fftw_global_petsc_output); CHKERRXX(this->ierr);

#endif

        }
        this->remeshed=PETSC_TRUE;
        this->clean=PETSC_FALSE;
        std::cout<<this->mpi->mpirank<<" finish initialyzeDiffusionFromMeanField "<<std::endl;

        return 0;

    }
    catch (const std::exception& e)
    {
        std::cout << "[" << this->mpi->mpirank << " -- ERROR]: " << e.what() << std::endl;
        return -1;
    }

}


int Diffusion::destruct_initial_data()
{
    // Check what does this destructor do exactly
    if(this->setup_finite_difference_solver)
        delete this->solver;
    // delete this->sol_history;
    //------------Petsc Object Destruction-----------------//
    this->ierr=VecDestroy(this->phi);       CHKERRXX(this->ierr);

    //this->ierr=VecDestroy(this->sol_0);      CHKERRXX(this->ierr);

    this->ierr=VecDestroy(this->sol_t);      CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->sol_tp1);     CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->b_t);         CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->r_t);         CHKERRXX(this->ierr);
    if( (this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson && this->setup_finite_difference_solver)||
            this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson||
            this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson||
            (!this->fast_algo))
    {
        this->ierr=MatDestroy(this->M1);        CHKERRXX(this->ierr);
        this->ierr=MatDestroy(this->M2);        CHKERRXX(this->ierr);
        this->ierr=KSPDestroy(this->myKsp);        CHKERRXX(this->ierr);
    }
    if( (this->my_casl_diffusion_method!=Diffusion::periodic_crank_nicholson) )
    {
        this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive); CHKERRXX(this->ierr);
        this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->is_crossed_neumann); CHKERRXX(this->ierr);
        if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            this->ierr=VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_is_below_epsilon); CHKERRXX(this->ierr);


        delete this->myNeumanPoissonSolverNodeBase;
        delete this->ix_fromLocal2Global;

    }
    return 0;
}


int Diffusion::reInitialyzeDiffusionFromMeanField( p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, p4est_connectivity_t *connectivity, my_p4est_brick_t *brick, my_p4est_hierarchy_t *hierarchy, my_p4est_node_neighbors_t *node_neighbors, double Lx,double Lx_physics,int i_mean_field)
{

    try{


        int n_games_i=1; int n_games_j=1;
        for(int i_game=0;i_game<n_games_i;i_game++)
        {

            // must get the new remapped statistical fields first from meanfield
            this->remesh=PETSC_TRUE;
            this->mapped=PETSC_FALSE;
            this->p4est = p4est;
            this->nodes=nodes;
            this->ghost=ghost;
            this->connectivity=connectivity;
            this->brick=*brick;
            this->nodes_neighbours=node_neighbors;



            if(this->setup_finite_difference_solver)
            {
                this->solver=new PoissonSolverNodeBase(node_neighbors);
            }
            this->Lx=Lx;
            this->Lx_physics=Lx_physics;
            this->ierr = VecCreateGhostNodes(this->p4est, this->nodes, &this->phi); CHKERRXX(this->ierr);
            VecSet(this->phi,-1.00);

            if(this->setup_finite_difference_solver)
            {
                this->solver->set_phi(this->phi);
                this->solver->set_mu(1.00);
            }

            std::cout<<"Print Data "<<std::endl;
            //       this->printForestNodes2TextFile();
            //        this->printForestOctants2TextFile();
            //        this->printForestNodesNeighborsCells2TextFile();
            //        this->printGhostCells();
            //        this->printMacroMesh();


            if(this->setup_finite_difference_solver)
            {
                switch(this->my_casl_diffusion_method)
                {
                case Diffusion::periodic_crank_nicholson:
                {

                    std::cout<<this->mpi->mpirank<<"--------------------setup M---------------------------"<<std::endl;
                    this->solver->setupM2();
                    for(int j_game=0;j_game<n_games_j;j_game++)
                    {
                        MatScale(this->solver->A,this->D);

                        std::cout<<this->mpi->mpirank<<"------------------finished to setup M-----------------"<<std::endl;
                        std::cout<<this->mpi->mpirank<<"--------------------start to print M---------------------------"<<std::endl;
                        if(this->print_remeshed_diffusion_matrices)
                            this->printDiffusionMatrices(&this->solver->A,"solver->A",i_mean_field);
                        std::cout<<this->mpi->mpirank<<"--------------------finish to print M---------------------------"<<std::endl;

                        std::cout<<this->mpi->mpirank<<"--------------------start to print Neighbor Information---------------------------"<<std::endl;
                        if(this->print_diffusion_matrices)
                            this->solver->print_quad_neighbor_nodes_of_node_t();
                        std::cout<<this->mpi->mpirank<<"--------------------finish to print Neighbor Information---------------------------"<<std::endl;


                        std::cout<<this->mpi->mpirank<<"--------------------start creating diffusion variables---------------------------"<<std::endl;
                        this->createDiffusionMatrices();
                        this->createKSPContext();

                        if(j_game<n_games_j-1)
                        {
                            // Check what does this destructor do exactly
                            // delete this->solver;
                            //------------Petsc Object Destruction-----------------//
                            //this->ierr=VecDestroy(this->phi);       CHKERRXX(this->ierr);

                            this->ierr=MatDestroy(this->M1);        CHKERRXX(this->ierr);
                            this->ierr=MatDestroy(this->M2);        CHKERRXX(this->ierr);
                            this->ierr=KSPDestroy(this->myKsp);        CHKERRXX(this->ierr);



                        }
                    }
                    break;
                }

                case Diffusion::neuman_backward_euler:
                {
                    //delete this->myNeumanPoissonSolverNodeBase;
                    this->setupNeumanSolver(node_neighbors);
                    if(this->fast_algo)
                    {
                        this->myNeumanPoissonSolverNodeBase->setupM2_Neuman();
                        this->createDiffusionMatricesNeuman();
                        this->createKSPContext();
                    }
                    break;
                }
                case Diffusion::neuman_crank_nicholson:
                {


                    // delete this->myNeumanPoissonSolverNodeBase;
                    this->setupNeumanSolver(node_neighbors);
                    if(this->fast_algo)
                    {

                        this->myNeumanPoissonSolverNodeBase->setupM2_Neuman();
                        for(int j_game=0;j_game<n_games_j;j_game++)
                        {
                            this->createDiffusionMatricesNeuman();
                            if(j_game<n_games_j-1)
                            {
                                // Check what does this destructor do exactly
                                // delete this->solver;
                                //------------Petsc Object Destruction-----------------//
                                //this->ierr=VecDestroy(this->phi);       CHKERRXX(this->ierr);

                                this->ierr=MatDestroy(this->M1);        CHKERRXX(this->ierr);
                                this->ierr=MatDestroy(this->M2);        CHKERRXX(this->ierr);
                                this->ierr=KSPDestroy(this->myKsp);        CHKERRXX(this->ierr);
                                //this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive); CHKERRXX(this->ierr);
                                //this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->is_crossed_neumann); CHKERRXX(this->ierr);
                                this->ierr=VecDestroy(this->myNeumanPoissonSolverNodeBase->add_); CHKERRXX(this->ierr);
                                //this->ierr=VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_); CHKERRXX(this->ierr);
                                //  this->ierr = MatDestroy(this->myNeumanPoissonSolverNodeBase->A);                      CHKERRXX(this->ierr);
                                //this->ierr = MatNullSpaceDestroy (this->myNeumanPoissonSolverNodeBase->A_null_space); CHKERRXX(this->ierr);
                                /*    this-> ierr = KSPDestroy(this->myNeumanPoissonSolverNodeBase->ksp);  */                  CHKERRXX(this->ierr);
                                //delete this->myNeumanPoissonSolverNodeBase;
                                //delete this->ix_fromLocal2Global;

                            }
                        }
                        this->createKSPContext();
                    }


                    break;
                }

                case Diffusion::dirichlet_crank_nicholson:
                {
                    //delete this->myNeumanPoissonSolverNodeBase;
                    this->setupDirichletSolver(node_neighbors);
                    if(this->fast_algo)
                    {
                        this->myNeumanPoissonSolverNodeBase->setupM2_Dirichlet();
                        this->createDiffusionMatricesDirichlet();
                        this->createKSPContext();
                    }
                    break;
                }


                }
            }


            //this->createInitialConditions(this->spatial);
            //this->printInitialConditions();

            this->createDiffusionSolution();
            //this->compute_mapping_fromLocal2Global();

            //            if(this->myPhase==Diffusion::from_text_file && this->i_mean_field==0)
            //            {
            //                this->get_potential_from_text_file_root_processor();
            //            }

            std::cout<<this->mpi->mpirank<<" w benchmark "<<this->w_benchmark<<std::endl;

            // if(this->i_mean_field>-1)
            //this->allocateMemory2History();

            if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
                    || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
            {
                this->filter_irregular_potential();
            }

            if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            {
                this->filter_petsc_vector_dirichlet(&this->wp);
                this->filter_petsc_vector_dirichlet(&this->wm);
            }

            if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive)
            {

                // this->printForestNodes2TextFile();
#ifdef P4_TO_P8
                // if(!this->real_to_complex_fftw_strategy)
                this->create_mapping_data_structure_fftw();
                //                else
                //                    this->create_mapping_data_structure_fftw_r2r();
#else
                //if(!this->real_to_complex_fftw_strategy)
                this->create_mapping_data_structure_fftw_2D();
                //                else
                //                    this->create_mapping_data_structure_fftw_2D_r2r();

#endif

            }

            if(i_game<n_games_i-1)
            {
                // Check what does this destructor do exactly
                if(this->setup_finite_difference_solver)
                    delete this->solver;
                //------------Petsc Object Destruction-----------------//
                this->ierr=VecDestroy(this->phi);       CHKERRXX(this->ierr);
                //this->ierr=VecDestroy(this->sol_0);      CHKERRXX(this->ierr);
                this->ierr=VecDestroy(this->sol_t);      CHKERRXX(this->ierr);
                this->ierr=VecDestroy(this->sol_tp1);     CHKERRXX(this->ierr);
                this->ierr=VecDestroy(this->b_t);         CHKERRXX(this->ierr);
                this->ierr=VecDestroy(this->r_t);         CHKERRXX(this->ierr);
                if( (this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson&& this->setup_finite_difference_solver) ||
                        this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson||
                        (!this->fast_algo))
                {
                    this->ierr=MatDestroy(this->M1);        CHKERRXX(this->ierr);
                    this->ierr=MatDestroy(this->M2);        CHKERRXX(this->ierr);
                    this->ierr=KSPDestroy(this->myKsp);        CHKERRXX(this->ierr);
                }
                if( (this->my_casl_diffusion_method!=Diffusion::periodic_crank_nicholson) )
                {
                    this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive); CHKERRXX(this->ierr);
                    this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->is_crossed_neumann); CHKERRXX(this->ierr);
                    delete this->myNeumanPoissonSolverNodeBase;
                    delete this->ix_fromLocal2Global;

                }
                if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive)
                {
                    this->ierr=ISDestroy(this->from_fftw); CHKERRXX(this->ierr);
                    this->ierr=ISDestroy(this->to_p4est); CHKERRXX(this->ierr);
                    this->ierr=VecScatterDestroy(&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
                    delete this->idx_from_fftw_to_p4est_fftw_ix;
                    delete this->idx_from_fftw_to_p4est_p4est_ix;

                    this->mapping_fftw_created=false;
                    this->mapping_fftw_destroyed=true;
                }
            }
        }
        this->remeshed=PETSC_TRUE;
        this->clean=PETSC_FALSE;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cout << "[" << this->mpi->mpirank << " -- ERROR]: " << e.what() << std::endl;
        return -1;
    }
}


Diffusion::~Diffusion()
{
    std::cout<<this->mpi->mpirank<<" Diffusion Destructor"<<std::endl;
    delete this->sol_history;
    //delete this->sol_p;
    //delete this->sol_p_cell;
    //delete this->rhs_p;
    //delete this->uex_p;
    //delete this->value;
}



// TODO: rewrite with volume integrals when time:
// This function gives correct results only on an uniform grid

int Diffusion::compute_energy_difference_analyticaly()
{



    if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive)
    {
        int Nx= (int)pow(2,this->max_level+log(this->nx_trees)/log(2));

        // This is just for checking analytical derivations
        //such that function will be debugged only on 1 core first

        Vec delta_w_m;
        PetscScalar *delta_w_local_m;
        Vec delta_w_p;
        PetscScalar *delta_w_local_p;
        PetscScalar *fp_local;
        PetscScalar *fm_local;
        this->ierr=VecDuplicate(this->phi,&delta_w_m); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(delta_w_m,&delta_w_local_m); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&delta_w_p); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(delta_w_p,&delta_w_local_p); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->fp,&fp_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->fm,&fm_local); CHKERRXX(this->ierr);
        int n_local;
        this->ierr=VecGetLocalSize(this->phi,&n_local); CHKERRXX(this->ierr);
        for(int i=0;i<n_local;i++)
        {
            delta_w_local_m[i]=-this->lambda_minus*fm_local[i];
            delta_w_local_p[i]=this->lambda_plus*fp_local[i];
        }


        double DEM=0;
        double DEP=0;
        for(int i=0;i<n_local;i++)
        {
            DEM+=delta_w_local_m[i]*fm_local[i];//+delta_w_local_p[i]*fp_local[i];
            DEP+=delta_w_local_p[i]*fp_local[i];
        }



        double nx_double=(double) Nx;



        std::cout<<" (nx_double,nx_double,nx_double) "<<(nx_double,nx_double,nx_double)<<std::endl;


#ifdef P4_TO_P8
        DEM=DEM/(nx_double*nx_double*nx_double);
        DEP=DEP/(nx_double*nx_double*nx_double);
#else
        DEM=DEM/(nx_double*nx_double);
        DEP=DEP/(nx_double*nx_double);
#endif
        this->ierr=VecRestoreArray(delta_w_m,&delta_w_local_m); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(delta_w_p,&delta_w_local_p); CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(this->fm,&fp_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->fp,&fm_local); CHKERRXX(this->ierr);


        this->ierr=VecDestroy(delta_w_m); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(delta_w_p); CHKERRXX(this->ierr);

        std::cout<<"DE predicted "<<DEM<<" "<<DEP<<" "<<DEM+DEP<<std::endl;


        this->analytic_energy_difference_predicted_current_time_step=this->analytic_energy_difference_predicted_next_time_step;
        double DE_LOCAL=DEM+DEP;
        this->ierr=MPI_Allreduce(&DE_LOCAL,& this->analytic_energy_difference_predicted_next_time_step,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);


    }
    else
    {


        // This is just for checking analytical derivations
        //such that function will be debugged only on 1 core first

        Vec delta_w_m;
        Vec delta_w_p;

        double DEM=0,DEP=0;

        this->ierr=VecDuplicate(this->phi,&delta_w_m); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&delta_w_p); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->fm,delta_w_m); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fp,delta_w_p); CHKERRXX(this->ierr);
        this->ierr=VecScale(delta_w_m,-this->lambda_minus); CHKERRXX(this->ierr);
        this->ierr=VecScale(delta_w_p,this->lambda_plus); CHKERRXX(this->ierr);

        this->ierr=VecPointwiseMult(delta_w_m,this->fm,delta_w_m); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(delta_w_p,this->fp,delta_w_p); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&delta_w_m);
        this->scatter_petsc_vector(&delta_w_p);



        if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        {
            my_p4est_level_set ls(this->nodes_neighbours);

            Vec bc_vec_fake;
            this->ierr= VecDuplicate(this->phi,&bc_vec_fake); CHKERRXX(this->ierr);
            this->ierr=VecSet(bc_vec_fake,0.00);              CHKERRXX(this->ierr);

            int extension_order=0,number_of_bands_to_extend=5;

            this->scatter_petsc_vector(&bc_vec_fake);
            ls.extend_Over_Interface(this->phi_polymer_shape,delta_w_m,NEUMANN,bc_vec_fake,extension_order,number_of_bands_to_extend);
            ls.extend_Over_Interface(this->phi_polymer_shape,delta_w_p,NEUMANN,bc_vec_fake,extension_order,number_of_bands_to_extend);

            this->scatter_petsc_vector(&delta_w_m);
            this->scatter_petsc_vector(&delta_w_p);

            DEM=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,delta_w_m);
            DEP=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,delta_w_p);

            this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);

        }
        if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
        {

            DEM=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,delta_w_m);
            DEP=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,delta_w_p);
        }
        if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
        {
            this->ierr=VecDuplicate(this->phi,&this->phiTemp); CHKERRXX(this->ierr);
            this->ierr=VecSet(this->phiTemp,-1.00); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phiTemp); CHKERRXX(this->ierr);
            DEM=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,delta_w_m);
            DEP=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,delta_w_p);

            this->ierr=VecDestroy(phiTemp); CHKERRXX(this->ierr);

        }

        DEM=DEM/this->V;
        DEP=DEP/this->V;



        this->ierr=VecDestroy(delta_w_m); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(delta_w_p); CHKERRXX(this->ierr);

        std::cout<<"DE predicted "<<DEM<<" "<<DEP<<" "<<DEM+DEP<<std::endl;


        this->analytic_energy_difference_predicted_current_time_step=this->analytic_energy_difference_predicted_next_time_step;
        this->analytic_energy_difference_predicted_next_time_step=DEM+DEP;
        std::cout<<"DE predicted "<<this->analytic_energy_difference_predicted_next_time_step<<std::endl;

    }



}


int Diffusion::computeStressTensor()
{

    this->stress_tensor_watch.start("stress tensor watch");

    Vec phi_is_all_positive_temp;
    this->ierr=VecDuplicate(this->phi,&phi_is_all_positive_temp); CHKERRXX(this->ierr);
    this->ierr=  VecSet(phi_is_all_positive_temp,0); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(phi_is_all_positive_temp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(phi_is_all_positive_temp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    Vec is_crossed_neuman_temp;
    this->ierr=VecDuplicate(this->phi,&is_crossed_neuman_temp); CHKERRXX(this->ierr);
    this->ierr=  VecSet(is_crossed_neuman_temp,0); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(is_crossed_neuman_temp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(is_crossed_neuman_temp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    switch(this->my_stress_tensor_computation_mode)
    {
    case StressTensor::qqcxx:
        this->my_stress_tensor=new StressTensor(this->mpi  ,this->p4est,this->nodes,this->ghost,this->connectivity,&this->brick,this->nodes_neighbours,&this->phi_polymer_shape,
                                                this->q_forward,this->q_backward,this->n_local_size_sol,this->local_size_sol,this->N_t,this->N_iterations,this->dt,
                                               &this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&this->myNeumanPoissonSolverNodeBase->is_crossed_neumann,this->ix_fromLocal2Global,
                                                0.5*(this->Q_forward+this->Q_backward),
                                                this->V,this->periodic_xyz,StressTensor::qqcxx,this->minimum_IO,this->IO_path,
                                                this->Lx,this->Lx_physics);
        break;

    case StressTensor::qxqcx_memory_optimized:
        this->my_stress_tensor=new StressTensor(this->mpi  ,this->p4est,this->nodes,this->ghost,this->connectivity,&this->brick,this->nodes_neighbours,&this->phi,
                                                this->q_forward,this->q_backward,this->n_local_size_sol,this->local_size_sol,this->N_t,this->N_iterations,this->dt,
                                                &phi_is_all_positive_temp,&is_crossed_neuman_temp,  this->ix_fromLocal2Global,0.5*(this->Q_forward+this->Q_backward),
                                                this->V,this->periodic_xyz,
                                                StressTensor::qxqcx_memory_optimized,this->minimum_IO,this->IO_path,
                                                this->Lx,this->Lx_physics);
        break;


    case StressTensor::shape_derivative:
        this->my_stress_tensor=new StressTensor(this->mpi  ,this->p4est,this->nodes,this->ghost,this->connectivity,&this->brick,this->nodes_neighbours,&this->phi_polymer_shape,
                                                this->q_forward,this->q_backward,this->n_local_size_sol,this->local_size_sol,this->N_t,this->N_iterations,this->dt,
                                                &this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&this->myNeumanPoissonSolverNodeBase->is_crossed_neumann   ,this->ix_fromLocal2Global,0.5*(this->Q_forward+this->Q_backward),
                                                this->V,this->periodic_xyz,
                                                StressTensor::shape_derivative,this->minimum_IO,this->IO_path,
                                                this->Lx,this->Lx_physics);
        break;

    }
    this->my_stress_tensor->set_some_params(this->nx_trees,this->max_level);
    this->my_stress_tensor->Compute_stress();
    this->ierr=VecDestroy(phi_is_all_positive_temp); CHKERRXX(this->ierr);
      this->ierr=VecDestroy(is_crossed_neuman_temp); CHKERRXX(this->ierr);
    this->stress_tensor_watch.stop();
}




int Diffusion::create_copy_and_scatter_shape_derivative_and_destroy_stress_tensor(Vec *snn)
{
    this->ierr=VecDuplicate(this->my_stress_tensor->snn_global,snn); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->my_stress_tensor->snn_global,*snn); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(snn);
    this->my_stress_tensor->cleanShapeDerivative();
}

int Diffusion::createDiffusionMatrices()
{

    this->ierr=MatDuplicate(this->solver->A,MAT_COPY_VALUES,&this->M1); CHKERRXX(this->ierr);
    this->ierr=MatCopy(this->solver->A,this->M1,SAME_NONZERO_PATTERN); CHKERRXX(this->ierr);
    this->ierr= MatDuplicate(this->solver->A,MAT_COPY_VALUES,&this->M2);CHKERRXX(this->ierr);
    this->ierr= MatCopy(this->solver->A,this->M2,SAME_NONZERO_PATTERN);CHKERRXX(this->ierr);
    this->ierr=  MatScale(this->M1,-1.00*0.5*this->dt);CHKERRXX(this->ierr);
    this->ierr=  MatScale(this->M2,-1.00*-0.5*this->dt);CHKERRXX(this->ierr);
    this->ierr=  MatShift(this->M1,1.00);CHKERRXX(this->ierr);
    this->ierr= MatShift(this->M2,1.00);CHKERRXX(this->ierr);


    if(this->print_diffusion_matrices)
    {
        this->printDiffusionMatrices(&this->M1,"M1",0);
        this->printDiffusionMatrices(&this->M2,"M2",0);
        this->printDiffusionMatrices(&this->solver->A,"M",0);
    }

    //    PetscBool check_matrices_symmetry=PETSC_TRUE;


    //    if(check_matrices_symmetry)
    //    {
    //        double symmetry_tolerance=0.0;
    //        PetscBool symmetry_flag=PETSC_FALSE;

    //        this->ierr=MatIsSymmetric(this->solver->A,symmetry_tolerance,&symmetry_flag); CHKERRXX(this->ierr);

    //        if(symmetry_flag)
    //            std::cout<<" A is symmetric "<<std::endl;
    //        else
    //            std::cout<<" A is not symmetric "<<std::endl;

    //        this->ierr=MatIsSymmetric(this->M1,symmetry_tolerance,&symmetry_flag); CHKERRXX(this->ierr);
    //        if(symmetry_flag)
    //            std::cout<<" M1 is symmetric "<<std::endl;
    //        else
    //            std::cout<<" M1 is not symmetric "<<std::endl;

    //        this->ierr=MatIsSymmetric(this->M2,symmetry_tolerance,&symmetry_flag); CHKERRXX(this->ierr);
    //        if(symmetry_flag)
    //            std::cout<<" M2 is symmetric "<<std::endl;
    //        else
    //            std::cout<<" M2 is not symmetric "<<std::endl;
    //    }

    return 0;
}


int Diffusion::createDiffusionMatricesRobin()
{
    if(true)
    {

    //if( !this->forward_stage && !this->first_stage)
    {
        this->ierr=MatDestroy(M1); CHKERRXX(this->ierr);
        this->ierr=MatDestroy(M2); CHKERRXX(this->ierr);


    }

    Vec v_interface;
    this->ierr=VecDuplicate(this->phi,&v_interface); CHKERRXX(this->ierr);
    this->myNeumanPoissonSolverNodeBase->setup_interface_matrix(&v_interface);

    PetscScalar scalar_before_insertion=-1;//*this->Lx_physics/this->Lx;

    // TODO: (easy) set the scalar for 3d

    if( (this->forward_stage && this->first_stage) || (!this->forward_stage && !this->first_stage))
        scalar_before_insertion=scalar_before_insertion*this->myMeanFieldPlan->kappaA;
    else
        scalar_before_insertion=scalar_before_insertion*this->myMeanFieldPlan->kappaB;

    this->ierr=VecScale(v_interface,scalar_before_insertion); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&v_interface);


    Vec Iv;
    Vec M2ii;
    Vec dii;
    Vec II;
    VecDuplicate(this->phi,&Iv);
    VecSet(Iv,1.00);

//    this->ierr=VecDestroy(this->myNeumanPoissonSolverNodeBase->add_); CHKERRXX(this->ierr);
//    this->myNeumanPoissonSolverNodeBase->set_diagonal(Iv);
//    this->myNeumanPoissonSolverNodeBase->setup_volume_matrix2();

//    VecGhostUpdateBegin(this->myNeumanPoissonSolverNodeBase->add_,INSERT_VALUES,SCATTER_FORWARD);
//    VecGhostUpdateEnd(this->myNeumanPoissonSolverNodeBase->add_,INSERT_VALUES,SCATTER_FORWARD);


    //this->scaleDiffusionMatricesNeuman();

    VecCopy(this->myNeumanPoissonSolverNodeBase->add_,Iv);

    VecGhostUpdateBegin(Iv,INSERT_VALUES,SCATTER_FORWARD);
    VecGhostUpdateEnd(Iv,INSERT_VALUES,SCATTER_FORWARD);


    if(this->print_diffusion_matrices)
    {
        this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M",0);
        this->printDiffusionArrayFromVector(&this->myNeumanPoissonSolverNodeBase->add_,"add");
        this->printDiffusionArrayFromVector(&Iv,"Iv");
    }

    // Load the free Neuman matrix A into M1 and M2
    // where A is computed as:

    // far from the interface:
    // Au=d2u/dx2+d2u/dy2+d2u/dz2

    // and far away from the interface
    // integral(Au)=integral(d2u/dx2+d2u/dy2+d2u/dz2)

    // such that if one has a Poisson problem to solve of Au=f
    // Au(!interface)+Au(interface)=f(!interface)+f(interface)
    // Au(!interface)+integral(Au(interface))=f(!interface)+integral(f(interface))

    // and usually the diffusion equation is solved with backward euler
    // in the following form
    // du/dt=d2u/dx2+d2u/dy2+d2u/dz2
    // Discretization in space:
    // u(t+1)-u(t)=dt x Au
    // CASL Irregular Algo Nodes Based
    // {u(t+1)-u(t)}(!interface)+{u(t+1)-u(t)}(interface)={dt x Au}(!interface) +{dt x Au}(interface)
    // {u(t+1)-u(t)}(!interface)+integral({u(t+1)-u(t)}(interface))={dt x Au}(!interface) +integral({dt x Au}(interface))
    // Implicit Backward Euler Algo
    // {u(t+1)-u(t)}(!interface)+integral({u(t+1)-u(t)}(interface))={dt x Au(t+1)}(!interface) +integral({dt x Au(t+1)}(interface))
    // (-dt x A+ I)(!interface) +integral((-dt x A + I)(interface))u(t+1)=
    //  {I x u(t)}(!interface)+integral({I x u(t)}(interface))

    // Due to the different scales present in the matrix from (!interface) and integral(interface)
    // the classical casl method is to build and scale the rhs annd matrix at each diffusion iteration
    // this is a very good idea to precondition the matrix

    // But at the end of the day the building of the matrix and rhs is not cheap cpu wise
    // Another way to do it would be:
    // Mv=A(!interface) +integral(A)(interface)
    // Iv=I(!interface) +I(A)(interface)
    // M2=Iv-dt*Mv
    // u(t+1)=(M2^(-1) x Iv )x u(t)
    // u(t+1)=(M2^(-1)xD^-1xDxIv)x u(t)
    // Iv will be called M1
    // where D is just the diagonal of M2

    // To implement the crank-nicholson a slight modification should be applied
    // M2=Iv-0.5*dt*Mv
    // M1=Iv+0.5*dt*Mv
    // u(t+1)=(M2^(-1) x M1 )     x u(t)
    // u(t+1)=(M2^(-1)xD^-1xDxM1) x u(t)
    // where D is just the diagonal of M2
    // such that we do the Jacobi preconditioning
    // which is efficient for diagonally dominant matrices







        PetscScalar scaler1=-1.00*1.00/2.00*this->dt;
        PetscScalar scaler2=-1.00*-1.00/2.00*this->dt;


        // M1 is just the identity matrix multiplied by the volume close to the interface
        this->ierr=  MatDuplicate(this->myNeumanPoissonSolverNodeBase->A,MAT_COPY_VALUES,&this->M1); CHKERRXX(this->ierr);
        this->ierr=   MatCopy(this->myNeumanPoissonSolverNodeBase->A,this->M1,SAME_NONZERO_PATTERN); CHKERRXX(this->ierr);
        this->ierr=   MatDiagonalSet(this->M1,v_interface,ADD_VALUES);  CHKERRXX(this->ierr);



        this->ierr=  MatScale(this->M1,scaler1);  CHKERRXX(this->ierr);
        this->ierr=   MatDiagonalSet(this->M1,this->myNeumanPoissonSolverNodeBase->add_,ADD_VALUES);  CHKERRXX(this->ierr);

        // M2 is the negative laplacian matrix mutliplied by the volume close
        // to the interface
        MatDuplicate(this->myNeumanPoissonSolverNodeBase->A,MAT_COPY_VALUES,&this->M2);
        MatCopy(this->myNeumanPoissonSolverNodeBase->A,this->M2,SAME_NONZERO_PATTERN);
        // The negative laplacian matrix is scaled by the time step

        if(this->print_diffusion_matrices)
            this->printDiffusionMatrices(&this->M2,"M2",0);


        this->ierr=   MatDiagonalSet(this->M2,v_interface,ADD_VALUES);  CHKERRXX(this->ierr);

        MatScale(this->M2,scaler2);

        if(this->print_diffusion_matrices)
            this->printDiffusionMatrices(&this->M2,"M2_scaled",0);
        // We add Iv to M2 as dscribed in details previously



        MatDiagonalSet(this->M2,this->myNeumanPoissonSolverNodeBase->add_,ADD_VALUES);

        if(this->print_diffusion_matrices)
            this->printDiffusionMatrices(&this->M2,"M2_add",0);

        // Make the Jacoby preconditioning
        VecDuplicate(Iv,&II);
        VecSet(II,1.00);

        VecGhostUpdateBegin(II,INSERT_VALUES,SCATTER_FORWARD);
        VecGhostUpdateEnd(II,INSERT_VALUES,SCATTER_FORWARD);


        VecDuplicate(Iv,&M2ii);
        MatGetDiagonal(this->M2,M2ii);

        VecGhostUpdateBegin(M2ii,INSERT_VALUES,SCATTER_FORWARD);
        VecGhostUpdateEnd(M2ii,INSERT_VALUES,SCATTER_FORWARD);

        VecDuplicate(M2ii,&dii);
        VecCopy(M2ii, dii);
        VecGhostUpdateBegin(dii,INSERT_VALUES,SCATTER_FORWARD);
        VecGhostUpdateEnd(dii,INSERT_VALUES,SCATTER_FORWARD);

        VecReciprocal(dii);

        VecGhostUpdateBegin(dii,INSERT_VALUES,SCATTER_FORWARD);
        VecGhostUpdateEnd(dii,INSERT_VALUES,SCATTER_FORWARD);

        MatDiagonalScale(this->M2,dii,II);
        MatDiagonalScale(this->M1,dii,II);

         MatAssemblyBegin(this->M1,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(this->M1,MAT_FINAL_ASSEMBLY);

        // is the assebmly necessary: to play with it to check
        // if it is necessaryy only after the call MatSetValues as
        // in the offical Petsc doucumentation

        MatAssemblyBegin(this->M2,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(this->M2,MAT_FINAL_ASSEMBLY);




    if(this->print_diffusion_matrices)
    {
        if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
            this->printDiffusionMatrices(&this->M1,"M1",0);

        this->printDiffusionMatrices(&this->M2,"M2_preconditionned",0);
        this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M",0);
        this->printDiffusionArrayFromVector(&this->myNeumanPoissonSolverNodeBase->add_,"Iv");
    }

    if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        this->filter_irregular_matrices();

    if(this->print_diffusion_matrices)
    {
        if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
            this->printDiffusionMatrices(&this->M1,"M1_after_insertion",0);
    }

    this->ierr=VecDestroy(Iv);   CHKERRXX(this->ierr);
    this->ierr=VecDestroy(M2ii); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(dii);  CHKERRXX(this->ierr);
    this->ierr=VecDestroy(II);   CHKERRXX(this->ierr);
    this->ierr=VecDestroy(v_interface); CHKERRXX(this->ierr);

    if(this->print_diffusion_matrices)
        this->printDiffusionVector(&this->myNeumanPoissonSolverNodeBase->add_,"neuman_add");
}
}

// See Papac and Gibou 2012 for how it is discretized on the interface
// Briefly: we could say that close to the inteface a finite volume approach is taken


int Diffusion::createDiffusionMatricesNeuman()
{

    int n_games_j=1;

    for(int j_game=0;j_game<n_games_j;j_game++)
    {

        Vec Iv;
        Vec M2ii;
        Vec dii;
        Vec II;
        VecDuplicate(this->phi,&Iv);
        VecSet(Iv,1.00);

        this->myNeumanPoissonSolverNodeBase->set_diagonal(Iv);
        this->myNeumanPoissonSolverNodeBase->setup_volume_matrix2();

        VecGhostUpdateBegin(this->myNeumanPoissonSolverNodeBase->add_,INSERT_VALUES,SCATTER_FORWARD);
        VecGhostUpdateEnd(this->myNeumanPoissonSolverNodeBase->add_,INSERT_VALUES,SCATTER_FORWARD);


        this->scaleDiffusionMatricesNeuman();

        VecCopy(this->myNeumanPoissonSolverNodeBase->add_,Iv);

        VecGhostUpdateBegin(Iv,INSERT_VALUES,SCATTER_FORWARD);
        VecGhostUpdateEnd(Iv,INSERT_VALUES,SCATTER_FORWARD);


        if(this->print_diffusion_matrices)
        {
            this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M",0);
            this->printDiffusionArrayFromVector(&this->myNeumanPoissonSolverNodeBase->add_,"add");
            this->printDiffusionArrayFromVector(&Iv,"Iv");
        }

        // Load the free Neuman matrix A into M1 and M2
        // where A is computed as:

        // far from the interface:
        // Au=d2u/dx2+d2u/dy2+d2u/dz2

        // and far away from the interface
        // integral(Au)=integral(d2u/dx2+d2u/dy2+d2u/dz2)

        // such that if one has a Poisson problem to solve of Au=f
        // Au(!interface)+Au(interface)=f(!interface)+f(interface)
        // Au(!interface)+integral(Au(interface))=f(!interface)+integral(f(interface))

        // and usually the diffusion equation is solved with backward euler
        // in the following form
        // du/dt=d2u/dx2+d2u/dy2+d2u/dz2
        // Discretization in space:
        // u(t+1)-u(t)=dt x Au
        // CASL Irregular Algo Nodes Based
        // {u(t+1)-u(t)}(!interface)+{u(t+1)-u(t)}(interface)={dt x Au}(!interface) +{dt x Au}(interface)
        // {u(t+1)-u(t)}(!interface)+integral({u(t+1)-u(t)}(interface))={dt x Au}(!interface) +integral({dt x Au}(interface))
        // Implicit Backward Euler Algo
        // {u(t+1)-u(t)}(!interface)+integral({u(t+1)-u(t)}(interface))={dt x Au(t+1)}(!interface) +integral({dt x Au(t+1)}(interface))
        // (-dt x A+ I)(!interface) +integral((-dt x A + I)(interface))u(t+1)=
        //  {I x u(t)}(!interface)+integral({I x u(t)}(interface))

        // Due to the different scales present in the matrix from (!interface) and integral(interface)
        // the classical casl method is to build and scale the rhs annd matrix at each diffusion iteration
        // this is a very good idea to precondition the matrix

        // But at the end of the day the building of the matrix and rhs is not cheap cpu wise
        // Another way to do it would be:
        // Mv=A(!interface) +integral(A)(interface)
        // Iv=I(!interface) +I(A)(interface)
        // M2=Iv-dt*Mv
        // u(t+1)=(M2^(-1) x Iv )x u(t)
        // u(t+1)=(M2^(-1)xD^-1xDxIv)x u(t)
        // Iv will be called M1
        // where D is just the diagonal of M2

        // To implement the crank-nicholson a slight modification should be applied
        // M2=Iv-0.5*dt*Mv
        // M1=Iv+0.5*dt*Mv
        // u(t+1)=(M2^(-1) x M1 )     x u(t)
        // u(t+1)=(M2^(-1)xD^-1xDxM1) x u(t)
        // where D is just the diagonal of M2
        // such that we do the Jacobi preconditioning
        // which is efficient for diagonally dominant matrices



        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler)
        {
            // M1 is just the identity matrix multiplied by the volume close to the interface
            // M1 sould be a diagonal matrix
            // M1 will just be a vector

            // M2 is the negative laplacian matrix mutliplied by the volume close
            // to the interface
            MatDuplicate(this->myNeumanPoissonSolverNodeBase->A,MAT_COPY_VALUES,&this->M2);
            MatCopy(this->myNeumanPoissonSolverNodeBase->A,this->M2,SAME_NONZERO_PATTERN);
            // The negative laplacian matrix is scaled by the time step


            MatScale(this->M2,-1.00*-1.00*this->dt);

            // We add Iv to M2 as dscribed in details previously



            MatDiagonalSet(this->M2,this->myNeumanPoissonSolverNodeBase->add_,ADD_VALUES);


            // Make the Jacoby preconditioning
            VecDuplicate(Iv,&II);
            VecSet(II,1.00);

            VecDuplicate(Iv,&M2ii);
            MatGetDiagonal(this->M2,M2ii);
            VecDuplicate(M2ii,&dii);
            VecCopy(M2ii, dii);

            VecReciprocal(dii);
            MatDiagonalScale(this->M2,dii,II);
            VecPointwiseMult(this->myNeumanPoissonSolverNodeBase->add_,dii,this->myNeumanPoissonSolverNodeBase->add_);

            MatAssemblyBegin(this->M2,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(this->M2,MAT_FINAL_ASSEMBLY);


        }

        if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        {


            PetscScalar scaler1=-1.00*1.00/2.00*this->dt;
            PetscScalar scaler2=-1.00*-1.00/2.00*this->dt;


            // M1 is just the identity matrix multiplied by the volume close to the interface
            this->ierr=  MatDuplicate(this->myNeumanPoissonSolverNodeBase->A,MAT_COPY_VALUES,&this->M1); CHKERRXX(this->ierr);
            this->ierr=   MatCopy(this->myNeumanPoissonSolverNodeBase->A,this->M1,SAME_NONZERO_PATTERN); CHKERRXX(this->ierr);
            this->ierr=  MatScale(this->M1,scaler1);  CHKERRXX(this->ierr);
            this->ierr=   MatDiagonalSet(this->M1,this->myNeumanPoissonSolverNodeBase->add_,ADD_VALUES);  CHKERRXX(this->ierr);

            // M2 is the negative laplacian matrix mutliplied by the volume close
            // to the interface
            MatDuplicate(this->myNeumanPoissonSolverNodeBase->A,MAT_COPY_VALUES,&this->M2);
            MatCopy(this->myNeumanPoissonSolverNodeBase->A,this->M2,SAME_NONZERO_PATTERN);
            // The negative laplacian matrix is scaled by the time step

            if(this->print_diffusion_matrices)
                this->printDiffusionMatrices(&this->M2,"M2",0);

            MatScale(this->M2,scaler2);

            if(this->print_diffusion_matrices)
                this->printDiffusionMatrices(&this->M2,"M2_scaled",0);
            // We add Iv to M2 as dscribed in details previously



            MatDiagonalSet(this->M2,this->myNeumanPoissonSolverNodeBase->add_,ADD_VALUES);

            if(this->print_diffusion_matrices)
                this->printDiffusionMatrices(&this->M2,"M2_add",0);

            // Make the Jacoby preconditioning
            VecDuplicate(Iv,&II);
            VecSet(II,1.00);

            VecGhostUpdateBegin(II,INSERT_VALUES,SCATTER_FORWARD);
            VecGhostUpdateEnd(II,INSERT_VALUES,SCATTER_FORWARD);


            VecDuplicate(Iv,&M2ii);
            MatGetDiagonal(this->M2,M2ii);

            VecGhostUpdateBegin(M2ii,INSERT_VALUES,SCATTER_FORWARD);
            VecGhostUpdateEnd(M2ii,INSERT_VALUES,SCATTER_FORWARD);

            VecDuplicate(M2ii,&dii);
            VecCopy(M2ii, dii);
            VecGhostUpdateBegin(dii,INSERT_VALUES,SCATTER_FORWARD);
            VecGhostUpdateEnd(dii,INSERT_VALUES,SCATTER_FORWARD);

            VecReciprocal(dii);

            VecGhostUpdateBegin(dii,INSERT_VALUES,SCATTER_FORWARD);
            VecGhostUpdateEnd(dii,INSERT_VALUES,SCATTER_FORWARD);

            MatDiagonalScale(this->M2,dii,II);
            MatDiagonalScale(this->M1,dii,II);

            // MatAssemblyBegin(this->M1,MAT_FLUSH_ASSEMBLY);
            //MatAssemblyEnd(this->M1,MAT_FLUSH_ASSEMBLY);

            // is the assebmly necessary: to play with it to check
            // if it is necessaryy only after the call MatSetValues as
            // in the offical Petsc doucumentation

            MatAssemblyBegin(this->M2,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(this->M2,MAT_FINAL_ASSEMBLY);

        }


        if(this->print_diffusion_matrices)
        {
            if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
                this->printDiffusionMatrices(&this->M1,"M1",0);

            this->printDiffusionMatrices(&this->M2,"M2_preconditionned",0);
            this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M",0);
            this->printDiffusionArrayFromVector(&this->myNeumanPoissonSolverNodeBase->add_,"Iv");
        }

        if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
            this->filter_irregular_matrices();

        if(this->print_diffusion_matrices)
        {
            if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
                this->printDiffusionMatrices(&this->M1,"M1_after_insertion",0);
        }

        this->ierr=VecDestroy(Iv);   CHKERRXX(this->ierr);
        this->ierr=VecDestroy(M2ii); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(dii);  CHKERRXX(this->ierr);
        this->ierr=VecDestroy(II);   CHKERRXX(this->ierr);
        if(this->print_diffusion_matrices)
            this->printDiffusionVector(&this->myNeumanPoissonSolverNodeBase->add_,"neuman_add");

        if(j_game<n_games_j-1)
        {
            // Check what does this destructor do exactly
            // delete this->solver;
            //------------Petsc Object Destruction-----------------//
            //this->ierr=VecDestroy(this->phi);       CHKERRXX(this->ierr);
            if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
                    || this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
            {
                this->ierr=MatDestroy(this->M1);        CHKERRXX(this->ierr);
                this->ierr=MatDestroy(this->M2);        CHKERRXX(this->ierr);
                //this->ierr=KSPDestroy(this->myKsp);        CHKERRXX(this->ierr);

            }
            if(!this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
            {
                //this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive); CHKERRXX(this->ierr);
                //this->ierr= VecDestroy(this->myNeumanPoissonSolverNodeBase->is_crossed_neumann); CHKERRXX(this->ierr);
                this->ierr=VecDestroy(this->myNeumanPoissonSolverNodeBase->add_); CHKERRXX(this->ierr);
                //this->ierr=VecDestroy(this->myNeumanPoissonSolverNodeBase->phi_); CHKERRXX(this->ierr);

                //  this->ierr = MatDestroy(this->myNeumanPoissonSolverNodeBase->A);                      CHKERRXX(this->ierr);
                //this->ierr = MatNullSpaceDestroy (this->myNeumanPoissonSolverNodeBase->A_null_space); CHKERRXX(this->ierr);
                /*    this-> ierr = KSPDestroy(this->myNeumanPoissonSolverNodeBase->ksp);  */                  CHKERRXX(this->ierr);
                //delete this->myNeumanPoissonSolverNodeBase;
                //delete this->ix_fromLocal2Global;

            }
        }
    }
}

int Diffusion::scaleDiffusionMatricesNeuman()
{
    this->ierr=VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol);CHKERRXX(this->ierr);
    this->ierr= VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol);CHKERRXX(this->ierr);
    this->compute_mapping_fromLocal2Global();

    // we do have three kind of nodes,
    // we segregate them by the sign of the level set function whose cell they do belong to
    // (1) all positive: outside the domain
    // (2) all negative: inside the domain
    // (3) at least one positive and at least one negative:
    // in other words they do not all have the same sign


    // We simply create a vector with 1 on where we are outside the domain
    // D=(Lx/L x Lx/L) where are inside the domain
    // dx/dy=1 where are on the interface:  see equation 9 of Papac and Gibou

    Vec matrix_scaler;
    this->ierr=VecDuplicate(this->phi,&matrix_scaler);CHKERRXX(this->ierr);
    PetscScalar *matrix_scaler_local;
    this->ierr=VecGetArray(matrix_scaler,&matrix_scaler_local);CHKERRXX(this->ierr);

    Vec vector_scaler;
    this->ierr=VecDuplicate(this->phi,&vector_scaler);CHKERRXX(this->ierr);
    PetscScalar *vector_scaler_local;
    this->ierr=VecGetArray(vector_scaler,&vector_scaler_local);CHKERRXX(this->ierr);


    PetscScalar *all_positive_array;
    this->ierr= VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&all_positive_array); CHKERRXX(this->ierr);


    PetscScalar *not_same_sign_array;
    this->ierr= VecGetArray(this->myNeumanPoissonSolverNodeBase->is_crossed_neumann,&not_same_sign_array);CHKERRXX(this->ierr);


    std::cout<<" local size "<<this->mpi->mpirank<<" "<<this->n_local_size_sol<<std::endl;
    std::cout<<" global size "<<this->mpi->mpirank<<" "<<this->n_global_size_sol<<std::endl;


    for(int i=0;i<this->n_local_size_sol;i++)
    {
        // internal nodes:
        // they all have the same sign and they are negative
        if(all_positive_array[i]<0.5 && not_same_sign_array[i]<0.5)
        {
            matrix_scaler_local[i]=this->D;
        }


        // they don't have the same sign: at the interface, we correct for the surface
        // if you see papac: L/dx=1 on 2d but on 3d S/dx=dx s.t we have to correct by dx_physics/dx=Lx_physics/Lx
        if(not_same_sign_array[i]>0.5 && all_positive_array[i]<0.5)
        {
#ifdef P4_TO_P8
            matrix_scaler_local[i]=this->Lx_physics/this->Lx;
#else
            matrix_scaler_local[i]=1.00;
#endif
        }

        // external nodes: they are all positive
        if(!all_positive_array[i]<0.5)
            matrix_scaler_local[i]=1.00;


        // here we correct the volume integral;
        if(not_same_sign_array[i]>0.5)
        {
            //in 3d
#ifdef P4_TO_P8
            vector_scaler_local[i]=pow(this->Lx_physics/this->Lx,3);
#else
            vector_scaler_local[i]=pow(this->Lx_physics/this->Lx,2);
#endif
        }
        else
        {
            vector_scaler_local[i]=1.00;
        }
    }

    this->ierr=VecRestoreArray(matrix_scaler,&matrix_scaler_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(vector_scaler,&vector_scaler_local);CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&all_positive_array); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->is_crossed_neumann,&not_same_sign_array);CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(matrix_scaler,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(matrix_scaler,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr= VecGhostUpdateBegin(vector_scaler,INSERT_VALUES,SCATTER_FORWARD);  CHKERRXX(this->ierr);
    this->ierr= VecGhostUpdateEnd(vector_scaler,INSERT_VALUES,SCATTER_FORWARD);    CHKERRXX(this->ierr);


    // Make the Jacoby preconditioning
    Vec II;
    this->ierr=VecDuplicate(this->phi,&II);CHKERRXX(this->ierr);
    this->ierr=VecSet(II,1.00);CHKERRXX(this->ierr);

    this->scatter_petsc_vector(&II);

    //        this->printDiffusionArrayFromVector(&matrix_scaler,"scl1");
    //        this->printDiffusionArrayFromVector(&vector_scaler,"scl2");
    //        this->printDiffusionArrayFromVector(&this->myNeumanPoissonSolverNodeBase->is_crossed_neumann,"interface");
    //        this->printDiffusionArrayFromVector(&this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,"outside");

    this->ierr=MatDiagonalScale(this->myNeumanPoissonSolverNodeBase->A,matrix_scaler,II);CHKERRXX(this->ierr);

    this->ierr=VecPointwiseMult(this->myNeumanPoissonSolverNodeBase->add_,this->myNeumanPoissonSolverNodeBase->add_,vector_scaler);CHKERRXX(this->ierr);


    this->ierr=  MatAssemblyBegin(this->myNeumanPoissonSolverNodeBase->A,MAT_FINAL_ASSEMBLY);CHKERRXX(this->ierr);
    this->ierr=MatAssemblyEnd(this->myNeumanPoissonSolverNodeBase->A,MAT_FINAL_ASSEMBLY);CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->myNeumanPoissonSolverNodeBase->add_,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->myNeumanPoissonSolverNodeBase->add_,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);


    this->ierr=VecDestroy(matrix_scaler);CHKERRXX(this->ierr);
    this->ierr=VecDestroy(vector_scaler);CHKERRXX(this->ierr);
    this->ierr=VecDestroy(II); CHKERRXX(this->ierr);
}

int Diffusion::createDiffusionMatricesDirichlet()
{

    int n_games_j=1;

    for(int j_game=0;j_game<n_games_j;j_game++)
    {

        Vec Iv;
        Vec M2ii;
        Vec dii;
        Vec II;
        this->ierr=VecDuplicate(this->phi,&Iv); CHKERRXX(this->ierr);
        this->ierr=VecSet(Iv,1.00); CHKERRXX(this->ierr);

        this->ierr=MatScale(this->myNeumanPoissonSolverNodeBase->A,this->D); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&Iv);

        if(this->print_diffusion_matrices)
        {
            this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M",0);
            this->printDiffusionArrayFromVector(&Iv,"Iv");
        }

        PetscScalar scaler1=-1.00*1.00/2.00*this->dt;
        PetscScalar scaler2=-1.00*-1.00/2.00*this->dt;


        // M1 is just the identity matrix multiplied by the volume close to the interface
        this->ierr=  MatDuplicate(this->myNeumanPoissonSolverNodeBase->A,MAT_COPY_VALUES,&this->M1); CHKERRXX(this->ierr);
        this->ierr=   MatCopy(this->myNeumanPoissonSolverNodeBase->A,this->M1,SAME_NONZERO_PATTERN); CHKERRXX(this->ierr);
        this->ierr=  MatScale(this->M1,scaler1);  CHKERRXX(this->ierr);
        this->ierr=   MatDiagonalSet(this->M1,Iv,ADD_VALUES);  CHKERRXX(this->ierr);

        // M2 is the negative laplacian matrix mutliplied by the volume close
        // to the interface
        this->ierr=MatDuplicate(this->myNeumanPoissonSolverNodeBase->A,MAT_COPY_VALUES,&this->M2); CHKERRXX(this->ierr);
        this->ierr=MatCopy(this->myNeumanPoissonSolverNodeBase->A,this->M2,SAME_NONZERO_PATTERN); CHKERRXX(this->ierr);
        // The negative laplacian matrix is scaled by the time step

        if(this->print_diffusion_matrices)
            this->printDiffusionMatrices(&this->M2,"M2",0);

        this->ierr=MatScale(this->M2,scaler2); CHKERRXX(this->ierr);

        if(this->print_diffusion_matrices)
            this->printDiffusionMatrices(&this->M2,"M2_scaled",0);
        // We add Iv to M2 as dscribed in details previously



        this->ierr=MatDiagonalSet(this->M2,Iv,ADD_VALUES); CHKERRXX(this->ierr);

        if(this->print_diffusion_matrices)
            this->printDiffusionMatrices(&this->M2,"M2_add",0);

        // Make the Jacoby preconditioning
        this->ierr=VecDuplicate(Iv,&II); CHKERRXX(this->ierr);
        this->ierr=VecSet(II,1.00); CHKERRXX(this->ierr);

        this->ierr= VecGhostUpdateBegin(II,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(II,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        this->ierr=VecDuplicate(Iv,&M2ii); CHKERRXX(this->ierr);
        this->ierr=MatGetDiagonal(this->M2,M2ii); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(M2ii,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(M2ii,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        this->ierr=VecDuplicate(M2ii,&dii);CHKERRXX(this->ierr);
        this->ierr=VecCopy(M2ii, dii);CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(dii,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(dii,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        this->ierr=VecReciprocal(dii);CHKERRXX(this->ierr);

        this->ierr= VecGhostUpdateBegin(dii,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(dii,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        this->ierr= MatDiagonalScale(this->M2,dii,II);CHKERRXX(this->ierr);
        this->ierr= MatDiagonalScale(this->M1,dii,II);CHKERRXX(this->ierr);

        this->filter_irregular_matrices_dirichlet();


        // M1 ia assembled already in filter_irregular_matrices_dirichlet
        //        this->ierr=MatAssemblyBegin(this->M1,MAT_FINAL_ASSEMBLY);CHKERRXX(this->ierr);
        //        this->ierr=MatAssemblyEnd(this->M1,MAT_FINAL_ASSEMBLY);CHKERRXX(this->ierr);

        //!\\
        // is the assebmly necessary: to play with it to check
        // if it is necessaryy only after the call MatSetValues as
        // in the offical Petsc doucumentation

        this->ierr=MatAssemblyBegin(this->M2,MAT_FINAL_ASSEMBLY);CHKERRXX(this->ierr);
        this->ierr= MatAssemblyEnd(this->M2,MAT_FINAL_ASSEMBLY);CHKERRXX(this->ierr);




        if(this->print_diffusion_matrices)
        {

            this->printDiffusionMatrices(&this->M1,"M1",0);

            this->printDiffusionMatrices(&this->M2,"M2_preconditionned",0);
            this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M",0);

        }


        this->ierr=VecDestroy(Iv);   CHKERRXX(this->ierr);
        this->ierr=VecDestroy(M2ii); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(dii);  CHKERRXX(this->ierr);
        this->ierr=VecDestroy(II);   CHKERRXX(this->ierr);

        if(j_game<n_games_j-1)
        {
            this->ierr=MatDestroy(this->M1);        CHKERRXX(this->ierr);
            this->ierr=MatDestroy(this->M2);        CHKERRXX(this->ierr);

        }

    }
}



int Diffusion::filter_irregular_matrices()
{
    VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol);
    VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol);
    this->compute_mapping_fromLocal2Global();
    PetscInt *filter_ix=new PetscInt[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
    //PetscScalar *filter_zero=new PetscScalar[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
    PetscScalar *filter_one=new PetscScalar[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];

    PetscScalar *phi_is_all_positive;

    VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive);
    int ix=0;
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_all_positive[i]>0.5)
        {
            filter_ix[ix]=i;
            ix++;
        }
    }

    for(int i=0;i<this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive;i++)
    {
        filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
        filter_one[i]=1.00;
        //filter_zero[i]=0.00;
    }

    Vec dd;
    VecDuplicate(this->phi_polymer_shape,&dd);
    MatGetDiagonal(this->M1,dd);
    VecSetValues(dd,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_one,INSERT_VALUES);

    // VecSetValues(this->wp,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);
    //VecSetValues(this->wm,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);

    VecAssemblyBegin(dd);
    VecAssemblyEnd(dd);


    //VecAssemblyBegin(this->wp);
    //VecAssemblyEnd(this->wp);

    //VecAssemblyBegin(this->wm);
    //VecAssemblyEnd(this->wm);

    if(this->print_diffusion_matrices)
    {
        this->printDiffusionVector(&dd,"dd");
    }


    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);

    MatAssemblyBegin(this->M1,MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(this->M1,MAT_FLUSH_ASSEMBLY);
    MatDiagonalSet(this->M1,dd,INSERT_VALUES);

    MatAssemblyBegin(this->M1,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(this->M1,MAT_FINAL_ASSEMBLY);

    VecDestroy(dd);
    delete filter_ix;
    delete filter_one;
    //   delete filter_zero;
    return 0;
}

int Diffusion::filter_irregular_matrices_dirichlet()
{
    this->ierr=VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol); CHKERRXX(this->ierr);
    this->ierr=VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol); CHKERRXX(this->ierr);
    this->compute_mapping_fromLocal2Global();
    // could use the same filter than in the potential filtering
    // consider making it a member class

    PetscScalar *phi_is_positive,*phi_is_negative_but_below_eps;
    PetscInt n2Filter=0;
    this->ierr=VecGetArray(this->phi_polymer_shape,&phi_is_positive); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_below_epsilon,&phi_is_negative_but_below_eps); CHKERRXX(this->ierr);
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_positive[i]>0.00 || phi_is_negative_but_below_eps[i]>0.5)
        {
            n2Filter++;
        }
    }

    PetscInt *filter_ix=new PetscInt[n2Filter];
    PetscScalar *filter_one=new PetscScalar[n2Filter];
    int ix=0;
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_positive[i]>0.00 || phi_is_negative_but_below_eps[i]>0.5)
        {
            filter_ix[ix]=i;
            ix++;
        }
    }
    // mapp filter ix from local to global
    for(int i=0;i<n2Filter;i++)
    {
        filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
        filter_one[i]=1.00;
    }

    Vec dd;
    VecDuplicate(this->phi_polymer_shape,&dd);
    MatGetDiagonal(this->M1,dd);
    VecSetValues(dd,n2Filter,filter_ix,filter_one,INSERT_VALUES);

    VecAssemblyBegin(dd);
    VecAssemblyEnd(dd);

    if(this->print_diffusion_matrices)
    {
        this->printDiffusionVector(&dd,"dd");
    }


    this->ierr=VecRestoreArray(this->phi_polymer_shape,&phi_is_positive); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_below_epsilon,&phi_is_negative_but_below_eps); CHKERRXX(this->ierr);

    MatAssemblyBegin(this->M1,MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(this->M1,MAT_FLUSH_ASSEMBLY);
    MatDiagonalSet(this->M1,dd,INSERT_VALUES);

    MatAssemblyBegin(this->M1,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(this->M1,MAT_FINAL_ASSEMBLY);

    VecDestroy(dd);
    delete filter_ix;
    delete filter_one;
    //   delete filter_zero;
    return 0;
}


int Diffusion::extractSubDiffusionMatrices()
{

    if(this->submatrices_destructed && !this->submatrices_extracted)
    {

        // step 1: create the parallel index scattering
        int n_not_phi_is_all_positive_local=0;
        PetscScalar *all_positive_array;
        PetscInt *idx_is;
        this->ierr= VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&all_positive_array); CHKERRXX(this->ierr);
        for(int i=0;i<this->n_local_size_sol;i++)
        {

            if(all_positive_array[i]>0.5)
                n_not_phi_is_all_positive_local++;
        }
        this->ierr= VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&all_positive_array); CHKERRXX(this->ierr);
        this->ierr=  PetscMalloc( n_not_phi_is_all_positive_local*sizeof(PetscInt),&idx_is);CHKERRXX(this->ierr);

        int j=0;

        for(int i=0;i<this->n_local_size_sol;i++)
        {

            if(all_positive_array[i]>0.5)
            {idx_is[j]=i;j++;}
        }

        this->ierr=ISCreateGeneral(MPI_COMM_WORLD,n_not_phi_is_all_positive_local,idx_is,PETSC_OWN_POINTER,&this->my_rows_parallel_indexes);
        this->ierr=ISCreateGeneral(MPI_COMM_WORLD,n_not_phi_is_all_positive_local,idx_is,PETSC_OWN_POINTER,&this->my_columns_parallel_indexes);



        // step 2: extract the submatrix
        this->ierr=MatGetSubMatrix(this->M1,this->my_rows_parallel_indexes,
                                   this->my_columns_parallel_indexes,MAT_INITIAL_MATRIX,&this->subM1); CHKERRXX(this->ierr);

        this->ierr=MatGetSubMatrix(this->M2,this->my_rows_parallel_indexes,
                                   this->my_columns_parallel_indexes,MAT_INITIAL_MATRIX,&this->subM2); CHKERRXX(this->ierr);
    }
    else
    {

        //std::
    }


}


int Diffusion::filter_irregular_potential()
{
    std::cout<<this->mpi->mpirank<<" start filtering potentials "<<std::endl;

    if(this->print_diffusion_matrices)
    {
        this->printDiffusionVector(&this->wp,"wp_before_filtering");
        this->printDiffusionVector(&this->wm,"wm_before_filtering");
    }

    VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol);
    VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol);
    this->compute_mapping_fromLocal2Global();
    PetscInt *filter_ix=new PetscInt[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
    PetscScalar *filter_zero=new PetscScalar[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];

    PetscScalar *phi_is_all_positive;

    VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive);
    int ix=0;
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_all_positive[i]>0.5)
        {
            filter_ix[ix]=i;
            ix++;
        }
    }

    for(int i=0;i<this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive;i++)
    {
        filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
        //filter_one[i]=1.00;
        filter_zero[i]=0.00;
    }

    VecSetValues(this->wp,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);
    VecSetValues(this->wm,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);


    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);

    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    if(this->print_diffusion_matrices)
    {
        this->printDiffusionVector(&this->wp,"wp_after_filtering");
        this->printDiffusionVector(&this->wm,"wm_after_filtering");
    }
    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);
    delete filter_ix;
    delete filter_zero;
    std::cout<<this->mpi->mpirank<<" finished filtering potentials "<<std::endl;

    return 0;
}


int Diffusion::filter_irregular_forces()
{

    if(this->print_diffusion_matrices)
    {
        this->printDiffusionVector(&this->fp,"wp_before_filtering");
        this->printDiffusionVector(&this->fm,"wm_before_filtering");
    }

    VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol);
    VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol);
    this->compute_mapping_fromLocal2Global();

    // could use the same filter than in the potential filtering
    // consider making it a member class
    PetscInt *filter_ix=new PetscInt[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
    PetscScalar *filter_zero=new PetscScalar[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];

    PetscScalar *phi_is_all_positive;

    VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive);
    int ix=0;
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_all_positive[i]>0.5)
        {
            filter_ix[ix]=i;
            ix++;
        }
    }

    for(int i=0;i<this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive;i++)
    {
        filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
        filter_zero[i]=0.00;
    }

    VecSetValues(this->fp,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);
    VecSetValues(this->fm,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);

    VecAssemblyBegin(this->fp);
    VecAssemblyEnd(this->fp);

    VecAssemblyBegin(this->fm);
    VecAssemblyEnd(this->fm);

    //    VecSetValues(this->rhoA,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);
    //    VecSetValues(this->rhoB,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES);

    //    VecAssemblyBegin(this->rhoA);
    //    VecAssemblyEnd(this->rhoA);

    //    VecAssemblyBegin(this->rhoB);
    //    VecAssemblyEnd(this->rhoB);

    if(this->print_diffusion_matrices)
    {
        this->printDiffusionVector(&this->fp,"wp_after_filtering");
        this->printDiffusionVector(&this->fm,"wm_after_filtering");
    }

    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);

    delete filter_ix;
    delete filter_zero;
    return 0;
}


int Diffusion::filter_petsc_vector(Vec *v2filter)
{


    VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol);
    VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol);
    this->compute_mapping_fromLocal2Global();

    // could use the same filter than in the potential filtering
    // consider making it a member class
    PetscInt *filter_ix=new PetscInt[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
    PetscScalar *filter_zero=new PetscScalar[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];

    PetscScalar *phi_is_all_positive;

    VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive);
    int ix=0;
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_all_positive[i]>0.5)
        {
            filter_ix[ix]=i;
            ix++;
        }
    }

    for(int i=0;i<this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive;i++)
    {
        filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
        filter_zero[i]=0.00;
    }

    this->ierr=VecSetValues(*v2filter,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES); CHKERRXX(this->ierr);

    this->ierr=VecAssemblyBegin(*v2filter); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(*v2filter);   CHKERRXX(this->ierr);

    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);

    delete filter_ix;
    delete filter_zero;
    return 0;
}
int Diffusion::filter_petsc_vector_dirichlet(Vec *v2filter)
{


    this->ierr=VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol); CHKERRXX(this->ierr);
    this->ierr=VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol); CHKERRXX(this->ierr);
    this->compute_mapping_fromLocal2Global();
    // could use the same filter than in the potential filtering
    // consider making it a member class

    PetscScalar *phi_is_positive,*phi_is_negative_but_below_eps;
    PetscInt n2Filter=0;
    this->ierr=VecGetArray(this->phi_polymer_shape,&phi_is_positive); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_below_epsilon,&phi_is_negative_but_below_eps); CHKERRXX(this->ierr);
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_positive[i]>0.00 || phi_is_negative_but_below_eps[i]>0.5)
        {
            n2Filter++;
        }
    }

    PetscInt *filter_ix=new PetscInt[n2Filter];
    PetscScalar *filter_zero=new PetscScalar[n2Filter];
    int ix=0;
    for(int i=0;i<this->n_local_size_sol;i++)
    {
        if(phi_is_positive[i]>0.00 || phi_is_negative_but_below_eps[i]>0.5)
        {
            filter_ix[ix]=i;
            ix++;
        }
    }
    for(int i=0;i<n2Filter;i++)
    {
        filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
        filter_zero[i]=0.00;
    }

    this->ierr=VecSetValues(*v2filter,n2Filter,filter_ix,filter_zero,INSERT_VALUES); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyBegin(*v2filter); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(*v2filter);   CHKERRXX(this->ierr);

    this->ierr=VecRestoreArray(this->phi_polymer_shape,&phi_is_positive); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_below_epsilon,&phi_is_negative_but_below_eps); CHKERRXX(this->ierr);
    delete filter_ix;
    delete filter_zero;
    return 0;
}

int Diffusion::extend_petsc_vector(Vec * v2extend)
{
    int extension_order=0;
    int number_of_bands_extension=5;
    my_p4est_level_set ls(this->nodes_neighbours);
    Vec bc_vec_fake;
    this->ierr= VecDuplicate(this->phi,&bc_vec_fake); CHKERRXX(this->ierr);
    this->ierr=VecSet(bc_vec_fake,0.00);              CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    ls.extend_Over_Interface(this->phi_polymer_shape,*v2extend,NEUMANN,bc_vec_fake,extension_order,number_of_bands_extension);

    this->ierr=VecGhostUpdateBegin(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);
}

int Diffusion::create_statistical_field()
{
    VecDuplicate(this->phi,&this->wp);
    VecDuplicate(this->phi,&this->wm);
    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    {
        switch(this->myPhase)
        {
        case Diffusion::benchmark_constant:
        {
            VecSet(this->wp,this->w_benchmark);
            VecSet(this->wm,this->w_benchmark);
            break;
        }
        case Diffusion::disordered:
        {
            this->generate_disordered_phase();
            break;
        }

        case Diffusion::lamellar:
        {
            this->generate_lamellar_phase();
            break;
        }
        case Diffusion::lamellar_x:
        {
            this->generate_lamellar_x_phase();
            break;
        }
        case Diffusion::lamellar_x_from_matlab:
        {
            this->generate_lamellar_x_phase_from_text_file();
            break;
        }

        case Diffusion::lamellar_y:
        {
            this->generate_lamellar_y_phase();
            break;
        }
        case Diffusion::lamellar_z:
        {
            this->generate_lamellar_z_phase();
            break;
        }
        case Diffusion::bcc:
        {

            this->generate_bcc_phase();


            break;
        }
        case Diffusion::bcc_masked:
        {
            this->generate_bcc_phase_masked();
            break;
        }
        case Diffusion::random:
        {
            this->generate_random_phase();
            break;
        }
        case Diffusion::single_ellipse:
        {
            this->generate_single_ellipse();
            break;
        }

        case Diffusion::smooth:
        {
            this->generate_smooth_fields();
            break;
        }
        case Diffusion::smooth_r:
        {
            this->generate_smooth_fields_r();
            break;
        }
        case Diffusion::smooth_y:
        {
            this->generate_smooth_fields_lamellar_y();
            break;
        }
        case Diffusion::smooth_x:
        {
            this->generate_smooth_fields_lamellar_x();
            break;
        }
        case Diffusion::constant_w:
        {
            this->generate_constant_field();
            break;
        }
        case Diffusion::gaussian_w:
        {
            this->generate_gaussian_field();
            break;
        }
        case Diffusion::clock_w:
        {
            this->generate_clock_field();
            break;
        }
        case Diffusion::cos_w:
        {
            this->generate_cos_field();
            break;
        }
        case Diffusion::smooth_xy:
        {
            this->generate_smooth_fields_xy();
            break;
        }
        case Diffusion::from_one_text_file:
        {
            this->get_fields_from_text_file();
            break;
        }
        case Diffusion::from_two_text_files:
        {
            this->get_fields_from_two_text_files();
            break;
        }
        case Diffusion::from_one_fine_text_file_to_coarse:
        {
            this->get_fine_fields_from_text_file();
            break;
        }
        case Diffusion::from_two_fine_text_file_to_coarse:
        {
            this->get_fine_fields_from_two_text_files();
            break;
        }
        }
    }
    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
    || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
    || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
    {
        switch(this->myPhase)
        {

        case Diffusion::disordered:
        {
            this->generate_disordered_phase();
            break;
        }
        case Diffusion::bcc_masked:
        {
            this->generate_bcc_phase_masked();
            break;
        }
        case Diffusion::lamellar:
        {
            this->generate_lamellar_phase();
            break;
        }
        case Diffusion::confined_spheres:
        {
            this->generate_bcc_phase_masked();
            break;
        }
        case Diffusion::confined_spheres_helix:
        {
            this->generate_spheres_helixed();
            break;
        }
        case Diffusion::confine_spheres_3d_l_shape:
        {
            this->generate_spheres_helixed_for_3d_l_shape();
            break;
        }


        case Diffusion::confined_cylinders_helix:
        {
            this->generate_cylinders_helixed();
            break;
        }

        case Diffusion::random:
        {
            this->generate_random_phase();
            break;
        }
        case Diffusion::single_ellipse:
        {
            this->generate_single_ellipse();
            break;
        }
        case Diffusion::constant_w:
        {
            this->generate_constant_field();
            break;
        }
        case Diffusion::gaussian_w:
        {
            this->generate_gaussian_field();
            break;
        }
        case Diffusion::clock_w:
        {
            this->generate_clock_field();
            break;
        }
        case Diffusion::cos_w:
        {
            this->generate_cos_field();
            break;
        }
        case Diffusion::smooth:
        {
            this->generate_smooth_fields();
            break;
        }
        case Diffusion::smooth_r:
        {
            this->generate_smooth_fields_r();
            break;
        }
        case Diffusion::smooth_y:
        {
            this->generate_smooth_fields_lamellar_y();
            break;
        }
        case Diffusion::smooth_x:
        {
            this->generate_smooth_fields_lamellar_x();
            break;
        }
        case Diffusion::from_one_text_file:
        {
            this->get_fields_from_text_file();
            break;
        }
        case Diffusion::from_two_text_files:
        {
            this->get_fields_from_two_text_files();
            break;
        }
        case Diffusion::from_one_fine_text_file_to_coarse:
        {
            this->get_fine_fields_from_text_file();
            break;
        }
        case Diffusion::smooth_xy:
        {
            this->generate_smooth_fields_xy();
            break;
        }
        }
    }
}




int Diffusion::get_fields_from_text_file()
{

    int wp_column_fetch=4,wpColumns2Continue=1;
    int wm_column_fetch=5,wmColumns2Continue=0;
    this->get_vec_from_text_file(&this->wp,this->input_potential_path_w_p,wp_column_fetch,wpColumns2Continue);
    this->get_vec_from_text_file(&this->wm,this->input_potential_path_w_m,wm_column_fetch,wmColumns2Continue);

}

int Diffusion::get_fine_fields_from_text_file()
{

    int wp_column_fetch=4,wpColumns2Continue=1;
    int wm_column_fetch=5,wmColumns2Continue=0;
    int file_level=5;
    this->get_coarse_field_from_fine_text_file(&wp,this->input_potential_path_w_p,file_level,wp_column_fetch,wpColumns2Continue);
    this->get_coarse_field_from_fine_text_file(&this->wm,this->input_potential_path_w_m,file_level,wm_column_fetch,wmColumns2Continue);

}
int Diffusion::get_fine_fields_from_two_text_files()
{

    int wp_column_fetch=6,wpColumns2Continue=0;
    int wm_column_fetch=6,wmColumns2Continue=0;
    int file_level=8;
    this->get_coarse_field_from_fine_text_file(&wp,this->input_potential_path_w_p,file_level,wp_column_fetch,wpColumns2Continue);
    this->get_coarse_field_from_fine_text_file(&this->wm,this->input_potential_path_w_m,file_level,wm_column_fetch,wmColumns2Continue);

}
int Diffusion::get_fields_from_two_text_files()
{

    int wp_column_fetch=5,wpColumns2Continue=0;
    int wm_column_fetch=5,wmColumns2Continue=0;
    this->get_vec_from_text_file(&this->wp,this->input_potential_path_w_p,wp_column_fetch,wpColumns2Continue);
    this->get_vec_from_text_file(&this->wm,this->input_potential_path_w_m,wm_column_fetch,wmColumns2Continue);

}

int Diffusion::get_coarse_fields_from_text_file()
{

    int wp_column_fetch=4;
    int wm_column_fetch=5;
    this->get_coarse_vec_from_text_file(&this->wp,this->field_text_file,wp_column_fetch);
    this->get_coarse_vec_from_text_file(&this->wm,this->field_text_file,wm_column_fetch);

}


int Diffusion::generate_lamellar_phase()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];


        double x,y,z;

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        x = node_x_fr_i(node) + tree_xmin;
        y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        z =node_z_fr_k(node) + tree_zmin;
#endif


        if(x<f*this->Lx/4.00 || x>this->Lx-this->f*this->Lx/4.00 || (x<this->Lx/2+this->f*this->Lx/4.00 && x>this->Lx/2-this->f*this->Lx/4.00))
            wm_local[i]=this->X_ab/2;
        else
            wm_local[i]=-this->X_ab/2;


        wp_local[i]=0;


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i, x ,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);

    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



}


int Diffusion::generate_lamellar_x_phase_from_text_file()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase_from_matlab")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();



    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2));

    double *wp_matlab=new double [N0];
    double *wm_matlab=new double [N0];

    std::string str=this->input_potential_path_w_m;
    fstream f;
    f.open(str.c_str(),ios_base::in);
    for(int i=0; i<N0; i++)
    {
        f>>wm_matlab[i] ;
        std::cout<<i<<" "<<wm_matlab[i]<<std::endl;
    }
    f.close();

    std::string str2=this->input_potential_path_w_p;
    fstream f2;
    f2.open(str2.c_str(),ios_base::in);
    for(int i=0; i<N0; i++)
    {
        f2>>wp_matlab[i] ;
        std::cout<<i<<" "<<wp_matlab[i]<<std::endl;
    }
    f2.close();



    //global_node_number=nodes->offset_owned_indeps;

    int n_local_size_amr;
    this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

    double i_node,j_node,k_node,x,y,z, offset_tree_x, offset_tree_y, offset_tree_z,
            x_normalized,y_normalized,z_normalized;

    double max_pow=log(P4EST_ROOT_LEN)/log(2);
    double normalizer=pow(2,max_pow-this->max_level);


    PetscScalar *wp_local=new PetscScalar[this->nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];


    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = this->connectivity->vertices[3*v_mm + 1];


        double x,y,z;

#ifdef P4_TO_P8
        double tree_zmin = this->connectivity->vertices[3*v_mm + 2];
#endif

        x = node_x_fr_i(node) + tree_xmin;
        y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        z =node_z_fr_k(node) + tree_zmin;
#endif

        x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
        y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);
#ifdef P4_TO_P8
        z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
#endif
        offset_tree_x=tree_xmin/this->Lx*N0;
        offset_tree_y=tree_ymin/this->Lx*N0;
#ifdef P4_TO_P8
        offset_tree_z=tree_zmin/this->Lx*N0;
#endif
        x_normalized=x/normalizer;
        y_normalized=y/normalizer;
#ifdef P4_TO_P8
        z_normalized=z/normalizer;
#endif
        i_node=x_normalized+offset_tree_x;
        j_node=y_normalized+offset_tree_y;
#ifdef P4_TO_P8
        k_node=z_normalized+offset_tree_z;
#endif

        wm_local[i]=wm_matlab[(int)i_node];
        wp_local[i]=wp_matlab[(int)i_node];


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i, x ,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);

    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



}


int Diffusion::generate_lamellar_x_phase()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];


        double x,y,z;

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        x = node_x_fr_i(node) + tree_xmin;
        y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        z =node_z_fr_k(node) + tree_zmin;
#endif


        if(x<f*this->Lx/4.00 || x>this->Lx-this->f*this->Lx/4.00 || (x<this->Lx/2+this->f*this->Lx/4.00 && x>this->Lx/2-this->f*this->Lx/4.00))
            wm_local[i]=this->X_ab/2;
        else
            wm_local[i]=-this->X_ab/2;


        wp_local[i]=0;


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i, x ,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);

    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



}

int Diffusion::generate_lamellar_y_phase()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];


        double x,y,z;

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        x = node_x_fr_i(node) + tree_xmin;
        y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        z =node_z_fr_k(node) + tree_zmin;
#endif


        if(y<f*this->Lx/4.00 || y>this->Lx-this->f*this->Lx/4.00 || (y<this->Lx/2+this->f*this->Lx/4.00 && y>this->Lx/2-this->f*this->Lx/4.00))
            wm_local[i]=this->X_ab/2;
        else
            wm_local[i]=-this->X_ab/2;


        wp_local[i]=0;


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i, x ,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);

    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



}

int Diffusion::generate_lamellar_z_phase()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];


        double x,y,z;

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        x = node_x_fr_i(node) + tree_xmin;
        y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        z =node_z_fr_k(node) + tree_zmin;
#endif


        if(z<f*this->Lx/4.00 || z>this->Lx-this->f*this->Lx/4.00 || (z<this->Lx/2+this->f*this->Lx/4.00 && z>this->Lx/2-this->f*this->Lx/4.00))
            wm_local[i]=this->X_ab/2;
        else
            wm_local[i]=-this->X_ab/2;


        wp_local[i]=0;


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i, x ,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);

    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


}



int Diffusion::generate_random_phase()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    int nb2Generate=nodes->num_owned_indeps;
    this->my_random_generator=new RandomGenerator(-this->X_ab/2,this->X_ab/2,nb2Generate);


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];


        double x,y,z;

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        z = int2double_coordinate_transform(node->x) + tree_xmin;
        y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        x = int2double_coordinate_transform(node->z) + tree_zmin;
#endif

        wm_local[i]=this->my_random_generator->random_Integer(i);
        wp_local[i]=0;
#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  z,y,x,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i, z ,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);

    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


    delete this->my_random_generator;

}

int Diffusion::get_potential_from_text_file_root_processor()
{

    // The potential should have been generated from the same mesh than
    // the mesh used in the current simulation

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    PetscScalar *wp_local=new PetscScalar[this->nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];



    std::cout<<"Started to Load Exchange Potential Seed"<<std::endl;
    fstream file_m;
    file_m.open(this->input_potential_path_w_m.c_str(),ios_base::in);
    std::cout<<this->n_global_size_sol<<"    "<<this->n_local_size_sol<<" "<<std::endl;
    double dummy_index; double dummyPotential;

    // first skip lines to get to the right number of lines

    for(int i=0; i<global_node_number; i++)
    {
        file_m >>dummy_index >>dummyPotential;
    }

    for(int i=0; i<this->nodes->num_owned_indeps; i++)
    {
        file_m >>dummy_index >> dummyPotential;
        //std::cout<<dummy_index<<" "<<dummyPotential<<std::endl;
        wm_local[i]=dummyPotential;
    }
    file_m.close();


    std::cout<<"Started to Load Pressure Potential Seed"<<std::endl;
    fstream file_p;
    file_p.open(this->input_potential_path_w_p.c_str(),ios_base::in);
    std::cout<<this->n_global_size_sol<<"    "<<this->n_local_size_sol<<" "<<std::endl;

    // first skip lines to get to the right number of lines

    for(int i=0; i<global_node_number; i++)
    {
        file_p >>dummy_index >>dummy_index;
    }

    for(int i=0; i<this->nodes->num_owned_indeps; i++)
    {
        file_p >>dummy_index >> wp_local[i];
    }
    file_p.close();


    this->printDiffusionArray(wm_local,this->nodes->num_owned_indeps,"wm_local_from_text");
    this->printDiffusionArray(wp_local,this->nodes->num_owned_indeps,"wp_local_from_text");



    std::cout<<"nodes->num_owned_indeps "<<this->nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<this->nodes->indep_nodes.elem_count<<std::endl;

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);


    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->printDiffusionArrayFromVector(&this->wp,"wp_from_text");
    this->printDiffusionArrayFromVector(&this->wm,"wm_from_text");

}


int Diffusion::generate_phase_from_mask(inverse_litography *my_litography)
{

    if(my_litography->reseed)
    {

        double ax_in=my_litography->ax,by_in=my_litography->by,cz_in=my_litography->cz;

        ax_in=ax_in*ax_in*my_litography->r_spot*my_litography->r_spot/my_litography->cut_r_spot;
        by_in=by_in*by_in*my_litography->r_spot*my_litography->r_spot/my_litography->cut_r_spot;
        cz_in=cz_in*cz_in*my_litography->r_spot*my_litography->r_spot/my_litography->cut_r_spot;





        for(int i=0;i<my_litography->n2Check;i++)
        {std::cout<<my_litography->xc[i]<<" "<<my_litography->yc[i]<<" "<<my_litography->zc[i]<<std::endl;}


        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("litography_phase")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        outFile=fopen(mystr2Debug.c_str(),"w");
        std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<myRank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        this->compute_mapping_fromLocal2Global();


        PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

        //global_node_number=nodes->offset_owned_indeps;

        for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
        {
            /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

            double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
            double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
            double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

            double x = int2double_coordinate_transform(node->x) + tree_xmin;
            double y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
            double z = int2double_coordinate_transform(node->z) + tree_zmin;
#endif


            bool isInsideASphere=false;


            for(int ii=0;ii<my_litography->n2Check;ii++)
            {
                if( pow(x- my_litography->xc[ii],2)/ax_in+pow(y-my_litography->yc[ii],2)/by_in
        #ifdef P4_TO_P8
                        //                    +pow(z-my_litography->zc[ii],2)/cz_in
        #endif
                        <1.00)
                {
                    isInsideASphere=true;
                    ii=my_litography->n2Check;
                }
            }
            double A=this->X_ab/2.00;
            if(isInsideASphere)
            {
                wp_local[i]=0;
                wm_local[i]=+A;
            }
            else
            {
                wp_local[i]=0;
                wm_local[i]=-A;
            }

#ifdef P4_TO_P8
            fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
            fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

        }
        std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
        std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
        fclose(outFile);

        this->ierr=VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES); CHKERRXX(this->ierr);
        this->ierr=VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES); CHKERRXX(this->ierr);


        this->ierr=VecAssemblyBegin(this->wp); CHKERRXX(this->ierr);
        this->ierr=VecAssemblyEnd(this->wp);   CHKERRXX(this->ierr);
        this->ierr=VecAssemblyBegin(this->wm); CHKERRXX(this->ierr);
        this->ierr=VecAssemblyEnd(this->wm);   CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    }
}

int Diffusion::generate_bcc_phase_masked()
{


    int n2Check;
#ifdef P4_TO_P8
    n2Check=9;
#else
    n2Check=5;
#endif

    double L=this->Lx;
    double V_temp=PI*this->Lx*this->Lx/16.00/n2Check;
    double r=pow((V_temp*this->f)/(2*PI),0.5);
    double *xc=new double[9];
    double *yc=new double[9];
    double *zc=new double[9];

    double R_mask=this->Lx/4.00;

    double dx=this->Lx/4.00/2.00;

    double x0;
    x0=this->Lx/2.00;

    double move_in=3*r;

    bool square_mask=false;
    bool L_mask=false;
    bool pyramid=true;

#ifdef P4_TO_P8


    xc[0]=x0;yc[0]=x0;zc[0]=x0;
    // (L,L,L) corner
    xc[1]=L-move_in;yc[1]=L-move_in;zc[1]=L-move_in;
    // (0,0,0) corner
    xc[2]=move_in;yc[2]=move_in;zc[2]=move_in;

    //(0,L,L) corner
    xc[3]=move_in;yc[3]=L-move_in;zc[3]=L-move_in;

    //(0,0,L) corner
    xc[4]=move_in;yc[4]=move_in;zc[4]=L-move_in;

    //(L,0,0)
    xc[5]=L-move_in;yc[5]=move_in;zc[5]=move_in;

    //(L,L,0)
    xc[6]=L-move_in;yc[6]=L-move_in;zc[6]=move_in;

    // (0,L,0)
    xc[7]=move_in;yc[7]=L-move_in;zc[7]=move_in;

    //L,0,L
    xc[8]=L-move_in;yc[8]=move_in;zc[8]=L-move_in;


#else

    if(square_mask)
    {
        xc[0]=x0-3*r;
        yc[0]=x0-3*r;
        zc[0]=x0;
        // (L,L,L) corner
        xc[1]=x0+3*r;
        yc[1]=x0-3*r;
        zc[1]=x0;
        // (0,0,0) corner
        xc[2]=x0+3*r;
        yc[2]=x0+3*r;
        zc[2]=x0;
        //(0,L,L) corner
        xc[3]=x0-3*r;
        yc[3]=x0+3*r;
        zc[3]=x0;
        //(0,0,L) corner
        xc[4]=x0;yc[4]=x0;zc[4]=L-move_in;

        //(L,0,0)
        xc[5]=L-move_in;yc[5]=move_in;zc[5]=move_in;
    }

    if(L_mask)
    {
        xc[0]=x0-3*dx;
        yc[0]=x0-3*r;
        zc[0]=x0;
        // (L,L,L) corner
        xc[1]=x0-3*r;
        yc[1]=x0;
        zc[1]=x0;
        // (0,0,0) corner
        xc[2]=x0+3*r;
        yc[2]=x0+3*r;
        zc[2]=x0;
        //(0,L,L) corner
        xc[3]=x0;
        yc[3]=x0+3*r;
        zc[3]=x0;
        //(0,0,L) corner
        xc[4]=x0+3*r;
        yc[4]=x0+3*r;
        zc[4]=L-move_in;

        //(L,0,0)
        xc[5]=L-move_in;yc[5]=move_in;zc[5]=move_in;
    }

    if(pyramid)
    {
        xc[0]=x0-dx;
        yc[0]=x0-dx;
        zc[0]=x0;

        xc[1]=x0+dx;
        yc[1]=x0-dx;
        zc[1]=x0;

        xc[2]=x0;
        yc[2]=x0-dx;
        zc[2]=x0;

        xc[3]=x0+0.75*dx;
        yc[3]=x0+dx;
        zc[3]=x0;

        xc[4]=x0-0.75*dx;
        yc[4]=x0+dx;
        zc[4]=L-move_in;


        xc[5]=L-move_in;yc[5]=move_in;zc[5]=move_in;
    }


    //(L,L,0)
    xc[6]=L-move_in;yc[6]=L-move_in;zc[6]=move_in;

    // (0,L,0)
    xc[7]=move_in;yc[7]=L-move_in;zc[7]=move_in;

    //L,0,L
    xc[8]=L-move_in;yc[8]=move_in;zc[8]=L-move_in;



#endif


    double ax_in=1.00,by_in=1.00,cz_in=1;

    ax_in=ax_in*ax_in*r*r;
    by_in=by_in*by_in*r*r;
    cz_in=cz_in*cz_in*r*r;





    for(int i=0;i<9;i++)
    {std::cout<<xc[i]<<" "<<yc[i]<<" "<<zc[i]<<std::endl;}


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("litography_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        double z = int2double_coordinate_transform(node->z) + tree_zmin;
#endif


        bool isInsideASphere=false;


        for(int ii=0;ii<n2Check;ii++)
        {
            if( pow(x-xc[ii],2)/ax_in+pow(y-yc[ii],2)/by_in
        #ifdef P4_TO_P8
                    +pow(z-zc[ii],2)/cz_in
        #endif
                    <1.00)
            {
                isInsideASphere=true;
                ii=n2Check;
            }
        }
        double A=this->X_ab/2.00;
        if(isInsideASphere)
        {
            wp_local[i]=0;
            wm_local[i]=+A;
        }
        else
        {
            wp_local[i]=0;
            wm_local[i]=-A;
        }

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);


    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



}


int Diffusion::generate_spheres_helixed_for_3d_l_shape()
{
#ifdef P4_TO_P8
    double r_helix=this->myMeanFieldPlan->alpha_r_helix*this->Lx;

    double r_mask=2*this->Lx/8.00;

    double t_start=0;
    double t_end=this->Lx;

    double xc=this->Lx/2.00;
    double yc=this->Lx/2.00;



    double helix_speed=2;

    int N_t_helix=4*(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    int tol=N_t_helix/8.00;

    int n_vector=N_t_helix-2*tol+1;

    double *x_vector=new double[n_vector];

    double dt_helix=(t_end-t_start)/N_t_helix;
    double *distance_helix=new double[n_vector];
    double *y_vector=new double [n_vector];
    double *z_vector=new double[n_vector];

    for(int i=0;i<n_vector;i++)
    {
        x_vector[i]=dt_helix*(i+tol);
        y_vector[i]=1.53*0.25*0.25/(x_vector[i]*x_vector[i])+0.22;
        z_vector[i]=1.53*0.25*0.25/((2-x_vector[i])*(2-x_vector[i]))+0.22;
        std::cout<<i<<" "<<x_vector[i]<<" "
                <<y_vector[i]<<" "
               << z_vector[i]<<std::endl;
    }




    //    double dt_sphere=(double)((t_end-t_start)/(n_spheres_double));



    for(int i=0;i<n_vector;i++)
    {std::cout<<x_vector[i]<<" "<<y_vector[i]<<" "<<z_vector[i]<<std::endl;}

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif


        bool isInsideASphere=false;


        for(int ii=0;ii<n_vector;ii++)
        {
            if( pow(x-x_vector[ii],2)+pow(y-y_vector[ii],2)+pow(z-z_vector[ii],2)<pow(r_helix/2.00,2.00))
            {
                isInsideASphere=true;

            }
        }
        double A=this->X_ab/2.00;
        if(isInsideASphere)
        {
            wp_local[i]=0;
            wm_local[i]=A;
        }
        else
        {
            wp_local[i]=0;
            wm_local[i]=-A;
        }

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);


    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



    delete x_vector;
    delete distance_helix;
    delete y_vector;
    delete z_vector;

#endif

}


int Diffusion::generate_spheres_helixed()
{
#ifdef P4_TO_P8
    double L=this->Lx;
    double r_sphere=1*this->Lx/12;
    double x0;
    x0=this->Lx/2.00;


    double r_helix=this->Lx/32.00;

    double r_mask=this->Lx/6.00;

    double t_start=0;
    double t_end=this->Lx;

    double x_c=this->Lx/2.00;
    double y_c=this->Lx/2.00;


    int tol=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))/4+5;

    double helix_speed=1;

    int N_t_helix=(int)pow(2.00,(double)this->max_level+log(8)/log(2));

    int n_spheres=N_t_helix-2*tol;
    double *xc=new double[n_spheres];
    double *yc=new double[n_spheres];
    double *zc=new double[n_spheres];


    double *t_vector=new double[N_t_helix-2*tol];

    double dt_helix=(t_end-t_start)/N_t_helix;

    double *sint_vector=new double [N_t_helix-2*tol];
    double *cost_vector=new double[N_t_helix-2*tol];

    for(int i=0;i<N_t_helix-2*tol;i++)
    {
        t_vector[i]=dt_helix*(i+tol);
        sint_vector[i]=r_mask*sin(2*PI*helix_speed*t_vector[i])+x_c;
        cost_vector[i]=r_mask*cos(2*PI*helix_speed*t_vector[i])+y_c;
        zc[i]=t_vector[i];
        xc[i]=cost_vector[i];
        yc[i]=sint_vector[i];
        std::cout<<i<<" "<<t_vector[i]<<" "
                <<sin(t_vector[i])<<" "
               <<cos(t_vector[i])<<std::endl;
    }


    //    double dt_sphere=(double)((t_end-t_start)/(n_spheres_double));



    for(int i=0;i<n_spheres;i++)
    {std::cout<<xc[i]<<" "<<yc[i]<<" "<<zc[i]<<std::endl;}

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif


        bool isInsideASphere=false;


        for(int ii=0;ii<n_spheres;ii++)
        {
            if( pow(x-xc[ii],2)+pow(y-yc[ii],2)+pow(z-zc[ii],2)<pow(r_sphere,2))
            {
                isInsideASphere=true;

            }
        }
        double A=this->X_ab/2.00;
        if(isInsideASphere)
        {
            wp_local[i]=0;
            wm_local[i]=A;
        }
        else
        {
            wp_local[i]=0;
            wm_local[i]=-A;
        }

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);


    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);




    delete t_vector;
    delete cost_vector;
    delete sint_vector;

#endif

}


int Diffusion::generate_cylinders_helixed()
{

#ifdef P4_TO_P8
    double L=this->Lx;
    double r_cylinder=1*this->Lx/32;

    double x0;
    x0=this->Lx/2.00;

    double my_eps=this->Lx/256;

    double r_helix=this->Lx/16.00+my_eps;

    double r_mask=3*this->Lx/8.00;

    double t_start=0;
    double t_end=this->Lx;

    double x_c=this->Lx/2.00;
    double y_c=this->Lx/2.00;


    int tol=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))/4;

    double helix_speed=2;

    int N_t_helix=(int)pow(2.00,(double)this->max_level+log(8)/log(2));

    double *t_vector=new double[N_t_helix-2*tol];

    double dt_helix=(t_end-t_start)/N_t_helix;

    double *sint_vector=new double [N_t_helix-2*tol];
    double *cost_vector=new double[N_t_helix-2*tol];

    //    double *sint_vector_cylinder=new double [N_t_helix-2*tol];
    //    double *cost_vector_cylinder=new double[N_t_helix-2*tol];

    double *distance_helix=new double [N_t_helix-2*tol];

    for(int i=0;i<N_t_helix-2*tol;i++)
    {
        t_vector[i]=dt_helix*(i+tol);
        sint_vector[i]=r_mask*sin(2*PI*helix_speed*t_vector[i]/t_end)+x_c;
        cost_vector[i]=r_mask*cos(2*PI*helix_speed*t_vector[i]/t_end)+y_c;

        //        sint_vector_cylinder[i]=r_cylinder*sin(helix_speed*t_vector[i])+x_c;
        //        cost_vector_cylinder[i]=r_cylinder*cos(helix_speed*t_vector[i])+y_c;
        std::cout<<i<<" "<<t_vector[i]<<" "
                <<sin(t_vector[i])<<" "
               <<cos(t_vector[i])<<std::endl;
    }






    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        double z = int2double_coordinate_transform(node->z) + tree_zmin;
#endif

        bool inside_cylinder=false;


        for(int j=0;j<N_t_helix-2*tol;j++)
        {
            distance_helix[j]=
                    (x-cost_vector[j])*(x-cost_vector[j])
                    +(y-sint_vector[j])*(y-sint_vector[j])
                    +(z-t_vector[j])*(z-t_vector[j]);


        }


        double minimum_proposed=this->Lx*this->Lx*this->Lx;

        //find minimum:
        int j_min=0;
        for(int j=0;j<N_t_helix-2*tol;j++)
        {
            if(distance_helix[j]<minimum_proposed)
            {
                minimum_proposed=distance_helix[j];
                j_min=j;
            }
        }


        if(minimum_proposed<r_cylinder)
            inside_cylinder=true;
        else
            inside_cylinder=false;

        bool condition1=inside_cylinder;
        bool condition2=true;//(z>t_start && z<t_end);
        bool condition3= true;//(x-x_start)*(x-x_start)+(y-y_start)*(y-y_start)+(z-t_start)*(z-t_start)<r_helix*r_helix;
        bool condition4=true;//(x-x_end)*(x-x_end)+(y-y_end)*(y-y_end)+(z-t_end)*(z-t_end)<r_helix*r_helix;


        double A=this->X_ab/2.00;

        if(condition1 && (condition2  ||condition3 ||condition4  ) )
            wm_local[i]=A;
        else
            wm_local[i]=-A;





#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);


    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    delete t_vector;
    delete cost_vector;
    delete sint_vector;
    //    delete cost_vector_cylinder;
    //    delete sint_vector_cylinder;
    delete distance_helix;
#endif
}


//int Diffusion::generate_cylinder_helixed()
//{
//    int n_spheres=16;

//    double L=this->Lx;
//    double r_sphere=1*this->Lx/16;
//    double *xc=new double[n_spheres];
//    double *yc=new double[n_spheres];
//    double *zc=new double[n_spheres];
//    double x0;
//    x0=this->Lx/2.00;

//    double my_eps=this->Lx/256;

//    double r_helix=this->Lx/16.00+my_eps;



//    double r_mask=3*this->Lx/8.00+my_eps;

//    double t_start=0;
//    double t_end=this->Lx;

//    double x_c=this->Lx/2.00;
//    double y_c=this->Lx/2.00;


//   int tol=10;

//    double helix_speed=1;

//    int N_t_helix=(int)pow(2.00,(double)this->max_level+log(8)/log(2));

//    double *t_vector=new double[N_t_helix-2*tol];

//    double dt_helix=(t_end-t_start)/N_t_helix;
//    double *distance_helix=new double[N_t_helix-2*tol];
//    double *sint_vector=new double [N_t_helix-2*tol];
//    double *cost_vector=new double[N_t_helix-2*tol];

//    for(int i=0;i<N_t_helix-2*tol;i++)
//    {
//        t_vector[i]=dt_helix*(i+tol);
//        sint_vector[i]=r_mask*sin(helix_speed*t_vector[i])+x_c;
//        cost_vector[i]=r_mask*cos(helix_speed*t_vector[i])+y_c;
//        std::cout<<i<<" "<<t_vector[i]<<" "
//                <<sin(t_vector[i])<<" "
//               <<cos(t_vector[i])<<std::endl;
//    }


//    int dt_sphere=(int)((N_t_helix-2*tol-tol)/(n_spheres));

//    for(int i_sphere=0;i_sphere<n_spheres;i_sphere++)
//    {
//        xc[i_sphere]=cost_vector[(i_sphere+1)*dt_sphere];
//        yc[i_sphere]=sint_vector[(i_sphere+1)*dt_sphere];
//        zc[i_sphere]=t_vector[(i_sphere+1)*dt_sphere];
//    }

//    for(int i=0;i<n_spheres;i++)
//    {std::cout<<xc[i]<<" "<<yc[i]<<" "<<zc[i]<<std::endl;}

//    int myRank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
//    std::stringstream oss2Debug;
//    std::string mystr2Debug;
//    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
//    mystr2Debug=oss2Debug.str();
//    FILE *outFile;
//    outFile=fopen(mystr2Debug.c_str(),"w");
//    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
//    p4est_topidx_t global_node_number=0;
//    for(int ii=0;ii<myRank;ii++)
//        global_node_number+=this->nodes->global_owned_indeps[ii];

//    this->compute_mapping_fromLocal2Global();


//    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

//    //global_node_number=nodes->offset_owned_indeps;

//    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
//    {
//        /* since we want to access the local nodes, we need to 'jump' over intial
//       * nonlocal nodes. Number of initial nonlocal nodes is given by
//       * nodes->offset_owned_indeps
//       */
//        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
//        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
//        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

//        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
//        double tree_ymin = connectivity->vertices[3*v_mm + 1];

//#ifdef P4_TO_P8
//        double tree_zmin = connectivity->vertices[3*v_mm + 2];
//#endif

//        double x = int2double_coordinate_transform(node->x) + tree_xmin;
//        double y = int2double_coordinate_transform(node->y) + tree_ymin;

//#ifdef P4_TO_P8
//        double z = int2double_coordinate_transform(node->z) + tree_zmin;
//#endif


//        bool isInsideASphere=false;


//        for(int ii=0;ii<9;ii++)
//        {
//            if( pow(x-xc[ii],2)+pow(y-yc[ii],2)+pow(z-zc[ii],2)<pow(r_sphere,2))
//            {
//                isInsideASphere=true;
//                ii=9;
//            }
//        }
//        double A=this->X_ab/2.00;
//        if(isInsideASphere)
//        {
//            wp_local[i]=0;
//            wm_local[i]=+A;
//        }
//        else
//        {
//            wp_local[i]=0;
//            wm_local[i]=-A;
//        }

//#ifdef P4_TO_P8
//        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
//#else
//        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

//#endif

//    }
//    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
//    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
//    fclose(outFile);

//    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
//    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);


//    VecAssemblyBegin(this->wp);
//    VecAssemblyEnd(this->wp);
//    VecAssemblyBegin(this->wm);
//    VecAssemblyEnd(this->wm);

//    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
//    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

//    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
//    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

//    delete t_vector;
//    delete cost_vector;
//    delete sint_vector;

//}


int Diffusion::generate_bcc_phase()
{
#ifdef P4_TO_P8
    double L=this->Lx;
    double r=this->Lx*pow(3.00*this->f/(8.00*PI),1.00/3.00);//
    double *xc=new double[9];
    double *yc=new double[9];
    double *zc=new double[9];
    double x0;
    x0=this->Lx/2.00;

    xc[0]=x0;yc[0]=x0;zc[0]=x0;
    // (L,L,L) corner
    xc[1]=L;yc[1]=L;zc[1]=L;

    // (0,0,0) corner
    xc[2]=0;yc[2]=0;zc[2]=0;

    //(0,L,L) corner
    xc[3]=0;yc[3]=L;zc[3]=L;

    //(0,0,L) corner
    xc[4]=0;yc[4]=0;zc[4]=L;

    //(L,0,0)
    xc[5]=L;yc[5]=0;zc[5]=0;

    //(L,L,0)
    xc[6]=L;yc[6]=L;zc[6]=0;

    // (0,L,0)
    xc[7]=0;yc[7]=L;zc[7]=0;

    //L,0,L
    xc[8]=L;yc[8]=0;zc[8]=L;







    for(int i=0;i<9;i++)
    {std::cout<<xc[i]<<" "<<yc[i]<<" "<<zc[i]<<std::endl;}


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        double z = int2double_coordinate_transform(node->z) + tree_zmin;
#endif


        bool isInsideASphere=false;


        for(int ii=0;ii<9;ii++)
        {
            if( pow(x-xc[ii],2)+pow(y-yc[ii],2)+pow(z-zc[ii],2)<pow(r,2))
            {
                isInsideASphere=true;
                ii=9;
            }
        }
        double A=this->X_ab/2.00;
        if(isInsideASphere)
        {
            wp_local[i]=0;
            wm_local[i]=+A;
        }
        else
        {
            wp_local[i]=0;
            wm_local[i]=-A;
        }

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);

    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);

    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

#endif

}

int Diffusion::generate_smooth_fields()
{


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("smooth_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double Am=0,Ap=  this->X_ab/2;

    PetscScalar *wp_local=new PetscScalar[this->nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    double x0=0.5/5.00*this->Lx/2.00;

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        wp_local[i]=Ap*cos(3*PI*(y-x0)/(4.5/5.00*this->Lx));
        wm_local[i]=Am*sin(this->sin_frequency*PI*(y-x0)/(4.5/5.00*this->Lx));

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
}


int Diffusion::generate_smooth_fields_xy()
{


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("smooth_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double Am=this->X_ab/2.00,Ap= 0;

    PetscScalar *wp_local=new PetscScalar[this->nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    double x0=0.5/5.00*this->Lx/2.00;

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        wp_local[i]=Ap*cos(this->myMeanFieldPlan->seed_frequency*2*PI*x/this->Lx);
        wm_local[i]=Am*cos(this->myMeanFieldPlan->seed_frequency*2*PI*x/this->Lx)*sin(this->myMeanFieldPlan->seed_frequency*2*PI*y/this->Lx);

#ifdef P4_TO_P8
        wm_local[i]=wm_local[i]*cos(this->myMeanFieldPlan->seed_frequency*2*PI*z/this->Lx);
#endif

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
}



int Diffusion::generate_smooth_fields_r()
{


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("smooth_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double Am=0,Ap=  this->X_ab/2;

    PetscScalar *wp_local=new PetscScalar[this->nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    double x0=this->Lx/2.00;

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        double d_from_r0=(x-x0)*(x-x0)+(y-x0)*(y-x0);
        d_from_r0=pow(d_from_r0,0.5);

        wp_local[i]=Ap*cos(PI*(d_from_r0/this->Lx));
        wm_local[i]=Am*sin(PI*(d_from_r0/this->Lx));

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
}


int Diffusion::generate_smooth_fields_lamellar_y()
{
    double L=this->Lx;

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("smooth_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double A=  this->X_ab/2;

    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];



    double wp_average_inside_wall;
    double wp_average_inside_mask;
    double wm_average_inside_mask;
    double wm_average_inside_wall;

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        double wp_local_i=A*cos(3*PI*y/L);
        double wm_local_i=0;//A*cos(sin_frequency*PI*y/L);
        wp_local[i]=wp_local_i;
        wm_local[i]=wm_local_i;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
}

int Diffusion::generate_smooth_fields_lamellar_x()
{
    double L=this->Lx;

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("smooth_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double A=  this->X_ab/2;

    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];



    double wp_average_inside_wall;
    double wp_average_inside_mask;
    double wm_average_inside_mask;
    double wm_average_inside_wall;

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        double wp_local_i=0*A*sin(sin_frequency*PI*x/L);
        double wm_local_i=A*cos(12.00*PI*x/L);
        wp_local[i]=wp_local_i;
        wm_local[i]=wm_local_i;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
}

int Diffusion::generate_constant_field()
{

    double L=this->Lx;

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("constant_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double A= 1;// this->X_ab/2;

    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];





    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        double wp_local_i=A;
        double wm_local_i=0;
        wp_local[i]=wp_local_i;
        wm_local[i]=wm_local_i;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


}
int Diffusion::generate_gaussian_field()
{
    double L=this->Lx;
    double xc,yc;
#ifdef P4_TO_P8
    double zc;
#endif
    xc=this->Lx/2;
    yc=this->Lx/2;
#ifdef P4_TO_P8
    zc=this->Lx/2;
#endif

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("gaussian_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double A= 1;// this->X_ab/2;

    double a_r=1;

    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];




    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        x=x-xc; y=y-yc;
#ifdef P4_TO_P8
        z=z-zc;
#endif

        double r2=x*x+y*y;
#ifdef P4_TO_P8
        r2+=z*z;
#endif

        double expr2=exp(-a_r*r2);

        double wp_local_i=A*expr2;
        double wm_local_i=0;
        wp_local[i]=wp_local_i;
        wm_local[i]=wm_local_i;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

}


int Diffusion::generate_clock_field()
{
    double L=this->Lx;
    double xc,yc;
#ifdef P4_TO_P8
    double zc;
#endif
    xc=this->Lx/2;
    yc=this->Lx/2;
#ifdef P4_TO_P8
    zc=this->Lx/2;
#endif

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("clock_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double A= 1;// this->X_ab/2;

    double a_r=0.1;

    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];




    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        x=x-xc; y=y-yc;
#ifdef P4_TO_P8
        z=z-zc;
#endif

        double r2=x*x+y*y;
#ifdef P4_TO_P8
        r2+=z*z;
#endif

        double expr2=exp(-a_r*r2);


        double wp_local_i;

        if(ABS(x*x+y*y)>=0.0001)
        {
            wp_local_i=A*x/(pow(x*x+y*y,0.5));
            wp_local_i=wp_local_i*wp_local_i;
        }
        else
        {
            wp_local_i=0.00;
        }
        double wm_local_i=0;
        wp_local[i]=wp_local_i;
        wm_local[i]=wm_local_i;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

}

int Diffusion::generate_cos_field()
{
    double L=this->Lx;
    double xc,yc;
#ifdef P4_TO_P8
    double zc;
#endif
    xc=this->Lx/2;
    yc=this->Lx/2;
#ifdef P4_TO_P8
    zc=this->Lx/2;
#endif

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("clock_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    this->compute_mapping_fromLocal2Global();

    double A= 1;// this->X_ab/2;

    double a_r=0.1;

    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps];
    PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];




    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        x=x-xc; y=y-yc;
#ifdef P4_TO_P8
        z=z-zc;
#endif

        double r2=x*x+y*y;
#ifdef P4_TO_P8
        r2+=z*z;
#endif

        double r=pow(r2,0.5)/(this->Lx/4.00);


        double wp_local_i;


        wp_local_i=A*(1+cos(PI*r));

        double wm_local_i=0;
        wp_local[i]=wp_local_i;
        wm_local[i]=wm_local_i;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

}


int Diffusion::generate_single_ellipse()
{

    double ax=1.00,by=1.00,cz=1.00/(ax*by);
    double critical_distance=this->Lx/2;

    int n_ellipses=9;
    int n_ellipses_bcc=1;

    double polymer_mask_radius=pow(3.00,0.5)*this->Lx/6.00;



    double L=this->Lx;
    // double r=polymer_mask_radius;//  this->Lx/16;     //*pow(3.00*this->f/(8.00*PI),1.00/3.00);//
    double r= this->Lx/16;     //*pow(3.00*this->f/(8.00*PI),1.00/3.00);//


    ax=ax*r; by=by*r; cz=cz*r;
    ax=ax*ax; by=by*by; cz=cz*cz;


    double *xc=new double[n_ellipses];
    double *yc=new double[n_ellipses];
    double *zc=new double[n_ellipses];
    double x0;
    x0=this->Lx/2.00;

    xc[0]=x0;yc[0]=x0;zc[0]=x0;
    // (L,L,L) corner
    xc[1]=L;yc[1]=L;zc[1]=L;

    // (0,0,0) corner
    xc[2]=0;yc[2]=0;zc[2]=0;

    //(0,L,L) corner
    xc[3]=0;yc[3]=L;zc[3]=L;

    //(0,0,L) corner
    xc[4]=0;yc[4]=0;zc[4]=L;

    //(L,0,0)
    xc[5]=L;yc[5]=0;zc[5]=0;

    //(L,L,0)
    xc[6]=L;yc[6]=L;zc[6]=0;

    // (0,L,0)
    xc[7]=0;yc[7]=L;zc[7]=0;

    //L,0,L
    xc[8]=L;yc[8]=0;zc[8]=L;


    for(int i=0;i<n_ellipses;i++)
    {std::cout<<xc[i]<<" "<<yc[i]<<" "<<zc[i]<<std::endl;}


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("lamelar_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    std::cout<<" start mapping "<<std::endl;
    this->compute_mapping_fromLocal2Global();


    PetscScalar *wp_local=new PetscScalar[nodes->num_owned_indeps]; PetscScalar *wm_local=new PetscScalar[this->nodes->num_owned_indeps];

    double wp_local_i;
    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        double z = int2double_coordinate_transform(node->z) + tree_zmin;
#endif


        bool isInsideASphere=false;

        double dp=0;
        for(int ii=0;ii<n_ellipses_bcc;ii++)
        {
            dp=(x-xc[ii])  *(x-xc[ii])/ax+(y-yc[ii])*(y-yc[ii])/by;
#ifdef P4_TO_P8
            dp +=(z-zc[ii])*(z-zc[ii])/cz;
#endif

            //dp=dp-1;
            if( dp-1<0)
            {
                isInsideASphere=true;
                ii=n_ellipses;
            }
        }

        //  isInsideASphere=PETSC_TRUE;
        if(isInsideASphere)
        {
            wp_local_i= -this->X_ab;//*(1+cos(PI*dp));
            wp_local[i]=wp_local_i;  //exp(-dp/critical_distance);
            wm_local[i]=0;

        }
        else
        {
            wp_local[i]=0;
            wm_local[i]=0;

        }

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    VecSetValues(this->wp,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wp_local,INSERT_VALUES);
    VecSetValues(this->wm,this->nodes->num_owned_indeps,this->ix_fromLocal2Global,wm_local,INSERT_VALUES);
    VecAssemblyBegin(this->wp);
    VecAssemblyEnd(this->wp);
    VecAssemblyBegin(this->wm);
    VecAssemblyEnd(this->wm);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    delete xc; delete yc; delete zc;
    delete wp_local; delete wm_local;

    this->printDiffusionArrayFromVector(&this->wp,"single_ellipse",PETSC_TRUE);


}


int Diffusion::printStatisticalFields()
{


    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    {
        VecDuplicate(this->phi,&this->phiTemp);
        VecSet(this->phiTemp,-1.00);
    }
    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
            || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
            || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson

            )
    {
        VecDuplicate(this->phi_polymer_shape,&this->phiTemp);
        VecCopy(this->phi_polymer_shape,this->phiTemp);

    }

    this->scatter_petsc_vector(&this->wp);
    this->scatter_petsc_vector(&this->wm);
    this->scatter_petsc_vector(&this->phiTemp);
    PetscScalar *phiTempArray;
    VecGetArray(this->phiTemp,&phiTempArray);


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("statistical_fields")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();

    std::cout<<mystr2Debug<<std::endl;

    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");

    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];


    // this->compute_mapping_fromLocal2Global();

    PetscScalar *wp_local;//=new PetscScalar[nodes->num_owned_indeps];
    PetscScalar *wm_local;//=new PetscScalar[this->nodes->num_owned_indeps];

    this->ierr=VecGetArray(this->wp,&wp_local);  CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->wm,&wm_local);  CHKERRXX(this->ierr);


    int n_local_wp;
    this->ierr= VecGetLocalSize(this->wp,&n_local_wp); CHKERRXX(this->ierr);
    std::cout<<n_local_wp<<" "<<this->nodes->num_owned_indeps<<std::endl;
    if(n_local_wp!=this->nodes->num_owned_indeps)
    {
        std::cout<<" data structure bug "<<std::endl;
        std::cout<<n_local_wp<<" "<<this->nodes->num_owned_indeps<<std::endl;
        throw("algo error; location: print statistical fields function 1");
    }
    this->ierr=VecGetLocalSize(phiTemp,&n_local_wp); CHKERRXX(this->ierr);
    std::cout<<n_local_wp<<" "<<this->nodes->num_owned_indeps<<std::endl;
    if(n_local_wp!=this->nodes->num_owned_indeps)
    {
        std::cout<<" data structure bug "<<std::endl;
        std::cout<<n_local_wp<<" "<<this->nodes->num_owned_indeps<<std::endl;
        throw("algo error; location: print statistical fields function 2");
    }

    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        double z = int2double_coordinate_transform(node->z) + tree_zmin;
#endif

        double phi_temp_i=phiTempArray[i];//std::cout<<" phi_temp_i "<<phi_temp_i<<std::endl;
        double wp_local_i=wp_local[i];//std::cout<<" wp_local_i "<<wp_local_i<<std::endl;
        double wm_local_i=wm_local[i];//std::cout<<" wm_local_i "<<wm_local_i<<std::endl;


        // std::cout<<" i "<<i<<std::endl;
#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,phi_temp_i,  wp_local_i,wm_local_i);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,phi_temp_i,wp_local_i,wm_local_i);

#endif
        //std::cout<<"after i "<<i<<std::endl;
    }
    this->ierr=VecRestoreArray(this->wp,&wp_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->wm,&wm_local); CHKERRXX(this->ierr);

    this->ierr=VecRestoreArray(this->phiTemp,&phiTempArray); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
}


int Diffusion::evolve_probability_equation(int T1, int T2)
{
    int n_games_i=1; int n_games_j=1;  int n_games_k=1;
    PetscBool minus_one_iteration=PETSC_FALSE;
    this->evolve_probability_equation_watch.start("evolve probability equation");
    this->convergence_test_watch_total_time=0;
    this->fftw_scaterring_watch_total_time=0;
    for(int i_game=0;i_game<n_games_i;i_game++)
    {
        for(int j_game=0;j_game<n_games_j;j_game++)
        {
            if(!this->FFTCreated && this->myNumericalScheme==Diffusion::splitting_spectral_adaptive)
            {
#ifdef P4_TO_P8
                if(!this->real_to_complex_fftw_strategy)
                    this->setup_spectral_solver_amr();
                else
                    this->setup_spectral_solver_amr_r2c();
#else
                if(!this->real_to_complex_fftw_strategy)
                    this->setup_spectral_solver_amr_2D();
                else
                    this->setup_spectral_solver_amr_2D_r2c();
#endif


            }


            this->allocateMemory2History(minus_one_iteration);
            this->forward_stage=true;
            this->first_stage=true;
            std::cout<<" evolve prob 1 "<<this->mpi->mpirank<<std::endl;
            this->updateDiffusionMatrices();
            this->it=0;
            for(int i=0;i<T1;i++)
            {
                this->evolveDiffusionIteration();
                this->sendSol2HitoryDataBase();
                //this->printDiffusionIteration();
            }
            first_stage=false;
            std::cout<<" evolve prob 2 "<<this->mpi->mpirank<<std::endl;
            //VecDestroy(this->w_t);
            this->updateDiffusionMatrices();
            for(int i=T1;i<this->N_iterations;i++)
            {
                this->evolveDiffusionIteration();
                this->sendSol2HitoryDataBase();
                //  this->printDiffusionIteration();
            }
            std::cout<<" evolve prob 3 "<<this->mpi->mpirank<<std::endl;
            this->q_forward=new double[this->local_size_sol];
            for(int i=0;i<local_size_sol;i++)
                this->q_forward[i]=this->sol_history[i];
            std::cout<<" evolve prob 4 "<<this->mpi->mpirank<<std::endl;
            forward_stage=true;
            this->printDiffusionHistory();


            for(int k_game=0;k_game<n_games_k;k_game++)
            {
                this->computeQ();
            }
            delete this->sol_history;
            if(j_game<n_games_j-1)
            {
                delete this->q_forward;
                VecDestroy(this->w_t);
            }

        }

        std::cout<<" evolve prob 5 "<<this->mpi->mpirank<<std::endl;


        std::cout<<" evolve prob 6 "<<this->mpi->mpirank<<std::endl;

        this->allocateMemory2History(minus_one_iteration);

        forward_stage=false;
        first_stage=true;
        std::cout<<" evolve prob 7 "<<this->mpi->mpirank<<std::endl;


        std::cout<<" evolve prob 8 "<<this->mpi->mpirank<<std::endl;

        this->updateDiffusionMatrices();
        std::cout<<" evolve prob 9 "<<this->mpi->mpirank<<std::endl;

        this->it=0;
        for(int i=0;i<T2;i++)
        {
            this->evolveDiffusionIteration();
            this->sendSol2HitoryDataBase();
            //  this->printDiffusionIteration();
        }
        first_stage=false;
        std::cout<<" evolve prob 10 "<<this->mpi->mpirank<<std::endl;

        // VecDestroy(this->w_t);
        this->updateDiffusionMatrices();

        std::cout<<" evolve prob 11 "<<this->mpi->mpirank<<std::endl;


        for(int i=T2;i<this->N_iterations;i++)
        {
            this->evolveDiffusionIteration();
            this->sendSol2HitoryDataBase();
            //  this->printDiffusionIteration();
        }
        std::cout<<" evolve prob 12 "<<this->mpi->mpirank<<std::endl;

        this->q_backward=new double [this->local_size_sol];
        for(int i=0;i<this->local_size_sol;i++)
            this->q_backward[i]=this->sol_history[i];

        forward_stage=false;
        std::cout<<" evolve prob 13 "<<this->mpi->mpirank<<std::endl;

        this->computeQ();
        this->printDiffusionHistory();
        delete this->sol_history;
        std::cout<<" evolve prob 14 "<<this->mpi->mpirank<<std::endl;

        if(i_game<n_games_i-1)
        {
            VecDestroy( this->w_t);
            delete this->q_backward;
            delete this->q_forward;
        }
    }
    this->evolve_probability_equation_watch.stop();

}


int Diffusion::scatterForwardMyParallelVectors()
{
    // When integrals are performed it is better to first scatter the values to the ghost values
    // in the parallel vectors

    // RhoA
    this->ierr=VecGhostUpdateBegin(this->rhoA,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->rhoA,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    // RhoB
    this->ierr=VecGhostUpdateBegin(this->rhoB,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->rhoB,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    // sol_tp1
    this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    // sol_tp
    this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    // fp
    this->ierr=VecGhostUpdateBegin(this->fp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->fp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


    // fm
    this->ierr=VecGhostUpdateBegin(this->fm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->fm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);




}

int Diffusion::computeQ()
{
    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    {

        if(this->forward_stage)
            this->computeVolume();

        if(this->myNumericalScheme!=Diffusion::splitting_spectral_adaptive)
        {

            int n_q_game=1;
            for(int q_game=0;q_game<n_q_game;q_game++)
            {
                VecDuplicate(this->phi,&this->phiTemp);

                VecSet(this->phiTemp,-1.00);

                ierr = VecGhostUpdateBegin(this->phiTemp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                ierr = VecGhostUpdateEnd  (this->phiTemp, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

                ierr = VecGhostUpdateBegin(this->sol_tp1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                ierr = VecGhostUpdateEnd  (this->sol_tp1, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

                //        this->ierr=VecAssemblyBegin(this->phiTemp);
                //        this->ierr=VecAssemblyEnd(this->phiTemp);

                if(this->forward_stage)
                {

                    std::cout<<this->mpi->mpirank<<" Qforward Computation"<<std::endl;
                    this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_forward");
                    this->Q_forward=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->sol_tp1);
                    this->Q_forward=this->Q_forward/this->V;
                    std::cout<<" rank Q forward "<<this->mpi->mpirank<<" "<<this->Q_forward<<std::endl;
                }
                else
                {
                    std::cout<<this->mpi->mpirank<<" Qbackward Computation"<<std::endl;
                    this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_backward");
                    this->Q_backward=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->sol_tp1);
                    this->Q_backward=this->Q_backward/this->V;
                    std::cout<<" rank Q backward "<<this->mpi->mpirank<<" "<<this->Q_backward<<std::endl;
                }

                //    double rescale_volume=this->Lx_physics/this->Lx;
                //    rescale_volume=rescale_volume*rescale_volume;

                //#ifdef P4_TO_P8
                //    rescale_volume=rescale_volume*rescale_volume;
                //#endif

                //     this->Q_forward=this->Q_forward/rescale_volume;
                //     this->Q_backward=this->Q_backward/rescale_volume;


                if(q_game<n_q_game-1)
                    VecDestroy(this->phiTemp);

            }
            VecDestroy(this->phiTemp);
        }

        if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive)
        {
            double Q_local=0;
            double Q_global=0;
#ifdef P4_TO_P8

            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));

            const ptrdiff_t N2_effective=2*(N2/2+1);

            if(!this->real_to_complex_fftw_strategy)
                for(int i=0;i<this->local_n0;i++)
                    for(int j=0;j<N1;j++)
                        for(int k=0;k<N2;k++)
                            Q_local+=this->input_forward[i*N1*N2+j*N2+k][0];
            else
                for(int i=0;i<this->local_n0;i++)
                    for(int j=0;j<N1;j++)
                        for(int k=0;k<N2;k++)
                            Q_local+=this->input_forward_real[i*N1*N2_effective+j*N2_effective+k];

            this->ierr=MPI_Allreduce(&Q_local,&Q_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); CHKERRXX(this->ierr);

            if(this->forward_stage)
            {
                this->Q_forward=Q_global/((double)(N0*N1*N2));
                std::cout<<" rank Q forward "<<this->mpi->mpirank<<" "<<this->Q_forward<<std::endl;

            }
            else
            {
                this->Q_backward=Q_global/((double) N0*N1*N2);
                std::cout<<" rank Q backward "<<this->mpi->mpirank<<" "<<this->Q_backward<<std::endl;

            }
#else

            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));

            const ptrdiff_t N1_effective=2*(N1/2+1);


            if(!this->real_to_complex_fftw_strategy)
                for(int i=0;i<this->local_n0;i++)
                    for(int j=0;j<N1;j++)
                        Q_local+=this->input_forward[i*N1+j][0];
            else
                for(int i=0;i<this->local_n0;i++)
                    for(int j=0;j<N1;j++)
                        Q_local+=this->input_forward_real[i*N1_effective+j];

            this->ierr=MPI_Allreduce(&Q_local,&Q_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); CHKERRXX(this->ierr);

            if(this->forward_stage)
            {
                this->Q_forward=Q_global/((double)(N0*N1));
                std::cout<<" rank Q forward "<<this->mpi->mpirank<<" "<<this->Q_forward<<std::endl;

            }
            else
            {
                this->Q_backward=Q_global/((double) N0*N1);
                std::cout<<" rank Q backward "<<this->mpi->mpirank<<" "<<this->Q_backward<<std::endl;

            }
#endif

        }
    }

    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
            || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
    {
        if(this->forward_stage)
            this->computeVolume();



        if(this->forward_stage)
            this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_forward_before_extension");
        else
            this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_backward_before_extension");


        my_p4est_level_set ls(this->nodes_neighbours);

        Vec bc_vec_fake;
        this->ierr= VecDuplicate(this->phi,&bc_vec_fake); CHKERRXX(this->ierr);
        this->ierr=VecSet(bc_vec_fake,0.00);              CHKERRXX(this->ierr);

        this->ierr = VecGhostUpdateBegin(bc_vec_fake, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateEnd  (bc_vec_fake, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(this->ierr);


        int extension_order=0;
        int number_of_bands_to_extend=5;

        ls.extend_Over_Interface(this->phi_polymer_shape,this->sol_tp1,NEUMANN,bc_vec_fake,extension_order,number_of_bands_to_extend);


        this->ierr = VecGhostUpdateBegin(this->sol_tp1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateEnd  (this->sol_tp1, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(this->ierr);

        this->ierr = VecGhostUpdateBegin(this->phi_polymer_shape, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_polymer_shape, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(this->ierr);

        VecDestroy(bc_vec_fake);

        if(this->forward_stage)
        {
            //this->computeVolume();
            std::cout<<this->mpi->mpirank<<" Qforward Computation"<<std::endl;
            this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_forward_after_extension");
            this->Q_forward=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->sol_tp1);
            //            this->printDiffusionVector(&this->sol_tp1,"q_last_forward");
            //            PetscScalar *sol_tp1_array;
            //            VecGetArray(this->sol_tp1,&sol_tp1_array);
            //            this->printDiffusionArray(sol_tp1_array,this->n_local_size_sol,"q_last_forward_array");
            //            VecRestoreArray(this->sol_tp1,&sol_tp1_array);
            this->Q_forward=this->Q_forward/this->V;
            std::cout<<" rank Q forward "<<this->mpi->mpirank<<" "<<this->Q_forward<<std::endl;
        }
        else
        {
            std::cout<<this->mpi->mpirank<<" Qbackward Computation"<<std::endl;
            this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_backward_before_extension");
            this->Q_backward=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->sol_tp1);
            //            this->printDiffusionVector(&this->sol_tp1,"q_last_backward");
            //            PetscScalar *sol_tp1_array;
            //            VecGetArray(this->sol_tp1,&sol_tp1_array);
            //            this->printDiffusionArray(sol_tp1_array,this->n_local_size_sol,"q_last_forward_array");
            //            VecRestoreArray(this->sol_tp1,&sol_tp1_array);
            this->Q_backward=this->Q_backward/this->V;
            std::cout<<" rank Q backward "<<this->mpi->mpirank<<" "<<this->Q_backward<<std::endl;
        }

    }

    if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
    {
        if(this->forward_stage)
            this->computeVolume();




        if(this->forward_stage)
        {

            std::cout<<this->mpi->mpirank<<" Qforward Computation"<<std::endl;
            this->Q_forward=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->sol_tp1);
            this->Q_forward=this->Q_forward/this->V;
            std::cout<<" rank Q forward "<<this->mpi->mpirank<<" "<<this->Q_forward<<std::endl;
        }
        else
        {
            std::cout<<this->mpi->mpirank<<" Qbackward Computation"<<std::endl;
            this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_backward_before_extension");
            this->Q_backward=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->sol_tp1);
            this->Q_backward=this->Q_backward/this->V;
            std::cout<<" rank Q backward "<<this->mpi->mpirank<<" "<<this->Q_backward<<std::endl;
        }

    }



}



int Diffusion::computeAverageProperties()
{

    this->computeVolume();
    //if(i_mean_field==0)
    {
        int n_game=1;


        for(int i_game=0;i_game<n_game;i_game++)
        {

            VecCreateGhostNodes(this->p4est,this->nodes,&phiTemp);
            if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
            {

                this->ierr=VecSet(phiTemp,-1.00); CHKERRXX(this->ierr);
            }
            if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
            || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
            || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            {

                this->ierr=VecCopy(this->phi_polymer_shape,phiTemp);  CHKERRXX(this->ierr);
            }

            this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



            VecCreateGhostNodes(this->p4est,this->nodes,&this->my_temp_vec);
            double *my_temp_vec_local=new  PetscScalar[this->n_local_size_sol];
            this->ierr=VecGetValues(this->fp,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local); CHKERRXX(this->ierr);
            for(int i=0;i<this->n_local_size_sol;i++)
            {
                my_temp_vec_local[i]=my_temp_vec_local[i]*my_temp_vec_local[i];

            }
            this->ierr=VecSetValues(this->my_temp_vec,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local,INSERT_VALUES); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateBegin(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
            this->ierr=VecAssemblyBegin(this->my_temp_vec); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyEnd(this->my_temp_vec); CHKERRXX(this->ierr);
            std::cout<<this->mpi->mpirank<<" Pressure Computation"<<std::endl;
            this->fp2=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->my_temp_vec);
            this->fp2=this->fp2/this->V;
            std::cout<<" rank pressure force norm 2 "<<this->mpi->mpirank<<" "<<this->fp2<<std::endl;
            //    VecDestroy(this->my_temp_vec);
            //    delete my_temp_vec_local;

            //    VecCreateGhostNodes(this->p4est,this->nodes,&this->my_temp_vec);
            //    my_temp_vec_local=new  PetscScalar[this->n_local_size_sol];
            this->ierr=VecGetValues(this->fm,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local); CHKERRXX(this->ierr);
            for(int i=0;i<this->n_local_size_sol;i++)
            {
                my_temp_vec_local[i]=my_temp_vec_local[i]*my_temp_vec_local[i];

            }
            this->ierr=VecSetValues(this->my_temp_vec,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local,INSERT_VALUES); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateBegin(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
            this->ierr=VecAssemblyBegin(this->my_temp_vec); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyEnd(this->my_temp_vec); CHKERRXX(this->ierr);
            std::cout<<this->mpi->mpirank<<" Exchange Computation"<<std::endl;
            this->fm2=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->my_temp_vec);
            this->fm2=this->fm2/this->V;
            std::cout<<" rank exchange force norm 2 "<<this->mpi->mpirank<<" "<<this->fm2<<std::endl;
            //    VecDestroy(this->my_temp_vec);
            //    delete my_temp_vec_local;


            //    VecCreateGhostNodes(this->p4est,this->nodes,&this->my_temp_vec);
            //    my_temp_vec_local=new  PetscScalar[this->n_local_size_sol];



            if(this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson ||
               this->my_casl_diffusion_method==Diffusion::neuman_backward_euler)
            {
                std::cout<<" compute average properties for the non periodic case "<<std::endl;

                Vec rhoA_extended;
                // this->printDiffusionArrayFromVector(&this->rhoA,"rhoa_before_ext");

                this->ierr=VecDuplicate(this->rhoA,&rhoA_extended); CHKERRXX(this->ierr);
                this->ierr=VecCopy(this->rhoA,rhoA_extended); CHKERRXX(this->ierr);

                this->extend_petsc_vector(&rhoA_extended);
                //this->printDiffusionArrayFromVector(&rhoA_extended,"rhoa_ext");


                Vec rhoB_extended;
                this->ierr=VecDuplicate(this->rhoB,&rhoB_extended); CHKERRXX(this->ierr);
                this->ierr=VecCopy(this->rhoB,rhoB_extended); CHKERRXX(this->ierr);
                this->extend_petsc_vector(&rhoB_extended);
                //this->printDiffusionArrayFromVector(&rhoB_extended,"rhob_ext");


                this->ierr=VecGetValues(rhoA_extended,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local); CHKERRXX(this->ierr);
                for(int i=0;i<this->n_local_size_sol;i++)
                {
                    my_temp_vec_local[i]=(my_temp_vec_local[i]-this->f)*(my_temp_vec_local[i]-this->f)/(this->f*(1-this->f));;

                }

                //this->printDiffusionArray(my_temp_vec_local,this->n_local_size_sol,"order_local");

                this->ierr=VecSetValues(this->my_temp_vec,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local,INSERT_VALUES); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyBegin(this->my_temp_vec); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyEnd(this->my_temp_vec);  CHKERRXX(this->ierr);
                this->ierr=VecGhostUpdateBegin(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
                this->ierr=VecGhostUpdateEnd(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
                std::cout<<this->mpi->mpirank<<" Order Computation"<<std::endl;


                //this->printDiffusionArrayFromVector(&this->my_temp_vec,"order_vec_before_integration");
                this->orderRatio=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->my_temp_vec);
                this->orderRatio=this->orderRatio/this->V;
                std::cout<<" order ratio "<<this->mpi->mpirank<<" "<<this->orderRatio<<std::endl;
                if(this->orderRatio>this->diverge2Crap)
                {
                    std::cout<<"order ratio "<<this->orderRatio<<std::endl;
                    throw std::runtime_error("scft divergence");
                }
                if(this->orderRatio<this->collapse2Disordered)
                {
                    std::cout<<"order ratio "<<this->orderRatio<<std::endl;
                    throw std::runtime_error("scft convergence to the homogeneous phase");
                }
                if(isnan(this->orderRatio) || isinf(this->orderRatio))
                {
                    std::cout<<"order ratio "<<this->orderRatio<<std::endl;
                    throw std::runtime_error("scft divergence or bug");
                }
                this->printDiffusionArrayFromVector(&this->my_temp_vec," order");
                //VecDestroy(this->my_temp_vec);
                delete my_temp_vec_local;


                //VecCreateGhostNodes(this->p4est,this->nodes,&this->my_temp_vec);
                this->ierr=VecSet(this->my_temp_vec,0.00); CHKERRXX(this->ierr);
                this->ierr=VecAXPBY(this->my_temp_vec,1,1,rhoA_extended); CHKERRXX(this->ierr);
                this->ierr=VecAXPBY(this->my_temp_vec,1,1,rhoB_extended); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyBegin(this->my_temp_vec); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyEnd(this->my_temp_vec); CHKERRXX(this->ierr);


                this->ierr=VecGhostUpdateBegin(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
                this->ierr=VecGhostUpdateEnd(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

                double surface_of_domain=0;
                if(this->myMeanFieldPlan->optimize_robin_kappa_b)
                 surface_of_domain=this->integrate_constant_over_interface(this->phiTemp);

                std::cout<<this->mpi->mpirank<<" rho 0 Computation"<<std::endl;
                this->rho0Average=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->my_temp_vec);
                this->rho0Average=this->rho0Average/this->V;
                if(this->myMeanFieldPlan->optimize_robin_kappa_b)
                this->rho0Average_surface=this->integrate_over_interface(this->phiTemp,this->my_temp_vec)/surface_of_domain;
                std::cout<<" rho 0 average  "<<this->mpi->mpirank<<" "<<this->rho0Average<<std::endl;
                this->ierr=VecDestroy(this->my_temp_vec); CHKERRXX(this->ierr);



                std::cout<<this->mpi->mpirank<<" Rhoa Computation"<<std::endl;
                //                if(!this->minimum_IO)
                //                    this->printDiffusionVector(&this->rhoA,"rhoav");
                this->rhoAAverage=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,rhoA_extended);
                this->rhoAAverage=this->rhoAAverage/this->V;
                if(this->myMeanFieldPlan->optimize_robin_kappa_b)
                this->rhoAAverage_surface=integrate_over_interface(this->phiTemp,rhoA_extended)/surface_of_domain;
                std::cout<<" rank density average rho a "<<this->mpi->mpirank<<" "<<this->rhoAAverage<<std::endl;
                 std::cout<<" rank surface density average rho a "<<this->mpi->mpirank<<" "<<this->rhoAAverage_surface<<std::endl;
                //this->printDiffusionArrayFromVector(&this->rhoA,"rho_a");

                std::cout<<this->mpi->mpirank<<" Rhob Computation"<<std::endl;
                this->rhoBAverage=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,rhoB_extended);
                this->rhoBAverage=this->rhoBAverage/this->V;
                if(this->myMeanFieldPlan->optimize_robin_kappa_b)
                this->rhoBAverage_surface=this->integrate_over_interface(this->phiTemp,rhoB_extended)/surface_of_domain;

                std::cout<<" rank density average rho b "<<this->mpi->mpirank<<" "<<this->rhoBAverage<<std::endl;
                std::cout<<" rank density surface average rho b "<<this->mpi->mpirank<<" "<<this->rhoBAverage_surface<<std::endl;
                this->ierr=VecDestroy(rhoA_extended); CHKERRXX(this->ierr);
                this->ierr=VecDestroy(rhoB_extended); CHKERRXX(this->ierr);

                Vec wp_extended,wm_extended;
                this->ierr=VecDuplicate(this->phi,&wp_extended); CHKERRXX(this->ierr);
                this->ierr=VecDuplicate(this->phi,&wm_extended); CHKERRXX(this->ierr);

                this->ierr=VecSet(wp_extended,1.00); CHKERRXX(this->ierr);
                this->scatter_petsc_vector(&wp_extended);
                double surface_value=this->integrate_over_interface(this->phiTemp,wp_extended);

                this->ierr=VecSet(wp_extended,0.00); CHKERRXX(this->ierr);
                this->scatter_petsc_vector(&wp_extended);

                this->ierr=VecCopy(this->wp,wp_extended); CHKERRXX(this->ierr);
                this->ierr=VecCopy(this->wm,wm_extended); CHKERRXX(this->ierr);
                this->extend_petsc_vector(&wp_extended);
                this->extend_petsc_vector(&wm_extended);



                this->wp_average_on_contour=integrate_over_interface(this->phiTemp,wp_extended)/surface_value;
                this->wm_average_on_contour=integrate_over_interface(this->phiTemp,wm_extended)/surface_value;

                this->ierr=VecShift(wp_extended,this->wp_average_on_contour); CHKERRXX(this->ierr);
                this->ierr=VecShift(wm_extended,this->wm_average_on_contour); CHKERRXX(this->ierr);

                this->ierr=VecPointwiseMult(wp_extended,wp_extended,wp_extended); CHKERRXX(this->ierr);
                this->ierr=VecPointwiseMult(wm_extended,wm_extended,wm_extended); CHKERRXX(this->ierr);

                this->scatter_petsc_vector(&wp_extended);
                this->scatter_petsc_vector(&wm_extended);

                this->wp_standard_deviation_on_contour=integrate_over_interface(this->phiTemp,wp_extended)/surface_value;
                this->wm_standard_deviation_on_contour=integrate_over_interface(this->phiTemp,wm_extended)/surface_value;

                this->wp_standard_deviation_on_contour=pow(this->wp_standard_deviation_on_contour,0.5);
                this->wm_standard_deviation_on_contour=pow(this->wm_standard_deviation_on_contour,0.5);

                this->ierr=VecDestroy(wp_extended); CHKERRXX(this->ierr);
                this->ierr=VecDestroy(wm_extended); CHKERRXX(this->ierr);
            }
            if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson
             ||this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            {

                std::cout<<" compute average properties for the periodic case "<<std::endl;
                this->ierr=VecGetValues(this->rhoA,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local); CHKERRXX(this->ierr);
                for(int i=0;i<this->n_local_size_sol;i++)
                {
                    my_temp_vec_local[i]=(my_temp_vec_local[i]-this->f)*(my_temp_vec_local[i]-this->f)/(this->f*(1-this->f));;

                }
                this->ierr=VecSetValues(this->my_temp_vec,this->n_local_size_sol,this->ix_fromLocal2Global,my_temp_vec_local,INSERT_VALUES); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyBegin(this->my_temp_vec); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyEnd(this->my_temp_vec);  CHKERRXX(this->ierr);


                this->ierr=VecGhostUpdateBegin(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
                this->ierr=VecGhostUpdateEnd(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

                std::cout<<this->mpi->mpirank<<" Order Computation"<<std::endl;
                this->orderRatio=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->my_temp_vec);
                this->orderRatio=this->orderRatio/this->V;
                std::cout<<" order ratio "<<this->mpi->mpirank<<" "<<this->orderRatio<<std::endl;

                // This is done mainly to not waste cpu time on the cluster//

                if(this->orderRatio>this->diverge2Crap)
                {
                    std::cout<<"order ratio "<<this->orderRatio<<std::endl;
                    throw std::runtime_error("scft divergence");
                }
                if(this->orderRatio<this->collapse2Disordered)
                {
                    std::cout<<"order ratio "<<this->orderRatio<<std::endl;
                    throw std::runtime_error("scft convergence to the homogeneous phase");
                }
                if(isnan(this->orderRatio) || isinf(this->orderRatio))
                {
                    std::cout<<"order ratio "<<this->orderRatio<<std::endl;
                    throw std::runtime_error("scft divergence or bug");
                }

                //this->printDiffusionArrayFromVector(&this->my_temp_vec," order");
                //VecDestroy(this->my_temp_vec);
                delete my_temp_vec_local;


                //VecCreateGhostNodes(this->p4est,this->nodes,&this->my_temp_vec);
                this->ierr=VecSet(this->my_temp_vec,0.00); CHKERRXX(this->ierr);
                this->ierr=VecAXPBY(this->my_temp_vec,1,1,this->rhoA); CHKERRXX(this->ierr);
                this->ierr=VecAXPBY(this->my_temp_vec,1,1,this->rhoB); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyBegin(this->my_temp_vec); CHKERRXX(this->ierr);
                this->ierr=VecAssemblyEnd(this->my_temp_vec); CHKERRXX(this->ierr);


                this->ierr=VecGhostUpdateBegin(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
                this->ierr=VecGhostUpdateEnd(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

                std::cout<<this->mpi->mpirank<<" rho 0 Computation"<<std::endl;
                this->rho0Average=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->my_temp_vec);
                this->rho0Average=this->rho0Average/this->V;
                std::cout<<" rho 0 average  "<<this->mpi->mpirank<<" "<<this->rho0Average<<std::endl;
                this->ierr=VecDestroy(this->my_temp_vec); CHKERRXX(this->ierr);

                std::cout<<this->mpi->mpirank<<" Rhoa Computation"<<std::endl;
                //                if(!this->minimum_IO)
                //                    this->printDiffusionVector(&this->rhoA,"rhoav");
                this->rhoAAverage=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->rhoA);
                this->rhoAAverage=this->rhoAAverage/this->V;
                std::cout<<" rank density average rho a "<<this->mpi->mpirank<<" "<<this->rhoAAverage<<std::endl;
                //this->printDiffusionArrayFromVector(&this->rhoA,"rho_a");

                std::cout<<this->mpi->mpirank<<" Rhob Computation"<<std::endl;
                this->rhoBAverage=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->rhoB);
                this->rhoBAverage=this->rhoBAverage/this->V;
                std::cout<<" rank density average rho b "<<this->mpi->mpirank<<" "<<this->rhoBAverage<<std::endl;



                if(this->advance_fields_scft_advance_mask_level_set
                || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
                {

                    this->rhoAAverage=this->rhoAAverage/this->phi_bar;
                    std::cout<<" rank density average rho a "<<this->mpi->mpirank<<" "<<this->rhoAAverage<<std::endl;


                    this->rhoBAverage=this->rhoBAverage/this->phi_bar;
                    std::cout<<" rank density average rho b "<<this->mpi->mpirank<<" "<<this->rhoBAverage<<std::endl;
                }

            }
            this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);



        }
    }

}

int Diffusion::computeVolume()
{

    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    {


        VecDuplicate(this->phi,&this->phiTemp);
        VecSet(this->phiTemp,-1.00);

        VecDuplicate(this->phi,&this->my_temp_vec);
        VecSet(this->my_temp_vec,1.00);
        std::cout<<this->mpi->mpirank<<" volume computation "<<std::endl;

        // phiTemp
        this->ierr=VecGhostUpdateBegin(this->phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        // fTemp
        this->ierr=VecGhostUpdateBegin(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->my_temp_vec,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
        this->V=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->my_temp_vec);

        VecDestroy(this->phiTemp);
        VecDestroy(this->my_temp_vec);
    }

    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
            || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
            || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
    {
        Vec fTemp;
        VecDuplicate(this->phi,&fTemp);
        VecSet(fTemp,1.00);

        //        Vec rhoa_temp;
        //        VecDuplicate(this->phi,&rhoa_temp);
        //        VecSet(rhoa_temp,0.5005);

        std::cout<<this->mpi->mpirank<<" volume computation "<<std::endl;
        // phiTemp
        this->ierr=VecGhostUpdateBegin(this->phi_polymer_shape,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->phi_polymer_shape,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
        // fTemp
        this->ierr=VecGhostUpdateBegin(fTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(fTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
        this->V=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,fTemp);



        // The next step is to estimate the error don gy the filtration extension

        this->filter_petsc_vector(&fTemp);
        this->printDiffusionArrayFromVector(&fTemp,"fTemp");
        double V_filtered=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,fTemp);

        this->extend_petsc_vector(&fTemp);
        this->printDiffusionArrayFromVector(&fTemp,"fTemp");
        double V_extended=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,fTemp);



        std::cout<<" V "<<this->V<<" V filtered "<<V_filtered<<" V extended "<<V_extended<<std::endl;
        std::cout<<" V_filt/V "<<V_filtered/this->V<<std::endl;
        std::cout<<" V ext/V "<<V_extended/this->V<<std::endl;


        // end of error estimation

        //double rhoa_temp_v=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,rhoa_temp);

        if(!this->minimum_IO)
            this->printDiffusionVector(&this->phi_polymer_shape,"mask4volume");

        VecDestroy(fTemp);

    }


}

int Diffusion::compute_energy()
{
    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {
        if( (this->my_casl_diffusion_method!=Diffusion::neuman_crank_nicholson)
                && (this->my_casl_diffusion_method!=Diffusion::neuman_backward_euler))
        {
            this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
            //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

            PetscScalar *energy_integrand_local;

            this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
            for(int i=0;i<this->n_local_size_sol;i++)
            {
                energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
            }

            //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);
            VecRestoreArray(this->energy_integrand,&energy_integrand_local);

            this->ierr=VecAssemblyBegin(this->energy_integrand);  CHKERRXX(this->ierr);
            this->ierr=VecAssemblyEnd(this->energy_integrand); CHKERRXX(this->ierr);

            //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_first_step");

            this->ierr=VecAXPBY(this->energy_integrand,-1.00,1.00/this->X_ab,this->wp); CHKERRXX(this->ierr);

            //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_second_step");


            VecDuplicate(this->phi,&this->phiTemp);
            VecSet(this->phiTemp,-1.00);
            std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


            // energy_integrand
            this->ierr=VecGhostUpdateBegin(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

            // phi_temp
            this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


            this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
            std::cout<<"E1 "<<this->E<<std::endl;
            this->E=this->E/this->V;
            this->E_w=this->E;
            std::cout<<"E2 "<<this->E<<std::endl;
            std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
            this->E_logQ=-log(0.5*this->Q_forward+0.5*this->Q_backward);
            this->E=this->E-log(0.5*this->Q_forward+0.5*this->Q_backward);
            std::cout<<" E "<<this->E<<std::endl;


            VecDestroy(this->phiTemp);

            //delete energy_integrand_local;
            VecDestroy(this->energy_integrand);

            double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;

            std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;
        }
        else
        {
            this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
            //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

            PetscScalar *energy_integrand_local;

            this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
            for(int i=0;i<this->n_local_size_sol;i++)
            {
                energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
            }

            //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);
            VecRestoreArray(this->energy_integrand,&energy_integrand_local);

            this->ierr=VecAssemblyBegin(this->energy_integrand);  CHKERRXX(this->ierr);
            this->ierr=VecAssemblyEnd(this->energy_integrand); CHKERRXX(this->ierr);

            //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_first_step");

            this->ierr=VecAXPBY(this->energy_integrand,-1.00,1.00/this->X_ab,this->wp); CHKERRXX(this->ierr);

            //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_second_step");



            this->extend_petsc_vector(&this->energy_integrand);

            // energy_integrand
            this->ierr=VecGhostUpdateBegin(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



            this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->energy_integrand);
            std::cout<<"E1 "<<this->E<<std::endl;
            this->E=this->E/this->V;
            this->E_w=this->E;
            std::cout<<"E2 "<<this->E<<std::endl;
            std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
            this->E_logQ=-log(0.5*this->Q_forward+0.5*this->Q_backward);
            this->E=this->E-log(0.5*this->Q_forward+0.5*this->Q_backward);
            std::cout<<" E "<<this->E<<std::endl;



            //delete energy_integrand_local;
            VecDestroy(this->energy_integrand);

            double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;

            std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;

        }
    }

}

int Diffusion::compute_phi_bar()
{
    if(this->my_casl_diffusion_method!=Diffusion::dirichlet_crank_nicholson)
    {

        this->ierr=VecDuplicate(this->phi,&this->phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->phiTemp,-1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->phiTemp);
        this->phi_bar=0;
        this->phi_bar=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->phi_wall);
        std::cout<<" phi_bar "<<this->phi_bar<<std::endl;
        this->phi_bar=1.00-this->phi_bar/this->V;
        std::cout<<" phi bar "<<this->phi_bar<<std::endl;
        this->ierr=VecDestroy(this->phiTemp);
    }
    if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
    {


        this->phi_bar=0;
        this->phi_bar=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->phi_wall);
        std::cout<<" phi_bar "<<this->phi_bar<<std::endl;
        this->phi_bar=1.00-this->phi_bar/this->V;
        std::cout<<" phi bar "<<this->phi_bar<<std::endl;

    }
}

int Diffusion::compute_energy_wall()
{
    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {


        this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
        this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

        this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;

        this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->n_local_size_sol;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(this->energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);



        this->ierr=VecDuplicate(this->phi,&this->phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->phiTemp,-1.00); CHKERRXX(this->ierr);
        std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


        // energy_integrand
        this->ierr=VecGhostUpdateBegin(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        // phi_temp
        this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


        this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        std::cout<<"E1 "<<this->E<<std::endl;
        this->E=this->E/this->V;
        this->E_w=this->E;
        std::cout<<"E2 "<<this->E<<std::endl;
        std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
        this->E_logQ=-this->phi_bar*log(0.5*this->Q_forward+0.5*this->Q_backward);
        this->E=this->E+this->E_logQ;
        std::cout<<" E "<<this->E<<std::endl;


        std::cout<<" start to compute compressibility and wall terms "<<std::endl;
        Vec  wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_1,wall_force_scalar_2;
        PetscScalar minus_one=-1.00;
        PetscScalar minus_half=-0.5;
        double E_p_compressible, E_p_wall,E_m_wall;



        // compute scalars
        compressibility_scalar=minus_half*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_1=this->xhi_w_p*this->zeta_n_inverse+1.00;
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure

        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,wall_force_scalar_1); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);






        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,this->wp); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,compressibility_scalar); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->energy_integrand); CHKERRXX(this->ierr);

        E_p_compressible=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_compressible=E_p_compressible/this->V;

        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,wall_force); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_wall=E_p_wall/this->V;


        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wm,this->phi_wall); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,2*this->xhi_w_m/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_m_wall=E_m_wall/this->V;


        std::cout<<" E_p_compressible "<<E_p_compressible<<" E_p_wall "<<E_p_wall<<
                   " E_m_wall "<<E_m_wall<<std::endl;

        this->E=this->E+E_p_compressible+E_p_wall+E_m_wall;
        this->E_compressible=E_p_compressible;
        this->E_wall_m=E_m_wall;
        this->E_wall_p=E_p_wall;



        this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);


        VecDestroy(this->energy_integrand);
        VecDestroy(wall_force);

        double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;

        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;
    }
}

int Diffusion::compute_energy_wall_neuman()
{
    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {
        this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
        this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

        this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;
        this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->n_local_size_sol;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        this->ierr=VecRestoreArray(this->energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->energy_integrand);
        std::cout<<"E1 "<<this->E<<std::endl;
        this->E=this->E/this->V;
        this->E_w=this->E;
        std::cout<<"E2 "<<this->E<<std::endl;
        std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
        this->E_logQ=(-1.00)*log(0.5*this->Q_forward+0.5*this->Q_backward);
        this->E=this->E+this->E_logQ;
        std::cout<<" E "<<this->E<<std::endl;


        //        std::cout<<" start to compute  wall terms "<<std::endl;
        //        Vec  wall_force;


        //        // Allocate memory using the apropriate petsc vector data structure

        //        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        //        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        //        this->ierr=VecShift(wall_force,-1.00); CHKERRXX(this->ierr);
        //        this->ierr=VecScale(wall_force,-1.00); CHKERRXX(this->ierr);
        //        this->ierr=VecPointwiseMult(wall_force,   wall_force,this->wp); CHKERRXX(this->ierr);


        //        this->scatter_petsc_vector(&wall_force); CHKERRXX(this->ierr);

        //        double E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,wall_force)/this->V;

        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wm,this->phi_wall); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,2.00*this->xhi_w_m/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);
        double E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->energy_integrand);
        E_m_wall=E_m_wall/this->V;

        std::cout<<" E_p_wall "<<0.00<<
                   " E_m_wall "<<E_m_wall<<std::endl;


        this->E_compressible=0.00;
        this->E_wall_m=E_m_wall;
        this->E_wall_p=0.00;
        this->E=this->E+this->E_wall_p+this->E_wall_m;

        this->ierr=VecDestroy(this->energy_integrand); CHKERRXX(this->ierr);
        //this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);

        double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;

        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;
    }
}


int Diffusion::compute_energy_wall_dirichlet()
{
    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {


        this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
        this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

        this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;

        this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->n_local_size_sol;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(this->energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);



        this->ierr=VecDuplicate(this->phi,&this->phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->phi_polymer_shape,this->phiTemp); CHKERRXX(this->ierr);
        std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


        // energy_integrand
        this->ierr=VecGhostUpdateBegin(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        // phi_temp
        this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


        this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        std::cout<<"E1 "<<this->E<<std::endl;
        this->E=this->E/this->V;
        this->E_w=this->E;
        std::cout<<"E2 "<<this->E<<std::endl;
        std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
        this->E_logQ=-this->phi_bar*log(0.5*this->Q_forward+0.5*this->Q_backward);
        this->E=this->E+this->E_logQ;
        std::cout<<" E "<<this->E<<std::endl;


        std::cout<<" start to compute compressibility and wall terms "<<std::endl;
        Vec  wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_1,wall_force_scalar_2;
        PetscScalar minus_one=-1.00;
        PetscScalar minus_half=-0.5;
        double E_p_compressible, E_p_wall,E_m_wall;



        // compute scalars
        compressibility_scalar=minus_half*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_1=this->xhi_w_p*this->zeta_n_inverse+1.00;
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure

        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,wall_force_scalar_1); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);






        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,this->wp); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,compressibility_scalar); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->energy_integrand); CHKERRXX(this->ierr);

        E_p_compressible=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_compressible=E_p_compressible/this->V;

        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,wall_force); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_wall=E_p_wall/this->V;


        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wm,this->phi_wall); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,2*this->xhi_w_m/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_m_wall=E_m_wall/this->V;


        std::cout<<" E_p_compressible "<<E_p_compressible<<" E_p_wall "<<E_p_wall<<
                   " E_m_wall "<<E_m_wall<<std::endl;

        this->E=this->E+E_p_compressible+E_p_wall+E_m_wall;
        this->E_compressible=E_p_compressible;
        this->E_wall_m=E_m_wall;
        this->E_wall_p=E_p_wall;



        this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);


        VecDestroy(this->energy_integrand);
        VecDestroy(wall_force);

        double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;

        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;
    }
}


int Diffusion::compute_energy_wall_with_spatial_selectivity()
{
    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {


        this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
        this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

        this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;

        this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->n_local_size_sol;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(this->energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);



        this->ierr=VecDuplicate(this->phi,&this->phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->phiTemp,-1.00); CHKERRXX(this->ierr);
        std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


        // energy_integrand
        this->ierr=VecGhostUpdateBegin(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        // phi_temp
        this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


        this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        std::cout<<"E1 "<<this->E<<std::endl;
        this->E=this->E/this->V;
        this->E_w=this->E;
        std::cout<<"E2 "<<this->E<<std::endl;
        std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
        this->E_logQ=-this->phi_bar*log(0.5*this->Q_forward+0.5*this->Q_backward);
        this->E=this->E+this->E_logQ;
        std::cout<<" E "<<this->E<<std::endl;


        std::cout<<" start to compute compressibility and wall terms "<<std::endl;
        Vec  wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_1,wall_force_scalar_2;
        PetscScalar minus_one=-1.00;
        PetscScalar minus_half=-0.5;
        double E_p_compressible, E_p_wall,E_m_wall;



        // compute scalars
        compressibility_scalar=minus_half*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_1=this->xhi_w_p*this->zeta_n_inverse+1.00;
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure

        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,this->zeta_n_inverse*this->xhi_w_p); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(wall_force,this->xhi_w_x,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(wall_force,1.00,this->phi_wall); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);

        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,this->wp); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,compressibility_scalar); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->energy_integrand); CHKERRXX(this->ierr);

        E_p_compressible=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_compressible=E_p_compressible/this->V;

        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,wall_force); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_wall=E_p_wall/this->V;


        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->xhi_w_x,this->phi_wall); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wm,this->energy_integrand); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,2*this->xhi_w_m/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_m_wall=E_m_wall/this->V;


        std::cout<<" E_p_compressible "<<E_p_compressible<<" E_p_wall "<<E_p_wall<<
                   " E_m_wall "<<E_m_wall<<std::endl;

        this->E=this->E+E_p_compressible+E_p_wall+E_m_wall;
        this->E_compressible=E_p_compressible;
        this->E_wall_m=E_m_wall;
        this->E_wall_p=E_p_wall;
        this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->energy_integrand); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);
        double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;
        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;
    }
}

int Diffusion::compute_energy_wall_neuman_with_spatial_selectivity()
{
    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {


        this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
        this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

        this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;

        this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->n_local_size_sol;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(this->energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);

        std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


        // energy_integrand
        this->scatter_petsc_vector(&this->energy_integrand);
        this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->energy_integrand);
        std::cout<<"E1 "<<this->E<<std::endl;
        this->E=this->E/this->V;
        this->E_w=this->E;
        std::cout<<"E2 "<<this->E<<std::endl;
        std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
        this->E_logQ=(-1.00)*log(0.5*this->Q_forward+0.5*this->Q_backward);
        this->E=this->E+this->E_logQ;
        std::cout<<" E "<<this->E<<std::endl;

        //        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        //        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,this->phi_wall); CHKERRXX(this->ierr);
        //        this->ierr=VecAXPY(this->energy_integrand,-1.00,this->wp); CHKERRXX(this->ierr);

        //        this->scatter_petsc_vector(&this->energy_integrand);

        //        double  E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        //        E_p_wall=E_p_wall/this->V;


        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wm,this->phi_wall); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->energy_integrand,this->xhi_w_x); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,2*this->xhi_w_m/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        double E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_polymer_shape,this->energy_integrand);
        E_m_wall=E_m_wall/this->V;


        std::cout   <<
                       " E_m_wall "<<E_m_wall<<std::endl;

        this->E_compressible=0;
        this->E_wall_m=E_m_wall;
        this->E_wall_p=0.00;
        this->E=this->E+E_m_wall;




        VecDestroy(this->energy_integrand);
        //    VecDestroy(wall_force);

        double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;

        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;
    }
}



int Diffusion::compute_energy_wall_terrace()
{
    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {


        this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
        this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

        this->ierr=VecDuplicate(this->wm,&this->energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wm,this->energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;

        this->ierr=VecGetArray(this->energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->n_local_size_sol;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(this->energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);



        this->ierr=VecDuplicate(this->phi,&this->phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->phiTemp,-1.00); CHKERRXX(this->ierr);
        std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


        // energy_integrand
        this->ierr=VecGhostUpdateBegin(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        // phi_temp
        this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


        this->E=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        std::cout<<"E1 "<<this->E<<std::endl;
        this->E=this->E/this->V;
        this->E_w=this->E;
        std::cout<<"E2 "<<this->E<<std::endl;
        std::cout<<"E3"<<-log(0.5*this->Q_forward+0.5*this->Q_backward)<<std::endl;
        this->E_logQ=-this->phi_bar*log(0.5*this->Q_forward+0.5*this->Q_backward);
        this->E=this->E+this->E_logQ;
        std::cout<<" E "<<this->E<<std::endl;


        std::cout<<" start to compute compressibility and wall terms "<<std::endl;
        Vec  wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_2;
        PetscScalar minus_one=-1.00;
        PetscScalar minus_half=-0.5;
        double E_p_compressible, E_p_wall,E_m_wall;



        // compute scalars
        compressibility_scalar=minus_half*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure

        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(wall_force,this->zeta_n_inverse,this->phi_wall_p); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);






        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,this->wp); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,compressibility_scalar); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->energy_integrand); CHKERRXX(this->ierr);

        E_p_compressible=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_compressible=E_p_compressible/this->V;

        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wp,wall_force); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_p_wall=E_p_wall/this->V;


        this->ierr=VecSet(this->energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->energy_integrand,this->wm,this->phi_wall_m); CHKERRXX(this->ierr);
        this->ierr=VecScale(this->energy_integrand,2/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->energy_integrand);

        E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->energy_integrand);
        E_m_wall=E_m_wall/this->V;


        std::cout<<" E_p_compressible "<<E_p_compressible<<" E_p_wall "<<E_p_wall<<
                   " E_m_wall "<<E_m_wall<<std::endl;

        this->E=this->E+E_p_compressible+E_p_wall+E_m_wall;

        this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);


        VecDestroy(this->energy_integrand);
        VecDestroy(wall_force);

        double disordered_energy=-0.25*(1-2*this->f)*(1-2*this->f)*this->X_ab;

        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<this->E<<" "<<disordered_energy<<std::endl;
    }
}


int Diffusion::compute_mapping_fromLocal2Global()
{
    if(!this->mapped)
    {
        this->ix_fromLocal2Global=new PetscInt[this->n_local_size_sol];
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        for(int i=0;i<this->n_local_size_sol;i++)
        {
            this->ix_fromLocal2Global[i]=global_node_number+i;
        }
        this->mapped=PETSC_TRUE;
    }
}

int Diffusion::compute_densities()
{


    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {

        std::cout<<"compute densities "<<this->mpi->mpirank<<std::endl;
        this->rhoA_local=new double[this->n_local_size_sol];
        this->rhoB_local=new double[this->n_local_size_sol];

        this->ierr=VecDuplicate(this->sol_t,&this->rhoA); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->sol_t,&this->rhoB); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->rhoA,0); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->rhoB,0); CHKERRXX(this->ierr);


        PetscScalar *mask_local;
        if(!this->periodic_xyz)
        {
            this->ierr=VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&mask_local); CHKERRXX(this->ierr);
        }
        else
        {
            mask_local=new PetscScalar[this->n_local_size_sol];
        }


        this->drho_t=new double[this->N_t];
        this->drho_x_backward=new double[this->N_t];
        this->drho_x_forward=new double[this->N_t];


        //this->compute_mapping_fromLocal2Global();


        for(int ix=0;ix<this->n_local_size_sol;ix++)
        {
            // std::cout<<"compute densities "<<this->mpi->mpirank<<" compute_time_integrand"<<ix<<std::endl;

            if(mask_local[ix]<0.5 || this->periodic_xyz)
            {
                for(int i=0;i<this->N_t;i++)
                {
                    this->drho_x_forward[i]=this->q_forward[i*this->n_local_size_sol+ix];
                    this->drho_x_backward[i]=this->q_backward[i*this->n_local_size_sol+ix];
                }
                // std::cout<<"compute densities "<<this->mpi->mpirank<<" compute_time_integral"<<ix<<std::endl;
                // this->printDiffusionArray(this->drho_x_forward,this->N_t,"drho_x_forward");
                // this->printDiffusionArray(this->drho_x_backward,this->N_t,"drho_x_backward");
                this->compute_time_integral4Densities(ix);
                //std::cout<<this->rhoA_local[ix]<<std::endl;
                this->rhoA_local[ix]=this->rhoA_local[ix]/(0.5*this->Q_forward+0.5*this->Q_backward);
                //std::cout<<this->rhoA_local[ix]<<std::endl;
                this->rhoB_local[ix]=this->rhoB_local[ix]/(0.5*this->Q_forward+0.5*this->Q_backward);
                //std::cout<<"compute densities "<<this->mpi->mpirank<<" "<<ix<<std::endl;
            }
            else
            {
                this->rhoA_local[ix]=0;
                this->rhoB_local[ix]=0;
            }
        }
        //std::cout<<"compute densities compute mapping"<<this->mpi->mpirank<<std::endl;
        this->compute_mapping_fromLocal2Global();

        ///    std::cout<<"compute densities set values "<<this->mpi->mpirank<<std::endl;
        this->ierr= VecSetValues(this->rhoA,this->n_local_size_sol,this->ix_fromLocal2Global,this->rhoA_local,INSERT_VALUES);CHKERRXX(this->ierr);
        this->ierr= VecSetValues(this->rhoB,this->n_local_size_sol,this->ix_fromLocal2Global,this->rhoB_local,INSERT_VALUES);CHKERRXX(this->ierr);



        this->ierr=VecAssemblyBegin(this->rhoA);  CHKERRXX(this->ierr);
        this->ierr=VecAssemblyEnd(this->rhoA);    CHKERRXX(this->ierr);

        this->ierr=VecAssemblyBegin(this->rhoB);CHKERRXX(this->ierr);
        this->ierr=VecAssemblyEnd(this->rhoB);CHKERRXX(this->ierr);


        if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
        {
            this->filter_petsc_vector_dirichlet(&this->rhoA);
            this->filter_petsc_vector_dirichlet(&this->rhoB);
        }

        //    this->ierr=  VecGhostUpdateBegin(this->rhoA,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        //    this->ierr=  VecGhostUpdateEnd(this->rhoA,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        //    this->ierr=VecGhostUpdateBegin(this->rhoB,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        //      this->ierr=VecGhostUpdateEnd(this->rhoB,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        if(!this->periodic_xyz)
        { this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&mask_local); CHKERRXX(this->ierr);}
        else
        {
            delete mask_local;
        }

        delete drho_t;
        delete drho_x_backward;
        delete drho_x_forward;

        if(i_game>0)
        {
            delete this->rhoA_local;
            delete this->rhoB_local;
            VecDestroy(this->rhoA);
            VecDestroy(this->rhoB);

        }

    }

    std::cout<<"finished to compute densities "<<this->mpi->mpirank<<std::endl;

}


int Diffusion::compute_time_integral4Densities(int ix)
{
    double *realIntegrand=new double[this->N_t];

    for(int it=0;it<this->N_t;it++)
    {
        realIntegrand[it]= this->drho_x_forward[it]*this->drho_x_backward[this->N_t-it-1];
    }


    //this->printDiffusionArray(realIntegrand,this->N_t,"real_integrand");

    double I=0;
    double I1=0;
    double I2=0;
    double I3=0;
    double I4=0;
    double I5=0;



    int n=(int)(this->f*this->N_iterations);
    int NL=n+1;
    if(n%2==0)
    {
        double h=this->dt;
        I1=realIntegrand[0];
        for(int i=1;i<=n/2-1;i++)
        {
            I2+=2*realIntegrand[2*i];
        }
        for(int i=1;i<=n/2;i++)
        {
            I3+=4*realIntegrand[2*i-1];
        }
        I4=realIntegrand[n];
        I=(h/3)*(I1+I2+I3+I4);
    }
    else
    {
        double h=this->dt;
        n--;
        I1=realIntegrand[0];
        for(int i=1;i<=n/2-1;i++)
        {
            I2+=2*realIntegrand[2*i];
        }

        for(int i=1;i<=n/2;i++)
        {
            I3+=4*realIntegrand[2*i-1];
        }
        I4=realIntegrand[n];

        I5=(this->dt/2)*(realIntegrand[NL-2]+realIntegrand[NL-1]);
        I=(h/3)*(I1+I2+I3+I4)+I5;

    }

    this->rhoA_local[ix]=I;
    I=0;I1=0;I2=0;I3=0;I4=0;I5=0;

    int nstart=(int)(this->f*this->N_iterations);

    n=(int)((1-this->f)*this->N_iterations);
    NL=n+1;
    if(n%2==0)
    {
        double h=this->dt;
        I1=realIntegrand[0+nstart];
        for(int i=1;i<=n/2-1;i++)
        {
            I2+=2*realIntegrand[2*i+nstart];
        }
        for(int i=1;i<=n/2;i++)
        {
            I3+=4*realIntegrand[nstart+2*i-1];
        }
        I4=realIntegrand[nstart+n];
        I=(h/3)*(I1+I2+I3+I4);
    }
    else
    {
        double h=this->dt;
        n--;
        I1=realIntegrand[0];
        for(int i=1;i<=n/2-1;i++)
        {
            I2+=2*realIntegrand[nstart+2*i];
        }

        for(int i=1;i<=n/2;i++)
        {
            I3+=4*realIntegrand[nstart+2*i-1];
        }
        I4=realIntegrand[nstart+n];

        I5=(this->dt/2)*(realIntegrand[nstart+NL-2]+realIntegrand[nstart+NL-1]);
        I=(h/3)*(I1+I2+I3+I4)+I5;

    }

    this->rhoB_local[ix]=I;
    delete realIntegrand;
    return I;
}


int Diffusion::compute_forces()
{

    int n_game=1;
    for(int i_game=0;i_game<n_game;i_game++)
    {
        this->ierr=VecDuplicate(this->phi,&this->fp);    CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&this->fm);    CHKERRXX(this->ierr);


        std::cout<<" compute forces "<<this->mpi->mpirank<<std::endl;
        // Pressure Force
        this->ierr=VecSet(this->fp,-1.00);              CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoA);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoB);  CHKERRXX(this->ierr);


        // Exchange Force
        this->ierr=VecCopy(this->wm,this->fm);          CHKERRXX(this->ierr);
        this->ierr=VecScale(this->fm,2.00/this->X_ab);     CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,+1.00,this->rhoB);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,-1.00,this->rhoA);  CHKERRXX(this->ierr);

        this->printDiffusionArrayFromVector(&this->fp,"fp");
        this->printDiffusionArrayFromVector(&this->fm,"fm");

        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)

        {
            this->filter_irregular_forces();
        }

        this->printDiffusionArrayFromVector(&this->fp,"fp_filtered");
        this->printDiffusionArrayFromVector(&this->fm,"fm_filtered");
        std::cout<<" finished to compute forces "<<this->mpi->mpirank<<std::endl;

        if(i_game>0)
        {
            VecDestroy(this->fp);
            VecDestroy(this->fm);
            //  VecDestroy(this->phi);


        }

    }




}

int Diffusion::compute_wall_forces()
{


    this->Q_backward=this->Q_backward*exp(-this->wp_average);
    this->Q_forward=this->Q_forward*exp(-this->wp_average);
    this->compute_phi_bar();
    this->ierr=VecScale(this->rhoA,this->phi_bar); CHKERRXX(this->ierr);
    this->ierr=VecScale(this->rhoB,this->phi_bar); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&this->rhoA);
    this->scatter_petsc_vector(&this->rhoB);
    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);


    int n_game=1;
    for(int i_game=0;i_game<n_game;i_game++)
    {




        std::cout<<" compute compressibility and wall forces "<<std::endl;

        // declare temporary variables

        Vec compressibility_force, wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_1,wall_force_scalar_2;
        PetscScalar minus_one=-1.00;



        // compute scalars
        compressibility_scalar=minus_one*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_1=this->xhi_w_p*this->zeta_n_inverse+1.00;
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure
        this->ierr=VecDuplicate(this->phi,&compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->wp,compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(compressibility_force,compressibility_scalar); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,wall_force_scalar_1); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);

        // Scatter

        this->scatter_petsc_vector(&compressibility_force);
        this->scatter_petsc_vector(&wall_force);

        this->printDiffusionArrayFromVector(&compressibility_force,"compressibility_force");


        std::cout<<" compute regular forces "<<this->mpi->mpirank<<std::endl;

        this->ierr=VecDuplicate(this->phi,&this->fp);    CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&this->fm);    CHKERRXX(this->ierr);


        // Pressure Force
        this->ierr=VecSet(this->fp,0.00);              CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoA);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoB);  CHKERRXX(this->ierr);


        // Exchange Force
        this->ierr=VecCopy(this->wm,this->fm);          CHKERRXX(this->ierr);
        this->ierr=VecScale(this->fm,2.00/this->X_ab);     CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,+1.00,this->rhoB);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,-1.00,this->rhoA);  CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->fp); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->fm); CHKERRXX(this->ierr);

        std::cout<<" add wall and compressibility forces to the usual forces "<<std::endl;

        this->ierr=VecAXPY(this->fp,1.00,compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,wall_force);

        this->ierr=VecAXPY(this->fm,2*this->xhi_w_m/this->X_ab,this->phi_wall); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->fm);
        this->scatter_petsc_vector(&this->fp);


        this->printDiffusionArrayFromVector(&this->fp,"fp");
        this->printDiffusionArrayFromVector(&this->fm,"fm");

        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
        || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)

        {
            this->filter_irregular_forces();
        }

        this->printDiffusionArrayFromVector(&this->fp,"fp_filtered");
        this->printDiffusionArrayFromVector(&this->fm,"fm_filtered");

        this->ierr=VecDestroy(compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);

        std::cout<<" finished to compute forces "<<this->mpi->mpirank<<std::endl;

        if(i_game>0)
        {
            VecDestroy(this->fp);
            VecDestroy(this->fm);
            //  VecDestroy(this->phi);


        }

    }


    this->compute_wp_averaged_on_saddle_point();

}

int Diffusion::compute_wall_forces_with_spatial_selectivity()
{
    this->Q_backward=this->Q_backward*exp(-this->wp_average);
    this->Q_forward=this->Q_forward*exp(-this->wp_average);
    this->compute_phi_bar();
    this->ierr=VecScale(this->rhoA,this->phi_bar); CHKERRXX(this->ierr);
    this->ierr=VecScale(this->rhoB,this->phi_bar); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&this->rhoA);
    this->scatter_petsc_vector(&this->rhoB);
    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);


    int n_game=1;
    for(int i_game=0;i_game<n_game;i_game++)
    {
        std::cout<<" compute compressibility and wall forces "<<std::endl;
        // declare temporary variables
        Vec compressibility_force, wall_force,weighted_selectivity;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_1,wall_force_scalar_2;
        PetscScalar minus_one=-1.00;
        // compute scalars
        compressibility_scalar=minus_one*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_1=this->xhi_w_p*this->zeta_n_inverse+1.00;
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure
        this->ierr=VecDuplicate(this->phi,&compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&weighted_selectivity); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wp,compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(compressibility_force,compressibility_scalar); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(wall_force,this->xhi_w_x,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,this->xhi_w_p); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,this->zeta_n_inverse); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);

        // Scatter

        this->scatter_petsc_vector(&compressibility_force);
        this->scatter_petsc_vector(&wall_force);

        this->printDiffusionArrayFromVector(&compressibility_force,"compressibility_force");
        std::cout<<" compute regular forces "<<this->mpi->mpirank<<std::endl;
        this->ierr=VecDuplicate(this->phi,&this->fp);    CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&this->fm);    CHKERRXX(this->ierr);
        // Pressure Force
        this->ierr=VecSet(this->fp,0.00);              CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoA);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoB);  CHKERRXX(this->ierr);


        // Exchange Force
        this->ierr=VecCopy(this->wm,this->fm);          CHKERRXX(this->ierr);
        this->ierr=VecScale(this->fm,2.00/this->X_ab);     CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,+1.00,this->rhoB);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,-1.00,this->rhoA);  CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->fp); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->fm); CHKERRXX(this->ierr);

        std::cout<<" add wall and compressibility forces to the usual forces "<<std::endl;

        this->ierr=VecAXPY(this->fp,1.00,compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,wall_force);


        this->ierr=VecCopy(this->xhi_w_x,weighted_selectivity); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(weighted_selectivity,this->phi_wall,weighted_selectivity); CHKERRXX(this->ierr);


        this->ierr=VecAXPY(this->fm,2*this->xhi_w_m/this->X_ab,weighted_selectivity); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->fm);
        this->scatter_petsc_vector(&this->fp);


        this->printDiffusionArrayFromVector(&this->fp,"fp");
        this->printDiffusionArrayFromVector(&this->fm,"fm");

        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)

        {
            this->filter_irregular_forces();
        }

        this->printDiffusionArrayFromVector(&this->fp,"fp_filtered");
        this->printDiffusionArrayFromVector(&this->fm,"fm_filtered");

        this->ierr=VecDestroy(compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(weighted_selectivity); CHKERRXX(this->ierr);



        std::cout<<" finished to compute forces "<<this->mpi->mpirank<<std::endl;

        if(i_game>0)
        {
            VecDestroy(this->fp);
            VecDestroy(this->fm);
            //  VecDestroy(this->phi);


        }

    }


    this->compute_wp_averaged_on_saddle_point();

}


int Diffusion::compute_wall_forces_neuman()
{

    //    this->compute_phi_bar();
    //    this->ierr=VecScale(this->rhoA,this->phi_bar); CHKERRXX(this->ierr);
    //    this->ierr=VecScale(this->rhoB,this->phi_bar); CHKERRXX(this->ierr);
    //    this->scatter_petsc_vector(&this->rhoA);
    //    this->scatter_petsc_vector(&this->rhoB);
    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);


    int n_game=1;
    for(int i_game=0;i_game<n_game;i_game++)
    {
        std::cout<<" compute wall forces in the Neuman case"<<std::endl;
        std::cout<<" compute regular forces "<<this->mpi->mpirank<<std::endl;

        this->ierr=VecDuplicate(this->phi,&this->fp);    CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&this->fm);    CHKERRXX(this->ierr);

        // Pressure Force
        this->ierr=VecSet(this->fp,0.00);              CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoA);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoB);  CHKERRXX(this->ierr);
        //this->ierr=VecAXPY(this->fp,1.00,this->phi_wall);  CHKERRXX(this->ierr);

        // Exchange Force
        this->ierr=VecCopy(this->wm,this->fm);          CHKERRXX(this->ierr);
        this->ierr=VecScale(this->fm,2.00/this->X_ab);     CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,+1.00,this->rhoB);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,-1.00,this->rhoA);  CHKERRXX(this->ierr);


        std::cout<<" add wall and compressibility forces to the usual forces "<<std::endl;
        this->ierr=VecShift(this->fp,-1.00); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,2*this->xhi_w_m/this->X_ab,this->phi_wall); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->fm);
        this->scatter_petsc_vector(&this->fp);


        this->printDiffusionArrayFromVector(&this->fp,"fp");
        this->printDiffusionArrayFromVector(&this->fm,"fm");
        this->filter_irregular_forces();
        this->printDiffusionArrayFromVector(&this->fp,"fp_filtered");
        this->printDiffusionArrayFromVector(&this->fm,"fm_filtered");
        std::cout<<" finished to compute forces "<<this->mpi->mpirank<<std::endl;
        if(i_game>0)
        {
            VecDestroy(this->fp);
            VecDestroy(this->fm);
        }
    }
}

int Diffusion::compute_wall_forces_dirichlet()
{

    this->compute_wall_forces();
    this->filter_petsc_vector_dirichlet(&this->fp);
    this->filter_petsc_vector_dirichlet(&this->fm);
}


int Diffusion::compute_wall_forces_with_spatial_selectivity_neuman()
{

    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);
    //    this->compute_phi_bar();
    //    this->ierr=VecScale(this->rhoA,this->phi_bar); CHKERRXX(this->ierr);
    //    this->ierr=VecScale(this->rhoB,this->phi_bar); CHKERRXX(this->ierr);
    //    this->scatter_petsc_vector(&this->rhoA);
    //    this->scatter_petsc_vector(&this->rhoB);

    int n_game=1;
    for(int i_game=0;i_game<n_game;i_game++)
    {
        std::cout<<" compute compressibility and wall forces "<<std::endl;
        // declare temporary variables
        Vec weighted_selectivity;


        // Allocate memory using the apropriate petsc vector data structure
        this->ierr=VecDuplicate(this->phi,&weighted_selectivity); CHKERRXX(this->ierr);



        std::cout<<" compute regular forces "<<this->mpi->mpirank<<std::endl;
        this->ierr=VecDuplicate(this->phi,&this->fp);    CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&this->fm);    CHKERRXX(this->ierr);
        // Pressure Force
        this->ierr=VecSet(this->fp,0.00);              CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoA);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoB);  CHKERRXX(this->ierr);
        // this->ierr=VecAXPY(this->fp,1.00,this->phi_wall);  CHKERRXX(this->ierr);
        this->ierr=VecShift(this->fp,-1.00); CHKERRXX(this->ierr);



        // Exchange Force
        this->ierr=VecCopy(this->wm,this->fm);          CHKERRXX(this->ierr);
        this->ierr=VecScale(this->fm,2.00/this->X_ab);     CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,+1.00,this->rhoB);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,-1.00,this->rhoA);  CHKERRXX(this->ierr);


        this->ierr=VecCopy(this->xhi_w_x,weighted_selectivity); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(weighted_selectivity,this->phi_wall,weighted_selectivity); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,2*this->xhi_w_m/this->X_ab,weighted_selectivity); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->fm);
        this->scatter_petsc_vector(&this->fp);
        this->printDiffusionArrayFromVector(&this->fp,"fp");
        this->printDiffusionArrayFromVector(&this->fm,"fm");
        this->filter_irregular_forces();
        this->printDiffusionArrayFromVector(&this->fp,"fp_filtered");
        this->printDiffusionArrayFromVector(&this->fm,"fm_filtered");
        this->ierr=VecDestroy(weighted_selectivity); CHKERRXX(this->ierr);
        std::cout<<" finished to compute forces "<<this->mpi->mpirank<<std::endl;
        if(i_game>0)
        {
            VecDestroy(this->fp);
            VecDestroy(this->fm);
        }
    }
}


int Diffusion::compute_wall_forces_terrace()
{

    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);


    int n_game=1;
    for(int i_game=0;i_game<n_game;i_game++)
    {




        std::cout<<" compute compressibility and wall forces "<<std::endl;

        // declare temporary variables

        Vec compressibility_force, wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_2;
        PetscScalar minus_one=-1.00;



        // compute scalars
        compressibility_scalar=minus_one*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure
        this->ierr=VecDuplicate(this->phi,&compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->wp,compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(compressibility_force,compressibility_scalar); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(wall_force,this->zeta_n_inverse,this->phi_wall_p); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);

        // Scatter

        this->scatter_petsc_vector(&compressibility_force);
        this->scatter_petsc_vector(&wall_force);

        this->printDiffusionArrayFromVector(&compressibility_force,"compressibility_force");


        std::cout<<" compute regular forces "<<this->mpi->mpirank<<std::endl;

        this->ierr=VecDuplicate(this->phi,&this->fp);    CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&this->fm);    CHKERRXX(this->ierr);


        // Pressure Force
        this->ierr=VecSet(this->fp,0.00);              CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoA);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,this->rhoB);  CHKERRXX(this->ierr);


        // Exchange Force
        this->ierr=VecCopy(this->wm,this->fm);          CHKERRXX(this->ierr);
        this->ierr=VecScale(this->fm,2.00/this->X_ab);     CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,+1.00,this->rhoB);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fm,-1.00,this->rhoA);  CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->fp); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->fm); CHKERRXX(this->ierr);

        std::cout<<" add wall and compressibility forces to the usual forces "<<std::endl;

        this->ierr=VecAXPY(this->fp,1.00,compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->fp,1.00,wall_force);

        this->ierr=VecAXPY(this->fm,2.00/this->X_ab,this->phi_wall_m); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->fm);
        this->scatter_petsc_vector(&this->fp);


        this->printDiffusionArrayFromVector(&this->fp,"fp");
        this->printDiffusionArrayFromVector(&this->fm,"fm");

        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        {
            this->filter_irregular_forces();
        }

        this->printDiffusionArrayFromVector(&this->fp,"fp_filtered");
        this->printDiffusionArrayFromVector(&this->fm,"fm_filtered");

        this->ierr=VecDestroy(compressibility_force); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);

        std::cout<<" finished to compute forces "<<this->mpi->mpirank<<std::endl;

        if(i_game>0)
        {
            VecDestroy(this->fp);
            VecDestroy(this->fm);
            //  VecDestroy(this->phi);


        }




        this->compute_wp_averaged_on_saddle_point();
    }

}


int Diffusion::compute_wp_averaged_on_saddle_point()
{

    int n_game=1;
    for(int i_game=0;i_game<n_game;i_game++)
    {


        std::cout<<" compute compressibility and wall forces "<<std::endl;

        // declare temporary variables


        Vec  wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_1,wall_force_scalar_2;
        PetscScalar minus_one=-1.00;



        // compute scalars
        compressibility_scalar=minus_one*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_1=this->xhi_w_p*this->zeta_n_inverse+1.00;
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure

        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);


        this->ierr=VecCopy(this->phi_wall,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,wall_force_scalar_1); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);


        this->scatter_petsc_vector(&wall_force);
        this->ierr=VecDuplicate(this->phi,&this->phiTemp); CHKERRXX(this->ierr);
        if(this->my_casl_diffusion_method!=Diffusion::dirichlet_crank_nicholson)
        {
            this->ierr=VecSet(this->phiTemp,-1.00); CHKERRXX(this->ierr);}
        else
        {
            this->ierr=VecCopy(this->phi_polymer_shape,this->phiTemp); CHKERRXX(this->ierr);}

        this->scatter_petsc_vector(&this->phiTemp);
        this->wp_average_at_saddle_point=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp, wall_force)/this->V;
        std::cout<< this->wp_average_at_saddle_point<<std::endl;


        this->ierr=VecShift(wall_force,this->phi_bar); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,-1/compressibility_scalar);
        // Scatter


        this->scatter_petsc_vector(&wall_force);





        double max_wall_force; int max_wall_force_i;
        double min_wall_force; int min_wall_force_i;
        double sum_wall_force;
        this->ierr=VecMax(wall_force,&max_wall_force_i,&max_wall_force); CHKERRXX(this->ierr);
        this->ierr=VecMin(wall_force,&min_wall_force_i,&min_wall_force); CHKERRXX(this->ierr);
        this->ierr=VecSum(wall_force,&sum_wall_force); CHKERRXX(this->ierr);

        double max_wall; int max_wall_i;
        double min_wall; int min_wall_i;
        double sum_wall;
        this->ierr=VecMax(this->phi_wall,&max_wall_i,&max_wall); CHKERRXX(this->ierr);
        this->ierr=VecMin(this->phi_wall,&min_wall_i,&min_wall); CHKERRXX(this->ierr);
        this->ierr=VecSum(this->phi_wall,&sum_wall); CHKERRXX(this->ierr);

        this->wp_average_at_saddle_point=integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp, wall_force)/this->V;
        std::cout<< this->wp_average_at_saddle_point<<std::endl;




        this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);

        std::cout<<" finished to compute wp_average_at_saddle_point "<<this->mpi->mpirank<<std::endl;

    }
}

//   Later on: insert the implicit evolution for the pressure field introducing
//  FFT transforms with FFTW
int Diffusion::evolve_pressure_field()
{


    this->printDiffusionArrayFromVector(&this->wp,"wp_before");
    this->printDiffusionArrayFromVector(&this->fp,"fp");

    this->ierr=VecCreateGhostNodes(this->p4est,this->nodes,&this->phiTemp);
    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    {

        this->ierr=VecSet(this->phiTemp,-1.00);
    }
    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
   || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
   || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
    {

        this->ierr=VecCopy(this->phi_polymer_shape,this->phiTemp);
    }

    this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


    this->ierr=VecAXPY(this->wp,this->lambda_plus,this->fp); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

    if(!this->seed_optimization
     && this->lambda_plus>0
     && !this->advance_fields_scft_advance_mask_level_set
     && this->my_casl_diffusion_method!=Diffusion::dirichlet_crank_nicholson
     && this->substract_average_from_wp)
    {
        Vec wp_extended;
        // this->printDiffusionArrayFromVector(&this->rhoA,"rhoa_before_ext");
        this->ierr=VecDuplicate(this->wp,&wp_extended); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->wp,wp_extended); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        if(!this->periodic_xyz)
            this->extend_petsc_vector(&wp_extended);
        //        PetscScalar wp_average;
        wp_average=   integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,wp_extended)/this->V;
        std::cout<<"wp spatial average before substraction "<<wp_average<<std::endl;
        Vec wp_average_vec;
        this->ierr=VecDuplicate(this->wp,&wp_average_vec); CHKERRXX(this->ierr);
        this->ierr=VecSet(wp_average_vec,wp_average);
        this->ierr=VecGhostUpdateBegin(wp_average_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(wp_average_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->wp,-1,wp_average_vec); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);


        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        {

            this->filter_irregular_potential();
            this->ierr=VecGhostUpdateBegin(this->wp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->wp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        }



        this->printDiffusionArrayFromVector(&this->wp,"wp_after");

        this->ierr=VecCopy(this->wp,wp_extended); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        if(!this->periodic_xyz)
            this->extend_petsc_vector(&wp_extended);

        //        wp_average=   integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,wp_extended)/this->V;

        std::cout<<"wp spatial average after substraction "<<integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,wp_extended)/this->V<<std::endl;




        this->ierr=VecCopy(this->fp,wp_extended); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        if(!this->periodic_xyz)
            this->extend_petsc_vector(&wp_extended);

        std::cout<<"fp spatial average  "<<integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,wp_extended)/this->V<<std::endl;

        this->ierr= VecDestroy(wp_average_vec); CHKERRXX(this->ierr);

        this->ierr=VecDestroy(wp_extended); CHKERRXX(this->ierr);
    }


    if(this->advance_fields_scft_advance_mask_level_set
    || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
    {

        //        Vec wp_extended;
        //        // this->printDiffusionArrayFromVector(&this->rhoA,"rhoa_before_ext");

        //        this->ierr=VecDuplicate(this->wp,&wp_extended); CHKERRXX(this->ierr);
        //        this->ierr=VecCopy(this->wp,wp_extended); CHKERRXX(this->ierr);
        //        this->ierr=VecGhostUpdateBegin(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        //        this->ierr=VecGhostUpdateEnd(wp_extended,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        //        if(!this->periodic_xyz)
        //            this->extend_petsc_vector(&wp_extended);

        this->wp_average=   integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->wp)/this->V;

        std::cout<<"wp spatial average before substraction "<<this->wp_average<<std::endl;


        if(this->substract_average_from_wp)
            this->ierr=VecShift(this->wp,this->wp_average_at_saddle_point-this->wp_average); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->wp);

        if(this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            this->filter_petsc_vector_dirichlet(&this->wp);

        //        this->wp_average=   integrate_over_negative_domain(this->p4est,this->nodes,this->phiTemp,this->wp)/this->V;
        //        std::cout<<"wp spatial average before substraction "<<this->wp_average<<std::endl;


        // this->ierr=VecDestroy(wp_extended); CHKERRXX(this->ierr);
        this->printDiffusionArrayFromVector(&this->wp,"wp_after");

    }


    this->ierr=VecDestroy(this->phiTemp); CHKERRXX(this->ierr);
}

int Diffusion::evolve_chemical_exchange_field()
{
    this->ierr=VecAXPY(this->wm,-this->lambda_minus,this->fm); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&this->wm);
}

int Diffusion::evolve_fields_explicit()
{
    this->evolve_pressure_field();
    this->evolve_chemical_exchange_field();
}

int Diffusion::generate_disordered_phase()
{

    VecSet(this->wp,0.00);
    PetscScalar wm_const=-(1-2*this->f)*this->X_ab/2.00;
    VecSet(this->wm,wm_const);
    this->scatter_petsc_vector(&this->wm);
}


int Diffusion::printDiffusionVector(Vec *V2Print, std::string file_name_str)
{

    if(!this->minimum_IO)
    {
        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2DebugVec;
        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<"_"<<this->i_mean_field<<"_"<<this->it<<".txt";
        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


        PetscViewer lab;
        this->ierr=PetscViewerASCIIOpen(MPI_COMM_WORLD,mystr2DebugVec.c_str(),&lab); CHKERRXX(this->ierr);
        // this->ierr=PetscViewerSetFormat(lab,PETSC_VIEWER_ASCII_INDEX); CHKERRXX(this->ierr);

        this->ierr=VecView(*V2Print,lab); CHKERRXX(this->ierr);
        this->ierr=PetscViewerDestroy(lab);
    }


}

int Diffusion::printDiffusionArray(double *x2Print, int Nx, string file_name_str)
{
    //
    //    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    //    {
    //        VecDuplicate(this->phi,&phiTemp);
    //        VecSet(phiTemp,-1.00);
    //    }
    //    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler|| this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
    //    {
    //        VecDuplicate(this->phi_polymer_shape,&phiTemp);
    //        VecCopy(this->phi_polymer_shape,phiTemp);
    //    }


    //    VecGetArray(phiTemp,&phiTempArray);
    if(!this->minimum_IO)
    {
        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2DebugVec;
        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";
        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(int i=0;i<Nx;i++)
        {
            double temp_x=x2Print[i];
            fprintf(outFileVec,"%d %f \n",i,temp_x);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        //VecRestoreArray(phiTemp,&phiTempArray);
        //VecDestroy(phiTemp);
    }
    return 0;


}

int Diffusion::printDiffusionArray(int *x2Print, int Nx, string file_name_str)
{
    if(!this->minimum_IO)
    {
        //
        //    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
        //    {
        //        VecDuplicate(this->phi,&phiTemp);
        //        VecSet(phiTemp,-1.00);
        //    }
        //    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler|| this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        //    {
        //        VecDuplicate(this->phi_polymer_shape,&phiTemp);
        //        VecCopy(this->phi_polymer_shape,phiTemp);
        //    }


        //    VecGetArray(phiTemp,&phiTempArray);
        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2DebugVec;
        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";
        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(int i=0;i<Nx;i++)
        {
            int temp_x=x2Print[i];
            fprintf(outFileVec,"%d %d \n",i,temp_x);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        //VecRestoreArray(phiTemp,&phiTempArray);
        //VecDestroy(phiTemp);
    }
    return 0;


}


int Diffusion::printDiffusionArrayFromVector(Vec *v2Print, string file_name_str, PetscBool write_nodes)
{
    // if(!this->minimum_IO)
    if(!this->minimum_IO)
    {
        PetscInt   Nx;

        VecGetLocalSize(*v2Print,&Nx);

        PetscScalar *x2Print;
        VecGetArray(*v2Print,&x2Print);


        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2DebugVec;
        //        if(this->forward_stage)
        //        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<< this->it<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<< this->it<<".txt";

        if(this->forward_stage)
            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<".txt";
        else
            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<".txt";

        //        if(this->forward_stage)
        //            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";

        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        if(!write_nodes)
        {

            std::cout<<" start to write simple "<<file_name_str<<" "<<Nx<<std::endl;
            std::cout<<" start to write simple "<<mystr2DebugVec<<" "<<Nx<<std::endl;
            for(int i=0;i<Nx;i++)
            {
                PetscScalar temp_x=x2Print[i];
                fprintf(outFileVec,"%d %e \n",i,temp_x);//,phiTempArray[i]);
            }
            fclose(outFileVec);

            VecRestoreArray(*v2Print,&x2Print);
            std::cout<<" finish to write simple "<<file_name_str<<std::endl;
        }
        else
        {
            std::cout<<" start to write with p4est "<<file_name_str<<std::endl;
            double x_local,y_local,z_local;
            p4est_topidx_t global_node_number=0;

            for(int ii=0;ii<mpi_rank;ii++)
                global_node_number+=this->nodes->global_owned_indeps[ii];

            for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++)
            {
                /* since we want to access the local nodes, we need to 'jump' over intial
           * nonlocal nodes. Number of initial nonlocal nodes is given by
           * nodes->offset_owned_indeps
           */
                p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
                p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

                bool isNodeWall=is_node_Wall(this->p4est,node);

                double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
                double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
                double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

                x_local = int2double_coordinate_transform(node->x) + tree_xmin;
                y_local = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
                z_local = int2double_coordinate_transform(node->z) + tree_zmin;
#endif


                PetscScalar temp_x=x2Print[i];

#ifdef P4_TO_P8
                fprintf(outFileVec,"%d %d %d %f %f %f %e\n",mpi_rank,global_node_number+i,i,x_local,y_local,z_local,temp_x);
#else
                fprintf(outFileVec,"%d %d %d %f %f %e\n",mpi_rank,global_node_number+i,i,x_local,y_local,temp_x);

#endif


            }
            std::cout<<" finish to write with p4est "<<file_name_str<<std::endl;
            VecRestoreArray(*v2Print,&x2Print);
            fclose(outFileVec);
        }

    }
    return 0;
}


int Diffusion::printForces()
{


    PetscScalar *phiTempArray=new PetscScalar[this->n_local_size_sol];
    if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    {
        VecGetValues(this->phi,this->n_local_size_sol,this->ix_fromLocal2Global,phiTempArray);

    }
    if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler
    || this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
    || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
    {
        VecGetValues(this->phi_polymer_shape,this->n_local_size_sol,this->ix_fromLocal2Global,phiTempArray);

    }




    int mpi_size;
    int mpi_rank;

    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

    std::stringstream oss2DebugVec;
    std::string mystr2DebugVec;
    oss2DebugVec << this->convert2FullPath("Forces")<<"_"<<mpi_rank<<".txt";
    mystr2DebugVec=oss2DebugVec.str();
    FILE *outFileVec;
    outFileVec=fopen(mystr2DebugVec.c_str(),"w");
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


    int n_local_size;
    int n_global_size;
    VecGetLocalSize(this->fp,&n_local_size);
    VecGetSize(this->fm,&n_global_size);

    PetscScalar *fp_local=new PetscScalar[n_local_size];
    VecGetValues(this->fp,this->n_local_size_sol,this->ix_fromLocal2Global,fp_local);

    PetscScalar *fm_local=new PetscScalar[n_local_size];
    VecGetValues(this->fm,this->n_local_size_sol,this->ix_fromLocal2Global,fm_local);

    PetscScalar x_local;
    PetscScalar y_local;
#ifdef P4_TO_P8
    PetscScalar z_local;
#endif

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        bool isNodeWall=is_node_Wall(this->p4est,node);

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        x_local = int2double_coordinate_transform(node->x) + tree_xmin;
        y_local = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        z_local = int2double_coordinate_transform(node->z) + tree_zmin;
#endif



        double temp_fp=fp_local[i];
        double temp_fm=fm_local[i];
#ifdef P4_TO_P8
        fprintf(outFileVec,"%d %d %d %f %f %f %f %f %f\n",mpi_rank,global_node_number+i,i,x_local,y_local,z_local,phiTempArray[i],temp_fp,temp_fm);
#else
        fprintf(outFileVec,"%d %d %d %f %f %f %f %f %\n",mpi_rank,global_node_number+i,i,x_local,y_local,phiTempArray[i],temp_fp,temp_fm);

#endif


    }
    fclose(outFileVec);


    delete fp_local;
    delete fm_local;
    delete phiTempArray;


    return 0;
}

int Diffusion::printDensities()
{

    //if(!this->minimum_IO)
    {
        PetscScalar *phiTempArray=new PetscScalar[this->n_local_size_sol];
        if(this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
        {
            VecGetValues(this->phi,this->n_local_size_sol,this->ix_fromLocal2Global,phiTempArray);

        }
        if(this->my_casl_diffusion_method==Diffusion::neuman_backward_euler|| this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        {
            VecGetValues(this->phi_polymer_shape,this->n_local_size_sol,this->ix_fromLocal2Global,phiTempArray);

        }
        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2DebugVec;
        //oss2DebugVec << this->convert2FullPath("Densities")<<"_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        oss2DebugVec << this->convert2FullPath("Densities")<<"_"<<mpi_rank<<".txt";
        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


        int n_local_size;
        int n_global_size;
        VecGetLocalSize(this->rhoA,&n_local_size);
        VecGetSize(this->rhoA,&n_global_size);



        PetscScalar x_local;
        PetscScalar y_local;
#ifdef P4_TO_P8
        PetscScalar z_local;
#endif


        p4est_topidx_t global_node_number=0;

        for(int ii=0;ii<myRank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];


        for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
        {
            /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

            bool isNodeWall=is_node_Wall(this->p4est,node);

            double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
            double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
            double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

            x_local = int2double_coordinate_transform(node->x) + tree_xmin;
            y_local = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
            z_local = int2double_coordinate_transform(node->z) + tree_zmin;
#endif





            double rho3=this->rhoA_local[i];
            double rho4=this->rhoB_local[i];
#ifdef P4_TO_P8
            fprintf(outFileVec,"%d %d %d %f %f %f %f %f %f\n",mpi_rank,global_node_number+i,i,x_local,y_local,z_local,phiTempArray[i],rho3,rho4);
#else
            fprintf(outFileVec,"%d %d %d %f %f %f %f %f\n",mpi_rank,global_node_number+i,i,x_local,y_local,phiTempArray[i],rho3,rho4);
#endif

        }
        fclose(outFileVec);


        delete phiTempArray;
    }
    return 0;
}


int Diffusion::printDensities_by_cells()
{


    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {

        //if(!this->minimum_IO)
        {

            int mpi_size;
            int mpi_rank;

            MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

            std::stringstream oss2DebugVec;
            std::string mystr2DebugVec;
            //oss2DebugVec << this->convert2FullPath("Densities")<<"_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
            oss2DebugVec << this->convert2FullPath("Densities_Cells")<<"_"<<mpi_rank<<".txt";
            mystr2DebugVec=oss2DebugVec.str();
            FILE *outFileVec;
            outFileVec=fopen(mystr2DebugVec.c_str(),"w");



            int n_local_size;
            int n_global_size;
            VecGetLocalSize(this->rhoA,&n_local_size);
            VecGetSize(this->rhoA,&n_global_size);





            Vec myCellsVector4Paraview_a;
            VecCreateGhostCells(this->p4est,this->ghost,&myCellsVector4Paraview_a);
            PetscScalar *Cells_Vector_local_a;
            this->ierr=VecGetArray(myCellsVector4Paraview_a,&Cells_Vector_local_a); CHKERRXX(this->ierr);

            Vec myCellsVector4Paraview_b;
            VecCreateGhostCells(this->p4est,this->ghost,&myCellsVector4Paraview_b);
            PetscScalar *Cells_Vector_local_b;
            this->ierr=VecGetArray(myCellsVector4Paraview_b,&Cells_Vector_local_b); CHKERRXX(this->ierr);


            PetscInt icell_size,icell_local_size;
            this->ierr=VecGetSize(myCellsVector4Paraview_a,&icell_size);CHKERRXX(this->ierr);
            this->ierr= VecGetLocalSize(myCellsVector4Paraview_a,&icell_local_size); CHKERRXX(this->ierr);


            this->ierr=VecGetArray(this->rhoA,&this->rhoA_local); CHKERRXX(this->ierr);
            this->ierr=VecGetArray(this->rhoB,&this->rhoB_local); CHKERRXX(this->ierr);


            double x0,x1,x2,x3;
            double y0,y1,y2,y3;
            double z0,z1,z2,z3;

#ifdef P4_TO_P8
            double x4,x5,x6,x7;
            double y4,y5,y6,y7;
            double z4,z5,z6,z7;
#endif

            const p4est_locidx_t *q2n = this->nodes->local_nodes;
            int jj=0;
            for (p4est_locidx_t i = p4est->first_local_tree; i<=p4est->last_local_tree; i++)
            {
                p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);
                //p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];
                //std::cout<<"tree2->quadrants.elem_count"<<tree2->quadrants.elem_count<<std::endl;

                for(p4est_locidx_t j=0;j<tree2->quadrants.elem_count;j++)
                {

                    /* since we want to access the local nodes, we need to 'jump' over intial
              * nonlocal nodes. Number of initial nonlocal nodes is given by
              * nodes->offset_owned_indeps
              */
                    p4est_quadrant_t *quadrant = (p4est_quadrant_t*)sc_array_index(&tree2->quadrants, j);
                    int n0,n1,n2,n3;

                    int quad_idx=j+tree2->quadrants_offset;

#ifdef P4_TO_P8
                    int n4,n5,n6,n7;
                    n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
                    n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
                    n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
                    n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];
                    n4 =  q2n[ quad_idx*P4EST_CHILDREN + 4 ];
                    n5 =  q2n[ quad_idx*P4EST_CHILDREN + 5 ];
                    n6 =  q2n[ quad_idx*P4EST_CHILDREN + 6 ];
                    n7 =  q2n[ quad_idx*P4EST_CHILDREN + 7 ];

#else
                    n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
                    n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
                    n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
                    n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];



#endif

                    this->get_x_y_z_from_node(n0,x0,y0,z0);
                    this->get_x_y_z_from_node(n1,x1,y1,z1);
                    this->get_x_y_z_from_node(n2,x2,y2,z2);
                    this->get_x_y_z_from_node(n3,x3,y3,z3);

#ifdef P4_TO_P8

                    this->get_x_y_z_from_node(n4,x4,y4,z4);
                    this->get_x_y_z_from_node(n5,x5,y5,z5);
                    this->get_x_y_z_from_node(n6,x6,y6,z6);
                    this->get_x_y_z_from_node(n7,x7,y7,z7);
#endif



#ifdef P4_TO_P8
                    Cells_Vector_local_a[jj]=(1.00/8.00)*
                            (this->rhoA_local[n0]+this->rhoA_local[n1]+this->rhoA_local[n2]+this->rhoA_local[n3]
                             +this->rhoA_local[n4]+this->rhoA_local[n5]+this->rhoA_local[n6]+this->rhoA_local[n7]);

                    Cells_Vector_local_b[jj]=(1.00/8.00)*
                            (this->rhoB_local[n0]+this->rhoB_local[n1]+this->rhoB_local[n2]+this->rhoB_local[n3]
                             +this->rhoB_local[n4]+this->rhoB_local[n5]+this->rhoB_local[n6]+this->rhoB_local[n7]);
#else
                    Cells_Vector_local_a[jj]=(1.00/4.00)*
                            (this->rhoA_local[n0]+this->rhoA_local[n1]+this->rhoA_local[n2]+this->rhoA_local[n3] );

                    Cells_Vector_local_b[jj]=(1.00/4.00)*
                            (this->rhoB_local[n0]+this->rhoB_local[n1]+this->rhoB_local[n2]+this->rhoB_local[n3]);

#endif
                    jj++;

#ifdef P4_TO_P8
                    fprintf(outFileVec,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
                            x0,x1,x2,x3,x4,x5,x6,x7,
                            y0,y1,y2,y3,y4,y5,y6,y7,
                            z0,z1,z2,z3,z4,z5,z6,z7,
                            Cells_Vector_local_a[jj],Cells_Vector_local_b[jj]);
#else
                    fprintf(outFileVec,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n",
                            x0,x1,x3,x2,
                            y0,y1,y3,y2,
                            this->rhoA_local[n0],this->rhoA_local[n1],this->rhoA_local[n3],this->rhoA_local[n2],
                            this->rhoB_local[n0],this->rhoB_local[n1],this->rhoB_local[n3],this->rhoB_local[n2]);
#endif

                }
            }

            fclose(outFileVec);






            this->ierr=VecRestoreArray(myCellsVector4Paraview_a,&Cells_Vector_local_a); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(myCellsVector4Paraview_a); CHKERRXX(this->ierr);

            this->ierr=VecRestoreArray(myCellsVector4Paraview_b,&Cells_Vector_local_b); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(myCellsVector4Paraview_b); CHKERRXX(this->ierr);

            this->ierr=VecRestoreArray(this->rhoA,&this->rhoA_local);   CHKERRXX(this->ierr);
            this->ierr=VecRestoreArray(this->rhoB,&this->rhoB_local);   CHKERRXX(this->ierr);

        }
    }
    return 0;
}


int Diffusion::print_vec_by_cells(Vec *v2print,std::string file_name)
{

    if(this->mpi->mpisize==1)
    {

        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2DebugVec;
        //oss2DebugVec << this->convert2FullPath("Densities")<<"_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        oss2DebugVec << this->convert2FullPath(file_name)<<"_"<<mpi_rank<<".txt";
        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");



        int n_local_size;
        int n_global_size;
        this->ierr=VecGetLocalSize(*v2print,&n_local_size); CHKERRXX(this->ierr);
        this->ierr=VecGetSize(*v2print,&n_global_size); CHKERRXX(this->ierr);





        Vec myCellsVector4Paraview;
        VecCreateGhostCells(this->p4est,this->ghost,&myCellsVector4Paraview);
        PetscScalar *Cells_Vector_local;
        this->ierr=VecGetArray(myCellsVector4Paraview,&Cells_Vector_local); CHKERRXX(this->ierr);



        PetscInt icell_size,icell_local_size;
        this->ierr=VecGetSize(myCellsVector4Paraview,&icell_size);CHKERRXX(this->ierr);
        this->ierr= VecGetLocalSize(myCellsVector4Paraview,&icell_local_size); CHKERRXX(this->ierr);


        PetscScalar *v2print_local;

        this->ierr=VecGetArray(*v2print,&v2print_local); CHKERRXX(this->ierr);

        double x0,x1,x2,x3;
        double y0,y1,y2,y3;
        double z0,z1,z2,z3;

#ifdef P4_TO_P8
        double x4,x5,x6,x7;
        double y4,y5,y6,y7;
        double z4,z5,z6,z7;
#endif

        const p4est_locidx_t *q2n = this->nodes->local_nodes;
        int jj=0;
        for (p4est_locidx_t i = p4est->first_local_tree; i<=p4est->last_local_tree; i++)
        {
            p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);
            //p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];
            //std::cout<<"tree2->quadrants.elem_count"<<tree2->quadrants.elem_count<<std::endl;

            for(p4est_locidx_t j=0;j<tree2->quadrants.elem_count;j++)
            {

                /* since we want to access the local nodes, we need to 'jump' over intial
              * nonlocal nodes. Number of initial nonlocal nodes is given by
              * nodes->offset_owned_indeps
              */
                p4est_quadrant_t *quadrant = (p4est_quadrant_t*)sc_array_index(&tree2->quadrants, j);
                int n0,n1,n2,n3;

                int quad_idx=j+tree2->quadrants_offset;

#ifdef P4_TO_P8
                int n4,n5,n6,n7;
                n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
                n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
                n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
                n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];
                n4 =  q2n[ quad_idx*P4EST_CHILDREN + 4 ];
                n5 =  q2n[ quad_idx*P4EST_CHILDREN + 5 ];
                n6 =  q2n[ quad_idx*P4EST_CHILDREN + 6 ];
                n7 =  q2n[ quad_idx*P4EST_CHILDREN + 7 ];

#else
                n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
                n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
                n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
                n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];



#endif

                this->get_x_y_z_from_node(n0,x0,y0,z0);
                this->get_x_y_z_from_node(n1,x1,y1,z1);
                this->get_x_y_z_from_node(n2,x2,y2,z2);
                this->get_x_y_z_from_node(n3,x3,y3,z3);

#ifdef P4_TO_P8

                this->get_x_y_z_from_node(n4,x4,y4,z4);
                this->get_x_y_z_from_node(n5,x5,y5,z5);
                this->get_x_y_z_from_node(n6,x6,y6,z6);
                this->get_x_y_z_from_node(n7,x7,y7,z7);
#endif



#ifdef P4_TO_P8
                Cells_Vector_local[jj]=(1.00/8.00)*
                        (v2print_local[n0]+v2print_local[n1]+v2print_local[n2]+v2print_local[n3]
                         +v2print_local[n4]+v2print_local[n5]+v2print_local[n6]+v2print_local[n7]);


#else
                Cells_Vector_local[jj]=(1.00/4.00)*
                        (v2print_local[n0]+v2print_local[n1]+v2print_local[n2]+v2print_local[n3] );


#endif
                jj++;

#ifdef P4_TO_P8
                fprintf(outFileVec,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
                        x0,x1,x2,x3,x4,x5,x6,x7,
                        y0,y1,y2,y3,y4,y5,y6,y7,
                        z0,z1,z2,z3,z4,z5,z6,z7,
                        Cells_Vector_local);
#else
                fprintf(outFileVec,"%f %f %f %f %f %f %f %f %f %f %f %f \n",
                        x0,x1,x3,x2,
                        y0,y1,y3,y2,
                        v2print_local[n0],v2print_local[n1],v2print_local[n3],v2print_local[n2]);
#endif

            }
        }

        fclose(outFileVec);






        this->ierr=VecRestoreArray(myCellsVector4Paraview,&Cells_Vector_local); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(myCellsVector4Paraview); CHKERRXX(this->ierr);


        this->ierr=VecRestoreArray(*v2print,&v2print_local);   CHKERRXX(this->ierr);



    }
    return 0;
}


int Diffusion::get_x_y_z_from_node(int n, double &x, double &y, double &z)
{z=0;
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;
    p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];



    double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

    x = node_x_fr_i(node) + tree_xmin;
    y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
    z = node_z_fr_k(node) + tree_zmin;
#endif

}

int Diffusion::updateDiffusionMatrices()
{

    if(this->myMeanFieldPlan->robin_bc)
        this->createDiffusionMatricesRobin();

    std::cout<<" update diffusion matrices 1 "<<this->mpi->mpirank<<std::endl;
    Vec Mii,dii;
    VecDuplicate(this->phi,&Mii);
    VecDuplicate(Mii,&dii);
    if( (this->myNumericalScheme==Diffusion::finite_difference_implicit
         || this->myNumericalScheme==Diffusion::finite_difference_explicit
         || this->myNumericalScheme==Diffusion::splitting_finite_difference_implicit)
            && this->my_casl_diffusion_method==Diffusion::periodic_crank_nicholson)
    {
        PetscInt number_of_local_rows;
        PetscInt number_of_local_columns;
        MatGetLocalSize(this->solver->A,&number_of_local_rows,&number_of_local_columns);
        //VecSetSizes(Mii,number_of_local_rows,PETSC_DECIDE);
        MatGetDiagonal(this->solver->A,Mii);
        VecScale(Mii,-this->dt/2.00);

    }
    std::cout<<" update diffusion matrices 2 "<<this->mpi->mpirank<<std::endl;
    if(forward_stage && first_stage)
    {
        VecDuplicate(this->phi,&this->w_t);
        std::cout<<" update diffusion matrices w1 "<<this->mpi->mpirank<<std::endl;
    }



    //-----------------------------------Forward Matrices---------------------------------//

    //-----------t<f----------//

    if(forward_stage && first_stage)
    {
        std::cout<<" update diffusion matrices 3 "<<this->mpi->mpirank<<std::endl;

        this->printDiffusionArrayFromVector(&this->wp,"wp");
        this->printDiffusionArrayFromVector(&this->wm,"wm");

        PetscInt size_w_t,size_w_m,size_w_p;
        VecGetLocalSize(this->w_t,&size_w_t);
        VecGetLocalSize(this->wm,&size_w_m);
        VecGetLocalSize(this->wp,&size_w_p);


        this->ierr=VecSet(this->w_t,0);    CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->w_t,1.00,this->wp);  CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->w_t,-1.00,this->wm); CHKERRXX(this->ierr);

        if(this->advance_fields_scft_advance_mask_level_set
                || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            this->ierr=VecShift(this->w_t,-this->wp_average); CHKERRXX(this->ierr);

        if(this->myNumericalScheme==splitting_spectral_adaptive)
        {
            this->ierr=VecGhostUpdateBegin(this->w_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->w_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyBegin(this->w_t); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyEnd(this->w_t); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
            this->interpolate_w_t_on_uniform_grid();
#else
            this->interpolate_w_t_on_uniform_grid_2D();
#endif
        }

        VecScale(this->w_t,this->dt/2.00);



        if( !this->strang_splitting)
        {
            // Update M1
            VecCopy(Mii,dii);
            VecShift(dii,1.00);
            VecAXPY(dii,-1.00,this->w_t);
            MatDiagonalSet(this->M1,dii,INSERT_VALUES);

            // Update M2 t<f
            VecCopy(Mii,dii);
            VecScale(dii,-1.00);
            VecShift(dii,1.00);
            VecAXPY(dii,1.00,this->w_t);
            MatDiagonalSet(this->M2,dii,INSERT_VALUES);
        }



        if(this->strang_splitting)

        {
            std::cout<<" update diffusion matrices 4 "<<this->mpi->mpirank<<std::endl;
            VecScale(this->w_t,-1);
            VecExp(this->w_t);
            // this->printDiffusionVector(&this->w_t,"wt");
        }
    }

    //----------------t>f----------//
    if(forward_stage && !first_stage)
    {
        std::cout<<" update diffusion matrices 5 "<<this->mpi->mpirank<<std::endl;
        VecSet(this->w_t,0);
        VecAXPY(this->w_t,1.00,this->wp);
        VecAXPY(this->w_t,1.00,this->wm);

        if(this->advance_fields_scft_advance_mask_level_set
                || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            this->ierr=VecShift(this->w_t,-this->wp_average); CHKERRXX(this->ierr);


        if(this->myNumericalScheme==splitting_spectral_adaptive)
        {
            this->ierr=VecGhostUpdateBegin(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecGhostUpdateEnd(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecAssemblyBegin(this->w_t);
            this->ierr=VecAssemblyEnd(this->w_t);
#ifdef P4_TO_P8
            this->interpolate_w_t_on_uniform_grid();
#else
            this->interpolate_w_t_on_uniform_grid_2D();
#endif
        }

        VecScale(this->w_t,this->dt/2.00);

        if( !this->strang_splitting)
        {
            // Update M1
            VecCopy(Mii,dii);
            VecShift(dii,1.00);
            VecAXPY(dii,-1.00,this->w_t);
            MatDiagonalSet(this->M1,dii,INSERT_VALUES);

            // Update M2 t<f
            VecCopy(Mii,dii);
            VecScale(dii,-1.00);
            VecShift(dii,1.00);
            VecAXPY(dii,1.00,this->w_t);
            MatDiagonalSet(this->M2,dii,INSERT_VALUES);
        }




        if(this->strang_splitting)

        {
            std::cout<<" update diffusion matrices 6 "<<this->mpi->mpirank<<std::endl;
            VecScale(this->w_t,-1);
            VecExp(this->w_t);
            this->ierr=VecGhostUpdateBegin(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecGhostUpdateEnd(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecAssemblyBegin(this->w_t);
            this->ierr=VecAssemblyEnd(this->w_t);
            // this->printDiffusionVector(&this->w_t,"wt");
        }
    }

    //-----------------------------------Backward Matrices---------------------------------//

    //-----------0<t<1-f----------//

    if(!forward_stage && first_stage)
    {
        std::cout<<" update diffusion matrices 7 "<<this->mpi->mpirank<<std::endl;
        VecSet(this->w_t,0);
        VecAXPY(this->w_t,1.00,this->wp);
        VecAXPY(this->w_t,1.00,this->wm);

        if(this->advance_fields_scft_advance_mask_level_set
                || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            this->ierr=VecShift(this->w_t,-this->wp_average); CHKERRXX(this->ierr);


        if(this->myNumericalScheme==splitting_spectral_adaptive)
        {
            this->ierr=VecGhostUpdateBegin(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecGhostUpdateEnd(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecAssemblyBegin(this->w_t);
            this->ierr=VecAssemblyEnd(this->w_t);
#ifdef P4_TO_P8
            this->interpolate_w_t_on_uniform_grid();
#else
            this->interpolate_w_t_on_uniform_grid_2D();
#endif
        }

        VecScale(this->w_t,this->dt/2.00);

        if( !this->strang_splitting)
        {
            // Update M1
            VecCopy(Mii,dii);
            VecShift(dii,1.00);
            VecAXPY(dii,-1.00,this->w_t);
            MatDiagonalSet(this->M1,dii,INSERT_VALUES);

            // Update M2 t<f
            VecCopy(Mii,dii);
            VecScale(dii,-1.00);
            VecShift(dii,1.00);
            VecAXPY(dii,1.00,this->w_t);
            MatDiagonalSet(this->M2,dii,INSERT_VALUES);
        }

        if(this->strang_splitting)
        {
            std::cout<<" update diffusion matrices 8 "<<this->mpi->mpirank<<std::endl;
            VecScale(this->w_t,-1);
            VecExp(this->w_t);
            //this->printDiffusionVector(&this->w_t,"wt");
        }
    }

    //----------------t>1-f----------//
    if(!forward_stage && !first_stage)
    {
        std::cout<<" update diffusion matrices 8 "<<this->mpi->mpirank<<std::endl;
        VecSet(this->w_t,0);
        VecAXPY(this->w_t,1.00,this->wp);
        VecAXPY(this->w_t,-1.00,this->wm);

        if(this->advance_fields_scft_advance_mask_level_set
                || this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
            this->ierr=VecShift(this->w_t,-this->wp_average); CHKERRXX(this->ierr);


        if(this->myNumericalScheme==splitting_spectral_adaptive)
        {
            this->ierr=VecGhostUpdateBegin(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecGhostUpdateEnd(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
            this->ierr=VecAssemblyBegin(this->w_t);
            this->ierr=VecAssemblyEnd(this->w_t);
#ifdef P4_TO_P8
            this->interpolate_w_t_on_uniform_grid();
#else
            this->interpolate_w_t_on_uniform_grid_2D();
#endif
        }

        VecScale(this->w_t,this->dt/2.00);

        if( !this->strang_splitting)
        {
            // Update M1
            VecCopy(Mii,dii);
            VecShift(dii,1.00);
            VecAXPY(dii,-1.00,this->w_t);
            MatDiagonalSet(this->M1,dii,INSERT_VALUES);

            // Update M2 t<f
            VecCopy(Mii,dii);
            VecScale(dii,-1.00);
            VecShift(dii,1.00);
            VecAXPY(dii,1.00,this->w_t);
            MatDiagonalSet(this->M2,dii,INSERT_VALUES);
        }


        if(this->strang_splitting)
        {
            std::cout<<" update diffusion matrices 9 "<<this->mpi->mpirank<<std::endl;
            VecScale(this->w_t,-1);
            VecExp(this->w_t);
            //this->printDiffusionVector(&this->w_t,"wt");
        }
    }
    std::cout<<" update diffusion matrices 10 "<<this->mpi->mpirank<<std::endl;

    this->ierr=VecGhostUpdateBegin(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
    this->ierr=VecGhostUpdateEnd(this->w_t,INSERT_VALUES,SCATTER_FORWARD);
    this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_update");
    this->printDiffusionArrayFromVector(&this->w_t,"w_t_update");

    //this->ierr=VecAssemblyBegin(this->w_t);
    //this->ierr=VecAssemblyEnd(this->w_t);


    if(this->print_diffusion_matrices && !this->strang_splitting)
    {
        this->printDiffusionMatrices(&this->M1,"M1W",this->i_mean_field);
        this->printDiffusionMatrices(&this->M2,"M2W",this->i_mean_field);
        //this->printDiffusionMatrices(&this->solver->A,"M",this->i_mean_field);
    }

    VecDestroy(Mii); VecDestroy(dii);

    return 0;


}

int Diffusion::printDiffusionMatrices(Mat *M2Print,std::string file_name_str,int i_mean_field)
{
    //if(this->print_diffusion_matrices)
    {
        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        int matrix_global_rows_number,matrix_global_columns_number;
        MatGetSize(*M2Print,&matrix_global_rows_number,&matrix_global_columns_number);


        if(mpi_size>-1)
        {

            // http://acts.nersc.gov/petsc/example1/ex2.c.html
            //                        PetscInt m_global,n_global;
            //                        PetscInt m_local,n_local;
            //                        PetscInt global_first_row,global_last_row;
            //                        MatGetSize(*M2Print,&m_global,&n_global);
            //                        MatGetLocalSize(*M2Print,&m_local,&n_local);
            //                        MatGetOwnershipRange(*M2Print,&global_first_row,&global_last_row);

            //                        std::cout<<"rank m_global n_global m_local n_local global_first_row global_last_row  "
            //                                <<mpi_rank<<" "<<m_global<<" "<<n_global<<" "<<m_local<<" "<<n_local<<" "
            //                               << global_first_row<<" "<< global_last_row<<" "
            //                               <<std::endl;

            //                        PetscInt number_of_rows=m_local;
            //                        PetscInt *global_indices_of_rows=new PetscInt[number_of_rows];
            //                        PetscInt number_of_columns=n_global;
            //                        PetscInt   *global_indices_of_columns=new PetscInt[number_of_columns];
            //                        PetscScalar *store_the_values=new PetscScalar[number_of_rows*number_of_columns];
            //                        for(int i=0;i<number_of_rows;i++)
            //                        {
            //                            global_indices_of_rows[i]=global_first_row+i;
            //                            //global_indices_of_rows[i]=i;
            //                        }
            //                        for(int j=0;j<number_of_columns;j++)
            //                        {
            //                            global_indices_of_columns[j]=j;//this->mpi->mpirank*n_global/this->mpi->mpisize+  j;
            //                            //   global_indices_of_columns[j]=  j;
            //                        }
            //                        std::stringstream oss2Debugix;
            //                        std::string mystr2Debugix;
            //                        oss2Debugix << this->convert2FullPath("IXmatrix_M2Print")<<"_"<<file_name_str<<"_"<<mpi_rank<<".txt";
            //                        mystr2Debugix=oss2Debugix.str();
            //                        FILE *outFileix;
            //                        outFileix=fopen(mystr2Debugix.c_str(),"w");
            //                        for(int i=0;i<number_of_rows;i++)
            //                        {
            //                            for(int j=0;j<number_of_columns;j++)
            //                            {
            //                                fprintf(outFileix,"%d %d %d %d %d\n",mpi_rank,i, j, global_indices_of_rows[i], global_indices_of_columns[j]);
            //                            }
            //                        }
            //                        fclose(outFileix);
            //                        std::cout<<"rank  "
            //                                <<mpi_rank
            //                               <<std::endl;
            //                        this->ierr= MatGetValues(*M2Print,number_of_rows,global_indices_of_rows,number_of_columns,global_indices_of_columns,store_the_values); CHKERRXX(this->ierr);
            //                        std::cout<<"rank  "
            //                                <<mpi_rank
            //                               <<std::endl;
            std::stringstream oss2Debug;
            std::string mystr2Debug;
            oss2Debug << this->convert2FullPath("M2Print")<<"_"<<file_name_str<<"_"<<mpi_rank<<"_"<<i_mean_field<<".txt";
            mystr2Debug=oss2Debug.str();
            //                    FILE *outFile;
            //                    outFile=fopen(mystr2Debug.c_str(),"w");
            //                    for(int i=0;i<number_of_rows;i++)
            //                    {
            //                        for(int j=0;j<number_of_columns;j++)
            //                        {
            //                            fprintf(outFile,"%d %d %d %d %d %f\n",mpi_rank,i, j, global_indices_of_rows[i], global_indices_of_columns[j],store_the_values[i*number_of_columns+j]);
            //                        }
            //                    }
            //                    fclose(outFile);

            PetscViewer lab;
            this->ierr=PetscViewerASCIIOpen(MPI_COMM_WORLD,mystr2Debug.c_str(),&lab); CHKERRXX(this->ierr);
            this->ierr=PetscViewerSetFormat(lab,PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(this->ierr);
            this->ierr=MatView(*M2Print,lab); CHKERRXX(this->ierr);
        }
        //    else
        //    {
        //        //http://acts.nersc.gov/petsc/example1/ex2.c.html
        //        PetscInt m_global,n_global;
        //        PetscInt m_local,n_local;
        //        PetscInt global_first_row,global_last_row;
        //        MatGetSize(*M2Print,&m_global,&n_global);
        //        MatGetLocalSize(*M2Print,&m_local,&n_local);
        //        MatGetOwnershipRange(*M2Print,&global_first_row,&global_last_row);

        //        std::cout<<"rank m_global n_global m_local n_local global_first_row global_last_row  "
        //                <<mpi_rank<<" "<<m_global<<" "<<n_global<<" "<<m_local<<" "<<n_local<<" "
        //               << global_first_row<<" "<< global_last_row<<" "
        //               <<std::endl;

        //        PetscInt number_of_rows=m_local;
        //        PetscInt *global_indices_of_rows=new PetscInt[number_of_rows];
        //        PetscInt number_of_columns=n_global;
        //        PetscInt   *global_indices_of_columns=new PetscInt[number_of_columns];
        //        PetscScalar *store_the_values=new PetscScalar[number_of_rows*number_of_columns];

        //        MatGetRowIJ(M2Print,0,PETSC_FALSE,PETSC_FALSE,)

        //        for(int i=0;i<number_of_rows;i++)
        //        {
        //            global_indices_of_rows[i]=global_first_row+i;
        //            //global_indices_of_rows[i]=i;
        //        }
        //        for(int j=0;j<number_of_columns;j++)
        //        {
        //            global_indices_of_columns[j]=j;//this->mpi->mpirank*n_global/this->mpi->mpisize+  j;
        //            //   global_indices_of_columns[j]=  j;
        //        }
        //        std::stringstream oss2Debugix;
        //        std::string mystr2Debugix;
        //        oss2Debugix << this->convert2FullPath("IXmatrix_M2Print")<<"_"<<file_name_str<<"_"<<mpi_rank<<".txt";
        //        mystr2Debugix=oss2Debugix.str();
        //        FILE *outFileix;
        //        outFileix=fopen(mystr2Debugix.c_str(),"w");
        //        for(int i=0;i<number_of_rows;i++)
        //        {
        //            for(int j=0;j<number_of_columns;j++)
        //            {
        //                fprintf(outFileix,"%d %d %d %d %d\n",mpi_rank,i, j, global_indices_of_rows[i], global_indices_of_columns[j]);
        //            }
        //        }
        //        fclose(outFileix);
        //        std::cout<<"rank  "
        //                <<mpi_rank
        //               <<std::endl;
        //        MatGetValues(*M2Print,number_of_rows,global_indices_of_rows,number_of_columns,global_indices_of_columns,store_the_values);
        //        std::cout<<"rank  "
        //                <<mpi_rank
        //               <<std::endl;
        //        std::stringstream oss2Debug;
        //        std::string mystr2Debug;
        //        oss2Debug << this->convert2FullPath("M2Print")<<"_"<<file_name_str<<"_"<<mpi_rank<<".txt";
        //        mystr2Debug=oss2Debug.str();
        //        FILE *outFile;
        //        outFile=fopen(mystr2Debug.c_str(),"w");
        //        for(int i=0;i<number_of_rows;i++)
        //        {
        //            for(int j=0;j<number_of_columns;j++)
        //            {
        //                fprintf(outFile,"%d %d %d %d %d %f\n",mpi_rank,i, j, global_indices_of_rows[i], global_indices_of_columns[j],store_the_values[i*number_of_columns+j]);
        //            }
        //        }
        //        fclose(outFile);
        //    }
    }
    return 0;
}



int Diffusion::allocateMemory2History(PetscBool minus_one_iteration)
{
    this->N_t=this->N_iterations+1;
    int mpi_size;
    int mpi_rank;

    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
    VecGetLocalSize(this->phi,&this->n_local_size_sol);
    VecGetSize(this->phi,&this->n_global_size_sol);
    int global_size=this->N_t*this->n_global_size_sol;
    this->local_size_sol=this->n_local_size_sol*this->N_t;

    if(!minus_one_iteration)
    {
        this->sol_history=new double[this->local_size_sol];

        for(int i=0;i<this->n_local_size_sol;i++)
            this->sol_history[i]=1.00;
    }


    std::cout<<"BooKeeping  " <<this->mpi->mpirank<<" "<<this->n_local_size_sol<<" "<<this->nodes->num_owned_indeps<<" "<<this->nodes->num_owned_shared<<" "<<this->nodes->indep_nodes.elem_count<<std::endl;

    return 0;

}

int Diffusion::sendSol2HitoryDataBase()
{
    if(this->print_my_current_debugger)
    {
        this->printDiffusionVector(&this->sol_t,"sol_t_before");
        this->printDiffusionVector(&this->sol_tp1,"sol_tp1_before");
        this->printDiffusionVector(&this->b_t,"b_t_before");
    }
    int n_local_size;
    VecGetLocalSize(this->sol_tp1,&n_local_size);
    PetscScalar *a;
    VecGetArray(this->sol_tp1,&a);

    for(int i=0;i<n_local_size;i++)
        this->sol_history[this->it*n_local_size+i]=a[i];

    VecRestoreArray(this->sol_tp1,&a);
    if(this->print_my_current_debugger)
    {
        this->printDiffusionVector(&this->sol_t,"sol_t_after");
        this->printDiffusionVector(&this->sol_tp1,"sol_tp1_after");
        this->printDiffusionVector(&this->b_t,"b_t_after");
    }
    return 0;
}

int Diffusion::sendFirstIteration2HitoryDataBase()
{
    if(this->print_my_current_debugger)
    {
        this->printDiffusionVector(&this->sol_t,"sol_t_before");

    }
    int n_local_size;
    VecGetLocalSize(this->sol_t,&n_local_size);
    PetscScalar *a;
    VecGetArray(this->sol_t,&a);

    for(int i=0;i<n_local_size;i++)
        this->sol_history[this->it*n_local_size+i]=a[i];

    VecRestoreArray(this->sol_t,&a);
    if(this->print_my_current_debugger)
    {
        this->printDiffusionVector(&this->sol_t,"sol_t_after");

    }
    return 0;
}


int Diffusion::createDiffusionSolution()
{
    this->ierr=VecDuplicate(this->phi,&this->sol_t);  CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->sol_tp1);  CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->b_t);  CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->r_t);  CHKERRXX(this->ierr);

    return 0;
}


int Diffusion::createKSPContext()
{

    this->preconditioner_algo_watch.start("preconditionning");
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                            Create the linear solver and set various options
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Create linear solver context
              */
    this->ierr = KSPCreate(PETSC_COMM_WORLD,&this->myKsp);CHKERRXX(this->ierr);

    /*
                 Set operators. Here the matrix that defines the linear system
                 also serves as the preconditioning matrix.
              */
    if(!this->extractSubDiffusionMatricesBool)
    { this->ierr = KSPSetOperators(this->myKsp,this->M2,this->M2,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);}
    else
    {   this->ierr = KSPSetOperators(this->myKsp,this->subM1,this->subM2,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);}

    /*
                 Set linear solver defaults for this problem (optional).
                 - By extracting the KSP and PC contexts from the KSP context,
                   we can then directly call any KSP and PC routines to set
                   various options.
                 - The following four statements are optional; all of these
                   parameters could alternatively be specified at runtime via
                   KSPSetFromOptions();
              */


    this->ierr = KSPGetPC(this->myKsp,&this->pc);CHKERRXX(this->ierr);

    switch(this->my_matrix_solver_algo)
    {
    case Diffusion::richardson_matrix_algo:
    {
        double damping_factor=0.1;
        this->ierr=KSPSetType(this->myKsp,KSPRICHARDSON); CHKERRXX(this->ierr);
        this->ierr=KSPRichardsonSetScale(this->myKsp,damping_factor); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::chebyshev_matrix_algo:
    {
        double lambda_min=0.01,lambda_max=100;
        this->ierr=KSPSetType(this->myKsp,KSPCHEBYSHEV); CHKERRXX(this->ierr);
        this->ierr=KSPChebyshevSetEigenvalues(this->myKsp,lambda_max,lambda_min); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::conjugate_gradient_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPCG); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::biConjugate_gradient_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPBICG); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::generalized_minimal_residual_matrix_algo:
    {
        int max_steps=30;
        this->ierr=KSPSetType(this->myKsp,KSPGMRES); CHKERRXX(this->ierr);
        this->ierr=KSPGMRESSetRestart(this->myKsp,max_steps); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::flexibe_generalized_minimal_residual_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPFGMRES); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::deflated_generalized_minimal_residual_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPDGMRES); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::generalized_conjugate_residual_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPGCR); CHKERRXX(this->ierr);
        break;
    }

    case Diffusion::bcgstab_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPBCGS); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::conjugate_gradient_squared_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPCGS); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::transpose_free_quasi_minimal_residual_matrix_algo_1:
    {
        this->ierr=KSPSetType(this->myKsp,KSPTFQMR); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::transpose_free_quasi_minimal_residual_matrix_algo_2:
    {
        this->ierr=KSPSetType(this->myKsp,KSPTCQMR); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::conjugate_residual_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPCR); CHKERRXX(this->ierr);
        break;
    }
    case Diffusion::least_squares_method_matrix_algo:
    {
        this->ierr=KSPSetType(this->myKsp,KSPLSQR); CHKERRXX(this->ierr);
        break;
    }
    }
    switch(this->my_preconditioner_algo)
    {
    case Diffusion::jacobi_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCJACOBI);CHKERRXX(this->ierr);
        break;


    case Diffusion::block_jacobi_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCBJACOBI);CHKERRXX(this->ierr);
        break;
    case Diffusion::sor_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCSOR);CHKERRXX(this->ierr);
        break;
    case Diffusion::sor_eisenstat_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCEISENSTAT);CHKERRXX(this->ierr);
        break;
    case Diffusion::incomplete_cholesky_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCICC);CHKERRXX(this->ierr);
        break;
    case Diffusion::incomplete_LU_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCILU);CHKERRXX(this->ierr);
        break;
    case Diffusion::additive_schwartz_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCASM);CHKERRXX(this->ierr);
        break;
    case Diffusion::algebraic_multi_grid_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCHYPRE);CHKERRXX(this->ierr);
        break;
    case Diffusion::linear_solver_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCKSP);CHKERRXX(this->ierr);
        break;
    case Diffusion::lu_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCLU);CHKERRXX(this->ierr);
        break;
    case Diffusion::cholesky_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCCHOLESKY);CHKERRXX(this->ierr);
        break;
    case Diffusion::no_preconditionning_preconditioner_algo:
        this->ierr = PCSetType(this->pc,PCNONE);CHKERRXX(this->ierr);
        break;
    }
    this->ierr = KSPSetTolerances(this->myKsp,1.e-32,1.e-12,5,1000);CHKERRXX(this->ierr);

    //this->ierr =  KSPSetTolerances(this->myKsp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRXX(this->ierr);

    /*
                Set runtime options, e.g.,
                    -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
                These options will override those specified above as long as
                KSPSetFromOptions() is called _after_ any other customization
                routines.
              */
    //this->ierr = KSPSetFromOptions(this->myKsp);

    CHKERRXX(this->ierr);
    this->preconditioner_algo_watch.stop();
    return 0;
}

int Diffusion::evolveDiffusionIteration()
{
    switch(this->my_casl_diffusion_method)
    {
    case Diffusion::periodic_crank_nicholson:
    {
        switch(this->myNumericalScheme)
        {
        case Diffusion::finite_difference_explicit:
            this->evolve_Equation_Explicit();
            break;
        case Diffusion::finite_difference_implicit:
            this->evolve_Equation_Implicit();
            break;
        case Diffusion::splitting_finite_difference_explicit:
            this->evolve_Equation_Finite_Difference_Splitting_Explicit();
            break;
        case Diffusion::splitting_finite_difference_implicit_scaled:
            this->evolve_Equation_Finite_Difference_Splitting_Implicit_Scaled();
            break;
        case Diffusion::splitting_finite_difference_implicit:
            this->evolve_Equation_Finite_Difference_Splitting_Implicit();
            break;

        case Diffusion::splitting_spectral:
            this->evolve_Equation_Spectral_Splitting();
            break;
        case Diffusion::splitting_spectral_adaptive:
        {
#ifdef P4_TO_P8
            if(!this->real_to_complex_fftw_strategy)
                this->evolve_Equation_Spectral_Splitting_AMR();
            else
                this->evolve_Equation_Spectral_Splitting_AMR_r2r();
#else
            if(!this->real_to_complex_fftw_strategy)
                this->evolve_Equation_Spectral_Splitting_AMR_2D();
            else
                this->evolve_Equation_Spectral_Splitting_AMR_2D_r2r();
#endif
        }

            break;


        default:
            throw(" diffusion solver mode must be set in order to run a scft simulation");

            break;

        }
        break;
    }
    case Diffusion::neuman_backward_euler:
    {
        if(!this->fast_algo)
        {
            // stopwatch
            parStopWatch slow_algo_watch;
            slow_algo_watch.start("slow algo: reconstruct the rhs at each iteration");
            this->evolve_Equation_Neuman_Backward_Euler();
            slow_algo_watch.stop(); slow_algo_watch.read_duration();
        }
        else
        {
            parStopWatch fast_algo_watch;
            fast_algo_watch.start("fast algo: do not reconstruct the rhs at each iteration\n Store the geometric info and resue it");
            this->evolve_Equation_Finite_Difference_Splitting_Implicit_Neuman_Backward_Euler_Fast();
            fast_algo_watch.stop(); fast_algo_watch.read_duration();
        }
        break;
    }
    case Diffusion::neuman_crank_nicholson:
    {
        if(this->fast_algo)
            this->evolve_Equation_Finite_Difference_Splitting_Implicit();
        else
            throw(" neuman cn has not been implemented with the slow algo");
        break;
    }
    case Diffusion::dirichlet_crank_nicholson:
    {
        if(this->fast_algo)
            this->evolve_Equation_Finite_Difference_Splitting_Implicit();
        else
            throw(" dirichlet cn has not been implemented with the slow algo");
        break;
    }
    default:
        throw(" casl_method for diffusion solving must be set for running an scft simulation ");
    }
    return 0;

}


int Diffusion::evolve_Equation_Neuman_Backward_Euler()
{

    if(this->it==0)
    {
        VecSet(this->sol_t,1.00);
        //this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M_NEUMAN",0);

    }

    if(this->it>0)
    {
        VecCopy(this->sol_tp1,this->sol_t);
    }

    this->it++;
    this->ierr=VecPointwiseMult(this->sol_t,this->sol_t,this->w_t); CHKERRXX(this->ierr);
    if(this->printIntemediateSteps)
        this->printDiffusionVector(&this->sol_t,"sol_tp1");



    this->myNeumanPoissonSolverNodeBase->set_rhs(this->sol_t);

    //this->printDiffusionVector(&this->myNeumanPoissonSolverNodeBase->rhs_,"rhs_before_solve");

    this->myNeumanPoissonSolverNodeBase->solve(this->sol_tp1);

    //this->printDiffusionVector(&this->myNeumanPoissonSolverNodeBase->rhs_,"rhs_after_solve");

    if(this->it==1)// && this->printIntemediateSteps)
    {

        this->printDiffusionMatrices(&this->myNeumanPoissonSolverNodeBase->A,"M_NEUMAN_Slow",0);

    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
       Check the error
    */


    PetscInt its; PetscScalar b_norm;
    this->ierr=MatMult(this->myNeumanPoissonSolverNodeBase->A,this->sol_tp1,this->r_t); CHKERRXX(this->ierr);
    this->ierr=VecAXPY(this->r_t,-1,this->sol_t); CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->r_t,NORM_2,&this->norm);CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->sol_t,NORM_2,&b_norm);CHKERRXX(this->ierr);
    this->ierr = KSPGetIterationNumber(this->myNeumanPoissonSolverNodeBase->ksp,&its);CHKERRXX(this->ierr);

    PetscScalar relative_residual=this->norm/b_norm;

    std::cout<<"my ksp view "<<std::endl;
    std::cout<<this->it<<" n_it "<<its<<" absolute residual "<<this->norm<<" relative residual "<<relative_residual<<std::endl;



    if(this->printIntemediateSteps)
        this->printDiffusionVector(&this->sol_tp1,"sol_tp2");
    this->ierr=VecPointwiseMult(this->sol_tp1,this->sol_tp1,this->w_t);CHKERRXX(this->ierr);
    if(this->printIntemediateSteps)
        this->printDiffusionVector(&this->sol_tp1,"sol_tp3");
    return 0;
}


int Diffusion::evolve_Equation_Finite_Difference_Splitting_Implicit_Neuman_Backward_Euler_Fast()
{



    if(this->it==0)
    {
        VecSet(this->sol_t,1.00);

    }
    if(this->it>0)
        VecCopy(this->sol_tp1,this->sol_t);



    this->it++;


    if(this->it==1)
    {
        this->printDiffusionMatrices(&this->M2,"M2_Neuman_Fast",this->i_mean_field);
        this->printDiffusionVector(&this->myNeumanPoissonSolverNodeBase->add_,"neuman_add");
    }

    PetscInt local_size_b,global_size_b;
    VecGetSize(this->b_t,&global_size_b);
    VecGetLocalSize(this->b_t,&local_size_b);
    std::cout<<"1 "<<global_size_b<<" "<<local_size_b<<std::endl;
    this->ierr=VecPointwiseMult(this->sol_t,this->sol_t,this->w_t); CHKERRXX(this->ierr);
    this->ierr=VecPointwiseMult(this->b_t,this->myNeumanPoissonSolverNodeBase->add_,this->sol_t); CHKERRXX(this->ierr);

    //  this->printDiffusionVector(&this->b_t,"b_t");
    VecGetSize(this->b_t,&global_size_b);
    VecGetLocalSize(this->b_t,&local_size_b);
    std::cout<<"2 "<<global_size_b<<" "<<local_size_b<<std::endl;



    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                  Solve the linear system
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Solve linear system
              */

    this->ierr = KSPSolve(this->myKsp,this->b_t,this->sol_tp1);CHKERRXX(this->ierr);


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
       Check the error
    */


    PetscInt its; PetscScalar b_norm;
    this->ierr=MatMult(this->M2,this->sol_tp1,this->r_t); CHKERRXX(this->ierr);
    this->ierr=VecAXPY(this->r_t,-1,this->b_t); CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->r_t,NORM_2,&this->norm);CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->b_t,NORM_2,&b_norm);CHKERRXX(this->ierr);
    this->ierr = KSPGetIterationNumber(this->myKsp,&its);CHKERRXX(this->ierr);


    PetscScalar relative_residual=this->norm/b_norm;

    std::cout<<"my ksp view "<<std::endl;
    std::cout<<this->it<<" n_it "<<its<<" absolute residual "<<this->norm<<" relative residual "<<relative_residual<<std::endl;
    //  this->printDiffusionVector(&this->sol_tp1,"sol_tp1_before");
    this->ierr=VecPointwiseMult(this->sol_tp1,this->sol_tp1,this->w_t);CHKERRXX(this->ierr);
    // this->printDiffusionVector(&this->sol_tp1,"sol_tp1");
    return 0;
}


int Diffusion::evolve_Equation_Implicit()
{


    if(this->it==0)
        VecSet(this->sol_t,1.00);
    if(this->it>0)
        VecCopy(this->sol_tp1,this->sol_t);



    this->it++;


    PetscInt local_size_b,global_size_b;
    VecGetSize(this->b_t,&global_size_b);
    VecGetLocalSize(this->b_t,&local_size_b);
    std::cout<<"1 "<<global_size_b<<" "<<local_size_b<<std::endl;
    this->ierr=MatMult(this->M1,this->sol_t,this->b_t); CHKERRXX(this->ierr);
    VecGetSize(this->b_t,&global_size_b);
    VecGetLocalSize(this->b_t,&local_size_b);
    std::cout<<"2 "<<global_size_b<<" "<<local_size_b<<std::endl;



    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                  Solve the linear system
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Solve linear system
              */

    this->ierr = KSPSolve(this->myKsp,this->b_t,this->sol_tp1);CHKERRXX(this->ierr);



    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
       Check the error
    */


    PetscInt its; PetscScalar b_norm;
    this->ierr=MatMult(this->M2,this->sol_tp1,this->r_t); CHKERRXX(this->ierr);
    this->ierr=VecAXPY(this->r_t,-1,this->b_t); CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->r_t,NORM_2,&this->norm);CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->b_t,NORM_2,&b_norm);CHKERRXX(this->ierr);
    this->ierr = KSPGetIterationNumber(this->myKsp,&its);CHKERRXX(this->ierr);

    PetscScalar relative_residual=this->norm/b_norm;

    std::cout<<"my ksp view "<<std::endl;
    std::cout<<this->it<<" n_it "<<its<<" absolute residual "<<this->norm<<" relative residual "<<relative_residual<<std::endl;

    return 0;

}

void Diffusion::solve_linear_system_slow()
{

    PetscReal abstol_in=1.e-12;
    PetscReal dtol_in=5;
    PetscInt maxits_in=10000;
    this->ierr = KSPSetTolerances(this->myKsp,this->rtol_in,abstol_in,dtol_in,maxits_in);CHKERRXX(this->ierr);
    this->ierr=KSPSetInitialGuessNonzero(this->myKsp,PETSC_TRUE);

    //    if(this->it>0)
    //        this->ierr = KSPSetOperators(this->myKsp,this->M2,this->M2,SAME_PRECONDITIONER);CHKERRXX(ierr);


    //    PetscLogEvent event;
    //    PetscLogEventRegister("Solve",PETSC_VIEWER_CLASSID,&event);
    //    PetscLogEventBegin(event,0,0,0,0);
    //    PetscLogEvent event2;
    //    PetscLogEventRegister("Solve2",PETSC_VIEWER_CLASSID,&event2);
    //    PetscLogEventBegin(event2,0,0,0,0);


    this->ierr = KSPSolve(this->myKsp,this->b_t,this->sol_tp1);CHKERRXX(this->ierr);

    //    PetscLogEventEnd(event,0,0,0,0);
    //    PetscLogEventEnd(event2,0,0,0,0);

    //    PetscViewer lab;
    //    this->ierr=PetscViewerASCIIOpen(MPI_COMM_WORLD,"/Users/gaddielouaknin/p4estLocal3/my_first_log2",&lab); CHKERRXX(this->ierr);
    //    // this->ierr=PetscViewerSetFormat(lab,PETSC_VIEWER_ASCII_INDEX); CHKERRXX(this->ierr);

    //    this->ierr=PetscLogView(lab); CHKERRXX(this->ierr);
    //    this->ierr=PetscViewerDestroy(lab);


    KSPConvergedReason myReason;
    this->ierr=KSPGetConvergedReason(this->myKsp,&myReason);
    //    std::cout<<myReason<<std::endl;
    //    switch ((int)myReason)
    //    {
    //    case 1:
    //        std::cout<<"KSP_CONVERGED_RTOL_NORMAL";
    //        break;
    //    case 9:
    //        std::cout<<"KSP_CONVERGED_ATOL_NORMA"<<std::endl;
    //        break;
    //    case 2:
    //        std::cout<<"KSP_CONVERGED_RTOL "<<std::endl;
    //        break;
    //    case 3:
    //        std::cout<< "KSP_CONVERGED_ATOL"<<std::endl;
    //        break;
    //    case 4:
    //        std::cout<<"KSP_CONVERGED_ITS  "<<std::endl;
    //        break;
    //    case 5:
    //        std::cout<<"KSP_CONVERGED_CG_NEG_CURVE"<<std::endl;
    //        break;
    //    case 6:
    //        std::cout<<"KSP_CONVERGED_CG_CONSTRAINED"<<std::endl;
    //        break;
    //    case 7:
    //        std::cout<<"KSP_CONVERGED_STEP_LENGTH"  <<std::endl;
    //        break;
    //    case 8:
    //        std::cout<<"KSP_CONVERGED_HAPPY_BREAKDOWN"  <<std::endl;
    //        break;
    //    case -2:
    //        /* diverged */
    //        std::cout<<"KSP_DIVERGED_NULL"            <<std::endl;
    //        break;
    //    case -3:
    //        std::cout<<"KSP_DIVERGED_ITS"                <<std::endl;
    //        break;
    //    case -4:
    //        std::cout<<"KSP_DIVERGED_DTOL"               <<std::endl;
    //        break;
    //    case -5:
    //        std::cout<< "KSP_DIVERGED_BREAKDOWN"          <<std::endl;
    //        break;
    //    case -6:
    //        std::cout<<"KSP_DIVERGED_BREAKDOWN_BICG"      <<std::endl;
    //        break;
    //    case -7:
    //        std::cout<<"KSP_DIVERGED_NONSYMMETRIC    "   <<std::endl;
    //        break;
    //    case -8:
    //        std::cout<< "KSP_DIVERGED_INDEFINITE_PC"      <<std::endl;
    //        break;
    //    case -9:
    //        std::cout<< "KSP_DIVERGED_NANORINF"          <<std::endl;
    //        break;
    //    case -10:
    //        std::cout<< "KSP_DIVERGED_INDEFINITE_MAT"     <<std::endl;
    //        break;
    //    case -0:

    //        std::cout<<KSP_CONVERGED_ITERATING<<std::endl;
    //        break;
    //    default:
    //        break;
    //    }
    if(!((int)myReason>=0))
    {
        std::cout<<"try to solve again with the Petsc solver"<<std::endl;
        this->ierr=KSPSetInitialGuessNonzero(this->myKsp,PETSC_FALSE);
        this->ierr = KSPSolve(this->myKsp,this->b_t,this->sol_tp1);CHKERRXX(this->ierr);
        this->ierr=KSPGetConvergedReason(this->myKsp,&myReason);
        std::cout<<myReason<<std::endl;
        //        switch ((int)myReason)
        //        {
        //        case 1:
        //            std::cout<<"KSP_CONVERGED_RTOL_NORMAL";
        //            break;
        //        case 9:
        //            std::cout<<"KSP_CONVERGED_ATOL_NORMA"<<std::endl;
        //            break;
        //        case 2:
        //            std::cout<<"KSP_CONVERGED_RTOL "<<std::endl;
        //            break;
        //        case 3:
        //            std::cout<< "KSP_CONVERGED_ATOL"<<std::endl;
        //            break;
        //        case 4:
        //            std::cout<<"KSP_CONVERGED_ITS  "<<std::endl;
        //            break;
        //        case 5:
        //            std::cout<<"KSP_CONVERGED_CG_NEG_CURVE"<<std::endl;
        //            break;
        //        case 6:
        //            std::cout<<"KSP_CONVERGED_CG_CONSTRAINED"<<std::endl;
        //            break;
        //        case 7:
        //            std::cout<<"KSP_CONVERGED_STEP_LENGTH"  <<std::endl;
        //            break;
        //        case 8:
        //            std::cout<<"KSP_CONVERGED_HAPPY_BREAKDOWN"  <<std::endl;
        //            break;
        //        case -2:
        //            /* diverged */
        //            std::cout<<"KSP_DIVERGED_NULL"            <<std::endl;
        //            break;
        //        case -3:
        //            std::cout<<"KSP_DIVERGED_ITS"                <<std::endl;
        //            break;
        //        case -4:
        //            std::cout<<"KSP_DIVERGED_DTOL"               <<std::endl;
        //            break;
        //        case -5:
        //            std::cout<< "KSP_DIVERGED_BREAKDOWN"          <<std::endl;
        //            break;
        //        case -6:
        //            std::cout<<"KSP_DIVERGED_BREAKDOWN_BICG"      <<std::endl;
        //            break;
        //        case -7:
        //            std::cout<<"KSP_DIVERGED_NONSYMMETRIC    "   <<std::endl;
        //            break;
        //        case -8:
        //            std::cout<< "KSP_DIVERGED_INDEFINITE_PC"      <<std::endl;
        //            break;
        //        case -9:
        //            std::cout<< "KSP_DIVERGED_NANORINF"          <<std::endl;
        //            break;
        //        case -10:
        //            std::cout<< "KSP_DIVERGED_INDEFINITE_MAT"     <<std::endl;
        //            break;
        //        case -0:

        //            std::cout<<KSP_CONVERGED_ITERATING<<std::endl;
        //            break;
        //        default:
        //            break;
        //        }
    }
    if(!((int)myReason>=0))
    {
        std::cout<<"try to solve again with the Petsc solver"<<std::endl;
        this->ierr = KSPSetTolerances(this->myKsp,1.e-32,1.e-8,5,1000);CHKERRXX(this->ierr);
        this->ierr=KSPSetInitialGuessNonzero(this->myKsp,PETSC_TRUE); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->sol_t,this->sol_tp1); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr = KSPSolve(this->myKsp,this->b_t,this->sol_tp1);CHKERRXX(this->ierr);
        this->ierr=KSPGetConvergedReason(this->myKsp,&myReason);
        std::cout<<myReason<<std::endl;
        //        switch ((int)myReason)
        //        {
        //        case 1:
        //            std::cout<<"KSP_CONVERGED_RTOL_NORMAL";
        //            break;
        //        case 9:
        //            std::cout<<"KSP_CONVERGED_ATOL_NORMA"<<std::endl;
        //            break;
        //        case 2:
        //            std::cout<<"KSP_CONVERGED_RTOL "<<std::endl;
        //            break;
        //        case 3:
        //            std::cout<< "KSP_CONVERGED_ATOL"<<std::endl;
        //            break;
        //        case 4:
        //            std::cout<<"KSP_CONVERGED_ITS  "<<std::endl;
        //            break;
        //        case 5:
        //            std::cout<<"KSP_CONVERGED_CG_NEG_CURVE"<<std::endl;
        //            break;
        //        case 6:
        //            std::cout<<"KSP_CONVERGED_CG_CONSTRAINED"<<std::endl;
        //            break;
        //        case 7:
        //            std::cout<<"KSP_CONVERGED_STEP_LENGTH"  <<std::endl;
        //            break;
        //        case 8:
        //            std::cout<<"KSP_CONVERGED_HAPPY_BREAKDOWN"  <<std::endl;
        //            break;
        //        case -2:
        //            /* diverged */
        //            std::cout<<"KSP_DIVERGED_NULL"            <<std::endl;
        //            break;
        //        case -3:
        //            std::cout<<"KSP_DIVERGED_ITS"                <<std::endl;
        //            break;
        //        case -4:
        //            std::cout<<"KSP_DIVERGED_DTOL"               <<std::endl;
        //            break;
        //        case -5:
        //            std::cout<< "KSP_DIVERGED_BREAKDOWN"          <<std::endl;
        //            break;
        //        case -6:
        //            std::cout<<"KSP_DIVERGED_BREAKDOWN_BICG"      <<std::endl;
        //            break;
        //        case -7:
        //            std::cout<<"KSP_DIVERGED_NONSYMMETRIC    "   <<std::endl;
        //            break;
        //        case -8:
        //            std::cout<< "KSP_DIVERGED_INDEFINITE_PC"      <<std::endl;
        //            break;
        //        case -9:
        //            std::cout<< "KSP_DIVERGED_NANORINF"          <<std::endl;
        //            break;
        //        case -10:
        //            std::cout<< "KSP_DIVERGED_INDEFINITE_MAT"     <<std::endl;
        //            break;
        //        case -0:

        //            std::cout<<KSP_CONVERGED_ITERATING<<std::endl;
        //            break;
        //        default:
        //            break;
        //        }
    }
    if(!((int)myReason>=0))


    {
        this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1_diverged");
        this->printDiffusionArrayFromVector(&this->sol_t,"solt_diverged");
        this->printDiffusionArrayFromVector(&this->wp,"wp_diverged");
        this->printDiffusionArrayFromVector(&this->wm,"wm_diverged");
        this->printDiffusionArrayFromVector(&this->w_t,"wt_diverged");
        this->printDiffusionArrayFromVector(&this->b_t,"bt_diverged");
        throw("PETSC can't make it sorry");
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
       Check the error
    */


    //    this->convergence_test_watch.start("convergence_test");

    this->scatter_petsc_vector(&this->sol_tp1);

    if(false)
    {
        PetscInt its; PetscScalar b_norm;
        this->ierr=MatMult(this->M2,this->sol_tp1,this->r_t); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->r_t,-1,this->b_t); CHKERRXX(this->ierr);
        this->ierr = VecNorm(this->r_t,NORM_2,&this->norm);CHKERRXX(this->ierr);
        this->ierr = VecNorm(this->b_t,NORM_2,&b_norm);CHKERRXX(this->ierr);
        this->ierr = KSPGetIterationNumber(this->myKsp,&its);CHKERRXX(this->ierr);



        PetscScalar relative_residual=this->norm/b_norm;

        std::cout<<"my ksp view "<<std::endl;
        std::cout<<this->it<<" n_it "<<its<<" absolute residual "<<this->norm<<" relative residual "<<relative_residual<<std::endl;
    }
    //   this->printDiffusionVector(&this->sol_tp1,"sol_tp1_before");

    ////    if(this->norm>pow(10,-11) && relative_residual>pow(10.00,-14.00) && its<500)
    ////        std::cout<<"petsc didn't work as expected from some unknown reason"<<std::endl;

    //    this->convergence_test_watch.stop();
    //    this->convergence_test_watch_total_time+=this->convergence_test_watch.read_duration();
}


void Diffusion::solve_linear_system_efficient()
{


    this->ierr = KSPSetTolerances(this->myKsp,1.e-32,1.e-12,5,1000);CHKERRXX(this->ierr);
    this->ierr=KSPSetInitialGuessNonzero(this->myKsp,PETSC_TRUE);

    Vec bTemp;
    Vec solTemp;
    PetscInt bTempSize,bTempSizeLocal;
    ISGetSize(this->my_rows_parallel_indexes,&bTempSize);
    ISGetLocalSize( this->my_rows_parallel_indexes,&bTempSizeLocal);
    this->ierr=VecCreate(MPI_COMM_WORLD,&bTemp); CHKERRXX(this->ierr);
    this->ierr=VecSetSizes(bTemp,bTempSizeLocal,bTempSize); CHKERRXX(this->ierr);
    this->ierr=VecSetFromOptions(bTemp); CHKERRXX(this->ierr);

    this->ierr=VecCreate(MPI_COMM_WORLD,&solTemp); CHKERRXX(this->ierr);
    this->ierr=VecSetSizes(solTemp,bTempSizeLocal,bTempSize); CHKERRXX(this->ierr);
    this->ierr=VecSetFromOptions(solTemp); CHKERRXX(this->ierr);


    this->ierr = KSPSolve(this->myKsp,this->b_t,this->sol_tp1);CHKERRXX(this->ierr);



    this->ierr=VecDestroy(bTemp); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(solTemp);


    KSPConvergedReason myReason;
    this->ierr=KSPGetConvergedReason(this->myKsp,&myReason);
    std::cout<<myReason<<std::endl;
    switch ((int)myReason)
    {
    case 1:
        std::cout<<"KSP_CONVERGED_RTOL_NORMAL";
        break;
    case 9:
        std::cout<<"KSP_CONVERGED_ATOL_NORMA"<<std::endl;
        break;
    case 2:
        std::cout<<"KSP_CONVERGED_RTOL "<<std::endl;
        break;
    case 3:
        std::cout<< "KSP_CONVERGED_ATOL"<<std::endl;
        break;
    case 4:
        std::cout<<"KSP_CONVERGED_ITS  "<<std::endl;
        break;
    case 5:
        std::cout<<"KSP_CONVERGED_CG_NEG_CURVE"<<std::endl;
        break;
    case 6:
        std::cout<<"KSP_CONVERGED_CG_CONSTRAINED"<<std::endl;
        break;
    case 7:
        std::cout<<"KSP_CONVERGED_STEP_LENGTH"  <<std::endl;
        break;
    case 8:
        std::cout<<"KSP_CONVERGED_HAPPY_BREAKDOWN"  <<std::endl;
        break;
    case -2:
        /* diverged */
        std::cout<<"KSP_DIVERGED_NULL"            <<std::endl;
        break;
    case -3:
        std::cout<<"KSP_DIVERGED_ITS"                <<std::endl;
        break;
    case -4:
        std::cout<<"KSP_DIVERGED_DTOL"               <<std::endl;
        break;
    case -5:
        std::cout<< "KSP_DIVERGED_BREAKDOWN"          <<std::endl;
        break;
    case -6:
        std::cout<<"KSP_DIVERGED_BREAKDOWN_BICG"      <<std::endl;
        break;
    case -7:
        std::cout<<"KSP_DIVERGED_NONSYMMETRIC    "   <<std::endl;
        break;
    case -8:
        std::cout<< "KSP_DIVERGED_INDEFINITE_PC"      <<std::endl;
        break;
    case -9:
        std::cout<< "KSP_DIVERGED_NANORINF"          <<std::endl;
        break;
    case -10:
        std::cout<< "KSP_DIVERGED_INDEFINITE_MAT"     <<std::endl;
        break;
    case -0:

        std::cout<<KSP_CONVERGED_ITERATING<<std::endl;
        break;
    default:
        break;
    }
    if((int)myReason<0)
    {
        std::cout<<"try to solve again with the Petsc solver"<<std::endl;
        this->ierr=KSPSetInitialGuessNonzero(this->myKsp,PETSC_FALSE);
        this->ierr = KSPSolve(this->myKsp,this->b_t,this->sol_tp1);CHKERRXX(this->ierr);
        this->ierr=KSPGetConvergedReason(this->myKsp,&myReason);
        std::cout<<myReason<<std::endl;
        switch ((int)myReason)
        {
        case 1:
            std::cout<<"KSP_CONVERGED_RTOL_NORMAL";
            break;
        case 9:
            std::cout<<"KSP_CONVERGED_ATOL_NORMA"<<std::endl;
            break;
        case 2:
            std::cout<<"KSP_CONVERGED_RTOL "<<std::endl;
            break;
        case 3:
            std::cout<< "KSP_CONVERGED_ATOL"<<std::endl;
            break;
        case 4:
            std::cout<<"KSP_CONVERGED_ITS  "<<std::endl;
            break;
        case 5:
            std::cout<<"KSP_CONVERGED_CG_NEG_CURVE"<<std::endl;
            break;
        case 6:
            std::cout<<"KSP_CONVERGED_CG_CONSTRAINED"<<std::endl;
            break;
        case 7:
            std::cout<<"KSP_CONVERGED_STEP_LENGTH"  <<std::endl;
            break;
        case 8:
            std::cout<<"KSP_CONVERGED_HAPPY_BREAKDOWN"  <<std::endl;
            break;
        case -2:
            /* diverged */
            std::cout<<"KSP_DIVERGED_NULL"            <<std::endl;
            break;
        case -3:
            std::cout<<"KSP_DIVERGED_ITS"                <<std::endl;
            break;
        case -4:
            std::cout<<"KSP_DIVERGED_DTOL"               <<std::endl;
            break;
        case -5:
            std::cout<< "KSP_DIVERGED_BREAKDOWN"          <<std::endl;
            break;
        case -6:
            std::cout<<"KSP_DIVERGED_BREAKDOWN_BICG"      <<std::endl;
            break;
        case -7:
            std::cout<<"KSP_DIVERGED_NONSYMMETRIC    "   <<std::endl;
            break;
        case -8:
            std::cout<< "KSP_DIVERGED_INDEFINITE_PC"      <<std::endl;
            break;
        case -9:
            std::cout<< "KSP_DIVERGED_NANORINF"          <<std::endl;
            break;
        case -10:
            std::cout<< "KSP_DIVERGED_INDEFINITE_MAT"     <<std::endl;
            break;
        case -0:

            std::cout<<KSP_CONVERGED_ITERATING<<std::endl;
            break;
        default:
            break;
        }

        if((int)myReason<0)
            throw std::runtime_error  ("PETSC can't make it sorry");
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
       Check the error
    */


    PetscInt its; PetscScalar b_norm;
    this->ierr=MatMult(this->M2,this->sol_tp1,this->r_t); CHKERRXX(this->ierr);
    this->ierr=VecAXPY(this->r_t,-1,this->b_t); CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->r_t,NORM_2,&this->norm);CHKERRXX(this->ierr);
    this->ierr = VecNorm(this->b_t,NORM_2,&b_norm);CHKERRXX(this->ierr);
    this->ierr = KSPGetIterationNumber(this->myKsp,&its);CHKERRXX(this->ierr);



    PetscScalar relative_residual=this->norm/b_norm;

    std::cout<<"my ksp view "<<std::endl;
    std::cout<<this->it<<" n_it "<<its<<" absolute residual "<<this->norm<<" relative residual "<<relative_residual<<std::endl;
    //  this->printDiffusionVector(&this->sol_tp1,"sol_tp1_before");

    if(this->norm>pow(10,-11) && relative_residual>pow(10.00,-14.00) && its<500)
        std::cout<<"petsc didn't work as expected from some unknown reason"<<std::endl;
}

int Diffusion::evolve_Equation_Finite_Difference_Splitting_Implicit()
{

    if(this->it==0)
    {
        this->ierr=VecSet(this->sol_t,1.00);    CHKERRXX(this->ierr);
        this->ierr=VecSet(this->sol_tp1,0.00);  CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);    CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);      CHKERRXX(this->ierr);

        if(!this->periodic_xyz && this->my_casl_diffusion_method==Diffusion::neuman_crank_nicholson)
        {
            this->ierr=VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol); CHKERRXX(this->ierr);
            this->ierr=VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol); CHKERRXX(this->ierr);
            this->compute_mapping_fromLocal2Global();
            // could use the same filter than in the potential filtering
            // consider making it a member class
            PetscInt *filter_ix=new PetscInt[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
            PetscScalar *filter_zero=new PetscScalar[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
            PetscScalar *phi_is_all_positive;
            this->ierr=VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);
            int ix=0;
            for(int i=0;i<this->n_local_size_sol;i++)
            {
                if(phi_is_all_positive[i]>0.5)
                {
                    filter_ix[ix]=i;
                    ix++;
                }
            }
            for(int i=0;i<this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive;i++)
            {
                filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
                filter_zero[i]=0.00;
            }

            this->ierr=VecSetValues(this->sol_t,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyBegin(this->sol_t); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyEnd(this->sol_t);   CHKERRXX(this->ierr);
            this->sendFirstIteration2HitoryDataBase();
            this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);
            delete filter_ix;
            delete filter_zero;

        }

        if(!this->periodic_xyz && this->my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson)
        {

            this->filter_petsc_vector_dirichlet(&this->sol_t);
            this->sendFirstIteration2HitoryDataBase();


        }

        if(this->forward_stage)
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_forward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_forward");
        }
        else
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_backward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_backward");
        }



    }
    if(this->it>0)
    {
        this->ierr=VecCopy(this->sol_tp1,this->sol_t); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }



    this->it++;


    PetscInt local_size_b,global_size_b;
    this->ierr=VecGetSize(this->b_t,&global_size_b); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(this->b_t,&local_size_b); CHKERRXX(this->ierr);
    //std::cout<<"1 "<<global_size_b<<" "<<local_size_b<<std::endl;
    this->ierr=VecPointwiseMult(this->sol_t,this->sol_t,this->w_t); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
    if(this->it==1)
    {

        this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_point");
        this->printDiffusionArrayFromVector(&this->w_t,"w_t_point");


    }
    //    this->ierr=VecAssemblyBegin(this->sol_t);
    //    this->ierr=VecAssemblyEnd(this->sol_t);

    // std::cout<<"1 1 "<<this->mpi->mpirank<<" "<<global_size_b<<" "<<local_size_b<<std::endl;
    this->ierr=MatMult(this->M1,this->sol_t,this->b_t); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->b_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->b_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    //  this->printDiffusionVector(&this->b_t,"b_t");
    this->ierr=VecGetSize(this->b_t,&global_size_b); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(this->b_t,&local_size_b);         CHKERRXX(this->ierr);
    //std::cout<<"2 "<<global_size_b<<" "<<local_size_b<<std::endl;



    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                  Solve the linear system
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Solve linear system
              */
    if(!this->extractSubDiffusionMatricesBool)
        this->solve_linear_system_slow();
    else
        this->solve_linear_system_efficient();

    this->ierr=VecPointwiseMult(this->sol_tp1,this->sol_tp1,this->w_t);CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    PetscScalar *sol_tp1_local; PetscInt n_local_vec_size;
    this->ierr=VecGetLocalSize(this->sol_tp1,&n_local_vec_size); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->sol_tp1,&sol_tp1_local); CHKERRXX(this->ierr);

    for(int i=0;i<n_local_vec_size;i++)
    {
        if(sol_tp1_local[i]<0)
            sol_tp1_local[i]=0;
    }

    this->ierr=VecRestoreArray(this->sol_tp1,&sol_tp1_local); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    if(this->printIntemediateSteps)
    {
        this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1");
        this->printDiffusionArrayFromVector(&this->sol_t,"sol_tp0");
    }


    return 0;
}

int Diffusion::evolve_Equation_Explicit(){return 0;}
int Diffusion::evolve_Equation_Finite_Difference_Splitting_Implicit_Scaled(){return 0;}
int Diffusion::evolve_Equation_Finite_Difference_Splitting_Explicit(){return 0;}
int Diffusion::evolve_Equation_Spectral_Splitting(){return 0;}

int Diffusion::setup_spectral_solver_amr()
{
    std::cout<<this->mpi->mpirank<<" started to setup the spectral solver amr "<<std::endl;

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));

    int n_fft=N0*N1*N2;
    int n_fft_p=n_fft/this->mpi->mpisize;



    fftw_mpi_init();


    /* get local data size and allocate */
    this->alloc_local = fftw_mpi_local_size_3d(N0, N1,N2, MPI_COMM_WORLD,
                                               &this->local_n0, &this->local_0_start);


    this->idx_fftw_local=new int[n_fft_p];

    for(int i=0;i<n_fft_p;i++)
    {
        this->idx_fftw_local[i]=2*(this->local_0_start*N1*N2+i);
    }

    this->input_forward= fftw_alloc_complex(this->alloc_local);
    this->input_backward= fftw_alloc_complex(this->alloc_local);
    this->output_forward= fftw_alloc_complex(this->alloc_local);
    this->output_backward= fftw_alloc_complex(this->alloc_local);

    /* create plan for in-place forward DFT */
    this->plan_fftw_forward= fftw_mpi_plan_dft_3d(N0, N1,N2, this->input_forward,this->output_forward, MPI_COMM_WORLD,
                                                  FFTW_FORWARD, FFTW_MEASURE);

    this->plan_fftw_backward=fftw_mpi_plan_dft_3d(N0,N1,N2,this->input_backward,this->output_backward,
                                                  MPI_COMM_WORLD,FFTW_BACKWARD,FFTW_MEASURE);



    //    this->ierr=VecCreate(MPI_COMM_WORLD,&this->fftw_global_petsc_output);              CHKERRXX(this->ierr);
    //    this->ierr=VecSetSizes(this->fftw_global_petsc_output,n_fft_p,n_fft);      CHKERRXX(this->ierr);
    //    this->ierr=VecSetFromOptions(this->fftw_global_petsc_output);              CHKERRXX(this->ierr);



    this->FFTCreated=true;

    std::cout<<this->mpi->mpirank<<" finished to setup the spectral solver "<<std::endl;
}

int Diffusion::setup_spectral_solver_amr_2D()
{
    std::cout<<this->mpi->mpirank<<" started to setup the spectral solver amr "<<std::endl;

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));

    int n_fft=N0*N1;
    int n_fft_p=n_fft/this->mpi->mpisize;



    fftw_mpi_init();


    /* get local data size and allocate */
    this->alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                               &this->local_n0, &this->local_0_start);


    this->idx_fftw_local=new int[n_fft_p];

    for(int i=0;i<n_fft_p;i++)
    {
        this->idx_fftw_local[i]=2*(this->local_0_start*N1+i);
    }

    this->input_forward= fftw_alloc_complex(this->alloc_local);
    this->input_backward= fftw_alloc_complex(this->alloc_local);
    this->output_forward= fftw_alloc_complex(this->alloc_local);
    this->output_backward= fftw_alloc_complex(this->alloc_local);

    /* create plan for in-place forward DFT */
    this->plan_fftw_forward= fftw_mpi_plan_dft_2d(N0, N1, this->input_forward,this->output_forward, MPI_COMM_WORLD,
                                                  FFTW_FORWARD, FFTW_MEASURE);

    this->plan_fftw_backward=fftw_mpi_plan_dft_2d(N0,N1,this->input_backward,this->output_backward,
                                                  MPI_COMM_WORLD,FFTW_BACKWARD,FFTW_MEASURE);



    //    this->ierr=VecCreate(MPI_COMM_WORLD,&this->fftw_global_petsc_output);              CHKERRXX(this->ierr);
    //    this->ierr=VecSetSizes(this->fftw_global_petsc_output,n_fft_p,n_fft);      CHKERRXX(this->ierr);
    //    this->ierr=VecSetFromOptions(this->fftw_global_petsc_output);              CHKERRXX(this->ierr);



    this->FFTCreated=true;

    std::cout<<this->mpi->mpirank<<" finished to setup the spectral solver "<<std::endl;
}


int Diffusion::setup_spectral_solver_amr_r2c()
{
    std::cout<<this->mpi->mpirank<<" started to setup the spectral solver amr "<<std::endl;

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));

    int n_fft=N0*N1*N2;
    int n_fft_p=n_fft/this->mpi->mpisize;



    fftw_mpi_init();


    /* get local data size and allocate */
    this->alloc_local = fftw_mpi_local_size_3d(N0, N1,N2/2+1, MPI_COMM_WORLD,
                                               &this->local_n0, &this->local_0_start);


    this->idx_fftw_local=new int[n_fft_p];

    for(int i=0;i<n_fft_p;i++)
    {
        this->idx_fftw_local[i]=2*(this->local_0_start*N1*N2+i);
    }

    this->input_forward_real= fftw_alloc_real(2*this->alloc_local);
    this->input_backward= fftw_alloc_complex(this->alloc_local);
    this->output_forward= fftw_alloc_complex(this->alloc_local);
    this->output_backward_real= fftw_alloc_real(2*this->alloc_local);

    /* create plan for out-of-place r2c DFT */
    this->plan_fftw_forward= fftw_mpi_plan_dft_r2c_3d(N0, N1,N2, this->input_forward_real,this->output_forward, MPI_COMM_WORLD,
                                                      FFTW_MEASURE);

    this->plan_fftw_backward=fftw_mpi_plan_dft_c2r_3d(N0,N1,N2,this->input_backward,this->output_backward_real,
                                                      MPI_COMM_WORLD,FFTW_MEASURE);



    //    this->ierr=VecCreate(MPI_COMM_WORLD,&this->fftw_global_petsc_output);              CHKERRXX(this->ierr);
    //    this->ierr=VecSetSizes(this->fftw_global_petsc_output,n_fft_p,n_fft);      CHKERRXX(this->ierr);
    //    this->ierr=VecSetFromOptions(this->fftw_global_petsc_output);              CHKERRXX(this->ierr);



    this->FFTCreated=true;

    std::cout<<this->mpi->mpirank<<" finished to setup the spectral solver "<<std::endl;
}

int Diffusion::setup_spectral_solver_amr_2D_r2c()
{
    std::cout<<this->mpi->mpirank<<" started to setup the spectral solver amr "<<std::endl;

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));

    int n_fft=N0*N1;
    int n_fft_p=n_fft/this->mpi->mpisize;



    fftw_mpi_init();


    /* get local data size and allocate */
    this->alloc_local = fftw_mpi_local_size_2d(N0, N1/2+1, MPI_COMM_WORLD,
                                               &this->local_n0, &this->local_0_start);


    this->idx_fftw_local=new int[n_fft_p];

    for(int i=0;i<n_fft_p;i++)
    {
        this->idx_fftw_local[i]=2*(this->local_0_start*N1+i);
    }

    this->input_forward_real= fftw_alloc_real(2*this->alloc_local);
    this->input_backward= fftw_alloc_complex(this->alloc_local);
    this->output_forward= fftw_alloc_complex(this->alloc_local);
    this->output_backward_real= fftw_alloc_real(2*this->alloc_local);

    /* create plan for in-place forward DFT */
    this->plan_fftw_forward= fftw_mpi_plan_dft_r2c_2d(N0, N1, this->input_forward_real,this->output_forward, MPI_COMM_WORLD,
                                                      FFTW_MEASURE);

    this->plan_fftw_backward=fftw_mpi_plan_dft_c2r_2d(N0,N1,this->input_backward,this->output_backward_real,
                                                      MPI_COMM_WORLD,FFTW_MEASURE);



    //    this->ierr=VecCreate(MPI_COMM_WORLD,&this->fftw_global_petsc_output);              CHKERRXX(this->ierr);
    //    this->ierr=VecSetSizes(this->fftw_global_petsc_output,n_fft_p,n_fft);      CHKERRXX(this->ierr);
    //    this->ierr=VecSetFromOptions(this->fftw_global_petsc_output);              CHKERRXX(this->ierr);



    this->FFTCreated=true;

    std::cout<<this->mpi->mpirank<<" finished to setup the spectral solver "<<std::endl;
}


int Diffusion::create_mapping_data_structure_fftw()
{
    this->fftw_setup_watch.start("fftw_setup");

    int n_games=1;

    for(int i=0;i<n_games;i++)
    {

        std::cout<<this->mpi->mpirank<<" start to create mapping data structure fftw "<<std::endl;

        if(this->mapping_fftw_destroyed && !this->mapping_fftw_created)
        {

            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));


            int n_local_size_amr;
            this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

            // create the mapping data structure
            p4est_topidx_t global_node_number=0;
            for(int ii=0;ii<this->mpi->mpirank;ii++)
                global_node_number+=this->nodes->global_owned_indeps[ii];

            std::cout<<this->mpi->mpirank<<" global number "<<global_node_number<<std::endl;
            std::cout<<this->mpi->mpirank<<" n_local_size_sol "<<n_local_size_amr<<std::endl;

            double i_node,j_node,k_node,x,y,z, offset_tree_x, offset_tree_y, offset_tree_z,
                    x_normalized,y_normalized,z_normalized;
            this->idx_from_fftw_to_p4est_fftw_ix=new int[n_local_size_amr];
            this->idx_from_fftw_to_p4est_p4est_ix=new int[n_local_size_amr];

            double max_pow=log(P4EST_ROOT_LEN)/log(2);
            double normalizer=pow(2,max_pow-this->max_level);

            std::cout<<max_pow<<" "<<normalizer<<" "<<P4EST_ROOT_LEN<<std::endl;

            for (int n = 0; n < n_local_size_amr; n++)
            {
                p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
                p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
                double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
                double tree_ymin = this->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
                double tree_zmin = this->connectivity->vertices[3*v_mm + 2];
#endif
                x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
                y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);
#ifdef P4_TO_P8
                z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
#endif

                offset_tree_x=tree_xmin/this->Lx*N0;
                offset_tree_y=tree_ymin/this->Lx*N0;
#ifdef P4_TO_P8
                offset_tree_z=tree_zmin/this->Lx*N0;
#endif

                x_normalized=x/normalizer;
                y_normalized=y/normalizer;
#ifdef P4_TO_P8
                z_normalized=z/normalizer;
#endif

                i_node=x_normalized+offset_tree_x;
                j_node=y_normalized+offset_tree_y;
#ifdef P4_TO_P8
                k_node=z_normalized+offset_tree_z;
#endif

                //  if(!this->periodic_xyz)
                {
                    i_node=(int)i_node%N0;
                    j_node=(int)j_node%N1;
#ifdef P4_TO_P8
                    k_node=(int)k_node%N2;
#endif
                }

                int idx_fftw_temp=(int)(i_node*((double)N1*N2) + j_node*(double)N2+k_node);

                this->idx_from_fftw_to_p4est_p4est_ix[n]=global_node_number+n;
                this->idx_from_fftw_to_p4est_fftw_ix[n]=idx_fftw_temp;
            }

            this->printDiffusionArray(this->idx_from_fftw_to_p4est_p4est_ix,n_local_size_amr,"idx_p4est");
            this->printDiffusionArray(this->idx_from_fftw_to_p4est_fftw_ix,n_local_size_amr,"idx_fftw");
            PetscInt       to_n,from_n;
            VecGetLocalSize(this->fftw_global_petsc_output,&from_n);
            VecGetLocalSize(this->sol_tp1,&to_n);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_fftw_ix,PETSC_COPY_VALUES,&this->from_fftw); CHKERRXX(this->ierr);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_p4est_ix,PETSC_COPY_VALUES,&this->to_p4est); CHKERRXX(this->ierr);
            this->ierr=VecScatterCreate(this->fftw_global_petsc_output,this->from_fftw,this->sol_tp1,this->to_p4est,&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
            this->mapping_fftw_created=true;
            this->mapping_fftw_destroyed=false;

            std::cout<<this->mpi->mpirank<<" finished to create mapping data structure fftw "<<std::endl;

        }

        if(i<n_games-1)
        {
            if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive && this->mapping_fftw_created && !this->mapping_fftw_destroyed)
            {
                this->ierr=ISDestroy(this->from_fftw); CHKERRXX(this->ierr);
                this->ierr=ISDestroy(this->to_p4est); CHKERRXX(this->ierr);
                this->ierr=VecScatterDestroy(&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
                delete this->idx_from_fftw_to_p4est_fftw_ix;
                delete this->idx_from_fftw_to_p4est_p4est_ix;

                this->mapping_fftw_created=false;
                this->mapping_fftw_destroyed=true;

            }
        }

    }

    this->fftw_setup_watch.stop();
}


int Diffusion::create_mapping_data_structure_fftw_2D()
{

    int n_games=1;

    for(int i=0;i<n_games;i++)
    {

        std::cout<<this->mpi->mpirank<<" start to create mapping data structure fftw "<<std::endl;

        if(this->mapping_fftw_destroyed && !this->mapping_fftw_created)
        {

            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));

            int n_local_size_amr;
            this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

            // create the mapping data structure
            p4est_topidx_t global_node_number=0;
            for(int ii=0;ii<this->mpi->mpirank;ii++)
                global_node_number+=this->nodes->global_owned_indeps[ii];

            std::cout<<this->mpi->mpirank<<" global number "<<global_node_number<<std::endl;
            std::cout<<this->mpi->mpirank<<" n_local_size_sol "<<n_local_size_amr<<std::endl;

            double i_node,j_node,x,y, offset_tree_x, offset_tree_y,
                    x_normalized,y_normalized;
            this->idx_from_fftw_to_p4est_fftw_ix=new int[n_local_size_amr];
            this->idx_from_fftw_to_p4est_p4est_ix=new int[n_local_size_amr];

            double max_pow=log(P4EST_ROOT_LEN)/log(2);
            double normalizer=pow(2,max_pow-this->max_level);

            std::cout<<max_pow<<" "<<normalizer<<" "<<P4EST_ROOT_LEN<<std::endl;

            for (int n = 0; n < n_local_size_amr; n++)
            {
                p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
                p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
                double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
                double tree_ymin = this->connectivity->vertices[3*v_mm + 1];

                x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
                y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);

                offset_tree_x=tree_xmin/this->Lx*N0;
                offset_tree_y=tree_ymin/this->Lx*N0;

                x_normalized=x/normalizer;
                y_normalized=y/normalizer;

                i_node=x_normalized+offset_tree_x;
                j_node=y_normalized+offset_tree_y;




                // if(!this->periodic_xyz)
                {
                    //                      if(i_node==N0 || j_node==N0)
                    //                          std::cout<<"hi"<<std::endl;
                    i_node=(int)i_node%N0;
                    j_node=(int)j_node%N1;
                }

                int idx_fftw_temp=(int)(i_node*((double)N1) + j_node);

                this->idx_from_fftw_to_p4est_p4est_ix[n]=global_node_number+n;
                this->idx_from_fftw_to_p4est_fftw_ix[n]=idx_fftw_temp;
            }

            this->printDiffusionArray(this->idx_from_fftw_to_p4est_p4est_ix,n_local_size_amr,"idx_p4est");
            this->printDiffusionArray(this->idx_from_fftw_to_p4est_fftw_ix,n_local_size_amr,"idx_fftw");
            PetscInt       to_n,from_n;
            this->ierr=VecGetLocalSize(this->fftw_global_petsc_output,&from_n); CHKERRXX(this->ierr);
            this->ierr=VecGetLocalSize(this->sol_tp1,&to_n); CHKERRXX(this->ierr);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_fftw_ix,PETSC_COPY_VALUES,&this->from_fftw); CHKERRXX(this->ierr);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_p4est_ix,PETSC_COPY_VALUES,&this->to_p4est); CHKERRXX(this->ierr);
            this->ierr=VecScatterCreate(this->fftw_global_petsc_output,this->from_fftw,this->sol_tp1,this->to_p4est,&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
            this->mapping_fftw_created=true;
            this->mapping_fftw_destroyed=false;

            std::cout<<this->mpi->mpirank<<" finished to create mapping data structure fftw "<<std::endl;

        }

        if(i<n_games-1)
        {
            if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive && this->mapping_fftw_created && !this->mapping_fftw_destroyed)
            {
                this->ierr=ISDestroy(this->from_fftw); CHKERRXX(this->ierr);
                this->ierr=ISDestroy(this->to_p4est); CHKERRXX(this->ierr);
                this->ierr=VecScatterDestroy(&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
                delete this->idx_from_fftw_to_p4est_fftw_ix;
                delete this->idx_from_fftw_to_p4est_p4est_ix;

                this->mapping_fftw_created=false;
                this->mapping_fftw_destroyed=true;

            }
        }

    }
}

int Diffusion::create_mapping_data_structure_fftw_r2r()
{
    this->fftw_setup_watch.start("fftw_setup");

    int n_games=1;

    for(int i=0;i<n_games;i++)
    {

        std::cout<<this->mpi->mpirank<<" start to create mapping data structure fftw "<<std::endl;

        if(this->mapping_fftw_destroyed && !this->mapping_fftw_created)
        {

            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));

            const ptrdiff_t N2_effective=2*(N2/2+1);
            int n_local_size_amr;
            this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

            // create the mapping data structure
            p4est_topidx_t global_node_number=0;
            for(int ii=0;ii<this->mpi->mpirank;ii++)
                global_node_number+=this->nodes->global_owned_indeps[ii];

            std::cout<<this->mpi->mpirank<<" global number "<<global_node_number<<std::endl;
            std::cout<<this->mpi->mpirank<<" n_local_size_sol "<<n_local_size_amr<<std::endl;

            double i_node,j_node,k_node,x,y,z, offset_tree_x, offset_tree_y, offset_tree_z,
                    x_normalized,y_normalized,z_normalized;
            this->idx_from_fftw_to_p4est_fftw_ix=new int[n_local_size_amr];
            this->idx_from_fftw_to_p4est_p4est_ix=new int[n_local_size_amr];

            double max_pow=log(P4EST_ROOT_LEN)/log(2);
            double normalizer=pow(2,max_pow-this->max_level);

            std::cout<<max_pow<<" "<<normalizer<<" "<<P4EST_ROOT_LEN<<std::endl;

            for (int n = 0; n < n_local_size_amr; n++)
            {
                p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
                p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
                double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
                double tree_ymin = this->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
                double tree_zmin = this->connectivity->vertices[3*v_mm + 2];
#endif
                x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
                y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);
#ifdef P4_TO_P8
                z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
#endif

                offset_tree_x=tree_xmin/this->Lx*N0;
                offset_tree_y=tree_ymin/this->Lx*N0;
#ifdef P4_TO_P8
                offset_tree_z=tree_zmin/this->Lx*N0;
#endif

                x_normalized=x/normalizer;
                y_normalized=y/normalizer;
#ifdef P4_TO_P8
                z_normalized=z/normalizer;
#endif

                i_node=x_normalized+offset_tree_x;
                j_node=y_normalized+offset_tree_y;
#ifdef P4_TO_P8
                k_node=z_normalized+offset_tree_z;
#endif

                //  if(!this->periodic_xyz)
                {
                    i_node=(int)i_node%N0;
                    j_node=(int)j_node%N1;
#ifdef P4_TO_P8
                    k_node=(int)k_node%N2;
#endif
                }

                int idx_fftw_temp=(int)(i_node*((double)N1*N2_effective) + j_node*(double)N2_effective+k_node);

                this->idx_from_fftw_to_p4est_p4est_ix[n]=global_node_number+n;
                this->idx_from_fftw_to_p4est_fftw_ix[n]=idx_fftw_temp;
            }

            this->printDiffusionArray(this->idx_from_fftw_to_p4est_p4est_ix,n_local_size_amr,"idx_p4est");
            this->printDiffusionArray(this->idx_from_fftw_to_p4est_fftw_ix,n_local_size_amr,"idx_fftw");
            PetscInt       to_n,from_n;
            VecGetLocalSize(this->fftw_global_petsc_output,&from_n);
            VecGetLocalSize(this->sol_tp1,&to_n);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_fftw_ix,PETSC_COPY_VALUES,&this->from_fftw); CHKERRXX(this->ierr);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_p4est_ix,PETSC_COPY_VALUES,&this->to_p4est); CHKERRXX(this->ierr);
            this->ierr=VecScatterCreate(this->fftw_global_petsc_output,this->from_fftw,this->sol_tp1,this->to_p4est,&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
            this->mapping_fftw_created=true;
            this->mapping_fftw_destroyed=false;

            std::cout<<this->mpi->mpirank<<" finished to create mapping data structure fftw "<<std::endl;

        }

        if(i<n_games-1)
        {
            if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive && this->mapping_fftw_created && !this->mapping_fftw_destroyed)
            {
                this->ierr=ISDestroy(this->from_fftw); CHKERRXX(this->ierr);
                this->ierr=ISDestroy(this->to_p4est); CHKERRXX(this->ierr);
                this->ierr=VecScatterDestroy(&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
                delete this->idx_from_fftw_to_p4est_fftw_ix;
                delete this->idx_from_fftw_to_p4est_p4est_ix;

                this->mapping_fftw_created=false;
                this->mapping_fftw_destroyed=true;

            }
        }

    }

    this->fftw_setup_watch.stop();
}


int Diffusion::create_mapping_data_structure_fftw_2D_r2r()
{

    int n_games=1;

    for(int i=0;i<n_games;i++)
    {

        std::cout<<this->mpi->mpirank<<" start to create mapping data structure fftw "<<std::endl;

        if(this->mapping_fftw_destroyed && !this->mapping_fftw_created)
        {

            const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
                    N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));

            const ptrdiff_t N1_effective=2*(N1/2+1);

            int n_local_size_amr;
            this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

            // create the mapping data structure
            p4est_topidx_t global_node_number=0;
            for(int ii=0;ii<this->mpi->mpirank;ii++)
                global_node_number+=this->nodes->global_owned_indeps[ii];

            std::cout<<this->mpi->mpirank<<" global number "<<global_node_number<<std::endl;
            std::cout<<this->mpi->mpirank<<" n_local_size_sol "<<n_local_size_amr<<std::endl;

            double i_node,j_node,x,y, offset_tree_x, offset_tree_y,
                    x_normalized,y_normalized;
            this->idx_from_fftw_to_p4est_fftw_ix=new int[n_local_size_amr];
            this->idx_from_fftw_to_p4est_p4est_ix=new int[n_local_size_amr];

            double max_pow=log(P4EST_ROOT_LEN)/log(2);
            double normalizer=pow(2,max_pow-this->max_level);

            std::cout<<max_pow<<" "<<normalizer<<" "<<P4EST_ROOT_LEN<<std::endl;

            for (int n = 0; n < n_local_size_amr; n++)
            {
                p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
                p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
                double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
                double tree_ymin = this->connectivity->vertices[3*v_mm + 1];

                x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
                y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);

                offset_tree_x=tree_xmin/this->Lx*N0;
                offset_tree_y=tree_ymin/this->Lx*N0;

                x_normalized=x/normalizer;
                y_normalized=y/normalizer;

                i_node=x_normalized+offset_tree_x;
                j_node=y_normalized+offset_tree_y;




                // if(!this->periodic_xyz)
                {
                    //                      if(i_node==N0 || j_node==N0)
                    //                          std::cout<<"hi"<<std::endl;
                    i_node=(int)i_node%N0;
                    j_node=(int)j_node%N1;
                }

                int idx_fftw_temp=(int)(i_node*((double)N1_effective) + j_node);

                this->idx_from_fftw_to_p4est_p4est_ix[n]=global_node_number+n;
                this->idx_from_fftw_to_p4est_fftw_ix[n]=idx_fftw_temp;
            }

            this->printDiffusionArray(this->idx_from_fftw_to_p4est_p4est_ix,n_local_size_amr,"idx_p4est");
            this->printDiffusionArray(this->idx_from_fftw_to_p4est_fftw_ix,n_local_size_amr,"idx_fftw");
            PetscInt       to_n,from_n;
            this->ierr=VecGetLocalSize(this->fftw_global_petsc_output,&from_n); CHKERRXX(this->ierr);
            this->ierr=VecGetLocalSize(this->sol_tp1,&to_n); CHKERRXX(this->ierr);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_fftw_ix,PETSC_COPY_VALUES,&this->from_fftw); CHKERRXX(this->ierr);
            this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,this->idx_from_fftw_to_p4est_p4est_ix,PETSC_COPY_VALUES,&this->to_p4est); CHKERRXX(this->ierr);
            this->ierr=VecScatterCreate(this->fftw_global_petsc_output,this->from_fftw,this->sol_tp1,this->to_p4est,&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
            this->mapping_fftw_created=true;
            this->mapping_fftw_destroyed=false;

            std::cout<<this->mpi->mpirank<<" finished to create mapping data structure fftw "<<std::endl;

        }

        if(i<n_games-1)
        {
            if(this->myNumericalScheme==Diffusion::splitting_spectral_adaptive && this->mapping_fftw_created && !this->mapping_fftw_destroyed)
            {
                this->ierr=ISDestroy(this->from_fftw); CHKERRXX(this->ierr);
                this->ierr=ISDestroy(this->to_p4est); CHKERRXX(this->ierr);
                this->ierr=VecScatterDestroy(&this->scatter_from_fftw_to_p4est); CHKERRXX(this->ierr);
                delete this->idx_from_fftw_to_p4est_fftw_ix;
                delete this->idx_from_fftw_to_p4est_p4est_ix;

                this->mapping_fftw_created=false;
                this->mapping_fftw_destroyed=true;

            }
        }

    }
}


int Diffusion::interpolate_w_t_on_uniform_grid()
{


    this->interpolation_on_uniform_watch.start("interp_watch");

    std::cout<<this->mpi->mpirank<<" start to interpolate wt on an uniform grid "<<std::endl;

    this->printDiffusionArrayFromVector(&this->w_t,"w_t_before_interpolation");

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));


    std::cout<<" N0 N1 N2 local_n0 "<<N0<<" "<<N1<<" "<<N2<<" "<<this->local_n0<< std::endl;


    // Create an interpolating function
    InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, &this->brick, this->nodes_neighbours);

    int n_p=0;
    for (int i = 0; i < this->local_n0; i++)
    {
        for (int j = 0; j < N1; j++)
        {
            for(int k=0;k<N2;k++)
            {

                double xyz [P4EST_DIM] =
                {
                    (double)(i+this->local_0_start)*this->Lx/(double)N0,
                    (double)j*this->Lx/(double) N1
    #ifdef P4_TO_P8
                    ,
                    (double)k*this->Lx/(double)N2
    #endif
                };

                // buffer the point
                w_func.add_point_to_buffer(n_p, xyz); n_p++;
            }
        }
    }



    // interpolate
    switch (this->my_interpolation_method)
    {
    case linear:
    {
        w_func.set_input_parameters(this->w_t, linear);
        w_func.interpolate(this->w3dGlobal);

        break;
    }
    case quadratic:
    {
        w_func.set_input_parameters(this->w_t, quadratic);
        w_func.interpolate(this->w3dGlobal);

        break;
    }
    case quadratic_non_oscillatory:
    {
        w_func.set_input_parameters(this->w_t, quadratic_non_oscillatory);
        w_func.interpolate(this->w3dGlobal);

        break;
    }
    default:
        throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }



    // Is it really needed

    this->printDiffusionArrayFromVector(&this->w3dGlobal,"w3DGlobal");

    this->interpolation_on_uniform_watch.stop();
    std::cout<<this->mpi->mpirank<<" finished to interpolate wt on an uniform grid "<<std::endl;


}

int Diffusion::interpolate_w_t_on_uniform_grid_2D()
{
    this->interpolation_on_uniform_watch.start("interp_watch");

    std::cout<<this->mpi->mpirank<<" start to interpolate wt on an uniform grid "<<std::endl;
    this->printDiffusionArrayFromVector(&this->w_t,"w_t_before_interpolation");
    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));
    std::cout<<" N0 N1 local_n0 "<<N0<<" "<<N1<<" "<<this->local_n0<< std::endl;

    // Create an interpolating function
    InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, &this->brick, this->nodes_neighbours);
    int n_p=0;
    for (int i = 0; i < this->local_n0; i++)
    {
        for (int j = 0; j < N1; j++)
        {

            double xy [P4EST_DIM] =
            {
                (double)(i+this->local_0_start)*this->Lx/(double)N0,
                (double)j*this->Lx/(double) N1

            };

            // buffer the point
            w_func.add_point_to_buffer(n_p, xy); n_p++;

        }
    }



    // interpolate
    switch (this->my_interpolation_method)
    {
    case linear:
    {
        w_func.set_input_parameters(this->w_t, linear);
        w_func.interpolate(this->w2dGlobal);

        break;
    }
    case quadratic:
    {
        w_func.set_input_parameters(this->w_t, quadratic);
        w_func.interpolate(this->w2dGlobal);

        break;
    }
    case quadratic_non_oscillatory:
    {
        w_func.set_input_parameters(this->w_t, quadratic_non_oscillatory);
        w_func.interpolate(this->w2dGlobal);

        break;
    }
    default:
        throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }



    // Is it really needed

    this->printDiffusionArrayFromVector(&this->w2dGlobal,"w2DGlobal");
    this->interpolation_on_uniform_watch.stop();
    std::cout<<this->mpi->mpirank<<" finished to interpolate wt on an uniform grid "<<std::endl;


}


int Diffusion::evolve_Equation_Spectral_Splitting_AMR()
{

    //std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    //    if(!this->FFTCreated)
    //        this->setup_spectral_solver_amr();

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));

    // this step should be done only after each change of the potential interpolated on an uniform grid
    this->ierr=VecGetArray(this->w3dGlobal,&this->w3D); CHKERRXX(this->ierr);

    if(!this->periodic_xyz)
    {
        throw std::runtime_error  ("no fftw for irregular domains have been implemented so far \n");
    }

    if(this->it==0)
    {
        this->ierr=VecSet(this->sol_t,1.00);    CHKERRXX(this->ierr);
        this->ierr=VecSet(this->sol_tp1,0.00);  CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);    CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);      CHKERRXX(this->ierr);
        if(this->forward_stage)
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_forward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_forward");
        }
        else
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_backward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_backward");
        }
    }
    if(this->it>0)
    {
        this->ierr=VecCopy(this->sol_tp1,this->sol_t); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }

    int local_size_fft_output;
    VecGetLocalSize(this->fftw_global_petsc_output,&local_size_fft_output);

    this->ierr=VecGetArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);


    if(this->it==0)
    {
        for (int i = 0; i < this->local_n0; i++)
        {
            for (int j = 0; j < N1; j++)
            {
                for(int k=0;k<N2;k++)
                {
                    this->input_forward[i*(N1*N2) + j*N2+k][0] =1.00;
                    this->input_forward[i*(N1*N2) + j*N2+k][1] =0;
                }
            }
        }
    }


    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {
            for(int k=0; k<N2;k++)
            {
                this->input_forward[k+j*N2+i*N2*N1][0]=this->input_forward[k+j*N2+i*N2*N1][0]*
                        exp(-this->w3D[i*(N1*N2) + j*N2+k]*this->dt/2.00);
                this->input_forward[k+j*N2+i*N2*N1][1]=0.00;
            }
        }
    }



    fftw_execute(this->plan_fftw_forward);
    double kx,ky,kz,k2;

    for(int ix=0;ix<this->local_n0;ix++)
    {
        if(ix+this->local_0_start>N0/2)
            kx=2*PI*(double) (ix+this->local_0_start-N0)/this->Lx_physics;
        else
            kx=2*PI*(double) (ix+this->local_0_start)/this->Lx_physics;


        for(int iy=0;iy<N1;iy++)
        {
            if(iy>N1/2)
                ky=2*PI*(double)(iy-N1)/this->Lx_physics;
            else
                ky=2*PI*(double)(iy)/this->Lx_physics;
            for(int iz=0;iz<N2;iz++)
            {
                if(iz>N2/2)
                    kz=2*PI*(double)(iz-N2)/this->Lx_physics;
                else
                    kz=2*PI*(double)iz/this->Lx_physics;

                k2=pow(kx,2)+pow(ky,2)+pow(kz,2);

                this->input_backward[iz+iy*N2+ix*N2*N1][0]=exp(-k2*this->dt)*this->output_forward[iz+iy*N2+ix*N2*N1][0];
                this->input_backward[iz+iy*N2+ix*N2*N1][1]=exp(-k2*this->dt)*this->output_forward[iz+iy*N2+ix*N2*N1][1];

            }
        }
    }

    fftw_execute(this->plan_fftw_backward);

    double n_fftw=N0*N1*N2;




    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {
            for(int k=0; k<N2;k++)
            {
                this->input_forward[k+j*N2+i*N2*N1][0]=(1/n_fftw)*this->output_backward[k+j*N2+i*N2*N1][0]*exp(-this->w3D[k+j*N2+i*N2*N1]*this->dt/2.00);
                if(this->input_forward[k+j*N2+i*N2*N1][0]<0)
                    this->input_forward[k+j*N2+i*N2*N1][0]=0;
                this->input_forward[k+j*N2+i*N2*N1][1]=0;
                this->fftw_local_petsc_output[k+j*N2+i*N2*N1]=this->input_forward[k+j*N2+i*N2*N1][0];
            }
        }
    }


    //this->printDiffusionArray(this->fftw_local_petsc_output,this->local_n0*N1*N2,"fftw_local_output");

    // Now set the fftw mpi output into a parallel petsc vector

    this->ierr=VecRestoreArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);

    this->ierr=VecAssemblyBegin(this->fftw_global_petsc_output); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(this->fftw_global_petsc_output); CHKERRXX(this->ierr);

    this->it++;
    PetscInt       to_n,from_n;
    VecGetLocalSize(this->fftw_global_petsc_output,&from_n);
    VecGetLocalSize(this->sol_tp1,&to_n);
    this->ierr=VecScatterBegin(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    if(this->printIntemediateSteps)
    {
        this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1");
        this->printDiffusionArrayFromVector(&this->sol_t,"sol_tp0");
    }

    this->ierr=VecRestoreArray(this->w3dGlobal,&this->w3D); CHKERRXX(this->ierr);

    // std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    return 0;
}


int Diffusion::evolve_Equation_Spectral_Splitting_AMR_2D()
{

    //std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    //    if(!this->FFTCreated)
    //        this->setup_spectral_solver_amr();

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));


    // this step should be done only after each change of the potential interpolated on an uniform grid
    this->ierr=VecGetArray(this->w2dGlobal,&this->w2D); CHKERRXX(this->ierr);

    //    if(!this->periodic_xyz)
    //    {
    //        throw std::runtime_error  ("no fftw for irregular domains have been implemented so far \n");
    //    }

    if(this->it==0)
    {
        this->ierr=VecSet(this->sol_t,1.00);    CHKERRXX(this->ierr);
        this->ierr=VecSet(this->sol_tp1,0.00);  CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);    CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);      CHKERRXX(this->ierr);
        if(this->forward_stage)
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_forward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_forward");
        }
        else
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_backward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_backward");
        }
    }
    if(this->it>0)
    {
        this->ierr=VecCopy(this->sol_tp1,this->sol_t); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }

    int local_size_fft_output;
    VecGetLocalSize(this->fftw_global_petsc_output,&local_size_fft_output);

    this->ierr=VecGetArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);


    if(this->it==0)
    {
        for (int i = 0; i < this->local_n0; i++)
        {
            for (int j = 0; j < N1; j++)
            {
                this->input_forward[i*N1 + j][0] =1.00;
                this->input_forward[i*N1 + j][1] =0;
            }

        }
    }


    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {

            this->input_forward[j+i*N1][0]=this->input_forward[j+i*N1][0]*
                    exp(-this->w2D[i*N1 + j]*this->dt/2.00);
            this->input_forward[j+i*N1][1]=0.00;

        }
    }



    fftw_execute(this->plan_fftw_forward);
    double kx,ky,k2;

    for(int ix=0;ix<this->local_n0;ix++)
    {
        if(ix+this->local_0_start>N0/2)
            kx=2*PI*(double) (ix+this->local_0_start-N0)/this->Lx_physics;
        else
            kx=2*PI*(double) (ix+this->local_0_start)/this->Lx_physics;


        for(int iy=0;iy<N1;iy++)
        {
            if(iy>N1/2)
                ky=2*PI*(double)(iy-N1)/this->Lx_physics;
            else
                ky=2*PI*(double)(iy)/this->Lx_physics;

            k2=pow(kx,2)+pow(ky,2);

            this->input_backward[iy+ix*N1][0]=exp(-k2*this->dt)*this->output_forward[iy+ix*N1][0];
            this->input_backward[iy+ix*N1][1]=exp(-k2*this->dt)*this->output_forward[iy+ix*N1][1];


        }
    }

    fftw_execute(this->plan_fftw_backward);

    double n_fftw=N0*N1;




    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {

            this->input_forward[j+i*N1][0]=(1/n_fftw)*this->output_backward[j+i*N1][0]*exp(-this->w2D[j+i*N1]*this->dt/2.00);
            if(this->input_forward[j+i*N1][0]<0)
                this->input_forward[j+i*N1][0]=0;
            this->input_forward[j+i*N1][1]=0;
            this->fftw_local_petsc_output[j+i*N1]=this->input_forward[j+i*N1][0];

        }
    }


    //this->printDiffusionArray(this->fftw_local_petsc_output,this->local_n0*N1*N2,"fftw_local_output");

    // Now set the fftw mpi output into a parallel petsc vector

    this->ierr=VecRestoreArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);

    this->ierr=VecAssemblyBegin(this->fftw_global_petsc_output); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(this->fftw_global_petsc_output); CHKERRXX(this->ierr);

    this->it++;
    PetscInt       to_n,from_n;
    VecGetLocalSize(this->fftw_global_petsc_output,&from_n);
    VecGetLocalSize(this->sol_tp1,&to_n);
    this->ierr=VecScatterBegin(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    if(this->printIntemediateSteps)
    {
        this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1",PETSC_TRUE);
        this->printDiffusionArrayFromVector(&this->sol_t,"sol_tp0",PETSC_TRUE);
    }

    this->ierr=VecRestoreArray(this->w2dGlobal,&this->w2D); CHKERRXX(this->ierr);

    // std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    return 0;
}



int Diffusion::evolve_Equation_Spectral_Splitting_AMR_r2r()
{

    //std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    //    if(!this->FFTCreated)
    //        this->setup_spectral_solver_amr();

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2));

    const ptrdiff_t N2_effective=2*(N2/2+1);
    const ptrdiff_t N2_half_effective=(N2/2+1);


    // this step should be done only after each change of the potential interpolated on an uniform grid
    this->ierr=VecGetArray(this->w3dGlobal,&this->w3D); CHKERRXX(this->ierr);

    if(!this->periodic_xyz)
    {
        throw std::runtime_error  ("no fftw for irregular domains have been implemented so far \n");
    }

    if(this->it==0)
    {
        this->ierr=VecSet(this->sol_t,1.00);    CHKERRXX(this->ierr);
        this->ierr=VecSet(this->sol_tp1,0.00);  CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);    CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);      CHKERRXX(this->ierr);
        if(this->forward_stage)
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_forward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_forward");
        }
        else
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_backward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_backward");
        }
    }
    if(this->it>0)
    {
        this->ierr=VecCopy(this->sol_tp1,this->sol_t); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }

    int local_size_fft_output;
    VecGetLocalSize(this->fftw_global_petsc_output,&local_size_fft_output);

    this->ierr=VecGetArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);


    if(this->it==0)
    {
        for (int i = 0; i < this->local_n0; i++)
        {
            for (int j = 0; j < N1; j++)
            {
                for(int k=0;k<N2;k++)
                {
                    this->input_forward_real[i*(N1*N2_effective) + j*N2_effective+k] =1.00;
                    //this->input_forward[i*(N1*N2) + j*N2+k][1] =0;
                }
            }
        }
    }


    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {
            for(int k=0; k<N2;k++)
            {
                this->input_forward_real[k+j*N2_effective+i*N2_effective*N1]=this->input_forward_real[k+j*N2_effective+i*N2_effective*N1]*
                        exp(-this->w3D[i*(N1*N2) + j*N2+k]*this->dt/2.00);
                //this->input_forward[k+j*N2+i*N2*N1][1]=0.00;
            }
        }
    }



    fftw_execute(this->plan_fftw_forward);
    double kx,ky,kz,k2;

    for(int ix=0;ix<this->local_n0;ix++)
    {
        if(ix+this->local_0_start>N0/2)
            kx=2*PI*(double) (ix+this->local_0_start-N0)/this->Lx_physics;
        else
            kx=2*PI*(double) (ix+this->local_0_start)/this->Lx_physics;


        for(int iy=0;iy<N1;iy++)
        {
            if(iy>N1/2)
                ky=2*PI*(double)(iy-N1)/this->Lx_physics;
            else
                ky=2*PI*(double)(iy)/this->Lx_physics;
            for(int iz=0;iz<=N2/2;iz++)
            {
                if(iz>N2/2)
                    kz=2*PI*(double)(iz-N2)/this->Lx_physics;
                else
                    kz=2*PI*(double)iz/this->Lx_physics;

                k2=pow(kx,2)+pow(ky,2)+pow(kz,2);

                this->input_backward[iz+iy*N2_half_effective+ix*N2_half_effective*N1][0]=
                        exp(-k2*this->dt)*this->output_forward[iz+iy*N2_half_effective+ix*N2_half_effective*N1][0];
                this->input_backward[iz+iy*N2_half_effective+ix*N2_half_effective*N1][1]=
                        exp(-k2*this->dt)*this->output_forward[iz+iy*N2_half_effective+ix*N2_half_effective*N1][1];

            }
        }
    }

    fftw_execute(this->plan_fftw_backward);

    double n_fftw=N0*N1*N2;




    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {
            for(int k=0; k<N2;k++)
            {
                this->input_forward_real[k+j*N2_effective+i*N2_effective*N1]=(1/n_fftw)*this->output_backward_real[k+j*N2_effective+i*N2_effective*N1]*exp(-this->w3D[k+j*N2+i*N2*N1]*this->dt/2.00);
                if(this->input_forward_real[k+j*N2_effective+i*N2_effective*N1]<0)
                {
                    //                                      std::cout<<"q<0 "<<this->input_forward_real[k+j*N2_effective+i*N2_effective*N1]<<std::endl;
                    //                    throw std::runtime_error("q<0");
                    this->input_forward_real[k+j*N2_effective+i*N2_effective*N1]=0;

                }

                this->fftw_local_petsc_output[k+j*N2+i*N2*N1]=this->input_forward_real[k+j*N2_effective+i*N2_effective*N1];
            }
        }
    }


    this->printDiffusionArray(this->fftw_local_petsc_output,this->local_n0*N1*N2,"fftw_local_output");
    if(this->it==(this->N_iterations-1))
    {
        if(this->myMeanFieldPlan->writeLastq2TextFile)
            if(this->forward_stage)
                this->print_uniform_local_array_to_text_file(this->fftw_local_petsc_output,"q_last_forward");
        else
                this->print_uniform_local_array_to_text_file(this->fftw_local_petsc_output,"q_last_backward");
    }
    // Now set the fftw mpi output into a parallel petsc vector

    this->ierr=VecRestoreArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);

    this->ierr=VecAssemblyBegin(this->fftw_global_petsc_output); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(this->fftw_global_petsc_output); CHKERRXX(this->ierr);




    this->it++;

    PetscInt       to_n,from_n;
    VecGetLocalSize(this->fftw_global_petsc_output,&from_n);
    VecGetLocalSize(this->sol_tp1,&to_n);
    this->ierr=VecScatterBegin(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    if(this->printIntemediateSteps)
    {
        this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1");
        this->printDiffusionArrayFromVector(&this->sol_t,"sol_tp0");
    }

    this->ierr=VecRestoreArray(this->w3dGlobal,&this->w3D); CHKERRXX(this->ierr);

    // std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    return 0;
}


int Diffusion::evolve_Equation_Spectral_Splitting_AMR_2D_r2r()
{

    //std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    //    if(!this->FFTCreated)
    //        this->setup_spectral_solver_amr();

    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2));


    const ptrdiff_t N1_effective=2*(N1/2+1);
    const ptrdiff_t N1_half_effective=(N1/2+1);



    // this step should be done only after each change of the potential interpolated on an uniform grid
    this->ierr=VecGetArray(this->w2dGlobal,&this->w2D); CHKERRXX(this->ierr);

    //    if(!this->periodic_xyz)
    //    {
    //        throw std::runtime_error  ("no fftw for irregular domains have been implemented so far \n");
    //    }

    if(this->it==0)
    {
        this->ierr=VecSet(this->sol_t,1.00);    CHKERRXX(this->ierr);
        this->ierr=VecSet(this->sol_tp1,0.00);  CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);    CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD);      CHKERRXX(this->ierr);
        if(this->forward_stage)
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_forward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_forward");
        }
        else
        {
            this->printDiffusionArrayFromVector(&this->sol_t,"sol_0_backward");
            this->printDiffusionArrayFromVector(&this->w_t,"w_t_backward");
        }
    }
    if(this->it>0)
    {
        this->ierr=VecCopy(this->sol_tp1,this->sol_t); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sol_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }

    int local_size_fft_output;
    VecGetLocalSize(this->fftw_global_petsc_output,&local_size_fft_output);

    this->ierr=VecGetArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);


    if(this->it==0)
    {
        for (int i = 0; i < this->local_n0; i++)
        {
            for (int j = 0; j < N1; j++)
            {
                this->input_forward_real[i*N1_effective + j] =1.00;
                //  this->input_forward[i*N1_effective + j][1] =0;
            }

        }
    }

    this->printDiffusionArray((double *)this->input_forward_real,this->local_n0*N1_effective,"input_forward_real_1");

    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {

            this->input_forward_real[j+i*N1_effective]=this->input_forward_real[j+i*N1_effective]*
                    exp(-this->w2D[i*N1 + j]*this->dt/2.00);
            //this->input_forward[j+i*N1][1]=0.00;

        }
    }


    this->printDiffusionArray((double *)this->input_forward_real,this->local_n0*N1_effective,"input_forward_real_2");

    fftw_execute(this->plan_fftw_forward);
    this->printDiffusionArray((double *)this->output_forward,this->local_n0*N1_effective,"output_forward_real");

    double kx,ky,k2;

    for(int ix=0;ix<this->local_n0;ix++)
    {
        if(ix+this->local_0_start>N0/2)
            kx=2*PI*(double) (ix+this->local_0_start-N0)/this->Lx_physics;
        else
            kx=2*PI*(double) (ix+this->local_0_start)/this->Lx_physics;


        for(int iy=0;iy<=N1/2;iy++)
        {
            if(iy>N1/2)
                ky=2*PI*(double)(iy-N1)/this->Lx_physics;
            else
                ky=2*PI*(double)(iy)/this->Lx_physics;

            k2=pow(kx,2)+pow(ky,2);

            this->input_backward[iy+ix*N1_half_effective][0]
                    =exp(-k2*this->dt)*this->output_forward[iy+ix*N1_half_effective][0];
            this->input_backward[iy+ix*N1_half_effective][1]
                    =exp(-k2*this->dt)*this->output_forward[iy+ix*N1_half_effective][1];


        }
    }
    this->printDiffusionArray((double *)this->input_backward,2*this->local_n0*N1_half_effective,"input_backward");

    fftw_execute(this->plan_fftw_backward);

    double n_fftw=N0*N1;


    this->printDiffusionArray(this->output_backward_real,this->local_n0*N1,"output_backward_real");


    for(int i=0;i<this->local_n0;i++)
    {
        for(int j=0;j<N1;j++)
        {

            this->input_forward_real[j+i*N1_effective]=
                    (1/n_fftw)*this->output_backward_real[j+i*N1_effective]*exp(-this->w2D[j+i*N1]*this->dt/2.00);
            if(this->input_forward_real[j+i*N1_effective]<0)
                this->input_forward_real[j+i*N1_effective]=0;

            this->fftw_local_petsc_output[j+i*N1]=this->input_forward_real[j+i*N1_effective];

        }
    }


    //this->printDiffusionArray(this->fftw_local_petsc_output,this->local_n0*N1*N2,"fftw_local_output");

    // Now set the fftw mpi output into a parallel petsc vector
    this->printDiffusionArray(this->fftw_local_petsc_output,this->local_n0*N1,"fftw_local_petsc_output");

    this->ierr=VecRestoreArray(this->fftw_global_petsc_output,&this->fftw_local_petsc_output); CHKERRXX(this->ierr);

    this->ierr=VecAssemblyBegin(this->fftw_global_petsc_output); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(this->fftw_global_petsc_output); CHKERRXX(this->ierr);


    this->it++;

    this->fftw_scaterring_watch.start_without_printing("");
    PetscInt       to_n,from_n;
    VecGetLocalSize(this->fftw_global_petsc_output,&from_n);
    VecGetLocalSize(this->sol_tp1,&to_n);
    this->ierr=VecScatterBegin(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(this->scatter_from_fftw_to_p4est,this->fftw_global_petsc_output,this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sol_tp1,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->fftw_scaterring_watch.stop();
    this->fftw_scaterring_watch_total_time+=this->fftw_scaterring_watch.get_duration();

    //if(this->printIntemediateSteps)
    {
        this->printDiffusionArrayFromVector(&this->sol_tp1,"sol_tp1",PETSC_TRUE);
        this->printDiffusionArrayFromVector(&this->sol_t,"sol_tp0",PETSC_TRUE);
    }

    this->ierr=VecRestoreArray(this->w2dGlobal,&this->w2D); CHKERRXX(this->ierr);



    // std::cout<<this->mpi->mpirank<<" "<<this->it<<" fftw "<<std::endl;

    return 0;
}


int Diffusion::print_uniform_local_array_to_text_file(PetscScalar *array2PrintOnUniformGrid, string str_file)
{


        std::cout<<" this->mpi->mpisize "<<this->mpi->mpisize<<std::endl;

        const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2));
        std::cout<<" N0 " <<N0<< std::endl;
        double i_local_start=(double)this->mpi->mpirank*(double)N0/(double)this->mpi->mpisize;
        int mpi_size;
        int mpi_rank;
        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
        std::stringstream oss2DebugVec;
        std::string mystr2Debug;
        oss2DebugVec << this->convert2FullPath(str_file)<<"_"<<mpi_rank<<".txt";
        mystr2Debug=oss2DebugVec.str();
        int n_p=0;
        FILE *outFile;
        outFile=fopen(mystr2Debug.c_str(),"w");
        std::cout<<this->mpi->mpirank<<" start to print on an uniform grid "<<str_file<<std::endl;
        for (int i = 0; i < N0/this->mpi->mpisize; i++)
        {
            for (int j = 0; j < N0; j++)
            {

#ifdef P4_TO_P8
                for(int k=0;k<N0;k++)
                {
#endif

                    double xyz [P4EST_DIM] =
                    {
                        (double)((double)i+i_local_start)*this->Lx/(double)N0,
                        (double)j*this->Lx/(double) N0
    #ifdef P4_TO_P8
                        ,(double) k*this->Lx/N0
    #endif
                    };


#ifdef P4_TO_P8
                    fprintf(outFile,"%d %d %d %d %f %f %f %f \n",mpi_rank, i,j,k,xyz[0],xyz[1],xyz[2],array2PrintOnUniformGrid[k+j*N0+i*N0*N0]); n_p++;
                }
#else
                    fprintf(outFile,"%d %d %d %f %f %f\n",mpi_rank, i,j,xyz[0],xyz[1],array2PrintOnUniformGrid[j+i*N0]);n_p++;
#endif
                }

            }

            fclose(outFile);


            std::cout<<this->mpi->mpirank<<" finished to print on an uniform grid "<<str_file<<std::endl;
        }















int Diffusion::get_vec_from_text_file(Vec *v2Fill, string file_name,int columns2Skip,int columns2Continue)
{

    //step 0: declare variables
    IS from_unf;
    IS to_p4est;
    VecScatter  scatter_from_unf_to_p4est;
    Vec         unf_global_petsc_input;
    PetscScalar *unf_local_petsc_input;


    // step 1: create parallel mapping data structure
    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2))
        #ifdef P4_TO_P8
            ,N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))
        #endif
            ;
    int n_unf=N0*N1;
#ifdef P4_TO_P8
    n_unf=n_unf*N2;
#endif
    int n_unf_local=n_unf/this->mpi->mpisize;


    this->ierr=VecCreate(MPI_COMM_WORLD,&unf_global_petsc_input); CHKERRXX(this->ierr);
    this->ierr=VecSetSizes(unf_global_petsc_input,n_unf_local,n_unf); CHKERRXX(this->ierr);
    this->ierr=VecSetFromOptions(unf_global_petsc_input); CHKERRXX(this->ierr);

    int n_local_size_amr;
    this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

    // create the mapping data structure
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    std::cout<<this->mpi->mpirank<<" global number "<<global_node_number<<std::endl;
    std::cout<<this->mpi->mpirank<<" n_local_size_sol "<<n_local_size_amr<<std::endl;

    double i_node,j_node,x,y, offset_tree_x, offset_tree_y,
            x_normalized,y_normalized, tree_xmin,tree_ymin;

#ifdef P4_TO_P8
    double k_node,z, offset_tree_z,z_normalized,tree_zmin;
#endif

    int *idx_from_unf_to_p4est_unf_ix=new int[n_local_size_amr];
    int *idx_from_unf_to_p4est_p4est_ix=new int[n_local_size_amr];

    double max_pow=log(P4EST_ROOT_LEN)/log(2);
    double normalizer=pow(2,max_pow-this->max_level);

    std::cout<<max_pow<<" "<<normalizer<<" "<<P4EST_ROOT_LEN<<std::endl;

    for (int n = 0; n < n_local_size_amr; n++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = this->connectivity->vertices[3*v_mm + 1];


        x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
        y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);



        offset_tree_x=tree_xmin/this->Lx*N0;
        offset_tree_y=tree_ymin/this->Lx*N0;

        x_normalized=x/normalizer;
        y_normalized=y/normalizer;

        i_node=x_normalized+offset_tree_x;
        j_node=y_normalized+offset_tree_y;


        i_node=(int)i_node%N0;
        j_node=(int)j_node%N1;

        int idx_unf_temp=(int)(i_node*((double)N1) + j_node);
#ifdef P4_TO_P8
        tree_zmin=this->connectivity->vertices[3*v_mm + 2];
        z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
        offset_tree_z=tree_zmin/this->Lx*N0;
        z_normalized=z/normalizer;
        k_node=z_normalized+offset_tree_z;
        k_node=(int)k_node%N2;
        idx_unf_temp=(int)(i_node*((double)N1*N2) + j_node*(double)N2+k_node);
#endif







        idx_from_unf_to_p4est_p4est_ix[n]=global_node_number+n;
        idx_from_unf_to_p4est_unf_ix[n]=idx_unf_temp;
    }


    PetscInt       to_n,from_n;
    this->ierr=VecGetLocalSize(unf_global_petsc_input,&from_n); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(*v2Fill,&to_n); CHKERRXX(this->ierr);
    this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,idx_from_unf_to_p4est_unf_ix,PETSC_COPY_VALUES,&from_unf); CHKERRXX(this->ierr);
    this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,idx_from_unf_to_p4est_p4est_ix,PETSC_COPY_VALUES,&to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterCreate(unf_global_petsc_input,from_unf,*v2Fill,to_p4est,&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);

    std::cout<<this->mpi->mpirank<<" finished to create mapping data structure uniform to p4est "<<std::endl;


    // step 2: read to uniform vector


    std::cout<<"Started to load uniform parallel vector"<<std::endl;
    fstream my_fstream;

    my_fstream.open(file_name.c_str(),ios_base::in);
    int ip,ix,ij;
    double x_dummy,y_dummy,ls_ij;

#ifdef P4_TO_P8
    int iz;
    double z_dummy;
#endif

    this->ierr=VecGetArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            for(int k=0; k<N2; k++)
            {

                for(int i_column=0;i_column<columns2Skip;i_column++)
                {
                    my_fstream>>ls_ij;
                }


                unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
                for(int i_column=0;i_column<columns2Continue;i_column++)
                {
                    my_fstream>>ls_ij;
                }
            }
        }
    }
#else
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {

            for(int i_column=0;i_column<columns2Skip;i_column++)
            {
                my_fstream>>ls_ij;
            }
            unf_local_petsc_input[j+i*N1]=ls_ij;
            for(int i_column=0;i_column<columns2Continue;i_column++)
            {
                my_fstream>>ls_ij;
            }
        }
    }
#endif
    my_fstream.close();
    this->ierr=VecRestoreArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&unf_global_petsc_input);
    // step 3: scatter from uniform grid to parallel AMR vector

    this->ierr=VecScatterBegin(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=ISDestroy(from_unf); CHKERRXX(this->ierr);
    this->ierr=ISDestroy(to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterDestroy(&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);
    delete idx_from_unf_to_p4est_unf_ix;
    delete idx_from_unf_to_p4est_p4est_ix;


    // step 4: interpolate back to uniform an print to matlab
    // (for debugging purposes)


}

int Diffusion::get_coarse_vec_from_text_file(Vec *v2Fill, string file_name,int column)
{

    //step 0: declare variables
    IS from_unf;
    IS to_p4est;
    VecScatter  scatter_from_unf_to_p4est;
    Vec         unf_global_petsc_input;
    PetscScalar *unf_local_petsc_input;


    // step 1: create parallel mapping data structure
    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2))
        #ifdef P4_TO_P8
            ,N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))
        #endif
            ;
    int n_unf=N0*N1;
#ifdef P4_TO_P8
    n_unf=n_unf*N2;
#endif
    int n_unf_local=n_unf/this->mpi->mpisize;


    this->ierr=VecCreate(MPI_COMM_WORLD,&unf_global_petsc_input); CHKERRXX(this->ierr);
    this->ierr=VecSetSizes(unf_global_petsc_input,n_unf_local,n_unf); CHKERRXX(this->ierr);
    this->ierr=VecSetFromOptions(unf_global_petsc_input); CHKERRXX(this->ierr);

    int n_local_size_amr;
    this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

    // create the mapping data structure
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    std::cout<<this->mpi->mpirank<<" global number "<<global_node_number<<std::endl;
    std::cout<<this->mpi->mpirank<<" n_local_size_sol "<<n_local_size_amr<<std::endl;

    double i_node,j_node,x,y, offset_tree_x, offset_tree_y,
            x_normalized,y_normalized, tree_xmin,tree_ymin;

#ifdef P4_TO_P8
    double k_node,z, offset_tree_z,z_normalized,tree_zmin;
#endif

    int *idx_from_unf_to_p4est_unf_ix=new int[n_local_size_amr];
    int *idx_from_unf_to_p4est_p4est_ix=new int[n_local_size_amr];

    double max_pow=log(P4EST_ROOT_LEN)/log(2);
    double normalizer=pow(2,max_pow-this->max_level);

    std::cout<<max_pow<<" "<<normalizer<<" "<<P4EST_ROOT_LEN<<std::endl;

    for (int n = 0; n < n_local_size_amr; n++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = this->connectivity->vertices[3*v_mm + 1];


        x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
        y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);



        offset_tree_x=tree_xmin/this->Lx*N0;
        offset_tree_y=tree_ymin/this->Lx*N0;

        x_normalized=x/normalizer;
        y_normalized=y/normalizer;

        i_node=x_normalized+offset_tree_x;
        j_node=y_normalized+offset_tree_y;


        i_node=(int)i_node%N0;
        j_node=(int)j_node%N1;

        int idx_unf_temp=(int)(i_node*((double)N1) + j_node);
#ifdef P4_TO_P8
        tree_zmin=this->connectivity->vertices[3*v_mm + 2];
        z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
        offset_tree_z=tree_zmin/this->Lx*N0;
        z_normalized=z/normalizer;
        k_node=z_normalized+offset_tree_z;
        k_node=(int)k_node%N2;
        idx_unf_temp=(int)(i_node*((double)N1*N2) + j_node*(double)N2+k_node);
#endif







        idx_from_unf_to_p4est_p4est_ix[n]=global_node_number+n;
        idx_from_unf_to_p4est_unf_ix[n]=idx_unf_temp;
    }


    PetscInt       to_n,from_n;
    this->ierr=VecGetLocalSize(unf_global_petsc_input,&from_n); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(*v2Fill,&to_n); CHKERRXX(this->ierr);
    this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,idx_from_unf_to_p4est_unf_ix,PETSC_COPY_VALUES,&from_unf); CHKERRXX(this->ierr);
    this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,idx_from_unf_to_p4est_p4est_ix,PETSC_COPY_VALUES,&to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterCreate(unf_global_petsc_input,from_unf,*v2Fill,to_p4est,&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);

    std::cout<<this->mpi->mpirank<<" finished to create mapping data structure uniform to p4est "<<std::endl;


    // step 2: read to uniform vector


    std::cout<<"Started to load uniform parallel vector"<<std::endl;
    fstream my_fstream;

    my_fstream.open(file_name.c_str(),ios_base::in);
    int ip,ix,ij;
    double x_dummy,y_dummy,ls_ij;

#ifdef P4_TO_P8
    int iz;
    double z_dummy;
#endif

    this->ierr=VecGetArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            for(int k=0; k<N2; k++)
            {

                if(i>=N0/4.00 && i<3.00*N0/4.00
                        &&j>=N0/4.00 && j<3.00*N0/4.00
                        &&k>=N0/4.00 && k<3.00*N0/4.00 )
                {
                    for(int i_column=0;i_column<column;i_column++)
                    {
                        my_fstream>>ls_ij;
                    }


                    unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
                }
                else
                {
                    unf_local_petsc_input[i*(N1*N2) + j*N2+k]=0.00;
                }
            }
        }
    }
#else
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {

            for(int i_column=0;i_column<column;i_column++)
            {
                my_fstream>>ls_ij;
            }
            unf_local_petsc_input[j+i*N1]=ls_ij;
        }
    }
#endif
    my_fstream.close();
    this->ierr=VecRestoreArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&unf_global_petsc_input);
    // step 3: scatter from uniform grid to parallel AMR vector

    this->ierr=VecScatterBegin(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=ISDestroy(from_unf); CHKERRXX(this->ierr);
    this->ierr=ISDestroy(to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterDestroy(&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);
    delete idx_from_unf_to_p4est_unf_ix;
    delete idx_from_unf_to_p4est_p4est_ix;


    // step 4: interpolate back to uniform an print to matlab
    // (for debugging purposes)


}


int Diffusion::get_coarse_field_from_fine_text_file(Vec *v2Fill, string file_name,int file_level,int columns2Skip,int columns2Continue)
{

    //step 0: declare variables
    IS from_unf;
    IS to_p4est;
    VecScatter  scatter_from_unf_to_p4est;
    Vec         unf_global_petsc_input;
    PetscScalar *unf_local_petsc_input;


    // step 1: create parallel mapping data structure
    const ptrdiff_t N0 =(int)pow(2,file_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,file_level+log(this->nx_trees)/log(2))
        #ifdef P4_TO_P8
            ,N2=(int)pow(2,file_level+log(this->nx_trees)/log(2))
        #endif
            ;
    int n_unf=N0*N1;
#ifdef P4_TO_P8
    n_unf=n_unf*N2;
#endif
    int n_unf_local=n_unf/this->mpi->mpisize;


    this->ierr=VecCreate(MPI_COMM_WORLD,&unf_global_petsc_input); CHKERRXX(this->ierr);
    this->ierr=VecSetSizes(unf_global_petsc_input,n_unf_local,n_unf); CHKERRXX(this->ierr);
    this->ierr=VecSetFromOptions(unf_global_petsc_input); CHKERRXX(this->ierr);

    int n_local_size_amr;
    this->ierr=VecGetLocalSize(this->phi,&n_local_size_amr); CHKERRXX(this->ierr);

    // create the mapping data structure
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    std::cout<<this->mpi->mpirank<<" global number "<<global_node_number<<std::endl;
    std::cout<<this->mpi->mpirank<<" n_local_size_sol "<<n_local_size_amr<<std::endl;

    double i_node,j_node,x,y, offset_tree_x, offset_tree_y,
            x_normalized,y_normalized, tree_xmin,tree_ymin;

#ifdef P4_TO_P8
    double k_node,z, offset_tree_z,z_normalized,tree_zmin;
#endif

    int *idx_from_unf_to_p4est_unf_ix=new int[n_local_size_amr];
    int *idx_from_unf_to_p4est_p4est_ix=new int[n_local_size_amr];

    double max_pow=log(P4EST_ROOT_LEN)/log(2);
    double normalizer=pow(2,max_pow-file_level);

    std::cout<<max_pow<<" "<<normalizer<<" "<<P4EST_ROOT_LEN<<std::endl;

    for (int n = 0; n < n_local_size_amr; n++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = this->connectivity->vertices[3*v_mm + 1];


        x=(node->x==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->x);
        y=(node->y==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->y);



        offset_tree_x=tree_xmin/this->Lx*N0;
        offset_tree_y=tree_ymin/this->Lx*N0;

        x_normalized=x/normalizer;
        y_normalized=y/normalizer;

        i_node=x_normalized+offset_tree_x;
        j_node=y_normalized+offset_tree_y;


        i_node=(int)i_node%N0;
        j_node=(int)j_node%N1;

        int idx_unf_temp=(int)(i_node*((double)N1) + j_node);

#ifdef P4_TO_P8
        tree_zmin=this->connectivity->vertices[3*v_mm + 2];
        z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
        offset_tree_z=tree_zmin/this->Lx*N0;
        z_normalized=z/normalizer;
        k_node=z_normalized+offset_tree_z;
        k_node=(int)k_node%N2;
        idx_unf_temp=(int)(i_node*((double)N1*N2) + j_node*(double)N2+k_node);
#endif







        idx_from_unf_to_p4est_p4est_ix[n]=global_node_number+n;
        idx_from_unf_to_p4est_unf_ix[n]=idx_unf_temp;
    }


    PetscInt       to_n,from_n;
    this->ierr=VecGetLocalSize(unf_global_petsc_input,&from_n); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(*v2Fill,&to_n); CHKERRXX(this->ierr);
    this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,idx_from_unf_to_p4est_unf_ix,PETSC_COPY_VALUES,&from_unf); CHKERRXX(this->ierr);
    this->ierr=ISCreateGeneral(PETSC_COMM_WORLD,n_local_size_amr,idx_from_unf_to_p4est_p4est_ix,PETSC_COPY_VALUES,&to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterCreate(unf_global_petsc_input,from_unf,*v2Fill,to_p4est,&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);

    std::cout<<this->mpi->mpirank<<" finished to create mapping data structure uniform to p4est "<<std::endl;


    // step 2: read to uniform vector


    std::cout<<"Started to load uniform parallel vector"<<std::endl;
    fstream my_fstream;

    my_fstream.open(file_name.c_str(),ios_base::in);
    int ip,ix,ij;
    double x_dummy,y_dummy,ls_ij;

#ifdef P4_TO_P8
    int iz;
    double z_dummy;
#endif

    this->ierr=VecGetArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            for(int k=0; k<N2; k++)
            {
                for(int i_column=0;i_column<columns2Skip;i_column++)
                {
                    my_fstream>>ls_ij;
                }


                unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
                for(int i_column=0;i_column<columns2Continue;i_column++)
                {
                    my_fstream>>ls_ij;
                }
            }
        }
    }
#else
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            for(int i_column=0;i_column<columns2Skip;i_column++)
            {
                my_fstream>>ls_ij;
            }
            unf_local_petsc_input[j+i*N1]=ls_ij;
            for(int i_column=0;i_column<columns2Continue;i_column++)
            {
                my_fstream>>ls_ij;
            }
        }
    }
#endif


    my_fstream.close();
    this->ierr=VecRestoreArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&unf_global_petsc_input);
    // step 3: scatter from uniform grid to parallel AMR vector

    this->ierr=VecScatterBegin(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=ISDestroy(from_unf); CHKERRXX(this->ierr);
    this->ierr=ISDestroy(to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterDestroy(&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);
    delete idx_from_unf_to_p4est_unf_ix;
    delete idx_from_unf_to_p4est_p4est_ix;


    // step 4: interpolate back to uniform an print to matlab
    // (for debugging purposes)


}


int Diffusion::interpolate_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid,std::string str_file)
{
    this->scatter_petsc_vector(vec2PrintOnUniformGrid);
        std::cout<<" this->mpi->mpisize "<<this->mpi->mpisize<<std::endl;
        Vec v_uniform;
        std::cout<<this->mpi->mpirank<<" start to interpolate on an uniform grid "<<str_file<<std::endl;
        const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2));
        std::cout<<" N0 " <<N0<< std::endl;

        int size2SetLocal,size2SetGlobal;
        size2SetGlobal=N0*N0;
#ifdef P4_TO_P8
        size2SetGlobal=size2SetGlobal*N0;
#endif
        size2SetLocal=size2SetGlobal/this->mpi->mpisize;

        this->ierr=VecCreate(MPI_COMM_WORLD,&v_uniform); CHKERRXX(this->ierr);
        this->ierr=VecSetSizes(v_uniform,size2SetLocal,size2SetGlobal); CHKERRXX(this->ierr);
        this->ierr=VecSetFromOptions(v_uniform); CHKERRXX(this->ierr);



        double i_local_start=(double)this->mpi->mpirank*(double)N0/(double)this->mpi->mpisize;
        // Create an interpolating function
        InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, &this->brick, this->nodes_neighbours);
        int n_p=0;
        std::cout<<this->mpi->mpirank<<" start to buffer on an uniform grid "<<str_file<<std::endl;
        for (int i = 0; i < N0/this->mpi->mpisize; i++)
        {
            for (int j = 0; j < N0; j++)
            {
#ifdef P4_TO_P8
                for(int k=0;k<N0;k++)
                {
#endif
                    double xyz [P4EST_DIM] =
                    {
                        (double)((double)i+i_local_start)*this->Lx/(double)N0,
                        (double)j*this->Lx/(double) N0
    #ifdef P4_TO_P8
                        ,(double) k*this->Lx/N0
    #endif
                    };
                    // buffer the point
                    w_func.add_point_to_buffer(n_p, xyz); n_p++;
#ifdef P4_TO_P8
                }
#endif

            }

        }
        std::cout<<this->mpi->mpirank<<" finished to buffer on an uniform grid "<<str_file<<std::endl;
        w_func.set_input_parameters(*vec2PrintOnUniformGrid, this->my_interpolation_method);
        std::cout<<this->mpi->mpirank<<" start to interpolate on an uniform grid "<<str_file<<std::endl;
        w_func.interpolate(v_uniform);
        std::cout<<this->mpi->mpirank<<" finished to interpolate on an uniform grid "<<str_file<<std::endl;
        // Is it really needed


        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2Debug;
        oss2DebugVec << this->convert2FullPath(str_file)<<"_"<<mpi_rank<<".txt";
        mystr2Debug=oss2DebugVec.str();

        n_p=0;
        FILE *outFile;
        outFile=fopen(mystr2Debug.c_str(),"w");


        PetscScalar *v_local;
        this->ierr=VecGetArray(v_uniform,&v_local); CHKERRXX(this->ierr);
        std::cout<<this->mpi->mpirank<<" start to print on an uniform grid "<<str_file<<std::endl;
        for (int i = 0; i < N0/this->mpi->mpisize; i++)
        {
            for (int j = 0; j < N0; j++)
            {

#ifdef P4_TO_P8
                for(int k=0;k<N0;k++)
                {
#endif

                    double xyz [P4EST_DIM] =
                    {
                        (double)((double)i+i_local_start)*this->Lx/(double)N0,
                        (double)j*this->Lx/(double) N0
    #ifdef P4_TO_P8
                        ,(double) k*this->Lx/N0
    #endif
                    };


#ifdef P4_TO_P8
                    fprintf(outFile,"%d %d %d %d %f %f %f %f \n",mpi_rank, i,j,k,xyz[0],xyz[1],xyz[2],v_local[k+j*N0+i*N0*N0]); n_p++;
                }
#else
                    fprintf(outFile,"%d %d %d %f %f %f\n",mpi_rank, i,j,xyz[0],xyz[1],v_local[j+i*N0]);n_p++;
#endif
                }

            }

            fclose(outFile);
            this->ierr=VecRestoreArray(v_uniform,&v_local); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(v_uniform); CHKERRXX(this->ierr);


            std::cout<<this->mpi->mpirank<<" finished to print on an uniform grid "<<str_file<<std::endl;
        }





int Diffusion::printDiffusionIteration()
{

    int mpi_size;
    int mpi_rank;

    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

    std::stringstream oss2DebugVec;
    std::string mystr2DebugVec;
    oss2DebugVec << this->convert2FullPath("Diffusion Iteration")<<"_"<<this->it<<"_"<<mpi_rank<<".txt";
    mystr2DebugVec=oss2DebugVec.str();
    FILE *outFileVec;
    outFileVec=fopen(mystr2DebugVec.c_str(),"w");
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


    int n_local_size;
    int n_global_size;
    VecGetLocalSize(this->sol_t,&n_local_size);
    VecGetSize(this->sol_t,&n_global_size);


    n_local_size=this->nodes->indep_nodes.elem_count;

    PetscScalar *local_solt=new PetscScalar[n_local_size];
    PetscScalar *local_bt=new PetscScalar[n_local_size];


    VecGetArray(this->sol_t,&local_solt);
    VecGetArray(this->b_t,&local_bt);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    for(int i=0;i<n_local_size;i++)
    {
        //std::cout<<mpi_rank<<" "<<n_local_size*mpi_rank+i<<" "<<i<<" "<<local_b[i]<<" "<<local_u[i]<<" "<<local_x[i]<<std::endl;
        //fprintf(outFileVec,"%d %d %d %f %f %f %f\n",mpi_rank,global_node_number+i,i,this->sol_p[i],this->uex_p[i],this->sol_p[i]-this->uex_p[i], local_bt[i]);
        fprintf(outFileVec,"%d %d %d %f %f\n",mpi_rank,global_node_number+i,i,local_solt[i], local_bt[i]);

    }
    fclose(outFileVec);

    return 0;

}

int Diffusion::printDiffusionHistory()
{
    if(this->printIntemediateSteps)
    {
        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2DebugVec;
        oss2DebugVec << this->convert2FullPath("Diffusion History")<<"_"<<mpi_rank<<".txt";
        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

        int n_local_size;
        int n_global_size;
        VecGetLocalSize(this->sol_t,&n_local_size);
        VecGetSize(this->sol_t,&n_global_size);

        p4est_topidx_t global_node_number=0;

        for(int ii=0;ii<myRank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        for(int tt=0;tt<this->N_t;tt++)
        {
            for(int i=0;i<n_local_size;i++)
            {
                fprintf(outFileVec,"%d %d %d %d %f\n",mpi_rank,tt,global_node_number+i,i,this->sol_history[tt*n_local_size+i]);
            }
        }
        fclose(outFileVec);
    }
    return 0;
}

void Diffusion::printForestNodes2TextFile()
{

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("forest_of_independent_nodes")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::stringstream oss2Debug2;
    std::string mystr2Debug2;
    oss2Debug2 << this->convert2FullPath("rhs")<<"_"<<myRank<<".txt";
    mystr2Debug2=oss2Debug2.str();
    FILE *outFile2;
    outFile2=fopen(mystr2Debug2.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        bool isNodeWall=is_node_Wall(this->p4est,node);

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = node_x_fr_i(node) + tree_xmin;
        double y =node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif


        //#ifdef P4_TO_P8
        //        fprintf(outFile,"%d %d %d %d %f %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,this->sol_p[i],this->uex_p[i],this->err[i]);
        //#else
        //        fprintf(outFile,"%d %d %d %d %d %f %f %f %f %f %f\n",isNodeWall,  myRank,tree_id,i,global_node_number+i,  x,y,this->sol_p[i],this->uex_p[i],this->rhs_p[i],this->err[i]);
        //        fprintf(outFile2,"%d %d %d %d %d %f %f %f\n",isNodeWall,  myRank,tree_id,i,global_node_number+i,  x,y,this->rhs_p[i]);
        //#endif


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f\n",isNodeWall, myRank,tree_id,i,global_node_number+i,  x,y,z);
#else
        fprintf(outFile,"%d %d %d %d %d %f %f\n",isNodeWall,  myRank,tree_id,i,global_node_number+i,  x,y);
        fprintf(outFile2,"%d %d %d %d %d %f %f\n",isNodeWall,  myRank,tree_id,i,global_node_number+i,  x,y);
#endif


    }


    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;


    //    for (p4est_locidx_t i = nodes->num_owned_indeps; i<nodes->indep_nodes.elem_count; ++i)
    //    {
    //      /* since we want to access the local nodes, we need to 'jump' over intial
    //       * nonlocal nodes. Number of initial nonlocal nodes is given by
    //       * nodes->offset_owned_indeps
    //       */
    //      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
    //      p4est_topidx_t tree_id = node->p.piggy3.which_tree;
    //      std::cout<<"myRank treeid "<<myRank<<" "<<tree_id<<std::endl;


    //      p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    //      double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
    //      double tree_ymin = connectivity->vertices[3*v_mm + 1];

    //     #ifdef P4_TO_P8
    //            double tree_zmin = connectivity->vertices[3*v_mm + 2];
    //     #endif

    //      double x = int2double_coordinate_transform(node->x) + tree_xmin;
    //      double y = int2double_coordinate_transform(node->y) + tree_ymin;

    //     #ifdef P4_TO_P8
    //      double z = int2double_coordinate_transform(node->z) + tree_zmin;
    //      #endif

    //      double phiphi=0;

    //      #ifdef P4_TO_P8
    //             BCCGenerator2 mySplitter(0.1,2.00);
    //      #else
    //             BCCGenerator2 mySplitter(0.1,2);
    //      #endif

    //  #ifdef P4_TO_P8
    //      phiphi = mySplitter(x,y,z);
    //  #else
    //      phiphi = mySplitter(x,y);
    //  #endif

    //#ifdef P4_TO_P8
    //fprintf(outFile,"%d  %d %d %d %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,phiphi);
    //#else
    //fprintf(outFile,"%d  %d %d %d %f %f %f\n",myRank,tree_id,i,global_node_number+  i,  x,y,phiphi);
    //#endif

    //    }

    fclose(outFile);

}

int Diffusion::printForestNodesNeighborsCells2TextFile()
{

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("forest_of_independent_cell_neighbor_nodes")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");



    std::stringstream oss2Debug2;
    std::string mystr2Debug2;
    oss2Debug2 << this->convert2FullPath("rhs")<<"_"<<myRank<<".txt";
    mystr2Debug2=oss2Debug2.str();
    FILE *outFile2;
    outFile2=fopen(mystr2Debug2.c_str(),"w");


    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    double *phi_copy=new double[this->nodes->num_owned_indeps];
    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        bool isNodeWall=is_node_Wall(this->p4est,node);

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

#ifdef P4_TO_P8
        double z = int2double_coordinate_transform(node->z) + tree_zmin;
#endif

        p4est_locidx_t quad_idx; p4est_topidx_t nb_tree_idx;

        int i_direction=-1;
        int j_direction=-1;
        int k_direction=1;


#ifdef P4_TO_P8
        this->nodes_neighbours->find_neighbor_cell_of_node(node,i_direction,j_direction,k_direction, quad_idx, nb_tree_idx);
#else
        this->nodes_neighbours->find_neighbor_cell_of_node(node,i_direction,j_direction, quad_idx, nb_tree_idx );
#endif


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %d %d\n", myRank,tree_id,i,global_node_number+i,  x,y,z,nb_tree_idx,quad_idx);
#else
        fprintf(outFile,"%d %d %d %d %f %f %d %d\n",  myRank,tree_id,i,global_node_number+i,  x,y,nb_tree_idx,quad_idx);
#endif


    }


    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;


    //    for (p4est_locidx_t i = nodes->num_owned_indeps; i<nodes->indep_nodes.elem_count; ++i)
    //    {
    //      /* since we want to access the local nodes, we need to 'jump' over intial
    //       * nonlocal nodes. Number of initial nonlocal nodes is given by
    //       * nodes->offset_owned_indeps
    //       */
    //      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
    //      p4est_topidx_t tree_id = node->p.piggy3.which_tree;
    //      std::cout<<"myRank treeid "<<myRank<<" "<<tree_id<<std::endl;


    //      p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    //      double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
    //      double tree_ymin = connectivity->vertices[3*v_mm + 1];

    //     #ifdef P4_TO_P8
    //            double tree_zmin = connectivity->vertices[3*v_mm + 2];
    //     #endif

    //      double x = int2double_coordinate_transform(node->x) + tree_xmin;
    //      double y = int2double_coordinate_transform(node->y) + tree_ymin;

    //     #ifdef P4_TO_P8
    //      double z = int2double_coordinate_transform(node->z) + tree_zmin;
    //      #endif

    //      double phiphi=0;

    //      #ifdef P4_TO_P8
    //             BCCGenerator2 mySplitter(0.1,2.00);
    //      #else
    //             BCCGenerator2 mySplitter(0.1,2);
    //      #endif

    //  #ifdef P4_TO_P8
    //      phiphi = mySplitter(x,y,z);
    //  #else
    //      phiphi = mySplitter(x,y);
    //  #endif

    //#ifdef P4_TO_P8
    //fprintf(outFile,"%d  %d %d %d %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,phiphi);
    //#else
    //fprintf(outFile,"%d  %d %d %d %f %f %f\n",myRank,tree_id,i,global_node_number+  i,  x,y,phiphi);
    //#endif

    //    }

    fclose(outFile);
    return 0;

}

void Diffusion::printForestQNodes2TextFile()
{

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("forest_of_octants_nodes")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");

    std::cout<<"p4est->trees->elem_count"<<p4est->trees->elem_count<<std::endl;
    std::cout<<"p4est->local_num_quadrants"<<p4est->local_num_quadrants<<std::endl;


    const p4est_locidx_t *q2n = this->nodes->local_nodes;
    for (p4est_locidx_t i = 0; i<p4est->trees->elem_count; i++)
    {
        p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);
        p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];

        double tree_xmin = connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        std::cout<<"tree2->quadrants.elem_count"<<tree2->quadrants.elem_count<<std::endl;

        for(p4est_locidx_t j=0;j<tree2->quadrants.elem_count;j++)
        {
            /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */


            p4est_quadrant_t *quadrant = (p4est_quadrant_t*)sc_array_index(&tree2->quadrants, j);


            double x = int2double_coordinate_transform(quadrant->x) + tree_xmin;
            double y = int2double_coordinate_transform(quadrant->y) + tree_ymin;

#ifdef P4_TO_P8
            double z = int2double_coordinate_transform(quadrant->z) + tree_zmin;
#endif



            int n0,n1,n2,n3;

            int quad_idx=j+tree2->quadrants_offset;

#ifdef P4_TO_P8
            int n4,n5,n6,n7;
            n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
            n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
            n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
            n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];
            n4 =  q2n[ quad_idx*P4EST_CHILDREN + 4 ];
            n5 =  q2n[ quad_idx*P4EST_CHILDREN + 5 ];
            n6 =  q2n[ quad_idx*P4EST_CHILDREN + 6 ];
            n7 =  q2n[ quad_idx*P4EST_CHILDREN + 7 ];

#else
            n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
            n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
            n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
            n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];
#endif


#ifdef P4_TO_P8
            fprintf(outFile,"%d %d %d %d %d %d %d %d %d %d %d\n",myRank, i,j,n0,n1,n2,n3,n4,n5,n6,n7);
#else
            fprintf(outFile,"%d %d %d %d %d %d %d\n",myRank, i,j,n0,n1,n2,n3);
#endif

        }

    }

    fclose(outFile);

}

void Diffusion::printForestOctants2TextFile()
{

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("forest_of_octants")<<"_"<<this->mpi->mpirank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"p4est->trees->elem_count"<<p4est->trees->elem_count<<std::endl;
    std::cout<<"p4est->local_num_quadrants"<<p4est->local_num_quadrants<<std::endl;
    int my_local_num=this->p4est->local_num_quadrants;
    int my_first_number=this->p4est->global_first_quadrant[this->mpi->mpirank];
    for (p4est_locidx_t i = 0; i<p4est->trees->elem_count; i++)
    {
        p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);
        p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];
        double tree_xmin = connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        std::cout<<"tree2->quadrants.elem_count"<<tree2->quadrants.elem_count<<std::endl;

        for(p4est_locidx_t j=0;j<tree2->quadrants.elem_count;j++)
        {
            /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */


            p4est_quadrant_t *quadrant = (p4est_quadrant_t*)sc_array_index(&tree2->quadrants, j);
            int my_tree_first_number=tree2->quadrants_offset;
            double x = int2double_coordinate_transform(quadrant->x) + tree_xmin;
            double y = int2double_coordinate_transform(quadrant->y) + tree_ymin;

#ifdef P4_TO_P8
            double z = int2double_coordinate_transform(quadrant->z) + tree_zmin;
#endif

#ifdef P4_TO_P8
            fprintf(outFile,"%d %d %d %d %d %d %f %f %f\n",this->mpi->mpirank, i,j,my_local_num,my_first_number,my_tree_first_number,   x,y,z);
#else
            fprintf(outFile,"%d %d %d %d %d %d %f %f\n",this->mpi->mpirank, i,j,my_local_num,my_first_number,my_tree_first_number,   x,y);
#endif

        }
    }
    fclose(outFile);
}

void Diffusion::printArrayOnForestOctants2TextFile(double *x2Print_cell,double *x2Print_nodes, int Nx, string file_name_str)
{


    int mpi_size;
    int mpi_rank;

    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

    std::stringstream oss2DebugVec;
    std::string mystr2Debug;
    oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";
    mystr2Debug=oss2DebugVec.str();


    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");

    std::cout<<"p4est->trees->elem_count"<<p4est->trees->elem_count<<std::endl;
    std::cout<<"p4est->local_num_quadrants"<<p4est->local_num_quadrants<<std::endl;


    const p4est_locidx_t *q2n = this->nodes->local_nodes;
    for (p4est_locidx_t i = 0; i<p4est->trees->elem_count; i++)
    {
        p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);
        p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];

        double tree_xmin = connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        std::cout<<"tree2->quadrants.elem_count"<<tree2->quadrants.elem_count<<std::endl;

        for(p4est_locidx_t j=0;j<tree2->quadrants.elem_count;j++)
        {
            /* since we want to access the local nodes, we need to 'jump' over intial
       * nonlocal nodes. Number of initial nonlocal nodes is given by
       * nodes->offset_owned_indeps
       */


            p4est_quadrant_t *quadrant = (p4est_quadrant_t*)sc_array_index(&tree2->quadrants, j);


            double x = int2double_coordinate_transform(quadrant->x) + tree_xmin;
            double y = int2double_coordinate_transform(quadrant->y) + tree_ymin;

#ifdef P4_TO_P8
            double z = int2double_coordinate_transform(quadrant->z) + tree_zmin;
#endif



            int n0,n1,n2,n3;

            int quad_idx=j+tree2->quadrants_offset;

#ifdef P4_TO_P8
            int n4,n5,n6,n7;
            n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
            n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
            n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
            n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];
            n4 =  q2n[ quad_idx*P4EST_CHILDREN + 4 ];
            n5 =  q2n[ quad_idx*P4EST_CHILDREN + 5 ];
            n6 =  q2n[ quad_idx*P4EST_CHILDREN + 6 ];
            n7 =  q2n[ quad_idx*P4EST_CHILDREN + 7 ];

#else
            n0 =  q2n[ quad_idx*P4EST_CHILDREN + 0 ];
            n1 =  q2n[ quad_idx*P4EST_CHILDREN + 1 ];
            n2 =  q2n[ quad_idx*P4EST_CHILDREN + 2 ];
            n3 =  q2n[ quad_idx*P4EST_CHILDREN + 3 ];
#endif


#ifdef P4_TO_P8
            fprintf(outFile,"%d %d %d %d %d %d %d %d %d %d %d %d\n",mpi_rank, i,j,quad_idx,n0,n1,n2,n3,n4,n5,n6,n7,x2Print_cell[quad_idx]);
#else
            fprintf(outFile,"%d %d %d %f %f %d %d %d %d %d %f %f %f %f %f\n",mpi_rank, i,j,x,y,quad_idx,n0,n1,n2,n3,
                    x2Print_nodes[n0],x2Print_nodes[n1],x2Print_nodes[n2],x2Print_nodes[n3],   x2Print_cell[quad_idx]);
#endif

        }

    }

    fclose(outFile);
}

void Diffusion::printMacroMesh()
{

    //    connectivity->vertices[0]=-1;
    //    connectivity->vertices[1]=-1;

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("macro_mesh")<<"_"<<this->mpi->mpirank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    for(int i=0;i<this->p4est->connectivity->num_vertices;i++)
    {
        fprintf(outFile,"%d %d %f %f %f \n",this->mpi->mpirank, i,this->connectivity->vertices[3*i+0],this->connectivity->vertices[3*i+1],this->connectivity->vertices[3*i+2]);

    }

    fprintf(outFile,"----int--------\n");

    for(int i=0;i<this->p4est->connectivity->num_vertices;i++)
    {
        fprintf(outFile,"%d %d %d %d %d \n",this->mpi->mpirank, i,(int)this->connectivity->vertices[3*i+0],(int)this->connectivity->vertices[3*i+1],(int)this->connectivity->vertices[3*i+2]);

    }

    fprintf(outFile,"----Trees--------\n");



    for (p4est_locidx_t i = 0; i<p4est->trees->elem_count; i++)
    {
        p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);

        p4est_topidx_t v_mm0 = connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];
        p4est_topidx_t v_mm1= connectivity->tree_to_vertex[P4EST_CHILDREN*i + 1];
        p4est_topidx_t v_mm2 = connectivity->tree_to_vertex[P4EST_CHILDREN*i + 2];
        p4est_topidx_t v_mm3 = connectivity->tree_to_vertex[P4EST_CHILDREN*i + 3];
        double tree_x0 = connectivity->vertices[3*v_mm0 + 0];
        double tree_y0 = connectivity->vertices[3*v_mm0 + 1];
#ifdef P4_TO_P8
        double tree_zmin0 = connectivity->vertices[3*v_mm0 + 2];
#endif
        double tree_x1 = connectivity->vertices[3*v_mm1 + 0];
        double tree_y1 = connectivity->vertices[3*v_mm1 + 1];

#ifdef P4_TO_P8
        double tree_zmin1 = connectivity->vertices[3*v_mm1 + 2];
#endif

        double tree_x2 = connectivity->vertices[3*v_mm2 + 0];
        double tree_y2 = connectivity->vertices[3*v_mm2 + 1];

#ifdef P4_TO_P8
        double tree_zmin2 = connectivity->vertices[3*v_mm2 + 2];
#endif

        double tree_x3 = connectivity->vertices[3*v_mm3 + 0];
        double tree_y3 = connectivity->vertices[3*v_mm3 + 1];

#ifdef P4_TO_P8
        double tree_zmin3 = connectivity->vertices[3*v_mm3 + 2];
#endif

        fprintf(outFile,"%d %d %d %d %f %f %f %f %f %f %f %f\n",this->mpi->mpirank, i,P4EST_ROOT_LEN,P4EST_MAXLEVEL,  tree_x0,tree_y0,  tree_x1,tree_y1  ,tree_x2,tree_y2, tree_x3,tree_y3);
    }



    //    fprintf(outFile,"----Max Distance--------\n");
    //    fprintf(outFile,"%d %d %d %d %f %f %f %f %f %f %f %f\n",this->mpi->mpirank, i,P4EST_ROOT_LEN,P4EST_MAXLEVEL,  tree_x0,tree_y0,  tree_x1,tree_y1  ,tree_x2,tree_y2, tree_x3,tree_y3);

    fclose(outFile);
}

void Diffusion::printGhostCells()
{

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("ghost_cells")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");


    p4est_locidx_t num_local=p4est->local_num_quadrants;
    std::vector<PetscInt> ghost_cells(this->ghost->ghosts.elem_count,0);


    PetscInt num_global=this->p4est->global_num_quadrants;

    for(int r=0;r<this->p4est->mpisize;r++)
    {
        for(p4est_locidx_t q=ghost->proc_offsets[r];q<ghost->proc_offsets[r+1];q++)
        {
            const p4est_quadrant_t *quad=(const p4est_quadrant_t*) sc_array_index(&ghost->ghosts,q);
            ghost_cells[q]=(PetscInt) quad->p.piggy3.local_num+(PetscInt) this->p4est->global_first_quadrant[r];
            fprintf(outFile,"%d %d %d %d %d\n",myRank, r,q,quad->p.piggy3.local_num,ghost_cells[q]);
        }
    }


    fclose(outFile);

}

void Diffusion::printGhostNodes()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("ghost_nodes")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");


    p4est_locidx_t num_local=this->nodes->num_owned_indeps;
    std::vector<PetscInt> ghost_nodes(this->nodes->indep_nodes.elem_count-num_local,0);
    std::vector<PetscInt> global_offset_sum(this->p4est->mpisize+1,0);

    // Calculate the global number of points

    for(int r=0;r<this->p4est->mpisize;r++)
    {
        global_offset_sum[r+1]=global_offset_sum[r]+(PetscInt)this->nodes->global_owned_indeps[r];
    }

    for(size_t i=0;i<ghost_nodes.size();i++)
    {
        p4est_indep_t *ni=(p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes,i+num_local);
        ghost_nodes[i]=(PetscInt) ni->p.piggy3.local_num+global_offset_sum[this->nodes->nonlocal_ranks[i]];

        fprintf(outFile,"%d %d %d %d %d\n",i,ghost_nodes[i],this->nodes->nonlocal_ranks[i],global_offset_sum[this->nodes->nonlocal_ranks[i]],ni->p.piggy3.local_num);
    }
    fclose(outFile);
}


double Diffusion::integrate_over_interface_in_one_quadrant(  p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
{
#ifdef P4_TO_P8
    OctValue phi_values;
    OctValue f_values;
#else
    QuadValue phi_values;
    QuadValue f_values;
#endif
    double *P, *F;
    PetscErrorCode ierr;
    ierr = VecGetArray(phi, &P); CHKERRXX(ierr);
    ierr = VecGetArray(f  , &F); CHKERRXX(ierr);

    const p4est_locidx_t *q2n = nodes->local_nodes;
#ifdef P4_TO_P8
    phi_values.val000 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
    phi_values.val100 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
    phi_values.val010 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
    phi_values.val110 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
    phi_values.val001 = P[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
    phi_values.val101 = P[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
    phi_values.val011 = P[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
    phi_values.val111 = P[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

    f_values.val000   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
    f_values.val100   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
    f_values.val010   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
    f_values.val110   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
    f_values.val001   = F[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
    f_values.val101   = F[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
    f_values.val011   = F[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
    f_values.val111   = F[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

#else
    phi_values.val00 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
    phi_values.val10 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
    phi_values.val01 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
    phi_values.val11 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];

    f_values.val00   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
    f_values.val10   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
    f_values.val01   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
    f_values.val11   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
#endif
    ierr = VecRestoreArray(phi, &P); CHKERRXX(ierr);
    ierr = VecRestoreArray(f  , &F); CHKERRXX(ierr);

    double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dy = dx;
#ifdef P4_TO_P8
    double dz = dx;
#endif

#ifdef P4_TO_P8
    Cube3 cube(0, dx, 0, dy, 0, dz);
#else
    Cube2 cube(0, dx, 0, dy);
#endif

    return cube.integrate_Over_Interface(f_values,phi_values);
}

double Diffusion::integrate_over_interface( Vec phi, Vec f)
{
    double sum = 0;
    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
        {
            p4est_quadrant_t *quad = ( p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
            sum += integrate_over_interface_in_one_quadrant( quad,
                                                             quad_idx + tree->quadrants_offset,
                                                             phi, f);
        }
    }

    /* compute global sum */
    double sum_global;
    PetscErrorCode ierr;
    ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
    return sum_global;
}

double Diffusion::integrate_constant_over_interface_in_one_quadrant(  p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi)
{
#ifdef P4_TO_P8
    OctValue phi_values;
    OctValue f_values;
#else
    QuadValue phi_values;
    QuadValue f_values;
#endif
    double *P, *F;
    PetscErrorCode ierr;
    ierr = VecGetArray(phi, &P); CHKERRXX(ierr);


    const p4est_locidx_t *q2n = nodes->local_nodes;
#ifdef P4_TO_P8
    phi_values.val000 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
    phi_values.val100 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
    phi_values.val010 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
    phi_values.val110 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
    phi_values.val001 = P[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
    phi_values.val101 = P[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
    phi_values.val011 = P[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
    phi_values.val111 = P[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

    f_values.val000   = 1.00;
    f_values.val100   = 1.00;
    f_values.val010   = 1.00;
    f_values.val110   = 1.00;
    f_values.val001   = 1.00;
    f_values.val101   = 1.00;
    f_values.val011   = 1.00;
    f_values.val111   = 1.00;

#else
    phi_values.val00 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
    phi_values.val10 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
    phi_values.val01 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
    phi_values.val11 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];

    f_values.val00   = 1.00;
    f_values.val10   = 1.00;
    f_values.val01   =1.00;
    f_values.val11   = 1.00;
#endif
    ierr = VecRestoreArray(phi, &P); CHKERRXX(ierr);


    double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dy = dx;
#ifdef P4_TO_P8
    double dz = dx;
#endif

#ifdef P4_TO_P8
    Cube3 cube(0, dx, 0, dy, 0, dz);
#else
    Cube2 cube(0, dx, 0, dy);
#endif

    return cube.integrate_Over_Interface(f_values,phi_values);
}

double Diffusion::integrate_constant_over_interface(Vec phi)
{
    double sum = 0;
    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
        {
            p4est_quadrant_t *quad = ( p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
            sum += integrate_constant_over_interface_in_one_quadrant( quad,
                                                                      quad_idx + tree->quadrants_offset,
                                                                      phi);
        }
    }

    /* compute global sum */
    double sum_global;
    PetscErrorCode ierr;
    ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
    return sum_global;
}
