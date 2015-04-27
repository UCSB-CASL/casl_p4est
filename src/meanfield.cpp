#include "meanfield.h"

#ifdef P4_TO_P8
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include<src/my_p8est_utils.h>
#else
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include<src/my_p4est_utils.h>
#endif

MeanField::MeanField()
{
    std::cout<<" Default Constructor "<<std::endl;
}

MeanField::MeanField(int argc, char *argv[], int nx_trees, int ny_trees, int nz_trees, int min_level, int max_level, int t_iterations, int mean_field_iterations, double f, double Xhi_AB, double Lx, double Lx_physics, Diffusion::phase myPhase
                     ,
                     Diffusion::casl_diffusion_method my_casl_diffusion_method, Diffusion::numerical_scheme my_numerical_scheme,
                     MeanField::mask_strategy my_mask_strategy, string my_io_path, double ax, double by, double cz,
                     int extension_advection_period, int stress_computation_period, int remesh_period, double lambda,
                     MeanField::strategy_meshing my_meshing_strategy, PetscBool setup_finite_difference_solver,
                     double polymer_mask_radius, string text_file_seed_str, string text_file_mask_str, string text_file_fields,
                     string text_file_field_wp, string text_file_field_wm, MeanFieldPlan *myMeanFieldPlan)
{
    this->LaunchMeanField(my_mask_strategy, my_meshing_strategy, polymer_mask_radius, setup_finite_difference_solver, text_file_seed_str,
                          remesh_period, f, myMeanFieldPlan, my_io_path, extension_advection_period, lambda, text_file_mask_str,
                          stress_computation_period);
    this->initialyze_MeanField(argc, argv,  nx_trees,  ny_trees,  nz_trees,  min_level,  max_level,  t_iterations,  mean_field_iterations,
                               f, Xhi_AB, Lx,Lx_physics, myPhase,
                               my_casl_diffusion_method, my_numerical_scheme, ax, by, cz,text_file_fields,text_file_field_wp,text_file_field_wm);

}



void MeanField::LaunchMeanField(MeanField::mask_strategy my_mask_strategy, MeanField::strategy_meshing my_meshing_strategy, double polymer_mask_radius, PetscBool setup_finite_difference_solver, string text_file_seed_str, int remesh_period, double f, MeanFieldPlan *myMeanFieldPlan, string my_io_path, int extension_advection_period, double lambda, string text_file_mask_str, int stress_computation_period)
{
    this->my_delta_H=new DeltaH();

    this->initialyze_default_parameters();
    this->myMeanFieldPlan=myMeanFieldPlan;
    this->my_robin_optimization=new RobinOptimization(this->myMeanFieldPlan->kappaA,this->myMeanFieldPlan->robin_lambda);

    this->text_file_mask_str=text_file_mask_str;
    this->text_file_seed_str=text_file_seed_str;
    if(this->myMeanFieldPlan->write2Vtk)
        this->do_not_write_to_vtk_in_any_case=PETSC_FALSE;
    else
        this->do_not_write_to_vtk_in_any_case=PETSC_TRUE;
    this->setup_finite_difference_solver=setup_finite_difference_solver;
    this->remesh_period=remesh_period;
    this->my_mask_strategy=my_mask_strategy;
    this->IO_path=my_io_path;
    this->extension_advection_period=extension_advection_period;
    this->stress_tensor_computation_period=stress_computation_period;

    this->polymer_mask_radius_for_initial_wall=polymer_mask_radius;


    if(this->myMeanFieldPlan->px
            && this->myMeanFieldPlan->py
            && this->myMeanFieldPlan->pz)
    {
        this->myMeanFieldPlan->periodic_xyz=PETSC_TRUE;

    }
    else
    {
        this->myMeanFieldPlan->periodic_xyz=PETSC_FALSE;

    }


    this->my_meshing_strategy=my_meshing_strategy;

    this->lambda_minus=lambda;
    this->lambda_plus=lambda;
    this->lambda_0=lambda;
    std::cout<<" Mean Field Constructor"<<std::endl;
    this->f_initial=f;


}

MeanField::MeanField(int argc, char *argv[], int nx_trees, int ny_trees, int nz_trees, int min_level, int max_level, int t_iterations, int mean_field_iterations, double f, double Xhi_AB, double Lx, double Lx_physics, Diffusion::phase myPhase
                     ,
                     Diffusion::casl_diffusion_method my_casl_diffusion_method, Diffusion::numerical_scheme my_numerical_scheme,
                     MeanField::mask_strategy my_mask_strategy, string my_io_path, double ax, double by, double cz,
                     int extension_advection_period, int stress_computation_period, int remesh_period, double lambda,
                     MeanField::strategy_meshing my_meshing_strategy, PetscBool setup_finite_difference_solver,
                     double polymer_mask_radius, string text_file_seed_str, string text_file_mask_str, string text_file_fields,
                     string text_file_field_wp, string text_file_field_wm, MeanFieldPlan *myMeanFieldPlan, inverse_litography *myInverseLitography)
{
    this->my_design_plan=myInverseLitography;
    this->LaunchMeanField(my_mask_strategy, my_meshing_strategy, polymer_mask_radius, setup_finite_difference_solver, text_file_seed_str,
                          remesh_period, f, myMeanFieldPlan, my_io_path, extension_advection_period, lambda, text_file_mask_str,
                          stress_computation_period);
    this->initialyze_MeanField(argc, argv,  nx_trees,  ny_trees,  nz_trees,  min_level,  max_level,  t_iterations,  mean_field_iterations,
                               f, Xhi_AB, Lx,Lx_physics, myPhase,
                               my_casl_diffusion_method, my_numerical_scheme, ax, by, cz,text_file_fields,text_file_field_wp,text_file_field_wm);


}

int MeanField::setNumericalParameters(int min_level, int max_level, int t_iterations, int mean_field_iterations)
{
    this->min_level=min_level;
    this->max_level=max_level;
    this->N_iterations=t_iterations;
    this->N_t=this->N_iterations+1;
    this->N_mean_field_iteration=mean_field_iterations;

    return 0;
}

int MeanField::setPolymerParameters(double f, double Xhi_AB, double Lx,Diffusion::phase myPhase)
{
    this->f=f;
    this->X_ab=Xhi_AB;
    this->Lx=Lx;
    this->myPhase=myPhase;
    return 0;
}

int MeanField::MeanField_initialyze_petsc(int argc, char *argv[])
{
    this->mpi = &mpi_context;
    this->mpi->mpicomm  = MPI_COMM_WORLD;
    this->mpi_session=new Session();
    this->mpi_session->init(argc, argv, this->mpi->mpicomm);

    MPI_Comm_size (this->mpi->mpicomm, &this->mpi->mpisize);
    MPI_Comm_rank (this->mpi->mpicomm, &this->mpi->mpirank);

    this->scft_clock=new my_clock();

    std::cout<<this->mpi->mpirank<<"--------------------initialyzed petsc---------------------------"<<std::endl;

    return 0;

}

int MeanField::constructForestOfTrees(int nx_trees,int ny_trees,int nz_trees)
{

    lamellar_level_set *myLS=new lamellar_level_set(this->Lx,this->f);
    this->data_p4est_cf=new splitting_criteria_cf_t(this->max_level, this->max_level,myLS, 1);
    this->nx_trees=nx_trees;
    this->ny_trees=ny_trees;
    this->nz_trees=nz_trees;
    this->brick=new my_p4est_brick_t();

#ifdef P4_TO_P8
    this->connectivity = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick,
                                            this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
#else
    this->connectivity = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick,this->myMeanFieldPlan->px,this->myMeanFieldPlan->py);
#endif


    // stopwatch
    parStopWatch w1, w2;


    /* create the p4est */
    this->p4est = p4est_new(this->mpi->mpicomm, this->connectivity, 0, NULL, NULL);
    this->p4est->user_pointer = (void*)(this->data_p4est_cf);
    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
    p4est_refine(this->p4est, P4EST_TRUE, refine_levelset_cf, NULL);


    /* partition the p4est */
    p4est_partition(this->p4est, NULL);

    /* create the ghost layer */
    this->ghost = p4est_ghost_new(this->p4est, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    this->nodes = my_p4est_nodes_new(this->p4est, this->ghost);

    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est, this->nodes, &this->phi); CHKERRXX(ierr);
    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy=new  my_p4est_hierarchy_t(this->p4est, this->ghost, this->brick);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors=new my_p4est_node_neighbors_t(this->hierarchy, this->nodes,this->myMeanFieldPlan->periodic_xyz,
                                                       this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
    this->node_neighbors->compute_max_distance();
    this->Lx=this->node_neighbors->max_distance;


    std::cout<<this->mpi->mpirank<<" "<<this->nodes->num_owned_indeps<< std::endl;
    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<" "<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->min_lvl<<std::endl;
    return 0;
}


int MeanField::constructForestOfTreesVisualization(int nx_trees,int ny_trees,int nz_trees)
{

    lamellar_level_set *myLS=new lamellar_level_set(this->Lx,this->f);
    this->data_p4est_cf_visualization=new splitting_criteria_cf_t(this->max_level+this->nb_splits, this->max_level+this->nb_splits,myLS, 1);

    this->nx_trees=nx_trees;
    this->ny_trees=ny_trees;
    this->nz_trees=nz_trees;
    this->brick_visualization=new my_p4est_brick_t();
    int periodic_a=0;  int periodic_b=0; int periodic_c=0;

#ifdef P4_TO_P8
    this->connectivity_visualization = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_visualization,periodic_a,periodic_b,periodic_c);
#else
    this->connectivity_visualization = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_visualization,periodic_a,periodic_b);
#endif
    // stopwatch
    parStopWatch w1, w2;
    /* create the p4est */
    this->p4est_visualization = p4est_new(this->mpi->mpicomm, this->connectivity_visualization, 0, NULL, NULL);
    this->p4est_visualization->user_pointer = (void*)(this->data_p4est_cf_visualization);
    p4est_refine(this->p4est_visualization, P4EST_TRUE, refine_levelset_cf, NULL);

    /* partition the p4est */
    p4est_partition(this->p4est_visualization, NULL);

    /* create the ghost layer */
    this->ghost_visualization = p4est_ghost_new(this->p4est_visualization, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    this->nodes_visualization = my_p4est_nodes_new(this->p4est_visualization, this->ghost_visualization);

    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est_visualization, this->nodes_visualization, &this->phi_vizualization); CHKERRXX(ierr);
    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy_visualization=new  my_p4est_hierarchy_t(this->p4est_visualization, this->ghost_visualization, this->brick_visualization);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors_visualization=new my_p4est_node_neighbors_t(this->hierarchy_visualization, this->nodes_visualization,false);
    this->node_neighbors_visualization->compute_max_distance();
    this->Lx=this->node_neighbors_visualization->max_distance;

    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
    return 0;
}



void MeanField::createPhiWall4Neuman()
{
    if(this->myMeanFieldPlan->neuman_with_mask  || this->myMeanFieldPlan->dirichlet_with_mask)
    {
        this->ierr=VecDuplicate(this->phi,&this->phi_wall); CHKERRXX(this->ierr);

        PetscScalar *phi_wall_local;
        this->ierr=VecGetArray(this->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);
        PetscScalar *polymer_mask_local;
        this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        //global_node_number=nodes->offset_owned_indeps;

        double tree_xmin , tree_ymin , tree_zmin ,x,  y,  z ;

        for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
        {
            /* since we want to access the local nodes, we need to 'jump' over intial
          * nonlocal nodes. Number of initial nonlocal nodes is given by
          * nodes->offset_owned_indeps
          */
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

            bool isNodeWall=is_node_Wall(this->p4est,node);

            tree_xmin = this->connectivity->vertices[3*v_mm + 0];
            tree_ymin = this->connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
            tree_zmin = this->connectivity->vertices[3*v_mm + 2];
#endif

            x = node_x_fr_i(node)   + tree_xmin;
            y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            z = node_z_fr_k(node) + tree_zmin;
#endif


            if(this->myMeanFieldPlan->neuman_with_mask)
                phi_wall_local[i]=0.5*exp(-(this->alpha_wall*polymer_mask_local[i]*this->alpha_wall*polymer_mask_local[i]));
            if(this->myMeanFieldPlan->dirichlet_with_mask)
                phi_wall_local[i]=1.00*(1.00-this->my_tanh_x( -this->alpha_wall*polymer_mask_local[i]));

        }




        this->ierr=VecRestoreArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->phi_wall);
        this->myDiffusion->set_phi_wall(this->phi_wall);
    }
}


void MeanField::reCreatePhiWall4Neuman()
{
    if(this->myMeanFieldPlan->neuman_with_mask || this->myMeanFieldPlan->dirichlet_with_mask)
    {
        this->ierr=VecDestroy(this->phi_wall); CHKERRXX(this->ierr);
        this->createPhiWall4Neuman();
    }
}

int MeanField::initialyze_MeanField(int argc, char* argv[],int nx_trees,int ny_trees,int nz_trees,int min_level,int max_level,int t_iterations,int mean_field_iterations,
                                    double f,double Xhi_AB,double Lx,double Lx_physics,Diffusion::phase myPhase,
                                    Diffusion::casl_diffusion_method my_casl_diffusion_method,Diffusion::numerical_scheme my_numerical_scheme,
                                    double ax,double by,double cz,string text_file_fields,string text_file_field_wp, string text_file_field_wm)
{
    this->MeanField_initialyze_petsc(argc,argv);
    std::cout<<this->mpi->mpirank<<"initialyzed in mean field"<<std::endl;
    this->a_ellipse=ax;
    this->b_ellipse=by;
    this->c_ellipse=cz;



    this->N_iterations=N_iterations;
    this->Lx_physics=Lx_physics;

    this->setPolymerParameters( f,Xhi_AB,Lx,myPhase);
    this->setNumericalParameters(min_level,max_level,t_iterations,mean_field_iterations);
    this->constructForestOfTrees(nx_trees,ny_trees,nz_trees);
    this->constructForestOfTreesVisualization(nx_trees,ny_trees,nz_trees);
    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
    this->myDiffusion=new Diffusion();
    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;

    if(my_casl_diffusion_method==Diffusion::neuman_backward_euler
            ||my_casl_diffusion_method==Diffusion::neuman_crank_nicholson
            ||my_casl_diffusion_method==Diffusion::dirichlet_crank_nicholson )
    {
        switch (this->my_mask_strategy)
        {
        case MeanField::sphere_mask:
            this->create_polymer_mask();
            break;
        case MeanField::annular_mask:
            this->create_polymer_mask_annular_cube();
            break;
        case MeanField::ellipse_mask:
            this->create_polymer_mask_moving_ellipse(ax,by,cz);
            break;
        case MeanField::cube_mask:
            this->create_polymer_mask_cube();
            break;
        case MeanField::helix_mask:
            this->create_polymer_mask_helix();
            break;
        case MeanField::drop_mask:
            this->create_polymer_mask_drop(this->a_ellipse,this->b_ellipse,this->c_ellipse);
            break;
        case MeanField::v_shape:
            this->create_polymer_mask_v_shape(this->a_ellipse,this->b_ellipse,this->c_ellipse);
            break;
        case MeanField::l_shape:
            this->create_polymer_mask_l_shape(this->a_ellipse,this->b_ellipse,this->c_ellipse);
            break;
        case MeanField::l_shape_3d:
            this->create_polymer_mask_3d_l_shape(this->a_ellipse,this->b_ellipse,this->c_ellipse);
            break;

        case MeanField::text_file_mask:
        {


            this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
            this->get_field_from_text_file(&this->polymer_mask,this->text_file_mask_str);

            this->scatter_petsc_vector(&this->polymer_mask);
            this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

            this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"text_file_mask",PETSC_TRUE);
            this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"text_file_mask");


            my_p4est_level_set PLS(this->node_neighbors);
            PLS.reinitialize_2nd_order(this->polymer_mask,50);

            this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

            int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
            PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

            this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"text_file_mask_reinitialyzed",PETSC_TRUE);
            this->myDiffusion->set_polymer_shape(this->polymer_mask);
            this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"text_file_vector");
            break;

        }

        case MeanField::ThreeD_from_TwoD_text_file_mask:
        {
            this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
            this->get_3Dfield_from_2Dtext_file(&this->polymer_mask,this->text_file_mask_str,this->polymer_mask_radius_for_initial_wall);
            this->scatter_petsc_vector(&this->polymer_mask);
            this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

            this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_ellipse",PETSC_TRUE);
            this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_ellipse_binary_vector");


            my_p4est_level_set PLS(this->node_neighbors);
            PLS.reinitialize_2nd_order(this->polymer_mask,50);

            this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

            int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
            PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

            this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"ellipse_reinitialyzed",PETSC_TRUE);
            this->myDiffusion->set_polymer_shape(this->polymer_mask);
            this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_ellipse_vector");
            break;
        }
        case MeanField::ThreeDParallel_from_TwoDSerial_text_file_mask:
        {
            this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
            this->get_3DParallelField_from_2DSerialtext_file(&this->polymer_mask,this->text_file_mask_str,this->polymer_mask_radius_for_initial_wall);
            this->scatter_petsc_vector(&this->polymer_mask);
            this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

            this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_ellipse",PETSC_TRUE);
            this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_ellipse_binary_vector");


            my_p4est_level_set PLS(this->node_neighbors);
            PLS.reinitialize_2nd_order(this->polymer_mask,50);

            this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

            int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
            PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

            this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"ellipse_reinitialyzed",PETSC_TRUE);
            this->myDiffusion->set_polymer_shape(this->polymer_mask);
            this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_ellipse_vector");
            break;
        }
        case MeanField::from_inverse_design:
            this->create_polymer_mask_from_inverse_litography();
            break;
        case MeanField::from_inverse_design_with_level_set:
            this->create_polymer_mask_from_inverse_litography_with_level_set();
            break;
        default:
            throw std::runtime_error  ("you need to define a mask");
            break;
        }


    }

    std::cout<<this->p4est<<std::endl;



    this->myDiffusion->initialyzeDiffusionFromMeanField(this->min_level,this->max_level,&this->phi,this->mpi_session,
                                                        &this->mpi_context,this->mpi,this->p4est,this->nodes,
                                                        this->ghost,this->connectivity,
                                                        this->brick,this->hierarchy,this->node_neighbors,this->Lx,this->Lx_physics,
                                                        this->f,this->X_ab,this->N_iterations,this->myPhase,
                                                        my_casl_diffusion_method, my_numerical_scheme,this->IO_path,this->stress_tensor_computation_period,
                                                        this->nx_trees,this->lambda_plus,this->setup_finite_difference_solver,this->myMeanFieldPlan->periodic_xyz,
                                                        text_file_fields,text_file_field_wp, text_file_field_wm,this->myMeanFieldPlan);


    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;

    std::cout<<" allocatio of mean field variables "<<std::endl;

    this->energy=new double[this->N_mean_field_iteration];
    this->energy_w=new double[this->N_mean_field_iteration];
    this->energy_logQ=new double[this->N_mean_field_iteration];
    this->pressure_force=new double[this->N_mean_field_iteration];
    this->exchange_force=new double[this->N_mean_field_iteration];
    this->shape_force=new double[this->N_mean_field_iteration];
    this->order_Ratio=new double[this->N_mean_field_iteration];
    this->rho0Average=new double[this->N_mean_field_iteration];
    this->rhoAAverage=new double[this->N_mean_field_iteration];
    this->rhoBAverage=new double[this->N_mean_field_iteration];
    this->length_series=new double[this->N_mean_field_iteration];
    this->VRealTime=new double[this->N_mean_field_iteration];

    for(int i=0;i<this->N_mean_field_iteration;i++)
    {
        this->energy[i]=0;
        this->energy_w[i]=0;
        this->energy_logQ[i]=0;
        this->pressure_force[i]=0;
        this->exchange_force[i]=0;
        this->shape_force[i]=0;
        this->order_Ratio[i]=0;
        this->rho0Average[i]=0;
        this->rhoAAverage[i]=0;
        this->rhoBAverage[i]=0;
        this->length_series[i]=0;
        this->VRealTime[i]=0;
    }





    int n_mask=(int)((double) this->N_mean_field_iteration/(double)this->mask_period);
    std::cout<<"n_mask"<<n_mask<<std::endl;

    //    this->de_predicted_from_level_set_change_current_step_before_lagrange=new double[n_mask];
    //     this->de_predicted_from_level_set_change_current_step_before_volume_conservation=new double[n_mask];
    this->de_predicted_from_level_set_change_current_step_after=new double[n_mask];
    this->prediction_error_level_set=new double[n_mask];
    this->de_mask=new double[n_mask];
    this->shape_force=new double[n_mask];
    this->dphi_step_1=new double[n_mask];
    this->dphi_step_2=new double[n_mask];
    this->dphi_step_3=new double[n_mask];
    this->dphi_step_4=new double[n_mask];
    this->dphi_step_5=new double[n_mask];

    this->dphi_step_0_3=new double[n_mask];
    this->dphi_step_0_4=new double[n_mask];
    this->dphi_step_0_5=new double[n_mask];

    this->wp_average_on_contour=new double[n_mask];
    this->wp_variance_on_contour=new double[n_mask];


    for(int i=0;i<n_mask;i++)
    {
        this->de_predicted_from_level_set_change_current_step_after[i]=0;
        this->prediction_error_level_set[i]=0;
        this->de_mask[i]=0;
        this->shape_force[i]=0;
        this->dphi_step_1[i]=0.000;
        this->dphi_step_2[i]=0.000;
        this->dphi_step_3[i]=0.000;
        this->dphi_step_4[i]=0.000;
        this->dphi_step_5[i]=0.000;
    }
    std::cout<<"phi_wall"<<n_mask<<std::endl;


    this->createPhiWall4Neuman();

    return 0;
}



int MeanField::create_polymer_mask_from_inverse_litography()
{

    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;
    double eps_radius=this->Lx/256;
    double polymer_mask_radius=this->polymer_mask_radius_for_initial_wall;


    double ax_in=1.00,by_in=1.00,cz_in=1;

    ax_in=ax_in*ax_in*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;
    by_in=by_in*by_in*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;
    cz_in=cz_in*cz_in*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;





    for(int i=0;i<this->my_design_plan->n2Check;i++)
    {
        std::cout<<this->my_design_plan->xc[i]<<" "<<this->my_design_plan->yc[i]<<" "<<this->my_design_plan->zc[i]<<std::endl;
    }


    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("litography_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;


    VecDuplicate(this->phi,&this->polymer_mask);
    PetscScalar *polymer_mask_local;
    VecGetArray(this->polymer_mask,&polymer_mask_local);
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];
    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++)
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

        double min_distance; double distance;
        for(int ii=0;ii<this->my_design_plan->n2Check;ii++)
        {
            distance= pow(x- this->my_design_plan->xc[ii],2)+pow(y-this->my_design_plan->yc[ii],2);
            if(ii==0)
                min_distance=distance;
            else
                min_distance=min(min_distance,distance);
        }


        polymer_mask_local[i]=pow(min_distance,0.5)-1.00*this->my_design_plan->r_spot;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,polymer_mask_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,polymer_mask_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    this->ierr=VecRestoreArray(this->polymer_mask,&polymer_mask_local);
    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask_from_distance_function",PETSC_TRUE);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_from_distance_function_binary_vector");



    my_p4est_level_set PLS(this->node_neighbors);
    PLS.reinitialize_2nd_order(this->polymer_mask,50);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/((double)Nx));
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_from_distance_function_vector");
}

int MeanField::create_polymer_mask_from_inverse_litography_with_level_set()
{

    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;
    double eps_radius=this->Lx/256;
    double polymer_mask_radius=this->polymer_mask_radius_for_initial_wall;


    double ax_in=1.00,by_in=1.00,cz_in=1;

    ax_in=this->my_design_plan->ax*this->my_design_plan->ax*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;
    by_in=this->my_design_plan->by*this->my_design_plan->by*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;
    cz_in=this->my_design_plan->cz*this->my_design_plan->cz*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;

    for(int i=0;i<this->my_design_plan->n2Check;i++)
    {
        std::cout<<this->my_design_plan->xc[i]<<" "<<this->my_design_plan->yc[i]<<" "<<this->my_design_plan->zc[i]<<std::endl;
    }

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("litography_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;

    VecDuplicate(this->phi,&this->polymer_mask);
    PetscScalar *polymer_mask_local;
    VecGetArray(this->polymer_mask,&polymer_mask_local);
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];
    //global_node_number=nodes->offset_owned_indeps;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++)
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

        double min_distance; double distance;
        for(int ii=0;ii<this->my_design_plan->n2Check;ii++)
        {
            distance= pow(x- this->my_design_plan->xc[ii],2)/ax_in+pow(y-this->my_design_plan->yc[ii],2)/by_in;
            if(ii==0)
                min_distance=distance;
            else
                min_distance=min(min_distance,distance);
        }


        polymer_mask_local[i]=this->my_design_plan->r_spot/(pow(this->my_design_plan->cut_r_spot,0.5))*(pow(min_distance,0.5)-1.00);

        //  polymer_mask_local[i]=(pow(min_distance,0.5)-100.00*this->my_design_plan->r_spot/(pow(this->my_design_plan->cut_r_spot,0.5)));


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,polymer_mask_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,polymer_mask_local[i]);

#endif

    }
    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);

    this->ierr=VecRestoreArray(this->polymer_mask,&polymer_mask_local);

    this->scatter_petsc_vector(&this->polymer_mask);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask_from_distance_function",PETSC_TRUE);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_from_distance_function_binary_vector");

    my_p4est_level_set PLS(this->node_neighbors);
    Vec temp_f;
    this->ierr=VecDuplicate(this->polymer_mask,&temp_f);
    this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_f);
    double V_it=integrate_over_negative_domain(this->p4est,this->nodes,this->polymer_mask,temp_f);;
    Vec vn_for_gudonov_advection;
    this->ierr=VecDuplicate(this->polymer_mask,&vn_for_gudonov_advection); CHKERRXX(this->ierr);
    this->ierr=VecSet(vn_for_gudonov_advection,0.0001); CHKERRXX(this->ierr);

    double dt_acvection;
    while(this->my_design_plan->Vspot/V_it>(this->f))
    {
        dt_acvection=PLS.advect_in_normal_direction(vn_for_gudonov_advection,this->polymer_mask,this->alpha_cfl);
        PLS.reinitialize_2nd_order(this->polymer_mask,50);
        this->scatter_petsc_vector(&this->polymer_mask);


        V_it =integrate_over_negative_domain(this->p4est,this->nodes,this->polymer_mask,temp_f);

    }

    //    this->f=round(100*this->my_design_plan->Vspot/V_it)/100;
    //    this->myDiffusion->set_f_a(this->f);

    this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(vn_for_gudonov_advection); CHKERRXX(this->ierr);

    PLS.reinitialize_2nd_order(this->polymer_mask,50);


    this->scatter_petsc_vector(&this->polymer_mask);
    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/((double)Nx));
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_from_distance_function_vector");
}



int MeanField::create_polymer_mask()
{
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;
    double eps_radius=this->Lx/256;
    double polymer_mask_radius=this->polymer_mask_radius_for_initial_wall;
    VecDuplicate(this->phi,&this->polymer_mask);
    PetscScalar *polymer_mask_local;
    VecGetArray(this->polymer_mask,&polymer_mask_local);
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<this->mpi->mpirank;ii++)
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


        double dp=  (x-polymer_mask_xc)  *(x-polymer_mask_xc)+(y-polymer_mask_yc)*(y-polymer_mask_yc);
#ifdef P4_TO_P8
        dp +=(z-polymer_mask_zc)*(z-polymer_mask_zc) ;
#endif

        dp=pow(dp,0.5);

        if(dp<polymer_mask_radius)
            polymer_mask_local[i]=  dp-polymer_mask_radius; // negative inside the circle
        if(dp>polymer_mask_radius)
            polymer_mask_local[i]=  dp-polymer_mask_radius; // positive outside the circle
        if(dp==0)
            polymer_mask_local[i]==0;


    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);



    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask",PETSC_TRUE);
    //  this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_sphere_binary_vector");



    my_p4est_level_set PLS(this->node_neighbors);
    PLS.reinitialize_2nd_order(this->polymer_mask,50);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/((double)Nx));


    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"mask_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    //    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_sphere_vector");


}

int MeanField::create_polymer_mask_moving_ellipse(double ax,double by,double cz)
{
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;

    this->a_ellipse=ax;
    this->b_ellipse=by;
    this->c_ellipse=cz;

    double polymer_mask_radius=this->polymer_mask_radius_for_initial_wall;

    ax=ax*polymer_mask_radius;
    by=by*polymer_mask_radius;
    cz=cz*polymer_mask_radius;

    ax=ax*ax;
    by=by*by;
    cz=cz*cz;

    this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
    PetscScalar *polymer_mask_local;
    this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
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

        double x =  node_x_fr_i(node) + tree_xmin;
        double y =  node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        double dp=  (x-polymer_mask_xc)  *(x-polymer_mask_xc)/ax+(y-polymer_mask_yc)*(y-polymer_mask_yc)/by;
#ifdef P4_TO_P8
        dp +=(z-polymer_mask_zc)*(z-polymer_mask_zc)/cz ;
#endif



        //        if(dp<1)
        //            polymer_mask_local[i]=dp-1; // negative inside the circle
        //        else
        //            polymer_mask_local[i]=1; // positive outside the circle

        dp=pow(dp,0.5);
        polymer_mask_local[i]=polymer_mask_radius*(dp-1); // negative inside the circle

    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);



    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_ellipse",PETSC_TRUE);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_ellipse_binary_vector");


    my_p4est_level_set PLS(this->node_neighbors);
    PLS.reinitialize_2nd_order(this->polymer_mask,50);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"ellipse_reinitialyzed",PETSC_TRUE);

    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);
    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);



//    PetscScalar dx_for_dt_min =this->Lx / (double)Nx;
//    PetscScalar kappa_bands=2;
//    FieldProcessor my_field_processor;
//    int n_smoothies=100;

//    my_field_processor.smooth_level_set(this->node_neighbors,&this->polymer_mask,this->nodes->num_owned_indeps,n_smoothies,kappa_bands*dx_for_dt_min);

    //PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

    //this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"l_shape_3d_smoothed",PETSC_TRUE);

      this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_ellipse_vector");




}


int MeanField::create_polymer_mask_drop(double ax,double by, double cz)
{
    double y_center=this->Lx/2.00;


    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=y_center;
    double polymer_mask_zc=this->Lx/2.00;

    this->a_ellipse=ax;
    this->b_ellipse=by;
    this->c_ellipse=cz;

    double polymer_mask_radius= this->a_radius*this->Lx/4.00; //*pow(3,0.5)

    ax=ax*polymer_mask_radius;
    by=by*polymer_mask_radius;
    cz=cz*polymer_mask_radius;

    ax=ax*ax;
    by=by*by;
    cz=cz*cz;

    double y_plane=y_center-(1.00/2.00)*this->a_radius*this->Lx/4.00*this->b_ellipse;

    double dp;
    this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
    PetscScalar *polymer_mask_local;
    this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
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

        double x =  node_x_fr_i(node) + tree_xmin;
        double y =  node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        double dp1=  (x-polymer_mask_xc)  *(x-polymer_mask_xc)/ax+(y-polymer_mask_yc)*(y-polymer_mask_yc)/by;
#ifdef P4_TO_P8
        dp1 +=(z-polymer_mask_zc)*(z-polymer_mask_zc)/cz ;
#endif

        dp1=pow(dp1,0.5);
        dp1=polymer_mask_radius*(dp1-1);
        double dp2=y_plane-y;

        if(dp2>=0)
            dp=dp2;
        else
            dp=dp1;



        polymer_mask_local[i]=dp; // negative inside the circle

    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_simple_drop",PETSC_TRUE);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_mask_drop_binary_vector");


    my_p4est_level_set PLS(this->node_neighbors);
    PLS.reinitialize_2nd_order(this->polymer_mask,50);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"drop_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"drop_ellipse_vector");

}


int MeanField::create_polymer_mask_v_shape(double ax,double by, double cz)
{
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;

    this->a_ellipse=ax;
    this->b_ellipse=by;
    this->c_ellipse=cz;

    double x_left=this->Lx/8;
    double x_right=this->Lx-this->Lx/8.00;



    double y_up,y_down,f_x=1;

    double gap_between_two_sinuses=this->Lx/4.00;
    double sin_amplitude=this->Lx/4.00;

    double y_up_x_left,y_down_x_left,y_up_x_right,y_down_x_right;


    y_up_x_left=polymer_mask_yc+cos(2*PI*f_x*(x_left-polymer_mask_xc)/this->Lx)*sin_amplitude
            +gap_between_two_sinuses/2.00;

    y_down_x_left=polymer_mask_yc+cos(2*PI*f_x*(x_left-polymer_mask_xc)/this->Lx)*sin_amplitude
            -gap_between_two_sinuses/2.00;


    y_up_x_right=polymer_mask_yc+cos(2*PI*f_x*(x_right-polymer_mask_xc)/this->Lx)*sin_amplitude
            +gap_between_two_sinuses/2.00;

    y_down_x_right=polymer_mask_yc+cos(2*PI*f_x*(x_right-polymer_mask_xc)/this->Lx)*sin_amplitude
            -gap_between_two_sinuses/2.00;


    double dp;
    this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
    PetscScalar *polymer_mask_local;
    this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
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

        double x =  node_x_fr_i(node) + tree_xmin;
        double y =  node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        y_up=polymer_mask_yc+cos(2*PI*f_x*(x-polymer_mask_xc)/this->Lx)*sin_amplitude
                +gap_between_two_sinuses/2.00;
        y_down=polymer_mask_yc+cos(2*PI*f_x*(x-polymer_mask_xc)/this->Lx)*sin_amplitude
                -gap_between_two_sinuses/2.00;

        // --------Between x_left and x_right: function of y only-------------//
        if(x>x_left && x<x_right)
        {
            if(y<=y_up && y>=y_down)
            {
                dp=MAX(y-y_up,y_down-y);
            }
            if(y>y_up)
            {
                dp=y-y_up;
            }
            if(y<y_down)
            {
                dp=y_down-y;
            }
        }//--------------------------Borders-------------------------------//
        else
        {
            double x_right_temp,x_left_temp;
            double y_right_temp,y_left_temp;
            //--------Right Border-------------------------------------------//

            //---------Right Border and too up or too low---------------------//
            if(x>=x_right && (y>=y_up_x_right || y<=y_down_x_right))
            {
                if(y>=y_up_x_right)
                    dp=y-y_up_x_right;
                if(y<=y_down_x_right)
                    dp=y_down_x_right-y;
            }

            //------Right Border but within y up and y down limits-----------//
            if(x>=x_right && (y<=y_up_x_right && y>=y_down_x_right))
            {
                y_right_temp=(y-y_down_x_right)/(y_up_x_right-y_down_x_right);
                x_right_temp=x_right+(this->Lx/16.00)*sin(PI*y_right_temp);
                if(x<x_right_temp)
                    dp=x-x_right_temp;
                else
                    dp=x-x_right_temp;
            }

            //--------Left Border-------------------------------------------//

            //---------Left Border and too up or too low---------------------//
            if(x<=x_left && (y>=y_up_x_left || y<=y_down_x_left))
            {
                if(y>=y_up_x_left)
                    dp=y-y_up_x_left;
                if(y<=y_down_x_left)
                    dp=y_down_x_left-y;
            }
            //------Left Border but within y up and y down limits-----------//
            if(x<=x_left && (y<=y_up_x_left && y>=y_down_x_left))
            {
                y_left_temp=(y-y_down_x_left)/(y_up_x_left-y_down_x_left);
                x_left_temp=x_left-(this->Lx/16.00)*sin(PI*y_left_temp);
                if(x>x_left_temp)
                    dp=x_left_temp-x;
                else
                    dp=x_left_temp-x;
            }


        }
        polymer_mask_local[i]=dp; // negative inside the circle\


    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_v_shape",PETSC_TRUE);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_v_shape_binary_vector");


    my_p4est_level_set PLS(this->node_neighbors);
    double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,0.0001,l1_error,l2_error);

    std::cout<<" l1_error "<<" l2_error "<<l1_error<<" "<<l2_error<<std::endl;

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"v_shape_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"v_shape_vector");


}

int MeanField::create_polymer_mask_l_shape(double ax,double by, double cz)
{
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;

    this->a_ellipse=ax;
    this->b_ellipse=by;
    this->c_ellipse=cz;

    double xc1=this->Lx/2-1.5*this->Lx/6;
    double yc1=this->Lx/2-1.5*this->Lx/6;

    double xc2=this->Lx/2+1.5*this->Lx/6;
    double yc2=this->Lx/2+1.5*this->Lx/6;

    //    double LL=(4.5/5.00)*  Lx/pow(8.00*5.00,0.5)*pow(5.00/4.875,0.5);

    double LL=this->a_ellipse*(4.5/5.00)*  Lx/pow(8.00*5.00,0.5)*pow(5.00/4.875,0.5);


    double x1=xc1-LL/2;double x2=xc1+LL/2;
    double y1=yc1-LL/2;double y2=xc1+LL/2;

    double x4=xc2+LL/2;double x5=xc2+LL/2;
    double y4=yc2-LL/2;double y5=yc2+LL/2;

    double dp;

    this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
    PetscScalar *polymer_mask_local;
    this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
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

        double x =  node_x_fr_i(node) + tree_xmin;
        double y =  node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        dp=1;
        bool outside=true;
        if(x<=x2 && x>=x1 && y>=y1 && y<=y5)
        {
            dp=-min(x2-x1,x-x1);
            outside=false;
        }
        if(x<=x4 && x>=x1 && y>=y4 && y<=y5)
        {
            outside=false;
            dp=-min(y5-y,y-y4);
        }
        if(outside)
        {
            if(y>y5)
                dp=y-y5;
            else
                if(x<x1)
                    dp=x1-x;
                else
                    if(x>x2 && y<y4)
                        dp=min(max(x-x2,0.00),max(y4-y,0.00));

            if(x<x2 && x>x1 && y<y2)
                dp=y2-y;
            if(y>y4 && y<y5 && x>x4)
                dp=x-x4;

        }

        polymer_mask_local[i]=dp; // negative inside the circle\


    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_l_shape",PETSC_TRUE);
    //this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"polymer_l_shape_binary_vector");


    my_p4est_level_set PLS(this->node_neighbors);
    double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,0.0000000001,l1_error,l2_error);

    std::cout<<" l1_error "<<" l2_error "<<l1_error<<" "<<l2_error<<std::endl;

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"l_shape_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    // this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_mask,"l_shape_vector");


}



int MeanField::create_polymer_mask_cube()
{
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;
    double ax, by,cz;


    ax=this->a_ellipse;
    by=this->b_ellipse;
    cz=this->c_ellipse;

    ax=ax*this->polymer_mask_radius_for_initial_wall;
    by=by*this->polymer_mask_radius_for_initial_wall;
    cz=cz*this->polymer_mask_radius_for_initial_wall;

    this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
    PetscScalar *polymer_mask_local;
    this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;


    double distance_x,distance_y,distance_z;
    double min_distance;

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

        double x =  node_x_fr_i(node) + tree_xmin;
        double y =  node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif


        double x_0=x-polymer_mask_xc;
        double y_0=y-polymer_mask_yc;

#ifdef P4_TO_P8
        double z_0=z-polymer_mask_zc;
#endif

#ifdef P4_TO_P8
        if( (x_0> -ax/2 && x_0<ax/2) && (y_0> -by/2 && y_0<by/2)

                && (z_0> -cz/2 && z_0<cz/2)
                )
        {
            // inside the domain
            polymer_mask_local[i]=-1.00;

            distance_x=min(x_0+ax/2,ax/2-x_0);
            distance_y=min(y_0+by/2,by/2-y_0);


            distance_z=min(z_0+cz/2,cz/2-z_0);

            distance_x=max(distance_x,0.00);
            distance_y=max(distance_y,0.00);
            distance_z=max(distance_z,0.00);


            min_distance=min(distance_x,distance_y);
            min_distance=min(min_distance,distance_z);

            if(min_distance<=0)
            {

            }

            polymer_mask_local[i]=-min_distance;
        }
        else
        {
            //outside the domain
            polymer_mask_local[i]=1.00;

            distance_x=10*this->Lx;
            if( (x_0-ax/2) >0)
                distance_x=x_0-ax/2;
            if( (-ax/2-x_0)>0)
                distance_x=-ax/2-x_0;
            if(! (x_0-ax/2) <=0 && (-ax/2-x_0)<=0)




                distance_y=10*this->Lx;
            if( (y_0-by/2) >0)
                distance_y=y_0-by/2;
            if( (-by/2-y_0)>0)
                distance_y=-by/2-y_0;


            distance_z=10*this->Lx;
            if( (z_0-cz/2) >0)
                distance_z=z_0-cz/2;
            if( (-cz/2-z_0)>0)
                distance_z=-cz/2-z_0;


            distance_x=max(distance_x,0.00);
            distance_y=max(distance_y,0.00);
            distance_z=max(distance_z,0.00);


            min_distance=min(distance_x,distance_y);

            min_distance=min(min_distance,distance_z);

            if(min_distance<=0)
            {

            }

            polymer_mask_local[i]=min_distance;
        }
#else
        if( (x_0> -ax/2 && x_0<ax/2) && (y_0> -by/2 && y_0<by/2))
            polymer_mask_local[i]=-1;
        else
            polymer_mask_local[i]=1;


#endif


    }

    this->ierr= VecRestoreArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);



    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask",PETSC_TRUE);

    my_p4est_level_set PLS(this->node_neighbors);
    int n_level_set_reinitialyzations=50;

#ifdef P4_TO_P8
    n_level_set_reinitialyzations=200;
    double l1_tolerance=0.0000001; double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(this->polymer_mask,n_level_set_reinitialyzations);
#else
//        double l1_tolerance=0.001; double l1_error,l2_error;
   //     PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(this->polymer_mask,n_level_set_reinitialyzations);

#endif

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);



    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));

    double effective_tolerance=this->ls_tolerance*this->Lx/(double)Nx;
    PLS.perturb_level_set_function(&this->polymer_mask,effective_tolerance);


    // this->interpolate_and_print_vec_to_uniform_grid(&this->polymer_mask,"initial_mask");

    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask_cube_reinitialyzed",PETSC_TRUE);
    ///this->myDiffusion->printDiffusionVector(&this->polymer_mask,"polymer_mask_vector");

}

int MeanField::create_polymer_mask_annular_cube()
{
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;
    double ax1, by1,cz1;


    ax1=this->a_ellipse;
    by1=this->b_ellipse;
    cz1=this->c_ellipse;

    ax1=ax1*this->polymer_mask_radius_for_initial_wall;
    by1=by1*this->polymer_mask_radius_for_initial_wall;
    cz1=cz1*this->polymer_mask_radius_for_initial_wall;


    Vec phi_square_out;
    Vec phi_square_in;

    this->create_polymer_mask_square(&phi_square_out,ax1,by1,cz1);

    double ax2=this->a_ellipse;
    double by2=this->b_ellipse;
    double cz2=this->c_ellipse;

    ax2=ax2*(this->polymer_mask_radius_for_initial_wall-this->Lx/20.00);
    by2=by2*(this->polymer_mask_radius_for_initial_wall-this->Lx/20.00);
    cz2=cz2*(this->polymer_mask_radius_for_initial_wall-this->Lx/20.00);

    this->create_polymer_mask_square(&phi_square_in,ax2,by2,cz2);


    this->ierr=VecDuplicate(this->phi,&this->polymer_mask); CHKERRXX(this->ierr);
    PetscScalar *polymer_mask_local,*phi_out_local,*phi_in_local;
    this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(phi_square_out,&phi_out_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(phi_square_in,&phi_in_local); CHKERRXX(this->ierr);


    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;


    double distance_x,distance_y,distance_z;
    double min_distance;

    bool picked_value;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {

        picked_value=false;
        if(phi_out_local[i]>0 && !picked_value)
        {
            polymer_mask_local[i]=phi_out_local[i];
            picked_value=true;
        }
        if(phi_in_local[i]<0 && !picked_value)
        {
            polymer_mask_local[i]=-phi_in_local[i];
            picked_value=true;
        }
        if(phi_in_local[i]>0 && phi_out_local[i]<=0 && !picked_value)
        {
            polymer_mask_local[i]=-min(phi_in_local[i],-phi_out_local[i]);
            picked_value=true;
        }

    }

    this->ierr= VecRestoreArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(phi_square_in,&phi_in_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(phi_square_out,&phi_out_local); CHKERRXX(this->ierr);


    this->scatter_petsc_vector(&this->polymer_mask);
    this->ierr=VecDestroy(phi_square_out); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(phi_square_in); CHKERRXX(this->ierr);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask",PETSC_TRUE);

    my_p4est_level_set PLS(this->node_neighbors);
    int n_level_set_reinitialyzations=50;

#ifdef P4_TO_P8
    n_level_set_reinitialyzations=200;
    double l1_tolerance=0.0000001; double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(this->polymer_mask,n_level_set_reinitialyzations);
#else
    //    double l1_tolerance=0.0000001; double l1_error,l2_error;
    //    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(this->polymer_mask,n_level_set_reinitialyzations);

#endif

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);



    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));

    double effective_tolerance=this->ls_tolerance*this->Lx/(double)Nx;
    PLS.perturb_level_set_function(&this->polymer_mask,effective_tolerance);


    this->interpolate_and_print_vec_to_uniform_grid(&this->polymer_mask,"initial_mask",false);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask_cube_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->printDiffusionVector(&this->polymer_mask,"polymer_mask_vector");


    this->ierr=VecGetArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);


    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;

    double eps_shape=1*this->Lx/300.00;
    double dl_tolerance=this->Lx/30.00;

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

        double x =  node_x_fr_i(node) + tree_xmin;
        double y =  node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif


        double x_0=x-polymer_mask_xc;
        double y_0=y-polymer_mask_yc;

#ifdef P4_TO_P8
        double z_0=z-polymer_mask_zc;
#endif


        if( (y_0<(-by2/2.00+dl_tolerance) )|| (y_0>(by2/2.00-dl_tolerance)) && ( (x_0>-ax2/2.00) && (x_0<ax2/2.00)))
        {
            polymer_mask_local[i]+=eps_shape*sin(16.00*2*PI*(x-(this->Lx-ax2)/2.00)/ax2);
        }
        if( (x_0<(-ax2/2.00+dl_tolerance)) || (x_0>(ax2/2.00-dl_tolerance)) && ((y_0>-by2/2.00) && (y_0<by2/2.00)))
        {
            polymer_mask_local[i]+=eps_shape*sin(16.00*2.00*PI*(y-(this->Lx-by2)/2.00)/by2);
        }
    }

    this->ierr= VecRestoreArray(this->polymer_mask,&polymer_mask_local); CHKERRXX(this->ierr);


    this->scatter_petsc_vector(&this->polymer_mask);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask",PETSC_TRUE);



#ifdef P4_TO_P8
    n_level_set_reinitialyzations=200;
    //  double l1_tolerance=0.0000001; double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(this->polymer_mask,n_level_set_reinitialyzations);
#else
    //    double l1_tolerance=0.0000001; double l1_error,l2_error;
    //    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(this->polymer_mask,n_level_set_reinitialyzations);

#endif

    this->scatter_petsc_vector(&this->polymer_mask);



    effective_tolerance=this->ls_tolerance*this->Lx/(double)Nx;
    PLS.perturb_level_set_function(&this->polymer_mask,effective_tolerance);


    this->interpolate_and_print_vec_to_uniform_grid(&this->polymer_mask,"initial_mask",false);

    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"polymer_mask_cube_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->printDiffusionVector(&this->polymer_mask,"polymer_mask_vector");


}


int MeanField::create_polymer_mask_square(Vec *phi_square,double ax,double by,double cz)
{
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;



    this->ierr=VecDuplicate(this->phi,phi_square); CHKERRXX(this->ierr);
    PetscScalar *polymer_mask_local;
    this->ierr=VecGetArray(*phi_square,&polymer_mask_local); CHKERRXX(this->ierr);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;


    double distance_x,distance_y,distance_z;
    double min_distance;

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

        double x =  node_x_fr_i(node) + tree_xmin;
        double y =  node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif


        double x_0=x-polymer_mask_xc;
        double y_0=y-polymer_mask_yc;

#ifdef P4_TO_P8
        double z_0=z-polymer_mask_zc;
#endif

#ifdef P4_TO_P8
        if( (x_0> -ax/2 && x_0<ax/2) && (y_0> -by/2 && y_0<by/2)

                && (z_0> -cz/2 && z_0<cz/2)
                )
        {
            // inside the domain
            polymer_mask_local[i]=-1.00;

            distance_x=min(x_0+ax/2,ax/2-x_0);
            distance_y=min(y_0+by/2,by/2-y_0);


            distance_z=min(z_0+cz/2,cz/2-z_0);

            distance_x=max(distance_x,0.00);
            distance_y=max(distance_y,0.00);
            distance_z=max(distance_z,0.00);


            min_distance=min(distance_x,distance_y);
            min_distance=min(min_distance,distance_z);

            if(min_distance<=0)
            {

            }

            polymer_mask_local[i]=-min_distance;
        }
        else
        {
            //outside the domain
            polymer_mask_local[i]=1.00;

            distance_x=10*this->Lx;
            if( (x_0-ax/2) >0)
                distance_x=x_0-ax/2;
            if( (-ax/2-x_0)>0)
                distance_x=-ax/2-x_0;
            if(! (x_0-ax/2) <=0 && (-ax/2-x_0)<=0)




                distance_y=10*this->Lx;
            if( (y_0-by/2) >0)
                distance_y=y_0-by/2;
            if( (-by/2-y_0)>0)
                distance_y=-by/2-y_0;


            distance_z=10*this->Lx;
            if( (z_0-cz/2) >0)
                distance_z=z_0-cz/2;
            if( (-cz/2-z_0)>0)
                distance_z=-cz/2-z_0;


            distance_x=max(distance_x,0.00);
            distance_y=max(distance_y,0.00);
            distance_z=max(distance_z,0.00);


            min_distance=min(distance_x,distance_y);

            min_distance=min(min_distance,distance_z);

            if(min_distance<=0)
            {

            }

            polymer_mask_local[i]=min_distance;
        }
#else
        if( (x_0> -ax/2 && x_0<ax/2) && (y_0> -by/2 && y_0<by/2))
            polymer_mask_local[i]=-1;
        else
            polymer_mask_local[i]=1;


#endif


    }

    this->ierr= VecRestoreArray(*phi_square,&polymer_mask_local); CHKERRXX(this->ierr);



    this->ierr=this->scatter_petsc_vector(phi_square);
    this->my_p4est_vtk_write_all_periodic_adapter(phi_square,"phi_square",PETSC_TRUE);

    my_p4est_level_set PLS(this->node_neighbors);
    int n_level_set_reinitialyzations=50;

#ifdef P4_TO_P8
    n_level_set_reinitialyzations=200;
    double l1_tolerance=0.0000001; double l1_error,l2_error;
    //PLS.reinitialize_2nd_order_with_tolerance(phi_square,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(this->polymer_mask,n_level_set_reinitialyzations);
#else
    //    double l1_tolerance=0.0000001; double l1_error,l2_error;
    //    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    PLS.reinitialize_2nd_order(*phi_square,n_level_set_reinitialyzations);

#endif

    this->scatter_petsc_vector(phi_square);


    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));

    double effective_tolerance=this->ls_tolerance*this->Lx/(double)Nx;
    PLS.perturb_level_set_function(phi_square,effective_tolerance);


    this->interpolate_and_print_vec_to_uniform_grid(phi_square,"initial_square",false);


    this->my_p4est_vtk_write_all_periodic_adapter(phi_square,"polymer_mask_cube_reinitialyzed",PETSC_TRUE);

}


int MeanField::create_polymer_mask_helix()
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
    int tol=N_t_helix/4;

    double *t_vector=new double[N_t_helix-2*tol];

    double dt_helix=(t_end-t_start)/N_t_helix;
    double *distance_helix=new double[N_t_helix-2*tol];
    double *sint_vector=new double [N_t_helix-2*tol];
    double *cost_vector=new double[N_t_helix-2*tol];

    for(int i=0;i<N_t_helix-2*tol;i++)
    {
        t_vector[i]=dt_helix*(i+tol);
        sint_vector[i]=r_mask*sin(2*PI*helix_speed*t_vector[i]/t_end)+xc;
        cost_vector[i]=r_mask*cos(2*PI*helix_speed*t_vector[i]/t_end)+yc;
        std::cout<<i<<" "<<t_vector[i]<<" "
                <<cost_vector[i]<<" "
               << sint_vector[i]<<std::endl;
    }


    //    double x_start=r_mask*cos(t_start+(tol-1)*dt_helix)+xc;
    //    double y_start=r_mask*sin(t_start+(tol-1)*dt_helix)+yc;

    //    double x_end=r_mask*cos(t_end-(tol-1)*dt_helix)+xc;
    //    double y_end=r_mask*sin(t_end-(tol-1)*dt_helix)+yc;

    VecDuplicate(this->phi,&this->polymer_mask);
    PetscScalar *polymer_mask_local;
    VecGetArray(this->polymer_mask,&polymer_mask_local);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;

    std::cout<<nodes->num_owned_indeps<<std::endl;
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

        double x =node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node)+ tree_zmin;
#endif

        bool inside_helix=false;


        for(int j=0;j<N_t_helix-2*tol;j++)
        {
            distance_helix[j]=
                    (x-cost_vector[j])*(x-cost_vector[j])
                    +(y-sint_vector[j])*(y-sint_vector[j])
                    +(z-t_vector[j])*(z-t_vector[j]);

            //distance_helix[j]=(z-t_vector[j])*(z-t_vector[j]);
        }

        // this->myDiffusion->printDiffusionArray(distance_helix,N_t_helix,"distance_helix");

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


        //        std::cout<<i<<" "<<x<<" "<<y<<" "<<z<<" "<<minimum_proposed<<" "<<r_helix<<
        //                   " "<<cost_vector[j_min]<<" "<<sint_vector[j_min]<<" "<<t_vector[j_min]<<   std::endl;



        //        double dx= (x-cost_vector[j_min])*(x-cost_vector[j_min]);
        //         double dy=(y-sint_vector[j_min])*(y-sint_vector[j_min]);
        //        double dz=(z-t_vector[j_min])*(z-t_vector[j_min]);

        //       double distance_helix_jmin= (x-cost_vector[j_min])*(x-cost_vector[j_min])+
        //        +(y-sint_vector[j_min])*(y-sint_vector[j_min])
        //        +(z-t_vector[j_min])*(z-t_vector[j_min]);

        //       std::cout<<distance_helix_jmin<<std::endl;

        if(minimum_proposed<r_helix)
            inside_helix=true;
        else
            inside_helix=false;

        //        bool condition1=inside_helix;
        //        bool condition2=true;//(z>t_start && z<t_end);
        //        bool condition3= true;//(x-x_start)*(x-x_start)+(y-y_start)*(y-y_start)+(z-t_start)*(z-t_start)<r_helix*r_helix;
        //        bool condition4=true;//(x-x_end)*(x-x_end)+(y-y_end)*(y-y_end)+(z-t_end)*(z-t_end)<r_helix*r_helix;


        polymer_mask_local[i]=minimum_proposed-r_helix;
        //        if(condition1 && (condition2  ||condition3 ||condition4  ) )
        //            polymer_mask_local[i]=-1;


    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);



    //  my_p4est_level_set PLS(this->node_neighbors);
    // PLS.reinitialize_1st_order(this->polymer_mask);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"helix_binary",PETSC_TRUE);

    my_p4est_level_set PLS(this->node_neighbors);
    double l1_tolerance=0.0000001; double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"helix_reinitialyzed",PETSC_TRUE);

    //this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"helix_reinitialyzed_");

    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionVector(&this->polymer_mask,"helix_vector");



    delete t_vector;
    delete distance_helix;
    delete cost_vector;
    delete sint_vector;

#endif
}

int MeanField::create_polymer_mask_3d_l_shape_old(double ax, double by, double cz)
{
#ifdef P4_TO_P8
    double r_helix=this->Lx/32.00;

    double r_mask=2*this->Lx/8.00;

    double t_start=0;
    double t_end=this->Lx;

    double xc=this->Lx/2.00;
    double yc=this->Lx/2.00;

    double x_first_one,y_first_one,z_first_one,
            x_second_one,y_second_one,z_second_one,
            x_first_two,y_first_two,z_first_two,
            x_second_two,y_second_two,z_second_two,
            x_first_three,y_first_three,z_first_three,
            x_second_three,y_second_three,z_second_three;

    double move_1=1.00/8.00*this->Lx,move_2=7.00/8.00*this->Lx;

    x_first_one=move_1,y_first_one=move_1,z_first_one=move_1;
    x_second_one=move_2,y_second_one=move_1,z_second_one=move_1;

    x_first_two=x_second_one,y_first_two=y_second_one,z_first_two=z_second_one;
    x_second_two=move_2,y_second_two=move_2,z_second_two=move_1;

    x_first_three=move_2,y_first_three=move_2,z_first_three=move_1;
    x_second_three=move_2,y_second_three=move_2,z_second_three=move_2;


    int tol=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))/4;

    double helix_speed=2;

    int N_t_helix=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));

    double *t_vector=new double[N_t_helix-2*tol];

    double dt_helix=(t_end-t_start)/N_t_helix;
    double *distance_helix=new double[N_t_helix-2*tol];
    double *sint_vector=new double [N_t_helix-2*tol];
    double *cost_vector=new double[N_t_helix-2*tol];

    for(int i=0;i<N_t_helix-2*tol;i++)
    {
        t_vector[i]=dt_helix*(i+tol);
        sint_vector[i]=r_mask*sin(2*PI*helix_speed*t_vector[i]/t_end)+xc;
        cost_vector[i]=r_mask*cos(2*PI*helix_speed*t_vector[i]/t_end)+yc;
        std::cout<<i<<" "<<t_vector[i]<<" "
                <<cost_vector[i]<<" "
               << sint_vector[i]<<std::endl;
    }


    //    double x_start=r_mask*cos(t_start+(tol-1)*dt_helix)+xc;
    //    double y_start=r_mask*sin(t_start+(tol-1)*dt_helix)+yc;

    //    double x_end=r_mask*cos(t_end-(tol-1)*dt_helix)+xc;
    //    double y_end=r_mask*sin(t_end-(tol-1)*dt_helix)+yc;

    VecDuplicate(this->phi,&this->polymer_mask);
    PetscScalar *polymer_mask_local;
    VecGetArray(this->polymer_mask,&polymer_mask_local);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;

    double d_1,d_2,d_3;double minimum_proposed;

    std::cout<<nodes->num_owned_indeps<<std::endl;
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

        double x =node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node)+ tree_zmin;
#endif

        bool inside_helix=false;


        d_1=MeanField::distance_of_a_point_from_a_line(x,y,z,x_first_one,y_first_one,z_first_one,
                                                       x_second_one,y_second_one,z_second_one);

        d_2=MeanField::distance_of_a_point_from_a_line(x,y,z,x_first_two,y_first_two,z_first_two,
                                                       x_second_two,y_second_two,z_second_two);


        d_3=MeanField::distance_of_a_point_from_a_line(x,y,z,x_first_three,y_first_three,z_first_three,
                                                       x_second_three,y_second_three,z_second_three);


        d_1=pow(d_1,0.5);
        d_2=pow(d_2,0.5);
        d_3=pow(d_3,0.5);

        minimum_proposed=min(d_1,d_2);
        minimum_proposed=min(d_3,minimum_proposed);

        if(minimum_proposed<r_helix)
            inside_helix=true;
        else
            inside_helix=false;


        polymer_mask_local[i]=minimum_proposed-r_helix;


        if(x>(move_2+r_helix) && inside_helix)
             polymer_mask_local[i]=x-(move_2+r_helix);
        if(x<(move_1-r_helix) && inside_helix)
             polymer_mask_local[i]=(move_1-r_helix)-x;

        if(y>(move_2+r_helix) && inside_helix)
             polymer_mask_local[i]=y-(move_2+r_helix);
        if(y<(move_1-r_helix) && inside_helix)
             polymer_mask_local[i]=(move_1-r_helix)-y;

        if(z>(move_2+r_helix) && inside_helix)
             polymer_mask_local[i]=z-(move_2+r_helix);
        if(z<(move_1-r_helix) && inside_helix)
             polymer_mask_local[i]=(move_1-r_helix)-z;

//                if(x>(move_2-r_helix) && inside_helix)
//                {
//                    minimum_proposed=min(d_3,d_2);
//                    polymer_mask_local[i]=minimum_proposed-r_helix;
//                }


//                if(x<(move_1+r_helix) && inside_helix)
//                {
//                    minimum_proposed=min(d_3,d_2);
//                    polymer_mask_local[i]=minimum_proposed-r_helix;
//                }

//                if(y>(move_2-r_helix) && inside_helix)
//                {
//                    minimum_proposed=min(d_3,d_1);
//                    polymer_mask_local[i]=minimum_proposed-r_helix;
//                }


//                if(y<(move_1+r_helix) && inside_helix)
//                {
//                    minimum_proposed=min(d_3,d_1);
//                    polymer_mask_local[i]=minimum_proposed-r_helix;
//                }

//                if(z>(move_2-r_helix) && inside_helix)
//                {
//                    minimum_proposed=min(d_2,d_1);
//                    polymer_mask_local[i]=minimum_proposed-r_helix;
//                }


//                if(z<(move_1+r_helix) && inside_helix)
//                {
//                    minimum_proposed=min(d_2,d_1);
//                    polymer_mask_local[i]=minimum_proposed-r_helix;
//                }
    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);



    //  my_p4est_level_set PLS(this->node_neighbors);
    // PLS.reinitialize_1st_order(this->polymer_mask);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"l_shape_3d_binary",PETSC_TRUE);

    my_p4est_level_set PLS(this->node_neighbors);
    double l1_tolerance=0.001; double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"l_shape_3d_reinitialyzed",PETSC_TRUE);

    PetscScalar Nx=pow(2.00,this->max_level+log(this->nx_trees)/log(2));
    PetscScalar dx_for_dt_min =this->Lx / (double)Nx;
    PetscScalar kappa_bands=2;
    FieldProcessor my_field_processor;
    int n_smoothies=20;

    my_field_processor.smooth_level_set(this->node_neighbors,&this->polymer_mask,this->nodes->num_owned_indeps,n_smoothies,kappa_bands*dx_for_dt_min);

    //PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"l_shape_3d_smoothed",PETSC_TRUE);

    //this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"helix_reinitialyzed_");

    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionVector(&this->polymer_mask,"l_shape_3d_vector");



    delete t_vector;
    delete distance_helix;
    delete cost_vector;
    delete sint_vector;

#endif
}

int MeanField::create_polymer_mask_3d_l_shape(double ax, double by, double cz)
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


    //    double x_start=r_mask*cos(t_start+(tol-1)*dt_helix)+xc;
    //    double y_start=r_mask*sin(t_start+(tol-1)*dt_helix)+yc;

    //    double x_end=r_mask*cos(t_end-(tol-1)*dt_helix)+xc;
    //    double y_end=r_mask*sin(t_end-(tol-1)*dt_helix)+yc;

    VecDuplicate(this->phi,&this->polymer_mask);
    PetscScalar *polymer_mask_local;
    VecGetArray(this->polymer_mask,&polymer_mask_local);

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;

    std::cout<<nodes->num_owned_indeps<<std::endl;
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

        double x =node_x_fr_i(node) + tree_xmin;
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node)+ tree_zmin;
#endif

        bool inside_helix=false;


        for(int j=0;j<N_t_helix-2*tol;j++)
        {
            distance_helix[j]=
                    (x-x_vector[j])*(x-x_vector[j])
                    +(y-y_vector[j])*(y-y_vector[j])
                    +(z-z_vector[j])*(z-z_vector[j]);

            //distance_helix[j]=(z-t_vector[j])*(z-t_vector[j]);
        }

        // this->myDiffusion->printDiffusionArray(distance_helix,N_t_helix,"distance_helix");

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


        //        std::cout<<i<<" "<<x<<" "<<y<<" "<<z<<" "<<minimum_proposed<<" "<<r_helix<<
        //                   " "<<cost_vector[j_min]<<" "<<sint_vector[j_min]<<" "<<t_vector[j_min]<<   std::endl;



        //        double dx= (x-cost_vector[j_min])*(x-cost_vector[j_min]);
        //         double dy=(y-sint_vector[j_min])*(y-sint_vector[j_min]);
        //        double dz=(z-t_vector[j_min])*(z-t_vector[j_min]);

        //       double distance_helix_jmin= (x-cost_vector[j_min])*(x-cost_vector[j_min])+
        //        +(y-sint_vector[j_min])*(y-sint_vector[j_min])
        //        +(z-t_vector[j_min])*(z-t_vector[j_min]);

        //       std::cout<<distance_helix_jmin<<std::endl;

        if(minimum_proposed<r_helix)
            inside_helix=true;
        else
            inside_helix=false;

        //        bool condition1=inside_helix;
        //        bool condition2=true;//(z>t_start && z<t_end);
        //        bool condition3= true;//(x-x_start)*(x-x_start)+(y-y_start)*(y-y_start)+(z-t_start)*(z-t_start)<r_helix*r_helix;
        //        bool condition4=true;//(x-x_end)*(x-x_end)+(y-y_end)*(y-y_end)+(z-t_end)*(z-t_end)<r_helix*r_helix;


        polymer_mask_local[i]=minimum_proposed-r_helix;
        //        if(condition1 && (condition2  ||condition3 ||condition4  ) )
        //            polymer_mask_local[i]=-1;


    }

    VecRestoreArray(this->polymer_mask,&polymer_mask_local);



    //  my_p4est_level_set PLS(this->node_neighbors);
    // PLS.reinitialize_1st_order(this->polymer_mask);

    this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"helix_binary",PETSC_TRUE);

    my_p4est_level_set PLS(this->node_neighbors);
    double l1_tolerance=0.0000001; double l1_error,l2_error;
    PLS.reinitialize_2nd_order_with_tolerance(this->polymer_mask,l1_tolerance,l1_error,l2_error);

    this->scatter_petsc_vector(&this->polymer_mask);
//    PetscScalar Nx2=pow(2.00,this->max_level+log(this->nx_trees)/log(2));
//    PetscScalar dx_for_dt_min =this->Lx / (double)Nx2;
//    PetscScalar kappa_bands=2;
//    FieldProcessor my_field_processor;
//    int n_smoothies=20;

//    my_field_processor.smooth_level_set(this->node_neighbors,&this->polymer_mask,this->nodes->num_owned_indeps,n_smoothies,kappa_bands*dx_for_dt_min);


//    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"l_shape_3d_smoothed",PETSC_TRUE);


    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->polymer_mask,this->ls_tolerance*this->Lx/(double)Nx);
 this->scatter_petsc_vector(&this->polymer_mask);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"helix_reinitialyzed",PETSC_TRUE);

    //this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"helix_reinitialyzed_");

    this->myDiffusion->set_polymer_shape(this->polymer_mask);
    this->myDiffusion->printDiffusionVector(&this->polymer_mask,"helix_vector");



    delete x_vector;
    delete distance_helix;
    delete y_vector;
    delete z_vector;

#endif
}


int MeanField::printNumericalDataEvolution()
{
    {
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("numerical data evolution")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %d %d %f %f %d %d \n",
                this->i_mean_field_iteration,  this->myDiffusion->get_n_global_size_sol(), this->myDiffusion->get_n_local_size_sol(),
                this->lambda_plus,this->lambda_minus,this->start_remesh_i,this->stop_remesh_i);

        fclose(outFile);
    }
}


int MeanField::printNumericalData()
{

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("numerical data")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();


    ofstream Data_Input_File ("input_file_for_mean_field.txt");         //Opening file to print info to
    Data_Input_File << "input data stream" << endl;          //Headings for file
    Data_Input_File <<"int min_level "<<6<<std::endl;
    Data_Input_File <<"int max_level "<<6<<std::endl;
    Data_Input_File <<"int t_iterations "<<100<<std::endl;
    Data_Input_File <<"int mean_field_iterations "<<1000<<std::endl;
    Data_Input_File <<"double f "<<0.14<<std::endl;
    Data_Input_File <<"double lambda " <<1.00<<std::endl;
    Data_Input_File <<"double Xhi_AB "<<100<<std::endl;
    Data_Input_File <<"Lx_physics "<<5.4559<<std::endl;
    Data_Input_File <<"MeanField::mask_strategy my_mask_strategy "<<"MeanField::sphere_mask"<<std::endl;
    Data_Input_File <<"Diffusion::phase myPhase "<<"Diffusion::disordered"<<std::endl;
    Data_Input_File <<"Diffusion::casl_diffusion_method my_casl_diffusion_method "<<"Diffusion::periodic_crank_nicholson"<<std::endl;
    Data_Input_File <<"Diffusion::numerical_scheme my_numerical_scheme "<<"Diffusion::splitting_spectral_adaptive"<<std::endl;
    Data_Input_File <<"PetscInt  extension_advection_period "<<1000<<std::endl;
    Data_Input_File <<"PetscInt stress_tensor_computation_period "<<1000<<std::endl;
    Data_Input_File<<"int remesh_period "<< 1<<std::endl;
    Data_Input_File<<"double ax "<<1.00<<std::endl;
    Data_Input_File<<"double by "<<1.00<<std::endl;
    Data_Input_File<<"double cz"<<1.00<<std::endl;
    Data_Input_File<<"std::string   my_io_path "<<"/Users/gaddielouaknin/p4estLocal3/PaperResultsFFT/"<<std::endl;
    Data_Input_File<<"MeanField::strategy_meshing my_strategy_meshing=MeanField::two_level_set "<<std::endl;
    Data_Input_File<<"setup_finite_difference_solver "<<"PETSC_FALSE"<<std::endl;
    Data_Input_File.close();
}


int MeanField::printOptimizationEvolution()
{
    {
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("mean_field_optimization")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %d %d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",this->i_mean_field_iteration,  this->myDiffusion->get_n_global_size_sol(), this->myDiffusion->get_n_local_size_sol(),
                this->energy[this->i_mean_field_iteration],this->pressure_force[this->i_mean_field_iteration],
                this->exchange_force[this->i_mean_field_iteration],this->order_Ratio[this->i_mean_field_iteration],this->rho0Average[this->i_mean_field_iteration],
                this->rhoAAverage[this->i_mean_field_iteration],this->rhoBAverage[this->i_mean_field_iteration],this->length_series[this->i_mean_field_iteration],
                this->myDiffusion->getQForward(),this->myDiffusion->getQBackward(),this->myDiffusion->getV(),this->volume_phi_seed,
                this->myDiffusion->getE_logQ(),this->myDiffusion->getE_w(), this->myDiffusion->getQForward()*this->myDiffusion->getV(),
                this->my_robin_optimization->E_t,this->my_robin_optimization->f_t,this->my_robin_optimization->kappa_b,this->myMeanFieldPlan->kappaB);

        fclose(outFile);
    }

    {
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("my_clocks")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }

        if(this->setup_finite_difference_solver)
        {
            //                1  2  3  4 5  6  7  8  9  10 11 12 13
            fprintf(outFile,"%d %d %d %e %e %e %e %e %e %e %e %e %e\n",
                    this->i_mean_field_iteration,this->myDiffusion->get_n_global_size_sol(), this->myDiffusion->get_n_local_size_sol(),
                    this->scft_clock->optimization_watch.read_duration(),this->scft_clock->remeshing_watch.read_duration(),
                    this->scft_clock->remapping_watch.read_duration(),this->scft_clock->k_means_watch.read_duration(),
                    this->scft_clock->ls_reinitialyzation_watch.read_duration(),this->scft_clock->setup_watch.read_duration(),
                    this->myDiffusion->evolve_probability_equation_watch.read_duration(),
                    this->myDiffusion->preconditioner_algo_watch.read_duration(),this->myDiffusion->convergence_test_watch_total_time,
                    this->myDiffusion->stress_tensor_watch.read_duration());
        }
        else
        {
            //                1  2  3  4 5  6  7  8  9  10 11 12 13
            fprintf(outFile,"%d %d %d %e %e %e %e %e %e %e %e %e %e %e\n",
                    this->i_mean_field_iteration,this->myDiffusion->get_n_global_size_sol(), this->myDiffusion->get_n_local_size_sol(),
                    this->scft_clock->optimization_watch.read_duration(),this->scft_clock->remeshing_watch.read_duration(),
                    this->scft_clock->remapping_watch.read_duration(),this->scft_clock->k_means_watch.read_duration(),
                    this->scft_clock->ls_reinitialyzation_watch.read_duration(),this->scft_clock->setup_watch.read_duration(),
                    this->myDiffusion->evolve_probability_equation_watch.read_duration(),
                    this->myDiffusion->interpolation_on_uniform_watch.read_duration(),this->myDiffusion->fftw_scaterring_watch_total_time,
                    this->myDiffusion->fftw_setup_watch.read_duration(),this->myDiffusion->stress_tensor_watch.read_duration());

        }

        fclose(outFile);
    }

    if(this->advance_fields_scft_advance_mask_level_set)
    {

        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("mask_optimization")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %d %d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",this->i_mean_field_iteration,
                this->myDiffusion->get_n_global_size_sol(), this->myDiffusion->get_n_local_size_sol(),
                this->energy[this->i_mean_field_iteration],this->pressure_force[this->i_mean_field_iteration],
                this->exchange_force[this->i_mean_field_iteration],this->order_Ratio[this->i_mean_field_iteration],
                this->rho0Average[this->i_mean_field_iteration],
                this->rhoAAverage[this->i_mean_field_iteration],
                this->rhoBAverage[this->i_mean_field_iteration],this->myDiffusion->get_phi_bar(),
                this->myDiffusion->getQForward(),this->myDiffusion->getQBackward(),this->myDiffusion->getV(),this->volume_phi_seed,
                this->myDiffusion->getE_logQ(),this->myDiffusion->getE_w(),this->myDiffusion->get_e_compressible(),
                this->myDiffusion->get_e_wall_m(),this->myDiffusion->get_e_wall_p());

        fclose(outFile);
    }


    {
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("prediction error")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %f %f %f\n",this->i_mean_field_iteration,this->de_predicted_total,this->de_in_fact_total,this->de_prediction_error);

        fclose(outFile);
    }



    {
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("energies_evolution")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %f %f %f %f %f %f \n",this->i_mean_field_iteration,
                this->energy[this->i_mean_field_iteration],
                this->myDiffusion->getE_logQ(),
                this->myDiffusion->getE_w(),
                this->myDiffusion->get_e_compressible(),
                this->myDiffusion->get_e_wall_p(),
                this->myDiffusion->get_e_wall_m());

        fclose(outFile);
    }

}


int MeanField::printShapeOptimizationEvolution()
{
    std::cout<<" start to write shape  evolution "<<std::endl;


    if(this->advance_fields_level_set || this->advance_fields_scft_advance_mask_level_set)
    {

        double xc,yc,zc;

        this->compute_center_of_mass(xc,yc,zc);

        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("shape_optimization")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %d %d %e %e %e %e %e %e %e %f %f %f\n",
                this->i_mean_field_iteration,
                this->myDiffusion->get_n_global_size_sol(),
                this->myDiffusion->get_n_local_size_sol(),
                this->length_series[this->i_mean_field_iteration],
                this->myDiffusion->getQForward(),this->myDiffusion->getQBackward(),
                this->volume_phi_seed,
                this->surface_phi_seed,
                this->surface_phi_seed*this->effective_radius/this->volume_phi_seed/3.00,
                this->myDiffusion->getQForward()*this->myDiffusion->getV(),xc,yc,zc);
        fclose(outFile);
    }
    else
    {
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("shape_optimization")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %d %d %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
                this->i_mean_field_iteration,
                this->myDiffusion->get_n_global_size_sol(),
                this->myDiffusion->get_n_local_size_sol(),
                this->length_series[this->i_mean_field_iteration],
                this->myDiffusion->getQForward(),this->myDiffusion->getQBackward(),
                this->domain_volume,
                this->domain_surface,
                this->domain_surface*this->effective_radius/this->domain_volume/3.00,
                this->myDiffusion->getQForward()*this->myDiffusion->getV(),
                this->wp_average_on_contour[this->mask_counter],
                this->wp_variance_on_contour[this->mask_counter],
                this->myDiffusion->wp_average_on_contour,
                this->myDiffusion->wm_average_on_contour,
                this->myDiffusion->wp_standard_deviation_on_contour,
                this->myDiffusion->wm_standard_deviation_on_contour);

        fclose(outFile);
    }

    std::cout<<" finish to write shape scft evolution "<<std::endl;

}

int MeanField::printSCFTShapeOptimizationEvolution()
{
    if(this->advance_fields_scft_advance_mask_level_set && this->i_mean_field_iteration>0)
    {

        std::cout<<" start to write shape scft evolution "<<std::endl;

        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("shape_optimization_scft")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->i_mean_field_iteration==1)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %d %d %f %f %f %f %f %f\n",
                this->i_mean_field_iteration,
                this->myDiffusion->get_n_global_size_sol(),
                this->myDiffusion->get_n_local_size_sol(),
                this->myDiffusion->get_wp_average(),
                this->myDiffusion->get_wp_average_at_saddle_point(),
                this->de_predicted_from_level_set_change_current_step_after[MAX(this->mask_counter-1,0)],
                this->prediction_error_level_set[MAX(this->mask_counter-1,0)],
                this->shape_force[MAX(this->mask_counter-1,0)],
                this->de_mask[MAX(this->mask_counter-1,0)]
                );
        fclose(outFile);


        std::cout<<" finish to write shape scft evolution "<<std::endl;

    }



}

int MeanField::printLevelSetEvolution()
{
    if(this->advance_fields_scft_advance_mask_level_set && this->i_mean_field_iteration>0)
    {

        std::cout<<" start to write level set scft evolution "<<std::endl;

        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("level_set_scft")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->mask_counter==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %d %d %f %f %f %f %f\n",
                this->mask_counter,
                this->myDiffusion->get_n_global_size_sol(),
                this->myDiffusion->get_n_local_size_sol(),
                this->de_predicted_from_level_set_change_current_step_before_lagrange_t,
                this->de_predicted_from_level_set_change_current_step_after_lagrange_t,
                this->de_predicted_from_level_set_change_current_step_after_cfl_t,
                this->de_predicted_from_level_set_change_current_step_after_reinitialyzation_t,
                this->de_predicted_from_level_set_change_current_step_after_t
                );
        fclose(outFile);


        std::cout<<" finish to write level set scft evolution "<<std::endl;

    }

    if(this->advance_fields_scft_advance_mask_level_set && this->i_mean_field_iteration>0)
    {

        std::cout<<" start to write level set scft evolution "<<std::endl;

        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("dphi_scft")<<"_"<<myRank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        if(this->mask_counter==0)
        {
            outFile=fopen(mystr2Debug.c_str(),"w");
        }
        else
        {
            outFile=fopen(mystr2Debug.c_str(),"a");
        }


        fprintf(outFile,"%d %e %e %e %e %e %e %e %e\n",
                this->mask_counter,
                this->dphi_step_1[this->mask_counter],
                this->dphi_step_2[this->mask_counter],
                this->dphi_step_3[this->mask_counter],
                this->dphi_step_4[this->mask_counter],
                this->dphi_step_5[this->mask_counter],
                this->dphi_step_0_3[this->mask_counter],
                this->dphi_step_0_4[this->mask_counter],
                this->dphi_step_0_5[this->mask_counter]);
        fclose(outFile);


        std::cout<<" finish to write level set scft evolution "<<std::endl;

    }

}


int MeanField::printStressTensorEvolution()
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("stress_tensor_evolution")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    if(this->i_mean_field_iteration==0)
    {
        outFile=fopen(mystr2Debug.c_str(),"w");
    }
    else
    {
        outFile=fopen(mystr2Debug.c_str(),"a");
    }


    fprintf(outFile,"%d %d %d %f %f %f %f %f %f %f\n",this->i_mean_field_iteration,  this->myDiffusion->get_n_global_size_sol(), this->myDiffusion->get_n_local_size_sol(),
            this->Sxx,this->Syy,this->Szz,this->Sxy,this->Sxz,this->Syz,this->myDiffusion->getV());

    fclose(outFile);
}

int MeanField::clean_mean_field_step()
{
    std::cout<<" start to clean mean field step from mean field"<<std::endl;

    if(this->remesh_my_forest)
    {
        //VecDestroy(this->wremesh);
        // VecDestroy(this->Icell);
        // VecDestroy(this->Ibin_m);
        //VecDestroy(this->Ibin_p);
        if(!this->my_meshing_strategy==MeanField::one_level_set)
        {
            this->ierr=VecDestroy(this->Ibin_nodes_m); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(this->Ibin_nodes_p); CHKERRXX(this->ierr);
        }
        else
            this->ierr=VecDestroy(this->Ibin_nodes); CHKERRXX(this->ierr);
    }
    //this->myDiffusion->clean_mean_field_step();
    std::cout<<" finished to clean mean field step from mean field"<<std::endl;
}

int MeanField::swap_p4ests()
{

}

void MeanField::extend_advect_shape_domain()
{

    std::cout<<this->i_mean_field_iteration<<" "<<this->extension_advection_counter<<" "
            <<this->mpi->mpirank<<" Started Extension Advection Algo "<<std::endl;


    // Here we change the mask by an advection velocity
    if(!this->change_manually_level_set)
    {
        // Duplicate and copy required values for computing the predicted energy change
        // due to the level set change. These values will be destroyed just after.

        int n_games=1;

        for(int i_game=0; i_game<n_games; i_game++)
        {
            //   this->ierr=VecDuplicate(this->phi,&this->polymer_shape_stored); CHKERRXX(this->ierr);
            this->ierr=VecDuplicate(this->phi,&this->wm_stored); CHKERRXX(this->ierr);
            this->ierr=VecDuplicate(this->phi,&this->wp_stored); CHKERRXX(this->ierr);

            // this->ierr=VecCopy(this->polymer_mask,this->polymer_shape_stored); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->myDiffusion->wm,this->wm_stored); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->myDiffusion->wp,this->wp_stored); CHKERRXX(this->ierr);

            //        this->ierr=VecGhostUpdateBegin(this->polymer_shape_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            //        this->ierr=VecGhostUpdateEnd(this->polymer_shape_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

            this->ierr=VecGhostUpdateBegin(this->wp_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->wp_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

            this->ierr=VecGhostUpdateBegin(this->wm_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->wm_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


            // Start Velocity Computation: Either a virtual numerical velocity or the result
            //  of a physical computation.
            //  Either way by the end of the velocity computation we should stay only with the velocity
            //  Memory Cost:
            // (1) construction of polymer mask velocity: Number of nodes x sizeof(PetscScalar)
            //this->compute_advection_velocity();
            // End of Velcocity Computation
            this->extension_advection_algo_euler_or_gudonov();



            //      this->ierr=VecDestroy(this->polymer_shape_stored); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(this->wp_stored); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(this->wm_stored); CHKERRXX(this->ierr);
        }
    }

    // Here we change the mask manually in a direct way
    if(this->change_manually_level_set)
    {
        this->change_level_sets();
    }

    this->extension_advection_counter++;

}

void MeanField::write2LogEnergyChanges()
{
    if(this->i_mean_field_iteration>0)
    {
        std::cout<<" E/V previous "<<this->energy[this->i_mean_field_iteration-1]
                <<" E/V now "<<this->energy[this->i_mean_field_iteration]
               <<" DE/V "<<this->energy[this->i_mean_field_iteration]-this->energy[this->i_mean_field_iteration-1]
                <<" Predicted from scft evolution "<<this->myDiffusion->analytic_energy_difference_predicted_current_time_step
               <<" Predicted from level set evolution "<<this->de_predicted_from_level_set_change_current_step
              <<" Total Prediction  "<<this->de_predicted_from_level_set_change_current_step+this->myDiffusion->analytic_energy_difference_predicted_current_time_step
             <<std::endl;


        if(this->extend_advect && mod(this->i_mean_field_iteration ,this->extension_advection_period)==0
                && this->i_mean_field_iteration>=this->extension_advection_period
                &&   this->i_mean_field_iteration>=this->start_remesh_i
                &&   this->i_mean_field_iteration<=this->stop_remesh_i)
        {


            this->my_delta_H->dEPredicted= this->de_predicted_from_level_set_change_current_step+this->myDiffusion->analytic_energy_difference_predicted_current_time_step;
            this->my_delta_H->dEComputed=this->energy[this->i_mean_field_iteration]-this->energy[this->i_mean_field_iteration-1];
            this->my_delta_H->dEwComputed=(this->energy_w[this->i_mean_field_iteration]-this->energy_w[this->i_mean_field_iteration-1]);
            this->my_delta_H->dlnQComputed=this->energy_logQ[this->i_mean_field_iteration]-this->energy_logQ[this->i_mean_field_iteration-1];
            this->my_delta_H->Q_previous=exp(-this->energy_logQ[this->i_mean_field_iteration-1]);
            this->my_delta_H->Q_now=exp(-this->energy_logQ[this->i_mean_field_iteration]);
            this->my_delta_H->dQComputed=this->my_delta_H->Q_now-this->my_delta_H->Q_previous;
            this->my_delta_H->dQdPhiSimpleTerm=-this->my_delta_H->Q_previous*this->my_delta_H->dlnQdphi_simple_term;
            this->my_delta_H->dQdPhiStressTerm=-this->my_delta_H->Q_previous*this->my_delta_H->dlnQdphi_stress_term;
            this->my_delta_H->dQ_q_n=(this->my_delta_H->dQdPhiSimpleTerm)/(this->my_delta_H->dQComputed);
            this->my_delta_H->dQ_stress_n=(this->my_delta_H->dQdPhiStressTerm)/(this->my_delta_H->dQComputed);
            this->my_delta_H->dEwError=(this->my_delta_H->dEwPredicted/this->my_delta_H->dEwComputed);
            this->my_delta_H->dlnQError=((this->my_delta_H->dlnQPredicted)/(this->my_delta_H->dlnQComputed));


            this->my_delta_H->dQ_joker=
                    (this->my_delta_H->dQdPhiSimpleTerm-0.5*this->my_delta_H->dQdPhiStressTerm)/this->my_delta_H->dQComputed;

            std::cout<<" dQdPhiSimpleTerm "<<this->my_delta_H->dQdPhiSimpleTerm<<std::endl
                    <<" dQdPhiStressTerm "<<this->my_delta_H->dQdPhiStressTerm<<std::endl;


            std::cout
                    <<" osher_and_santosa_term  "<<this->osher_and_santosa_term<<std::endl
                   <<" naive_term  "<<this->naive_term<<std::endl
                  <<" stress_term "<<this->stress_term<<std::endl
                 <<" simple_term_wm "<<this->simple_term_wm<<std::endl
                <<" simple_term_wp "<<this->simple_term_wp<<std::endl;

            std::cout
                    <<" correction_term_for_uncounstrained_volume log Q "<<
                      this->correction_term_for_uncounstrained_volume_log_Q<<std::endl;
            std::cout
                    <<" correction_term_for_uncounstrained_volume Ew "<<
                      this->correction_term_for_uncounstrained_volume_E_w<<std::endl;


            std::cout<<" E_w/V previous "<<this->energy_w[this->i_mean_field_iteration-1]
                    <<" E_w/V now "<<this->energy_w[this->i_mean_field_iteration]
                   <<" DE_w/V "<< this->my_delta_H->dEwComputed
                  <<" Predicted from level set evolution wm"<<this->my_delta_H->dEwmdphi<<std::endl
                 <<" Predicted from level set evolution wp"<<this->my_delta_H->dEwpdphi<<std::endl
                <<" Predicted from level set evolution "<<this->my_delta_H->dEwdphi<<std::endl;



            std::cout<<" E_logQ/V previous "<<this->energy_logQ[this->i_mean_field_iteration-1]
                    <<" E_logQ/V now "<<this->energy_logQ[this->i_mean_field_iteration]
                   <<" DE_logQ/V "<<this->my_delta_H->dlnQComputed
                  <<" Predicted from level set evolution "<<
                    this->naive_term+
                    this->stress_term+
                    this->correction_term_for_uncounstrained_volume_log_Q<<std::endl;



            std::cout<<" check matching terms "<<std::endl
                    <<"   this->my_delta_H->dQ_q_n "<<this->my_delta_H->dQ_q_n<<std::endl
                   <<"   this->my_delta_H->dQ_stress_n "<<this->my_delta_H->dQ_stress_n<<std::endl;




            int myRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
            std::stringstream oss2Debug;
            std::string mystr2Debug;
            oss2Debug << this->convert2FullPath("shape_derivative")<<"_"<<myRank<<".txt";
            mystr2Debug=oss2Debug.str();


            ofstream Data_Input_File (mystr2Debug.c_str(), ios::out | ios::app );         //Opening file to print info to


            Data_Input_File
                    <<" dQdPhiSimpleTerm "<<this->my_delta_H->dQdPhiSimpleTerm<<std::endl
                   <<" dQdPhiStressTerm "<<this->my_delta_H->dQdPhiStressTerm<<std::endl;


            Data_Input_File<<" E/V previous "<<this->energy[this->i_mean_field_iteration-1]
                          <<" E/V now "<<this->energy[this->i_mean_field_iteration]
                         <<" DE/V "<<this->my_delta_H->dEComputed
                        <<" Predicted from scft evolution "<<this->my_delta_H->dHdw
                       <<" Predicted from level set evolution "<<this->my_delta_H->dHdphi
                      <<" Total Prediction  "<<this->my_delta_H->dEPredicted
                     <<" Predicted/Computed "<<this->my_delta_H->dEPredicted/this->my_delta_H->dEComputed<<    std::endl;

            Data_Input_File <<" osher_and_santosa_term  "<<this->osher_and_santosa_term<<std::endl;
            Data_Input_File<<" naive_term  "<<this->naive_term<<std::endl;
            Data_Input_File    <<" stress_term "<<this->stress_term<<std::endl;
            Data_Input_File  <<" simple_term wm"<<this->simple_term_wm<<std::endl;
            Data_Input_File  <<" simple_term wp"<<this->simple_term_wp<<std::endl;
            Data_Input_File  <<" simple_term"<<this->simple_term_wm+this->simple_term_wp<<std::endl;

            Data_Input_File
                    <<" correction_term_for_uncounstrained_volume log Q "<<
                      this->correction_term_for_uncounstrained_volume_log_Q<<std::endl;
            Data_Input_File
                    <<" correction_term_for_uncounstrained_volume Ew "<<
                      this->correction_term_for_uncounstrained_volume_E_w<<std::endl;


            Data_Input_File<<" E_w/V previous "<<this->energy_w[this->i_mean_field_iteration-1]
                          <<" E_w/V now "<<this->energy_w[this->i_mean_field_iteration]
                         <<" DE_w/V "<<this->my_delta_H->dEwComputed
                        <<" Predicted from scft evolution "<<this->my_delta_H->dEwdw
                       <<" Predicted from level set evolution wm"<<this->my_delta_H->dEwmdphi
                      <<" Predicted from level set evolution wp"<<this->my_delta_H->dEwpdphi
                     <<" Predicted from level set evolution "<<this->my_delta_H->dEwdphi
                    <<" Total Prediction  "<<this->my_delta_H->dEwPredicted
                   <<" Predicted/Computed "<<this->my_delta_H->dEwError<<    std::endl;



            Data_Input_File<<" E_logQ/V previous "<<this->energy_logQ[this->i_mean_field_iteration-1]
                          <<" E_logQ/V now "<<this->energy_logQ[this->i_mean_field_iteration]
                         <<" DE_logQ/V "<<this->my_delta_H->dlnQComputed
                        <<" Predicted from scft evolution "<<this->my_delta_H->dlnQdw
                       <<" Predicted from level set evolution "<<this->my_delta_H->dlnQdphi
                      <<" Total Prediction  "<<this->my_delta_H->dlnQPredicted
                     <<" Predicted/Computed "<<this->my_delta_H->dlnQError<<    std::endl;


            Data_Input_File <<" dQ "<<this->my_delta_H->dQComputed<<std::endl
                           <<" dQdPhiSimpleTerm "<<this->my_delta_H->dQdPhiSimpleTerm<<std::endl
                          <<" dQdPhiStressTerm "<<this->my_delta_H->dQdPhiStressTerm<<std::endl
                         <<" dQPredicted/dQComputed "<<(this->my_delta_H->dQdPhiStressTerm+ this->my_delta_H->dQdPhiSimpleTerm)/this->my_delta_H->dQComputed<<std::endl
                        <<" dQdPhiSimpleTerm "<<this->my_delta_H->dQdPhiSimpleTerm/this->my_delta_H->dQComputed<<std::endl
                       <<" dQdPhiStressTerm "<<this->my_delta_H->dQdPhiStressTerm/this->my_delta_H->dQComputed<<std::endl
                      <<" dQJoker "<<this->my_delta_H->dQ_joker<<std::endl;




            Data_Input_File<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl
                          <<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;

            Data_Input_File.close();

        }




        double total_prediction_error=0;

        this->de_predicted_total=(this->de_predicted_from_level_set_change_current_step+this->myDiffusion->analytic_energy_difference_predicted_current_time_step);
        this->de_in_fact_total=(this->energy[this->i_mean_field_iteration]-this->energy[this->i_mean_field_iteration-1]);
        this->de_prediction_error=
                100*ABS(this->de_predicted_total
                        -this->de_in_fact_total)/
                ABS(this->de_in_fact_total);

        std::cout<<" energy prediction error "<<this->de_prediction_error<<"[%]"<<std::endl;


        if(this->advance_fields_level_set|| this->advance_fields_scft_advance_mask_level_set)
        {
            std::cout<<" ElogQ/V previous "<<this->energy_logQ[this->i_mean_field_iteration-1]
                    <<" ElogQ/V now "<<this->energy_logQ[this->i_mean_field_iteration]
                   <<" DElogQ/V "<<this->energy_logQ[this->i_mean_field_iteration]-this->energy_logQ[this->i_mean_field_iteration-1]
                    <<" Predicted from scft evolution "<<this->myDiffusion->analytic_energy_difference_predicted_current_time_step
                   <<" Predicted from level set evolution "<<this->de_predicted_from_level_set_change_current_step
                  <<std::endl;

            std::cout<<" Ew/V previous "<<this->energy_w[this->i_mean_field_iteration-1]
                    <<" Ew/V now "<<this->energy_w[this->i_mean_field_iteration]
                   <<" DEw/V "<<this->energy_w[this->i_mean_field_iteration]-this->energy_w[this->i_mean_field_iteration-1]
                    <<" Predicted from scft evolution "<<this->myDiffusion->analytic_energy_difference_predicted_current_time_step
                   <<" Predicted from level set evolution "<<this->de_predicted_from_level_set_change_current_step
                  <<std::endl;


            double v_source=this->compute_volume_source();
            std::cout<<" volume source "<<v_source<<" volume loss " <<this->source_volume_initial-v_source  << std::endl;
        }
    }
}

void MeanField::create_diffuse_mask()
{
    if(this->advance_fields_scft_advance_mask_level_set && this->i_mean_field_iteration ==0 )
    {
        switch(this->my_seed_strategy)
        {
        case MeanField::sphere_seed:
            this->set_initial_wall_from_level_set();
            break;
        case MeanField::bcc_seed:
            this->set_initial_seed_from_level_set_bcc();
            break;
        case MeanField::helix_seed:
            this->set_initial_seed_from_level_set_helix();
            break;
        case MeanField::from_wp:
            this->set_initial_seed_from_level_set_wp();
            break;
        case MeanField::terrace:
        {
            this->set_initial_wall_from_level_set_terracing();

            break;
        }

        case MeanField::rectangle:
        {
            this->set_initial_wall_from_level_set_rectangle();

            break;
        }
        case MeanField::l_shape_seed:
        {
            this->create_polymer_mask_l_shape(this->a_ellipse,this->b_ellipse,this->c_ellipse);
            this->ierr=VecDuplicate(this->polymer_mask,&this->phi_seed); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->polymer_mask,this->phi_seed); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed);
            this->ierr=VecDuplicate(this->polymer_mask,&this->phi_seed_initial); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->polymer_mask,this->phi_seed_initial); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_initial);
            this->ierr=VecDestroy(this->polymer_mask); CHKERRXX(this->ierr);
            this->interpolate_and_print_vec_to_uniform_grid(&this->phi_seed_initial,"ps",false);
            this->source_volume_initial=   this->compute_volume_source();
            this->volume_phi_seed=this->source_volume_initial;

            break;
        }
        case MeanField::text_file_seed:
        {
            this->ierr=VecDuplicate(this->phi,&this->phi_seed); CHKERRXX(this->ierr);
            this->get_field_from_text_file(&this->phi_seed,this->text_file_mask_str);
            this->scatter_petsc_vector(&this->phi_seed);
            this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_initial); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_initial); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_initial);
            this->interpolate_and_print_vec_to_uniform_grid(&this->phi_seed_initial,"ps",false);
            this->source_volume_initial=   this->compute_volume_source();
            this->volume_phi_seed=this->source_volume_initial;
            break;
        }
        case MeanField::text_file_seed_from_fine_to_coarse:
        {
            this->ierr=VecDuplicate(this->phi,&this->phi_seed); CHKERRXX(this->ierr);
            int file_level=6;
            this->get_coarse_field_from_fine_text_file(&this->phi_seed,this->text_file_mask_str,file_level);
            this->scatter_petsc_vector(&this->phi_seed);
            this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_initial); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_initial); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_initial);
            this->interpolate_and_print_vec_to_uniform_grid(&this->phi_seed_initial,"ps",false);
            this->source_volume_initial=   this->compute_volume_source();
            this->volume_phi_seed=this->source_volume_initial;
            break;


        }
        case MeanField::ThreeD_from_TwoD_text_file_seed:
        {
            this->ierr=VecDuplicate(this->phi,&this->phi_seed); CHKERRXX(this->ierr);
            this->get_3Dfield_from_2Dtext_file(&this->phi_seed,this->text_file_mask_str,this->polymer_mask_radius_for_initial_wall);
            this->scatter_petsc_vector(&this->phi_seed);
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_crude_from_text",PETSC_TRUE);
            my_p4est_level_set PLS(this->node_neighbors);
            double l1_error,l2_error;
            double ls_tol=0.001;
            PLS.reinitialize_2nd_order_with_tolerance(this->phi_seed,ls_tol,l1_error,l2_error);

            std::cout<<" l1_error "<<" l2_error "<<l1_error<<" "<<l2_error<<std::endl;
            double epsilon_for_level_set_perturbation=0.000018;
            PLS.perturb_level_set_function(&this->phi_seed,epsilon_for_level_set_perturbation);

            this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


            this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_initial); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_initial); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_initial);
            this->interpolate_and_print_vec_to_uniform_grid(&this->phi_seed_initial,"ps",false);
            this->source_volume_initial=   this->compute_volume_source();
            this->volume_phi_seed=this->source_volume_initial;
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed_initial,"phi_seed_reinitialyzed_from_text",PETSC_TRUE);
            if(this->myDiffusion->spatial_xhi_w)
                this->create_spatial_xhi_wall();




            break;
        }

        case MeanField::ThreeD_from_TwoD_text_file_seed_from_fine_to_coarse:
        {
            this->ierr=VecDuplicate(this->phi,&this->phi_seed); CHKERRXX(this->ierr);

            int file_level=6;
            this->get_3D_coarse_field_from_2D_fine_text_file(&this->phi_seed,this->text_file_mask_str,this->polymer_mask_radius_for_initial_wall,file_level);
            this->scatter_petsc_vector(&this->phi_seed);
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_crude_from_text",PETSC_TRUE);
            my_p4est_level_set PLS(this->node_neighbors);
            double l1_error,l2_error;
            double ls_tol=0.001;
            PLS.reinitialize_2nd_order_with_tolerance(this->phi_seed,ls_tol,l1_error,l2_error);

            std::cout<<" l1_error "<<" l2_error "<<l1_error<<" "<<l2_error<<std::endl;
            double epsilon_for_level_set_perturbation=0.000018;
            PLS.perturb_level_set_function(&this->phi_seed,epsilon_for_level_set_perturbation);

            this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


            this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_initial); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_initial); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_initial);
            this->interpolate_and_print_vec_to_uniform_grid(&this->phi_seed_initial,"ps",false);
            this->source_volume_initial=   this->compute_volume_source();
            this->volume_phi_seed=this->source_volume_initial;
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed_initial,"phi_seed_reinitialyzed_from_text",PETSC_TRUE);
            if(this->myDiffusion->spatial_xhi_w)
                this->create_spatial_xhi_wall();


            break;
        }



        }
        PetscBool destroy_phi_wall=PETSC_FALSE,compute_on_remeshed_p4est=PETSC_FALSE;

        this->set_advected_wall_from_level_set(destroy_phi_wall,compute_on_remeshed_p4est);
        if(this->terracing)
            this->correct_advected_wall_from_level_set_terrace(destroy_phi_wall,compute_on_remeshed_p4est);

        if(!this->inverse_design_litography)
            this->generate_masked_fields();
        //                if(!this->periodic_xyz\approx)
        //                    this->myDiffusion->filter_irregular_potential();

    }
}

int MeanField::create_spatial_xhi_wall()
{
    this->ierr=VecDuplicate(this->phi_seed,&this->myDiffusion->xhi_w_x); CHKERRXX(this->ierr);
    PetscScalar *phi_seed_initial_local;
    PetscScalar *spatial_xhi_w_local;

    this->ierr=VecGetArray(this->myDiffusion->xhi_w_x,&spatial_xhi_w_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->phi_seed,&phi_seed_initial_local); CHKERRXX(this->ierr);




    bool isInsideASphere;
    double tree_xmin , tree_ymin , tree_zmin ;

    double x , y , z ;
    double ax_in=1.00,by_in=1.00,cz_in=1;

    ax_in=ax_in*ax_in*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;
    by_in=by_in*by_in*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;
    cz_in=cz_in*cz_in*this->my_design_plan->r_spot*this->my_design_plan->r_spot/this->my_design_plan->cut_r_spot;





    for(int i=0;i<this->my_design_plan->n2Check;i++)
    {std::cout<<this->my_design_plan->xc[i]<<" "<<this->my_design_plan->yc[i]<<" "<<this->my_design_plan->zc[i]<<std::endl;}



    for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
      * nonlocal nodes. Number of initial nonlocal nodes is given by
      * nodes->offset_owned_indeps
      */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];


        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif
        x =  node_x_fr_i(node) + tree_xmin;
        y =  node_y_fr_j(node) + tree_ymin;
#ifdef P4_TO_P8
        z = node_z_fr_k(node) + tree_zmin;
#endif
        isInsideASphere=false;
        for(int ii=0;ii<this->my_design_plan->n2Check;ii++)
        {
            if( pow(x- this->my_design_plan->xc[ii],2)/ax_in+pow(y-this->my_design_plan->yc[ii],2)/by_in
        #ifdef P4_TO_P8
                    //                    +pow(z-my_litography->zc[ii],2)/cz_in
        #endif
                    <1.00)
            {
                isInsideASphere=true;
                ii=this->my_design_plan->n2Check;
            }
        }



        if(isInsideASphere)
        {
            spatial_xhi_w_local[i]=-1.00;
        }
        else
        {
            spatial_xhi_w_local[i]=+1.00;
        }
    }


    this->ierr=VecRestoreArray(this->myDiffusion->xhi_w_x,&spatial_xhi_w_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->phi_seed,&phi_seed_initial_local); CHKERRXX(this->ierr);

    this->scatter_petsc_vector(&this->myDiffusion->xhi_w_x);

    this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->xhi_w_x,"xhi_w_x",PETSC_TRUE);

    return 0;
}

void MeanField::advect_diffuse_mask()
{
    if(this->advance_fields_scft_advance_mask_level_set && this->i_mean_field_iteration>0
            &&   mod(this->i_mean_field_iteration,this->mask_period)==0)
    {
        if(this->i_mean_field_iteration>=this->start_remesh_i  && this->i_mean_field_iteration<=this->stop_remesh_i)
        {
            int n_games=1;
            for(int i_game=0;i_game<n_games;i_game++)
            {
                this->change_internal_interface();

                this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_before_remesh",PETSC_TRUE);

                PetscBool destroy_phi_wall=PETSC_TRUE,compute_on_remeshed_p4est=PETSC_FALSE;

                this->set_advected_wall_from_level_set(destroy_phi_wall,compute_on_remeshed_p4est);

                if(this->terracing)
                    this->correct_advected_wall_from_level_set_terrace(destroy_phi_wall,compute_on_remeshed_p4est);

                if(i_game<n_games-1)
                {
                    this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
                }
            }
        }
        else
        {
            this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
            this->ierr = VecGhostUpdateBegin(this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        }
    }
}

void MeanField::create_internal_interface()
{
    if(this->advance_fields_level_set && this->i_mean_field_iteration ==0 )
    {
        switch(this->my_seed_strategy)
        {
        case MeanField::sphere_seed:
            this->set_initial_seed_from_level_set();
            break;
        case MeanField::bcc_seed:
            this->set_initial_seed_from_level_set_bcc();
            break;
        case MeanField::helix_seed:
            this->set_initial_seed_from_level_set_helix();
            break;
        case MeanField::from_wp:
            this->set_initial_seed_from_level_set_wp();
            break;

        }
        this->set_advected_fields_from_level_set();
        //                if(!this->periodic_xyz)
        //                    this->myDiffusion->filter_irregular_potential();

    }
}

void MeanField::optimize_internal_interface()
{
    if(this->advance_fields_level_set && this->i_mean_field_iteration>0 )
    {
        int n_games=1;
        for(int i_game=0;i_game<n_games;i_game++)
        {
            bool option_advection=false;
            bool option_change=!option_advection;
            if(option_advection)
            {
                this->compute_advection_velocity();
                this->advect_internal_interface();
            }
            if(option_change)
                this->change_internal_interface();

            this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_before_remesh",PETSC_TRUE);
            //this->set_initial_fields_from_level_set();
            this->set_advected_fields_from_level_set();
            //this->myDiffusion->filter_irregular_potential();
            if(i_game<n_games-1)
            {
                this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
            }
        }
    }
}

void MeanField::write_mean_field_to_txt_and_vtk()
{
    int i_games=1;
    for (int i_game=0;i_game<i_games;i_game++)
    {

        //this->interpolate_and_print_vec_to_uniform_grid(&this->Ibin_nodes_m,"phi_m");
        //this->interpolate_and_print_vec_to_uniform_grid(&this->Ibin_nodes_p,"phi_p");

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->fm_stored,"fm_stored",PETSC_FALSE);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->fp_stored,"fp_stored",PETSC_FALSE);

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->wp,"rosh_hashana_wp",PETSC_FALSE);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->wm,"rosh_hashana_wm",PETSC_FALSE);

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->rhoA,"rho_a",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->rhoB,"rho_b",PETSC_FALSE);

        if(!this->myMeanFieldPlan->periodic_xyz)
        {
            Vec wp_extended;
            this->ierr=VecDuplicate(this->myDiffusion->wp,&wp_extended); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->myDiffusion->wp,wp_extended); CHKERRXX(this->ierr);
            this->extend_petsc_vector(&this->polymer_mask,&wp_extended,pow(2,this->max_level)); CHKERRXX(this->ierr);
            this->interpolate_and_print_vec_to_uniform_grid(&wp_extended,"wp_uniform",this->myMeanFieldPlan->compressed_io);
            this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->wp,"pure_wp_uniform",this->myMeanFieldPlan->compressed_io);
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&wp_extended,"wp_uniform",PETSC_FALSE);
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->polymer_mask,"mask_uniform",PETSC_FALSE);
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->last_q_stored,"last_q",PETSC_FALSE);

            this->interpolate_and_print_vec_to_uniform_grid(&this->polymer_mask,"mask_uniform",this->myMeanFieldPlan->compressed_io);
            this->interpolate_and_print_vec_to_uniform_grid(&this->last_q_stored,"q_uniform",this->myMeanFieldPlan->compressed_io);
            this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->wm,"wm",this->myMeanFieldPlan->compressed_io);
            this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->rhoA,"rho_A",false);
            this->ierr=VecDestroy(wp_extended); CHKERRXX(this->ierr);
        }
        else
        {
            this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->wp,"wp_uniform",this->myMeanFieldPlan->compressed_io);
            if(this->advance_fields_scft_advance_mask_level_set)
                this->interpolate_and_print_vec_to_uniform_grid(&this->phi_seed,"mask_uniform",this->myMeanFieldPlan->compressed_io);
            this->interpolate_and_print_vec_to_uniform_grid(&this->last_q_stored,"q_uniform",this->myMeanFieldPlan->compressed_io);
            this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->wm,"wm",this->myMeanFieldPlan->compressed_io);
            this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->rhoA,"rho_A",false);
            if(this->myDiffusion->get_my_phase()==Diffusion::from_one_fine_text_file_to_coarse
                    ||this->myDiffusion->get_my_phase()==Diffusion::from_two_fine_text_file_to_coarse)
            {
                int file_level_to_print=5;
                this->interpolate_to_fine_and_print_vec_to_uniform_grid(&this->myDiffusion->rhoA,"rho_A",file_level_to_print);
            }
        }
        if(this->advance_fields_scft_advance_mask_level_set)
        {
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"shape",  this->write2VtkShapeOptimization);
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->phi_wall,"phi_wall",PETSC_TRUE);
        }
        if(this->extend_advect)
        {
            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->polymer_mask,"mask_t",PETSC_TRUE);
        }
    }
}

int MeanField::Optimize()
{

    if(this->advance_fields_scft_advance_mask_level_set)
    {
        this->myDiffusion->set_advance_fields_scft_advance_mask_level_set_2_true();
        if(this->terracing)
            this->myDiffusion->setTerraccing2True();
    }
    this->myDiffusion->setupEnergyAndForces();

    this->myDiffusion->set_xhi_values(this->xhi_w,this->xhi_w_a,this->xhi_w_b,this->xhi_w_m,this->xhi_w_p);


    //print the statistical fields into text files
    //this->myDiffusion->printStatisticalFields();



    if(this->inverse_design_litography)
        this->myDiffusion->generate_phase_from_mask(this->my_design_plan);
    std::cout<<"destruct"<<std::endl;
    this->myDiffusion->destruct_initial_data();
    std::cout<<"just_after"<<std::endl;

    //print the statistical files into vtk files
    //    this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wp,"seed_wp",PETSC_TRUE);
    //    this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wm,"seed_wm",PETSC_TRUE);


    //    this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->wp,"seed_wp");
    //    this->interpolate_and_print_vec_to_uniform_grid(&this->myDiffusion->wm,"seed_wm");


    this->i_mean_field_iteration=0;
    std::cout<<"just_before"<<std::endl;

    while(i_mean_field_iteration<this->N_mean_field_iteration)
    {


        this->scft_clock->optimization_watch.start("mean field step time");

        this->de_predicted_from_level_set_change_current_step=0;
        this->osher_and_santosa_term=0;
        this->naive_term=0;
        this->stress_term=0;
        this->simple_term_wm=0;
        this->simple_term_wp=0;
        this->correction_term_for_uncounstrained_volume_log_Q=0;
        this->correction_term_for_uncounstrained_volume_E_w=0;


        if(this->advance_fields_scft_advance_mask_level_set)
        {
            this->de_predicted_from_level_set_change_current_step=0.00;

        }

        if(this->inverse_design_litography)
        {

            this->myDiffusion->set_lambda_plus_and_lambda_minus(this->lambda_plus,this->lambda_minus);
            this->my_design_plan->compute_litography_stage(this->i_mean_field_iteration,this->lambda_0,this->lambda_minus,this->lambda_plus,
                                                           this->start_remesh_i,this->stop_remesh_i,
                                                           this->source_volume_initial,this->VRealTime,
                                                           this->myDiffusion->wp_average_on_contour,
                                                           this->myDiffusion->wp_standard_deviation_on_contour);
            this->myDiffusion->set_lambda_plus_and_lambda_minus(this->lambda_plus,this->lambda_minus);

            std::cout<<"volume phi seed"<<this->volume_phi_seed;//<<std::endl;
            std::cout<<" volume initial "<<this->source_volume_initial<<" ";//<<std::endl;
            std::cout<<this->volume_phi_seed/this->source_volume_initial<<std::endl;

            if(this->i_mean_field_iteration>10 &&(this->my_design_plan->dynamic_f_A)&&
                    (this->my_design_plan->get_my_inverse_litography_stage() ==inverse_litography::move_pressure_and_exchange_field))

            {
                this->f=round(100*this->f_initial*(this->source_volume_initial/this->volume_phi_seed)/this->my_design_plan->cut_r_spot)/100;
                this->myDiffusion->set_f_a((this->f));
                this->my_design_plan->dynamic_f_A=false;
                this->conserve_shape_volume=PETSC_TRUE;
                this->conserve_reaction_source_volume=  this->conserve_shape_volume;
                this->source_volume_initial=this->domain_volume;
            }


        }
        if(this->multi_alpha_cfl)
        {
            if(this->i_mean_field_iteration<200)
                this->alpha_cfl=1.00;
            //this->alpha_cfl=0.00;
            if(this->i_mean_field_iteration>=200 && this->i_mean_field_iteration<300)
                this->alpha_cfl=0.9;
            if(this->i_mean_field_iteration>=300 && this->i_mean_field_iteration<400)
                this->alpha_cfl=1.8;
            if(this->i_mean_field_iteration>=400)
                this->alpha_cfl=2.00;
        }
        this->printNumericalDataEvolution();
        std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
        // The mean field optimization should work the same if the forest is remeshed and the variables are remapped
        // The algo must be totally immune to this forest exchange (not forex)
        if(this->remesh_my_forest &&( mod(this->i_mean_field_iteration,this->remesh_period)==0 ||this->i_mean_field_iteration==0 ) )
        {

            // The mean field optimization should work the same if the potential fields are extended
            // and the mask advected
            if(this->extend_advect && mod(this->i_mean_field_iteration ,this->extension_advection_period)==0
                    && this->i_mean_field_iteration>=this->extension_advection_period
                    &&   this->i_mean_field_iteration>=this->start_remesh_i
                    &&   this->i_mean_field_iteration<=this->stop_remesh_i)
            {
                int extension_games;

                extension_games=1;
                for(int i_extension_game=0;i_extension_game<extension_games;i_extension_game++)
                {
                    this->extend_advect_shape_domain();
                }
            }
            else
            {
                if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->myDiffusion->stress_tensor_computation_period)==0)
                   {
                    this->ierr=VecDestroy(this->snn); CHKERRXX(this->ierr);
                }
            }


            this->optimize_internal_interface();
            this->create_internal_interface();
            this->advect_diffuse_mask();
            this->create_diffuse_mask();




            //-------------Remeshing: from old forest to new forest -----------------//
            //-----------------------Algo-------------------------------------------//
            //---Input: Forest +Evolved Diffusion Potentials From SCFT--------------//
            //---Output: Remeshed Forest+Remapped variables to the new forest-------//



            //-------------Remeshing: from old forest to new forest -----------------//
            //-----------------------Algo-------------------------------------------//
            //---Input: Forest +Diffusion Potentials-------------------------------//
            //---Output: Remeshed Forest+Remapped variables to the new forest-------//

            //(1) Segment  Potentials Kmeans
            //Input:   {wp,wm}(Forest_Old)
            //Output:  {Ibin}(Forest_Old)
            //Algo: kmeans algo: creates a binary representation of the potential with +1/-1 values
            std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;

            switch(this->my_meshing_strategy)
            {
            case MeanField::one_level_set:
            {


                this->scft_clock->k_means_watch.start("k-means");
                this->segmentDiffusionPotentialsWbyNodes(&this->myDiffusion->wm,&this->Ibin_nodes);
                this->scft_clock->k_means_watch.stop();


                //(2) Create  Level Set from Kmeans
                // Input:  {binary_w}(Forest_old)
                // Output: {level_set_w}(Forest_old)
                // Algo: reinitialyze the binary potential to a level set function

                my_p4est_level_set PLS(this->node_neighbors);
                this->scft_clock->ls_reinitialyzation_watch.start("ls reinitialyzation");
                if(this->myMeanFieldPlan->first_order_reinitialyzation)
                {
                    PLS.reinitialize_1st_order(this->Ibin_nodes);
                }
                else
                {
                    PLS.reinitialize_2nd_order(this->Ibin_nodes);
                }
                this->scatter_petsc_vector(&this->Ibin_nodes);
                this->my_p4est_vtk_write_all_periodic_adapter(&this->Ibin_nodes,"LevelSetReinitialyzed",PETSC_FALSE);
                this->scft_clock->ls_reinitialyzation_watch.stop();
                break;
            }

            case MeanField::two_level_set:
            {


                int n_games=1;
                for(int i_game=0;i_game<n_games;i_game++)
                {

                    this->scft_clock->k_means_watch.start("kmeans watch");
                    this->segmentDiffusionPotentialsWbyNodes(&this->myDiffusion->wp,&Ibin_nodes_p);
                    this->segmentDiffusionPotentialsWbyNodes(&this->myDiffusion->wm,&Ibin_nodes_m);
                    this->scft_clock->k_means_watch.stop();

                    //(2) Create  Level Set from Kmeans
                    // Input:  {binary_w}(Forest_old)
                    // Output: {level_set_w}(Forest_old)
                    // Algo: reinitialyze the binary potential to a level set function

                    my_p4est_level_set PLS(this->node_neighbors);
                    this->scft_clock->ls_reinitialyzation_watch.start("ls reinitialyzation");

                    if(this->myMeanFieldPlan->first_order_reinitialyzation)
                    {
                        PLS.reinitialize_1st_order(this->Ibin_nodes_m);
                        PLS.reinitialize_1st_order(this->Ibin_nodes_p);
                    }
                    else
                    {
                        PLS.reinitialize_2nd_order(this->Ibin_nodes_m);
                        PLS.reinitialize_2nd_order(this->Ibin_nodes_p);
                    }
                    this->scatter_petsc_vector(&this->Ibin_nodes_m);
                    this->scatter_petsc_vector(&this->Ibin_nodes_p);
                    this->my_p4est_vtk_write_all_periodic_adapter(&this->Ibin_nodes_m,"LevelSetReinitialyzed_m",PETSC_FALSE);
                    this->my_p4est_vtk_write_all_periodic_adapter(&this->Ibin_nodes_p,"LevelSetReinitialyzed_p",PETSC_FALSE);
                    this->scft_clock->ls_reinitialyzation_watch.stop();
                    if(n_games>1 && i_game!=n_games-1)
                    {
                        VecDestroy(this->Ibin_nodes_m);
                        VecDestroy(this->Ibin_nodes_p);
                    }
                }
                break;
            }

            case MeanField::two_level_set_with_interface:
            {
                int n_games=1;
                for(int i_game=0;i_game<n_games;i_game++)
                {

                    //Vec Ibin_p;
                    //Vec Ibin_m;
                    this->scft_clock->k_means_watch.start("kmeans watch");
                    this->segmentDiffusionPotentialsWbyNodes(&this->myDiffusion->wm,&Ibin_nodes_m);
                    this->segmentDiffusionPotentialsWbyNodes(&this->myDiffusion->wp,&Ibin_nodes_p);
                    this->scft_clock->k_means_watch.stop();

                    //this->segmentDiffusionPotentials();


                    //this->myDiffusion->printDiffusionVector(&this->Ibin_m,"IbinCells_m");
                    //this->myDiffusion->printDiffusionVector(&this->Ibin_p,"IbinCells_p");

                    //(2) Create  Level Set from Kmeans
                    // Input:  {binary_w}(Forest_old)
                    // Output: {level_set_w}(Forest_old)
                    // Algo: reinitialyze the binary potential to a level set function

                    my_p4est_level_set PLS(this->node_neighbors);
                    //this->mapBinaryImageFromCell2NodesW(&Ibin_m,&this->Ibin_nodes_m);
                    //VecDestroy(Ibin_m);
                    //this->mapBinaryImageFromCell2NodesW(&Ibin_p,&this->Ibin_nodes_p);
                    //VecDestroy(Ibin_p);

                    //this->mapBinaryImageFromCell2Nodes();
                    this->scft_clock->ls_reinitialyzation_watch.start("ls reinitialyzation watch");
                    if(this->myMeanFieldPlan->first_order_reinitialyzation)
                    {
                        PLS.reinitialize_1st_order(this->Ibin_nodes_m);
                        PLS.reinitialize_1st_order(this->Ibin_nodes_p);
                    }
                    else
                    {
                        PLS.reinitialize_2nd_order(this->Ibin_nodes_m);
                        PLS.reinitialize_2nd_order(this->Ibin_nodes_p);
                    }

                    this->scatter_petsc_vector(&this->Ibin_nodes_m);
                    this->scatter_petsc_vector(&this->Ibin_nodes_p);
                    this->scft_clock->ls_reinitialyzation_watch.stop();
                    if(n_games>1 && i_game!=n_games-1)
                    {
                        VecDestroy(this->Ibin_nodes_m);
                        VecDestroy(this->Ibin_nodes_p);
                    }
                }
                break;
            }


            default:
                //  throw  (std::cout<<" no valid meshing strategy has been set");
                break;

            }

            for(int k_game=0;k_game<1;k_game++)
            {
                if(this->i_mean_field_iteration==0 && this->myMeanFieldPlan->regenerate_potentials)
                {
                    switch(this->my_meshing_strategy)
                    {
                    case one_level_set:
                    this->regenerate_potentials_from_level_set(&this->Ibin_nodes);
                        break;
                    case two_level_set:
                        this->regenerate_potentials_from_level_set(&this->Ibin_nodes_m);
                        break;
                    case two_level_set_with_interface:
                        this->regenerate_potentials_from_level_set(&this->Ibin_nodes_m);
                        break;
                    }
                    this->myMeanFieldPlan->regenerate_potentials=PETSC_FALSE;
                }


                std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;

                //(3) Create New Forest
                // Input:  {level_set_w,Tree_old} (Forest_old)
                // Output: {Tree_new}(Forest_new)
                // Algo:   create forest from level set/ update forest from level set
                this->scft_clock->remeshing_watch.start("remeshing watch");
                switch(this->my_meshing_strategy)
                {
                case one_level_set:
                {
                    this->remeshForestOfTrees_parallel(this->nx_trees,this->ny_trees,this->nz_trees);
                    if(this->myMeanFieldPlan->periodic_xyz)
                        this->remeshForestOfTreesVisualization_parallel(this->nx_trees,this->ny_trees,this->nz_trees );
                    break;
                }

                case two_level_set:
                {

                    this->remeshForestOfTrees_parallel_two_level_set(this->nx_trees,this->ny_trees,this->nz_trees);
                    if(this->myMeanFieldPlan->periodic_xyz && !this->do_not_write_to_vtk_in_any_case)
                        this->remeshForestOfTreesVisualization_parallel_two_level_set(this->nx_trees,this->ny_trees,this->nz_trees );
                    break;
                }
                case two_level_set_with_interface:
                {
                    if(this->advance_fields_scft_advance_mask_level_set)
                    {
                        this->ierr=VecDuplicate(this->phi_seed,&this->polymer_mask); CHKERRXX(this->ierr);
                        this->ierr=VecCopy(this->phi_seed,this->polymer_mask); CHKERRXX(this->ierr);
                        this->scatter_petsc_vector(&this->polymer_mask); CHKERRXX(this->ierr);
                    }

                    this->remeshForestOfTrees_parallel_two_level_set_with_interface(this->nx_trees,this->ny_trees,this->nz_trees);
                    if(this->myMeanFieldPlan->periodic_xyz)// && this->write_to_vtk)
                        this->remeshForestOfTreesVisualization_parallel_two_level_set_with_interface(this->nx_trees,this->ny_trees,this->nz_trees );


                    break;
                }

                default:
                    throw "";
                    break;
                }
                this->scft_clock->remeshing_watch.stop();



                //(4) Remap all variables to New Forest
                //  Recreate the diffusion variables in terms of the space variable: reallocate memory , resize etc where needed
                // Input: {Forest_new,Diffusion}(Forest_old)
                // Output: {Diffusion}(Forest_old)
                // Algo: interpolation from one tree to another tree+ memory resizing destruction + construction of diffusion variables
                // Destruct Old Forest
                // Assign Forest to New Forest

                std::cout<<" remap variables from old forest to new forest "<<std::endl;


                int n_game=1;
                for(int i_game=0;i_game<n_game;i_game++)
                {
                    this->scft_clock->remapping_watch.start("remap");
                    if(       this->myDiffusion->get_my_casl_diffusion_method()==Diffusion::neuman_backward_euler
                              || this->myDiffusion->get_my_casl_diffusion_method()==Diffusion::neuman_crank_nicholson
                              || this->myDiffusion->get_my_casl_diffusion_method()==Diffusion::dirichlet_crank_nicholson
                              ||(this->myDiffusion->get_myNumericalScheme()==Diffusion::splitting_spectral_adaptive
                                 && this->advance_fields_scft_advance_mask_level_set))
                    {
                        this->remapFromOldForestToNewForest2();
                    }
                    if((this->myDiffusion->get_myNumericalScheme()==Diffusion::splitting_spectral_adaptive
                        && !this->advance_fields_scft_advance_mask_level_set)
                            ||(this->myDiffusion->get_myNumericalScheme()==Diffusion::splitting_finite_difference_implicit
                               &&this->myDiffusion->get_my_casl_diffusion_method()==Diffusion::periodic_crank_nicholson ))
                    {
                        this->remapFromOldForestToNewForest();
                    }

                    this->scft_clock->remapping_watch.stop();



                    if(this->advance_fields_level_set && this->i_mean_field_iteration>0)
                    {

                        this->ierr=VecDestroy(this->phi_seed); CHKERRXX(this->ierr);
                        this->remapFromOldForestToNewForest(this->p4est_remeshed,this->nodes_remeshed,this->ghost_remeshed,this->brick_remeshed,
                                                            &this->phi_seed_new_on_old_p4est,&this->phi_seed);
                        this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
                    }

                    if(this->advance_fields_scft_advance_mask_level_set)
                    {
                        this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
                        this->ierr=VecDestroy(this->phi_seed);CHKERRXX(this->ierr);
                        this->ierr=VecDuplicate(this->polymer_mask,&this->phi_seed); CHKERRXX(this->ierr);
                        this->ierr=VecCopy(this->polymer_mask,this->phi_seed); CHKERRXX(this->ierr);
                        this->scatter_petsc_vector(&this->phi_seed);

                        int n_local_phi_wall,n_local_polymer_mask,
                                n_local_phi_seed;

                        this->ierr=VecGetLocalSize(this->myDiffusion->phi_wall,&n_local_phi_wall); CHKERRXX(this->ierr);
                        this->ierr=VecGetLocalSize(this->polymer_mask,&n_local_polymer_mask); CHKERRXX(this->ierr);
                        this->ierr=VecGetLocalSize(this->phi_seed,&n_local_phi_seed); CHKERRXX(this->ierr);


                        std::cout<<" n_local_phi_wall "<<n_local_phi_wall
                                <<" n_local_polymer_mask_local "<<n_local_polymer_mask
                               <<" n_local_phi_seed "<<n_local_phi_seed<<std::endl;



                        PetscBool destroy_phi_wall=PETSC_TRUE,compute_on_remeshed_p4est=PETSC_TRUE;

                        this->set_advected_wall_from_level_set(destroy_phi_wall,compute_on_remeshed_p4est);
                        if(this->terracing)
                            this->correct_advected_wall_from_level_set_terrace(destroy_phi_wall,compute_on_remeshed_p4est);

                        this->ierr=VecGetLocalSize(this->myDiffusion->phi_wall,&n_local_phi_wall); CHKERRXX(this->ierr);
                        this->ierr=VecGetLocalSize(this->polymer_mask,&n_local_polymer_mask); CHKERRXX(this->ierr);
                        this->ierr=VecGetLocalSize(this->phi_seed,&n_local_phi_seed); CHKERRXX(this->ierr);


                        std::cout<<" n_local_phi_wall "<<n_local_phi_wall
                                <<" n_local_polymer_mask_local "<<n_local_polymer_mask
                               <<" n_local_phi_seed "<<n_local_phi_seed<<std::endl;

                        this->ierr=VecDestroy(this->polymer_mask); CHKERRXX(this->ierr);

                        my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_t",  this->write2VtkShapeOptimization);
                        my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->phi_wall,"phi_wall_t",  this->write2VtkShapeOptimization);


                    }

                }




                int n_games=1;
                for(int i_game=0;i_game<n_games;i_game++)
                {


                    p4est_nodes_destroy(this->nodes);
                    p4est_ghost_destroy(this->ghost);
                    p4est_destroy(this->p4est);
                    p4est_connectivity_destroy(this->connectivity);

                    delete this->brick;
                    delete this->hierarchy;
                    delete this->node_neighbors;

                    this->p4est=this->p4est_remeshed;
                    this->nodes= this->nodes_remeshed;
                    this->ghost= this->ghost_remeshed;
                    this->connectivity= this->connectivity_remeshed;
                    this->brick= this->brick_remeshed;
                    this->hierarchy= this->hierarchy_remeshed;
                    this->node_neighbors=this->node_neighbors_remeshed;

                    if(i_game<n_games-1)
                        this->remeshForestOfTrees_parallel_two_level_set(this->nx_trees,this->ny_trees,this->nz_trees);

                }

                if(this->myMeanFieldPlan->neuman_with_mask || this->myMeanFieldPlan->dirichlet_with_mask)
                    this->reCreatePhiWall4Neuman();
                if(this->myDiffusion->spatial_xhi_w)
                {
                    this->ierr=VecDestroy(this->myDiffusion->xhi_w_x); CHKERRXX(this->ierr);
                    this->create_spatial_xhi_wall();

                }


            }

            this->myDiffusion->set_i_mean_field(this->i_mean_field_iteration);
            this->scft_clock->setup_watch.start("setup");
            this->myDiffusion->reInitialyzeDiffusionFromMeanField(this->p4est,this->nodes,this->ghost,this->connectivity,this->brick,this->hierarchy,
                                                                  this->node_neighbors,this->Lx,this->Lx_physics,this->i_mean_field_iteration);


            this->scft_clock->setup_watch.stop();

        }

        {
            //if(!this->remesh_my_forest)// || (this->remesh_my_forest && mod(this->i_mean_field_iteration,this->remesh_period)!=0))
            this->myDiffusion->set_i_mean_field(this->i_mean_field_iteration);


            int n_games=1;
            for(int i_game=0;i_game<n_games;i_game++)
            {

                if(this->advance_fields_level_set && this->i_mean_field_iteration ==0 )
                {
                    switch(this->my_seed_strategy)
                    {
                    case MeanField::sphere_seed:
                        this->set_initial_seed_from_level_set();
                        break;
                    case MeanField::bcc_seed:
                        this->set_initial_seed_from_level_set_bcc();
                        break;
                    case MeanField::helix_seed:
                        this->set_initial_seed_from_level_set_helix();
                        break;
                    case MeanField::from_wp:
                        this->set_initial_seed_from_level_set_wp();
                    case MeanField::l_shape_seed:
                        this->create_polymer_mask_l_shape(this->a_ellipse,this->b_ellipse,this->c_ellipse);
                        this->ierr=VecDuplicate(this->polymer_mask,&this->phi_seed); CHKERRXX(this->ierr);
                        this->ierr=VecCopy(this->polymer_mask,this->phi_seed); CHKERRXX(this->ierr);
                        this->scatter_petsc_vector(&this->phi_seed);
                        this->ierr=VecDuplicate(this->polymer_mask,&this->phi_seed_initial); CHKERRXX(this->ierr);
                        this->ierr=VecCopy(this->polymer_mask,this->phi_seed_initial); CHKERRXX(this->ierr);
                        this->scatter_petsc_vector(&this->phi_seed_initial);
                        this->ierr=VecDestroy(this->polymer_mask); CHKERRXX(this->ierr);

                        break;

                    }

                    this->set_advected_fields_from_level_set();
                    if(!this->myMeanFieldPlan->periodic_xyz)
                        this->myDiffusion->filter_irregular_potential();

                }



                this->myDiffusion->evolve_mean_field_step();

                if(this->myMeanFieldPlan->optimize_robin_kappa_b && (this->i_mean_field_iteration%this->myMeanFieldPlan->robin_optimization_period)==0)
                {
                    this->my_robin_optimization->evolve_kappa_b(this->myDiffusion->get_RhoAAverage_surface(),
                                                                this->myDiffusion->get_RhoBAverage_surface());
                    this->myMeanFieldPlan->kappaB=this->my_robin_optimization->kappa_b;
                }


                n_games=1;
                for(int i_game=0;i_game<n_games;i_game++)
                {

                    std::cout<<this->mpi->mpirank<<" "<<this->i_mean_field_iteration<<" "<<"  Mean Field Steps Evolved "<<std::endl;


                    //Evolve Diffusion Potentials up to the force:
                    // can be done explicitly (Euler Step)
                    // implicitly with Fast Fourier Transforms and Inverse Fast Fourier Transforms
                    // can be done with predictor/corrector numerical scheme

                    if(this->advance_fields_scft)
                        this->evolve_statistical_fields_explicit();

                    if(this->i_mean_field_iteration>0)
                    {
                        this->ierr=VecDestroy(this->fp_stored); CHKERRXX(this->ierr);
                        this->ierr=VecDestroy(this->fm_stored); CHKERRXX(this->ierr);
                    }

                    this->ierr=VecDuplicate(this->phi,&this->fm_stored); CHKERRXX(this->ierr);
                    this->ierr=VecDuplicate(this->phi,&this->fp_stored); CHKERRXX(this->ierr);

                    this->ierr=VecCopy(this->myDiffusion->fm,this->fm_stored); CHKERRXX(this->ierr);
                    this->ierr=VecCopy(this->myDiffusion->fp,this->fp_stored); CHKERRXX(this->ierr);


                    if(this->conserve_reaction_source_volume && this->advance_fields_level_set)
                    {
                        this->ierr=VecShift(this->fp_stored,1.00); CHKERRXX(this->ierr);
                    }

                    this->ierr=VecGhostUpdateBegin(this->fm_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
                    this->ierr=VecGhostUpdateEnd(this->fm_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

                    this->ierr=VecGhostUpdateBegin(this->fp_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
                    this->ierr=VecGhostUpdateEnd(this->fp_stored,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


                    if(this->extend_advect)
                    {
                        if(this->i_mean_field_iteration>0)
                        {
                            this->ierr=VecDestroy(this->rho_a_stored); CHKERRXX(this->ierr);
                            this->ierr=VecDestroy(this->rho_b_stored); CHKERRXX(this->ierr);
                        }

                        this->ierr=VecDuplicate(this->phi,&this->rho_a_stored); CHKERRXX(this->ierr);
                        this->ierr=VecDuplicate(this->phi,&this->rho_b_stored); CHKERRXX(this->ierr);

                        this->ierr=VecCopy(this->myDiffusion->rhoA,this->rho_a_stored); CHKERRXX(this->ierr);
                        this->ierr=VecCopy(this->myDiffusion->rhoB,this->rho_b_stored); CHKERRXX(this->ierr);

                        this->scatter_petsc_vector(&this->rho_a_stored);
                        this->scatter_petsc_vector(&this->rho_b_stored);

                    }



                    //if(this->advance_fields_scft   &&this->extend_advect && this->my_shape_optimization_strategy==MeanField::level_set_optimization)
                    {
                        if(this->i_mean_field_iteration>0)
                            this->ierr=VecDestroy(this->last_q_stored); CHKERRXX(this->ierr);
                        this->myDiffusion->copy_and_scatter_sol_tp1(&this->last_q_stored);
                        this->plot_log_vec(&this->last_q_stored,"last_log_q",PETSC_FALSE);
                        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->last_q_stored,"last_q",PETSC_FALSE);
                        this->myDiffusion->printDiffusionArrayFromVector(&this->last_q_stored,"last_q");
                    }


                    if(i_game<n_games-1)
                    {
                        this->ierr=VecDestroy(this->fp_stored); CHKERRXX(this->ierr);
                        this->ierr=VecDestroy(this->fm_stored); CHKERRXX(this->ierr);
                        this->ierr=VecDestroy(this->last_q_stored); CHKERRXX(this->ierr);
                    }
                }



                write_mean_field_to_txt_and_vtk();


                bool remesh_next_iteration;
                remesh_next_iteration=this->remesh_my_forest
                        &&( mod((this->i_mean_field_iteration+1),this->remesh_period)==0
                            || (this->i_mean_field_iteration+1==0) );


                if(this->myMeanFieldPlan->get_stride_from_neuman && this->myDiffusion->neuman_crank_nicholson)
                {
                    if(this->i_mean_field_iteration>0)
                        this->ierr=VecDestroy(this->neuman_stride);

                    this->ierr=VecDuplicate(this->phi,&this->neuman_stride); CHKERRXX(this->ierr);
                    this->ierr=VecCopy(this->myDiffusion->myNeumanPoissonSolverNodeBase->phi_is_all_positive,this->neuman_stride); CHKERRXX(this->ierr);
                    this->scatter_petsc_vector(&this->neuman_stride);
                }

                this->myDiffusion->clean_mean_field_step(remesh_next_iteration);

                std::cout<<this->mpi->mpirank<<" "<<this->i_mean_field_iteration<<" "<<"   Field Steps Evolved "<<std::endl;

            }

            this->energy[this->i_mean_field_iteration]=this->myDiffusion->get_E();
            this->energy_logQ[this->i_mean_field_iteration]=this->myDiffusion->getE_logQ();
            this->energy_w[this->i_mean_field_iteration]=this->myDiffusion->getE_w();
            this->exchange_force[this->i_mean_field_iteration]=this->myDiffusion->get_Fm2();
            this->pressure_force[this->i_mean_field_iteration]=this->myDiffusion->get_Fp2();
            this->rho0Average[this->i_mean_field_iteration]=this->myDiffusion->get_Rho0Average();
            this->rhoAAverage[this->i_mean_field_iteration]=this->myDiffusion->get_RhoAAverage();
            this->rhoBAverage[this->i_mean_field_iteration]=this->myDiffusion->get_RhoBAverage();
            this->order_Ratio[this->i_mean_field_iteration]=this->myDiffusion->get_OrderRatio();
            this->length_series[this->i_mean_field_iteration]=this->Lx_physics;

            if(this->extend_advect)
                this->VRealTime[this->i_mean_field_iteration]=this->domain_volume;



            if(this->i_mean_field_iteration==this->N_mean_field_iteration-1 && this->setup_finite_difference_solver && !this->myMeanFieldPlan->periodic_xyz)
            {
                Vec wp_extended;
                this->ierr=VecDuplicate(this->myDiffusion->wp,&wp_extended); CHKERRXX(this->ierr);
                this->ierr=VecCopy(this->myDiffusion->wp,wp_extended); CHKERRXX(this->ierr);
                if(this->myMeanFieldPlan->get_stride_from_neuman)
                    this->extend_petsc_vector_with_stride(&this->polymer_mask,&wp_extended,pow(2,this->max_level));
                else
                    this->extend_petsc_vector(&this->polymer_mask,&wp_extended,pow(2,this->max_level));
                this->myDiffusion->print_vec_by_cells(&wp_extended,"wp_last");
                this->myDiffusion->print_vec_by_cells(&this->polymer_mask,"mask_shape");
                this->ierr=VecDestroy(wp_extended); CHKERRXX(this->ierr);
            }


            if(this->advance_fields_scft_advance_mask_level_set)
            {
                double mask_energy;
                this->compute_mask_energy_from_level_set(mask_energy,&this->phi_seed);

                std::cout<<" mask_energy "<<mask_energy<<" E "<<this->myDiffusion->get_E()<<std::endl;
            }

            this->write2LogEnergyChanges();

            std::cout<<this->mpi->mpirank<<" "<<this->i_mean_field_iteration<<" "<<"   Evolution Quantities "<<std::endl;

            //this->myDiffusion->printStatisticalFields();

            std::cout<<this->mpi->mpirank<<" "<<this->i_mean_field_iteration<<" "<<"   Statistical Fields Printed "<<std::endl;

            if(this->advance_fields_scft_advance_mask_level_set)
            {
                this->printSCFTShapeOptimizationEvolution();
            }


            this->compute_volume_and_surface_domain();
            this->printShapeOptimizationEvolution();


            this->scft_clock->optimization_watch.stop();
            this->printOptimizationEvolution();

            std::cout<<this->mpi->mpirank<<" "<<this->i_mean_field_iteration<<" "<<"   Optimization Evolution Printed "<<std::endl;


            this->clean_mean_field_step();

            if(this->myDiffusion->compute_stress_tensor_bool &&mod( this->i_mean_field_iteration+1,this->myDiffusion->stress_tensor_computation_period)==0)
            {
                if(this->myMeanFieldPlan->periodic_xyz)
                {
                    //                this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->my_stress_tensor->sxx_global,"sxx",PETSC_TRUE);
                    //                this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->my_stress_tensor->syy_global,"syy",PETSC_TRUE);
                    //                this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->my_stress_tensor->szz_global,"szz",PETSC_TRUE);
                    //                this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->my_stress_tensor->sxy_global,"sxy",PETSC_TRUE);
                    //                this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->my_stress_tensor->sxz_global,"sxz",PETSC_TRUE);
                    //                this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->my_stress_tensor->syz_global,"syz",PETSC_TRUE);

                    this->Sxx=this->myDiffusion->my_stress_tensor->Sxx;
                    this->Syy=this->myDiffusion->my_stress_tensor->Syy;
                    this->Szz=this->myDiffusion->my_stress_tensor->Szz;
                    this->Sxy=this->myDiffusion->my_stress_tensor->Sxy;
                    this->Sxz=this->myDiffusion->my_stress_tensor->Sxz;
                    this->Syz=this->myDiffusion->my_stress_tensor->Syz;

                    this->stress_tensor_trace=this->Sxx;//+this->Syy+this->Szz);

                    //            this->Sxx=this->Sxx-this->stress_tensor_trace;
                    //            this->Syy=this->Syy-this->stress_tensor_trace;
                    //            this->Szz=this->Szz-this->stress_tensor_trace;
                    //            this->Sxy=this->Sxy-this->stress_tensor_trace;
                    //            this->Sxz=this->Sxz-this->stress_tensor_trace;
                    //            this->Syz=this->Syz-this->stress_tensor_trace;

                    this->printStressTensorEvolution();

                    delete this->myDiffusion->my_stress_tensor;
                    if(this->stress_tensor_trace>0)
                    {
                        this->Lx_physics=this->Lx_physics+MIN(this->stress_tensor_trace,0.1);
                        this->myDiffusion->resetLxPhysics(this->Lx_physics);
                    }
                    else
                    {
                        this->Lx_physics=this->Lx_physics+MAX(this->stress_tensor_trace,-0.1);
                        this->myDiffusion->resetLxPhysics(this->Lx_physics);
                    }


                }
                else
                {
                    if(this->myDiffusion->my_stress_tensor_computation_mode==StressTensor::shape_derivative)
                        this->myDiffusion->create_copy_and_scatter_shape_derivative_and_destroy_stress_tensor(&this->snn);
                }
            }


            this->i_mean_field_iteration++;
        }

    }
}

int MeanField::regenerate_potentials_from_level_set(Vec *field_seed)
{
    // do not change any value of the level set in this function

    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start regenerate_potentials_from_level_set"<<std::endl;
        // step 1
        // seed/ set the statistical fields from the level set function which acts as an Heaviside function

        PetscScalar *field_seed_local;
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" get array set_advected_fields_from_level_set"<<std::endl;
        this->ierr=VecGetArray(*field_seed,&field_seed_local); CHKERRXX(this->ierr);

        PetscScalar *wm_local;
        this->ierr=VecGetArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);

        double field_seed_local_double;
        double tanh_phi;

        for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
        {
            field_seed_local_double=field_seed_local[i];
            tanh_phi=MeanField::my_tanh_x(this->alpha_wall*field_seed_local_double);
            wm_local[i]=ABS(tanh_phi)*wm_local[i];
        }
        this->ierr=VecRestoreArray(*field_seed,&field_seed_local); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" restore array regenerate_potentials_from_level_set"<<std::endl;

        this->ierr=VecRestoreArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->myDiffusion->wm);

        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wm,"wm_reseeded",PETSC_FALSE);


        this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wm,"wm_reseeded",PETSC_TRUE);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<"end regenerate_potentials_from_level_set "<<std::endl;
    }
}


int MeanField::compute_phi_bar()
{
    Vec phiTemp;
    this->ierr=VecDuplicate(this->phi,&phiTemp); CHKERRXX(this->ierr);
    this->ierr=VecSet(phiTemp,-1.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&phiTemp);
    this->phi_bar=0;
    this->phi_bar=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,this->myDiffusion->phi_wall);
    std::cout<<" phi_bar "<<this->phi_bar<<std::endl;
    this->phi_bar=(1.00-this->phi_bar)/this->volume_phi_seed;
    std::cout<<" phi bar "<<this->phi_bar<<std::endl;
    this->ierr=VecDestroy(phiTemp);
}

void MeanField::smooth_seed_for_papers_1_and_2()
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


    double A=  this->X_ab/2;

    PetscScalar *wp_local;
    PetscScalar *wm_local;

    this->ierr=VecGetArray(this->myDiffusion->wp,&wp_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);

    PetscScalar *phi_wall_local;
    this->ierr=VecGetArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);


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


        if(!this->terracing)
        {
            wp_local[i]=A*sin(6*PI*x/L)*exp(-phi_wall_local[i]*phi_wall_local[i])+this->phi_bar*this->X_ab/2.00*(1-exp(-phi_wall_local[i]*phi_wall_local[i]))/(1-this->phi_bar);
            wm_local[i]=A*cos(6*PI*x/L)*exp(-phi_wall_local[i]*phi_wall_local[i]);
        }
        else
        {
            wp_local[i]=A*sin(6*PI*y/L)*exp(-this->alpha_wall*phi_wall_local[i]*phi_wall_local[i])+this->phi_bar*this->X_ab/2.00*(1-exp(-this->alpha_wall*phi_wall_local[i]*phi_wall_local[i]))/(1-this->phi_bar);
            wm_local[i]=A*exp(-this->alpha_wall*  phi_wall_local[i]*phi_wall_local[i])*(cos(6*PI*y/L)+0.0*cos(PI*x/L));

        }
#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i]);

#endif

    }
    this->ierr=VecRestoreArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myDiffusion->wp,&wp_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    this->ierr=VecGhostUpdateBegin(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
}

int MeanField::design_seed()
{

    double wp_average_inside_wall;
    double wp_average_inside_mask;
    double wm_average_inside_mask;
    double wm_average_inside_wall;
    PetscScalar compressibility_scalar;
    PetscScalar wall_force_scalar_1,wall_force_scalar_2;
    PetscScalar minus_one=-1.00;
    PetscScalar phi_wall_inside_wall=1.00;
    PetscScalar phi_wall_inside_mask=0.00;



    // compute scalars

    compressibility_scalar=minus_one*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
    wall_force_scalar_1=this->xhi_w_p*this->zeta_n_inverse+1.00;
    wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;

    //--------------------Compute wp_average outside the mask and inside the mask---------------------------//
    wp_average_inside_wall=wall_force_scalar_1*phi_wall_inside_wall+minus_one;
    wp_average_inside_wall=wp_average_inside_wall/wall_force_scalar_2;
    wp_average_inside_wall=-wp_average_inside_wall/compressibility_scalar;

    wp_average_inside_mask=1-1/wall_force_scalar_2;
    wp_average_inside_mask=-wp_average_inside_mask/compressibility_scalar;

    //-----------------Compute wm_average outside the mask and inside the mask--------------------//
    wm_average_inside_wall=-this->xhi_w_m*phi_wall_inside_wall;
    wm_average_inside_mask=-this->X_ab/2.00*(1.00-2.00*this->f);



    //------------------------------------------------------------------------------------------//

    double L=this->Lx;

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("design_phase")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");
    std::cout<<"nodes->num_owned_indeps"<<this->nodes->num_owned_indeps<<std::endl;
    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];


    double A=  this->X_ab/2;

    PetscScalar *wp_local;
    PetscScalar *wm_local;

    this->ierr=VecGetArray(this->myDiffusion->wp,&wp_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);

    PetscScalar *phi_wall_local;
    this->ierr=VecGetArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);




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

        wp_local[i]=wp_average_inside_mask*(1-phi_wall_local[i])+phi_wall_local[i]*wp_average_inside_wall;

        wm_local[i]=wm_local[i]*(1.00-phi_wall_local[i])+phi_wall_local[i]*wm_average_inside_wall;

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,wp_local[i],wm_local[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f\n",  myRank,tree_id,i,global_node_number+i,  x,y,wp_local[i],wm_local[i],phi_wall_local[i]);

#endif

    }
    this->ierr=VecRestoreArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myDiffusion->wp,&wp_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);

    std::cout<<"nodes->num_owned_indeps "<<nodes->num_owned_indeps<<std::endl;
    std::cout<<"nodes->indep_nodes.elem_count"<<nodes->indep_nodes.elem_count<<std::endl;
    fclose(outFile);
    this->ierr=VecGhostUpdateBegin(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


    this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wm,"wm_reseeded",PETSC_TRUE);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wp,"wp_reseeded",PETSC_TRUE);
}


int MeanField::generate_masked_fields()
{
    this->compute_phi_bar();


    if(this->regenerate_potentials_from_mask)
    {
        if(this->papers_1_and_2_seed)
            this->smooth_seed_for_papers_1_and_2();
        else
            this->design_seed();
    }
}


int MeanField::plot_log_vec(Vec *v2plogplot, string file_name, PetscBool write2vtk)
{

    this->ierr=VecDuplicate(*v2plogplot,&this->log_vector_for_plot); CHKERRXX(this->ierr);
    this->ierr=VecCopy(*v2plogplot,this->log_vector_for_plot); CHKERRXX(this->ierr);

    PetscScalar *v2plot_local;
    this->ierr=VecGetArray(this->log_vector_for_plot,&v2plot_local); CHKERRXX(this->ierr);

    for(int i=0;i<this->nodes->num_owned_indeps;i++)
    {
        if(v2plot_local[i]>0)
        {
            v2plot_local[i]=log(v2plot_local[i]);
        }
        else
            v2plot_local[i]=0;
    }
    this->ierr=VecRestoreArray(this->log_vector_for_plot,&v2plot_local); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&this->log_vector_for_plot);
    this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->log_vector_for_plot,file_name,write2vtk);
    this->ierr=VecDestroy(this->log_vector_for_plot); CHKERRXX(this->ierr);
}


int MeanField::compute_volume_and_surface_domain()
{

    if(this->extend_advect)
    {
        Vec temp_f;

        this->ierr=VecDuplicate(this->polymer_mask,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);
        this->extend_petsc_vector(&this->polymer_mask,&temp_f);
        this->domain_volume=integrate_over_negative_domain(this->p4est,this->nodes,this->polymer_mask,temp_f);
        this->domain_surface=this->integrate_over_interface(this->polymer_mask,temp_f);
        this->effective_radius=this->domain_volume/(4.00*PI/3.00);
        this->effective_radius=pow(this->effective_radius,1.00/3.00);

        this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    }
    if(this->advance_fields_level_set || this->advance_fields_scft_advance_mask_level_set)
    {
        Vec temp_f;

        this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);
        this->extend_petsc_vector(&this->phi_seed,&temp_f);
        this->volume_phi_seed=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,temp_f);
        this->surface_phi_seed=this->integrate_over_interface(this->phi_seed,temp_f);
        this->effective_radius=this->volume_phi_seed/(4.00*PI/3.00);
        this->effective_radius=pow(this->effective_radius,1.00/3.00);

        this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);



    }
}

void MeanField::set_fake_vx_vy_vz()
{
    PetscScalar fake_scalar_velocity=0.1;

    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_x); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_y); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_z); CHKERRXX(this->ierr);



    // (2) Assign a velocity either a fake one or
    //     a velocity which is a result of a physical
    //     computation


    this->ierr=VecSet(this->polymer_velocity_x,fake_scalar_velocity);
    this->ierr=VecSet(this->polymer_velocity_y,fake_scalar_velocity);
    this->ierr=VecSet(this->polymer_velocity_z,fake_scalar_velocity);
}


int MeanField::compute_normal_to_the_interface(Vec *level_set_interface)
{


    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start to compute the normal to the interface "<<std::endl;

    this->ierr=VecDuplicate(*level_set_interface,&this->dmask_dx); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(*level_set_interface,&this->dmask_dy); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
    this->ierr=VecDuplicate(*level_set_interface,&this->dmask_dz); CHKERRXX(this->ierr);
#endif

    PetscScalar *dmask_local;
    this->ierr=VecGetArray(*level_set_interface,&dmask_local); CHKERRXX(this->ierr);

    this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // memory increase local to the function 5
    this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // memory increase local to the function 6
#ifdef P4_TO_P8
    this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // memory increase local to the function 7
#endif
    PetscScalar normalize_normal;
    for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
    {
        this->dmask_dx_local[ix]=this->node_neighbors->neighbors[ix].dx_central(dmask_local);
        this->dmask_dy_local[ix]=this->node_neighbors->neighbors[ix].dy_central(dmask_local);

#ifdef P4_TO_P8
        this->dmask_dz_local[ix]=this->node_neighbors->neighbors[ix].dz_central(dmask_local);
#endif
        normalize_normal=this->dmask_dx_local[ix]*this->dmask_dx_local[ix]+this->dmask_dy_local[ix]*this->dmask_dy_local[ix];

#ifdef P4_TO_P8
        normalize_normal+=this->dmask_dz_local[ix]*this->dmask_dz_local[ix];
#endif

        //        if(normalize_normal!=0)
        //        {
        //            normalize_normal=pow(normalize_normal,0.5);
        //            this->dmask_dx_local[ix]=this->dmask_dx_local[ix]/normalize_normal;
        //            this->dmask_dy_local[ix]=this->dmask_dy_local[ix]/normalize_normal;
        //#ifdef P4_TO_P8
        //            this->dmask_dz_local[ix]=this->dmask_dz_local[ix]/normalize_normal;
        //#endif
        //        }
        //        else
        //        {
        //            std::cout<<"grad phi is zero "<<std::endl;
        //        }
    }

    this->ierr=VecRestoreArray(*level_set_interface,&dmask_local);CHKERRXX(this->ierr); // no need to decrease memory local to the function 4
    this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 5
    this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 6
#ifdef P4_TO_P8
    this->ierr=VecRestoreArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 7
#endif
    // scatter again

    this->ierr=VecGhostUpdateBegin(this->dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
    this->ierr=VecGhostUpdateBegin(this->dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
#endif


#ifdef P4_TO_P8
    this->print_3D_Vector(&this->dmask_dx,&this->dmask_dy,&this->dmask_dz," normal");
#else
    this->print_2D_Vector(&this->dmask_dx,&this->dmask_dy,"normal");
    this->print_2D_VectorWithForest(&this->dmask_dx,&this->dmask_dy,"normalWithForest");



#endif

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" finish to compute the normal to the interface "<<std::endl;
}

int MeanField::compute_advection_velocity()
{
    switch(this->my_shape_optimization_strategy)
    {
    case MeanField::stress_tensor_optimization:
        this->compute_advection_velocity_from_stress_tensor();
        break;
    case MeanField::level_set_optimization:
        this->compute_advection_velocity_from_level_set();
        break;

    case MeanField::set_velocity_manually:
        this->set_advection_velocity();
        break;


    }

    return 0;
}


int MeanField::compute_advection_velocity_from_stress_tensor()
{



    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_x); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_y); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_z); CHKERRXX(this->ierr);




    PetscScalar *vx_local,*vy_local,*vz_local;
    this->ierr=VecGetArray(this->polymer_velocity_x,&vx_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->polymer_velocity_y,&vy_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->polymer_velocity_z,&vz_local); CHKERRXX(this->ierr);

    this->compute_normal_to_the_interface(&this->polymer_mask);

    this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // memory increase local to the function 5
    this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // memory increase local to the function 6
    this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // memory increase local to the function 7

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    // Point by Point project the stress tensor on the normal like:
    //
    // vx=Sxx*nx+Sxy*ny+Sxz*nz
    // vy=Syx*nx+Syy*ny+Szz*nz
    // vz=Szx*nx+Szy*ny+Szz*nz

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        vx_local[i]=this->Sxx*this->dmask_dx_local[i]+this->Sxy*this->dmask_dy_local[i]+this->Sxz*this->dmask_dz_local[i];
        vy_local[i]=this->Sxy*this->dmask_dx_local[i]+this->Syy*this->dmask_dy_local[i]+this->Syz*this->dmask_dz_local[i];
        vz_local[i]=this->Sxz*this->dmask_dx_local[i]+this->Syz*this->dmask_dy_local[i]+this->Szz*this->dmask_dz_local[i];
    }

    this->ierr= VecRestoreArray(this->polymer_velocity_x,&vx_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->polymer_velocity_y,&vy_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->polymer_velocity_z,&vz_local); CHKERRXX(this->ierr);


    this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 5
    this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 6
    this->ierr=VecRestoreArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 7


    // scatter the ghost vector to all the processors
    VecGhostUpdateBegin(this->polymer_velocity_x,INSERT_VALUES,SCATTER_FORWARD);
    VecGhostUpdateEnd(this->polymer_velocity_x,INSERT_VALUES,SCATTER_FORWARD);

    // scatter the ghost vector to all the processors
    VecGhostUpdateBegin(this->polymer_velocity_y,INSERT_VALUES,SCATTER_FORWARD);
    VecGhostUpdateEnd(this->polymer_velocity_y,INSERT_VALUES,SCATTER_FORWARD);

    // scatter the ghost vector to all the processors
    VecGhostUpdateBegin(this->polymer_velocity_z,INSERT_VALUES,SCATTER_FORWARD);
    VecGhostUpdateEnd(this->polymer_velocity_z,INSERT_VALUES,SCATTER_FORWARD);

    // (3) plot the velocities map


    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_x,"vx",PETSC_TRUE);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_y,"vy",PETSC_TRUE);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_z,"vz",PETSC_TRUE);

    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_x,"vx");
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_y,"vy");
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_z,"vz");

    this->print_3D_Vector(&this->polymer_velocity_x,&this->polymer_velocity_y,&this->polymer_velocity_z,"v_xyz");

    this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr);


    return 0;
}


int MeanField::compute_advection_velocity_from_level_set()
{
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_x); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_y); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_z); CHKERRXX(this->ierr);

    PetscScalar *vx_local,*vy_local,*vz_local;
    this->ierr=VecGetArray(this->polymer_velocity_x,&vx_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->polymer_velocity_y,&vy_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->polymer_velocity_z,&vz_local); CHKERRXX(this->ierr);

    this->compute_normal_to_the_interface(&this->polymer_mask);

    this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // memory increase local to the function 5
    this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // memory increase local to the function 6
    this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // memory increase local to the function 7

    PetscScalar *last_q_stored_local;
    this->ierr=VecGetArray(this->last_q_stored,&last_q_stored_local); CHKERRXX(this->ierr);


    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    // Point by Point project the stress tensor on the normal like:
    //
    // vx=Sxx*nx+Sxy*ny+Sxz*nz
    // vy=Syx*nx+Syy*ny+Szz*nz
    // vz=Szx*nx+Szy*ny+Szz*nz

    this->dphi=0.05;

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        //        vx_local[i]=last_q_stored_local[i]*this->dmask_dx_local[i];
        //        vy_local[i]=last_q_stored_local[i]*this->dmask_dx_local[i];
        //        vz_local[i]= last_q_stored_local[i]*this->dmask_dx_local[i];
        vx_local[i]=this->dphi*   this->dmask_dx_local[i];
        vy_local[i]=this->dphi*this->dmask_dx_local[i];
        vz_local[i]= this->dphi*this->dmask_dx_local[i];
    }


    this->ierr=VecRestoreArray(this->last_q_stored,&last_q_stored_local); CHKERRXX(this->ierr);

    this->ierr= VecRestoreArray(this->polymer_velocity_x,&vx_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->polymer_velocity_y,&vy_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->polymer_velocity_z,&vz_local); CHKERRXX(this->ierr);


    this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 5
    this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 6
    this->ierr=VecRestoreArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 7


    // scatter the ghost vector to all the processors
    this->ierr=VecGhostUpdateBegin(this->polymer_velocity_x,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->polymer_velocity_x,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    // scatter the ghost vector to all the processors
    this->ierr=VecGhostUpdateBegin(this->polymer_velocity_y,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->polymer_velocity_y,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    // scatter the ghost vector to all the processors
    this->ierr=VecGhostUpdateBegin(this->polymer_velocity_z,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->polymer_velocity_z,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    // (3) plot the velocities map


    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_x,"vx",PETSC_FALSE);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_y,"vy",PETSC_FALSE);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_z,"vz",PETSC_FALSE);

    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_x,"vx");
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_y,"vy");
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_z,"vz");

    this->print_3D_Vector(&this->polymer_velocity_x,&this->polymer_velocity_y,&this->polymer_velocity_z,"v_xyz");

    this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr);

    return 0;
}

int MeanField::set_advection_velocity()
{
#ifdef P4_TO_P8
    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;

    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_x); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_y); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi,&this->polymer_velocity_z); CHKERRXX(this->ierr);

    PetscScalar *vx_local,*vy_local,*vz_local;
    this->ierr=VecGetArray(this->polymer_velocity_x,&vx_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->polymer_velocity_y,&vy_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->polymer_velocity_z,&vz_local); CHKERRXX(this->ierr);

    if(this->advance_fields_level_set && ! this->extend_advect)
        this->compute_normal_to_the_interface(&this->phi_seed);
    if(!this->advance_fields_level_set && this->extend_advect)
        this->compute_normal_to_the_interface(&this->polymer_mask);

    this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // memory increase local to the function 5
    this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // memory increase local to the function 6
    this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // memory increase local to the function 7

    p4est_topidx_t global_node_number=0;

    for(int ii=0;ii<this->mpi->mpirank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    // Point by Point project the stress tensor on the normal like:
    //
    // vx=Sxx*nx+Sxy*ny+Sxz*nz
    // vy=Syx*nx+Syy*ny+Szz*nz
    // vz=Szx*nx+Szy*ny+Szz*nz
    double acos_argument,denominator,teta;
    double v0=1;//0.1;
    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
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
        double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        double z = node_z_fr_k(node) + tree_zmin;
#endif

        denominator=(pow((x- polymer_mask_xc)*(x- polymer_mask_xc)+
                         (y- polymer_mask_yc)*(y- polymer_mask_yc)+
                         (z- polymer_mask_zc)*(z- polymer_mask_zc),0.5));
        if(denominator>0)
        {
            acos_argument=  (z- polymer_mask_zc)/denominator;



            if(acos_argument>1 || acos_argument<-1)
                throw std::logic_error("cos argument must be between -1 and 1");

            teta=acos(acos_argument);

            vx_local[i]=v0;//0
            vy_local[i]=v0;//0;//sin(2*teta);
            vz_local[i]=v0;//0;
        }
        else
        {
            vx_local[i]=v0;
            vy_local[i]=v0;
            vz_local[i]=v0;
        }
    }

    this->ierr= VecRestoreArray(this->polymer_velocity_x,&vx_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->polymer_velocity_y,&vy_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(this->polymer_velocity_z,&vz_local); CHKERRXX(this->ierr);


    this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 5
    this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 6
    this->ierr=VecRestoreArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // no need to decrease memory local to the function 7


    // scatter the ghost vector to all the processors
    this->ierr=VecGhostUpdateBegin(this->polymer_velocity_x,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->polymer_velocity_x,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    // scatter the ghost vector to all the processors
    this->ierr= VecGhostUpdateBegin(this->polymer_velocity_y,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=  VecGhostUpdateEnd(this->polymer_velocity_y,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    // scatter the ghost vector to all the processors
    this->ierr= VecGhostUpdateBegin(this->polymer_velocity_z,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr= VecGhostUpdateEnd(this->polymer_velocity_z,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    // (3) plot the velocities map


    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_x,"vx",PETSC_TRUE);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_y,"vy",PETSC_TRUE);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_z,"vz",PETSC_TRUE);

    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_x,"vx");
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_y,"vy");
    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_velocity_z,"vz");

    this->print_3D_Vector(&this->polymer_velocity_x,&this->polymer_velocity_y,&this->polymer_velocity_z,"v_xyz");

    this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr);




    return 0;
#endif
}


int  MeanField::print_3D_Vector(Vec *nx,Vec *ny,Vec *nz, string file_name_str)
{

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start to print 3d vector "<<std::endl;

    if(!this->myMeanFieldPlan->periodic_xyz)
    {
        PetscInt   Nx;

        this->ierr= VecGetLocalSize(*nx,&Nx); CHKERRXX(this->ierr);

        PetscScalar *x2Print;
        this->ierr=VecGetArray(*nx,&x2Print); CHKERRXX(this->ierr);
        PetscScalar *y2Print;
        this->ierr=VecGetArray(*ny,&y2Print); CHKERRXX(this->ierr);
        PetscScalar *z2Print;
        this->ierr=VecGetArray(*nz,&z2Print); CHKERRXX(this->ierr);

        PetscScalar *phiTemp;
        this->ierr=VecGetArray(this->polymer_mask,&phiTemp); CHKERRXX(this->ierr);

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


        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";

        //        if(this->forward_stage)
        //            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";

        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(int i=0;i<Nx;i++)
        {
            PetscScalar temp_x=x2Print[i];
            PetscScalar temp_y=y2Print[i];
            PetscScalar temp_z=z2Print[i];
            PetscScalar temp_temp_phi=phiTemp[i];
            fprintf(outFileVec,"%d %f %f %f %f\n",i, temp_x,temp_y,temp_z,temp_temp_phi);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        this->ierr=VecRestoreArray(*nx,&x2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*ny,&y2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*nz,&z2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->polymer_mask,&phiTemp); CHKERRXX(this->ierr);

    }
    if(this->myMeanFieldPlan->periodic_xyz)
    {
        PetscInt   Nx;

        this->ierr= VecGetLocalSize(*nx,&Nx); CHKERRXX(this->ierr);

        PetscScalar *x2Print;
        this->ierr=VecGetArray(*nx,&x2Print); CHKERRXX(this->ierr);
        PetscScalar *y2Print;
        this->ierr=VecGetArray(*ny,&y2Print); CHKERRXX(this->ierr);
        PetscScalar *z2Print;
        this->ierr=VecGetArray(*nz,&z2Print); CHKERRXX(this->ierr);


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


        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";

        //        if(this->forward_stage)
        //            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";

        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(int i=0;i<Nx;i++)
        {
            PetscScalar temp_x=x2Print[i];
            PetscScalar temp_y=y2Print[i];
            PetscScalar temp_z=z2Print[i];

            fprintf(outFileVec,"%d %f %f %f\n",i, temp_x,temp_y,temp_z);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        this->ierr=VecRestoreArray(*nx,&x2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*ny,&y2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*nz,&z2Print); CHKERRXX(this->ierr);

    }
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" finished to print 3d vector "<<std::endl;

    return 0;
}


int  MeanField::print_2D_Vector(Vec *nx,Vec *ny, string file_name_str)
{

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start to print 3d vector "<<std::endl;

    if(!this->myMeanFieldPlan->px && !this->myMeanFieldPlan->py && !this->myMeanFieldPlan->pz)
    {
        PetscInt   Nx;

        this->ierr= VecGetLocalSize(*nx,&Nx); CHKERRXX(this->ierr);

        PetscScalar *x2Print;
        this->ierr=VecGetArray(*nx,&x2Print); CHKERRXX(this->ierr);
        PetscScalar *y2Print;
        this->ierr=VecGetArray(*ny,&y2Print); CHKERRXX(this->ierr);

        PetscScalar *phiTemp;
        this->ierr=VecGetArray(this->polymer_mask,&phiTemp); CHKERRXX(this->ierr);

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


        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";

        //        if(this->forward_stage)
        //            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";

        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(int i=0;i<Nx;i++)
        {
            PetscScalar temp_x=x2Print[i];
            PetscScalar temp_y=y2Print[i];

            PetscScalar temp_temp_phi=phiTemp[i];
            fprintf(outFileVec,"%d %f %f %f\n",i, temp_x,temp_y,temp_temp_phi);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        this->ierr=VecRestoreArray(*nx,&x2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*ny,&y2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->polymer_mask,&phiTemp); CHKERRXX(this->ierr);

    }
    if(this->myMeanFieldPlan->px ||this->myMeanFieldPlan->py ||this->myMeanFieldPlan->pz)
    {
        PetscInt   Nx;

        this->ierr= VecGetLocalSize(*nx,&Nx); CHKERRXX(this->ierr);

        PetscScalar *x2Print;
        this->ierr=VecGetArray(*nx,&x2Print); CHKERRXX(this->ierr);
        PetscScalar *y2Print;
        this->ierr=VecGetArray(*ny,&y2Print); CHKERRXX(this->ierr);


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


        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";

        //        if(this->forward_stage)
        //            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";

        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(int i=0;i<Nx;i++)
        {
            PetscScalar temp_x=x2Print[i];
            PetscScalar temp_y=y2Print[i];

            fprintf(outFileVec,"%d %f %f\n",i, temp_x,temp_y);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        this->ierr=VecRestoreArray(*nx,&x2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*ny,&y2Print); CHKERRXX(this->ierr);

    }
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" finished to print 3d vector "<<std::endl;

    return 0;
}


int  MeanField::print_2D_VectorWithForest(Vec *nx,Vec *ny, string file_name_str)
{

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start to print 3d vector "<<std::endl;

    if(!this->myMeanFieldPlan->px && !this->myMeanFieldPlan->py && !this->myMeanFieldPlan->pz)
    {
        PetscInt   Nx;

        this->ierr= VecGetLocalSize(*nx,&Nx); CHKERRXX(this->ierr);

        PetscScalar *x2Print;
        this->ierr=VecGetArray(*nx,&x2Print); CHKERRXX(this->ierr);
        PetscScalar *y2Print;
        this->ierr=VecGetArray(*ny,&y2Print); CHKERRXX(this->ierr);

        PetscScalar *phiTemp;
        this->ierr=VecGetArray(this->polymer_mask,&phiTemp); CHKERRXX(this->ierr);

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


        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";

        //        if(this->forward_stage)
        //            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";

        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {



            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
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
            PetscScalar temp_x=x2Print[n];
            PetscScalar temp_y=y2Print[n];

            PetscScalar temp_temp_phi=phiTemp[n];
            fprintf(outFileVec,"%d %f %f %f %f %f\n",n,x,y,temp_temp_phi, temp_x,temp_y);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        this->ierr=VecRestoreArray(*nx,&x2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*ny,&y2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->polymer_mask,&phiTemp); CHKERRXX(this->ierr);

    }
    if(this->myMeanFieldPlan->px ||this->myMeanFieldPlan->py ||this->myMeanFieldPlan->pz)
    {
        PetscInt   Nx;

        this->ierr= VecGetLocalSize(*nx,&Nx); CHKERRXX(this->ierr);

        PetscScalar *x2Print;
        this->ierr=VecGetArray(*nx,&x2Print); CHKERRXX(this->ierr);
        PetscScalar *y2Print;
        this->ierr=VecGetArray(*ny,&y2Print); CHKERRXX(this->ierr);


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


        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";

        //        if(this->forward_stage)
        //            oss2DebugVec << this->convert2FullPath(file_name_str)<<"_forward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";
        //        else
        //             oss2DebugVec << this->convert2FullPath(file_name_str)<<"_backward_"<<mpi_rank<<"_"<<this->i_mean_field<<".txt";

        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");

        for(int i=0;i<Nx;i++)
        {
            PetscScalar temp_x=x2Print[i];
            PetscScalar temp_y=y2Print[i];

            fprintf(outFileVec,"%d %f %f\n",i, temp_x,temp_y);//,phiTempArray[i]);
        }
        fclose(outFileVec);

        this->ierr=VecRestoreArray(*nx,&x2Print); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*ny,&y2Print); CHKERRXX(this->ierr);

    }
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" finished to print 3d vector "<<std::endl;

    return 0;
}



int MeanField::extension_advection_algo_semi_lagrangian()
{
    // This is one of the two components at the core of the
    // extension addvection mask algorithm
    // We will extend the potentials at the frontier of the level set
    // whose zero countours describe the levet set mask

    // One paper among others which can describe the idea is the paper by
    // Adalsteinton and Sethian: The fast construction of extension velocities
    // in level set methods

    int n_games=1;

    for(int i_game=0; i_game<n_games; i_game++)
    {
        //(1) create level set and parameters for extension

        my_p4est_level_set LS4extension(this->node_neighbors);
        int order_of_extension=2;
        int number_of_bands_to_extend=5;
        int number_of_bands_to_extend_on_all_domain=10;
        // Vecs=0+1=1

        Vec bc_vec_fake;

        this->ierr=VecDuplicate(this->polymer_mask,&bc_vec_fake); CHKERRXX(this->ierr); // memory increase local to the function 1
        this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        // (2) Extend statistical potentials on the current grid with Neuman boundary conditions at the interface

        LS4extension.extend_Over_Interface(this->polymer_mask,this->myDiffusion->wp,NEUMANN,bc_vec_fake,order_of_extension,number_of_bands_to_extend);
        LS4extension.extend_Over_Interface(this->polymer_mask,this->myDiffusion->wm,NEUMANN,bc_vec_fake,order_of_extension,number_of_bands_to_extend);

        this->ierr=VecGhostUpdateBegin(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wp,"wp_extended",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wm,"wm_extended",PETSC_TRUE);

        // (3) Extend cartesian velocities on the current grid from interface to whole domain


        // Vecs=1+3=4
        Vec polymer_velocity_x_extended;
        Vec polymer_velocity_y_extended;
#ifdef P4_TO_P8
        Vec polymer_velocity_z_extended;
#endif

        this->ierr=VecDuplicate(this->phi,&polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory increase local to the function 2
        this->ierr=VecDuplicate(this->phi,&polymer_velocity_y_extended); CHKERRXX(this->ierr); // memory increase local to the function 3

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(this->phi,&polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory increase local to the function 4
#endif

        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_x,polymer_velocity_x_extended,number_of_bands_to_extend_on_all_domain);
        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_y,polymer_velocity_y_extended,number_of_bands_to_extend_on_all_domain);
#ifdef P4_TO_P8
        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_z,polymer_velocity_z_extended,number_of_bands_to_extend_on_all_domain);
#endif


        this->ierr=VecGhostUpdateBegin(polymer_velocity_x_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(polymer_velocity_x_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(polymer_velocity_y_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(polymer_velocity_y_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_x_extended,"vx_extended",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_y_extended,"vy_extended",PETSC_TRUE);

#ifdef P4_TO_P8
        this->ierr=VecGhostUpdateBegin(polymer_velocity_z_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(polymer_velocity_z_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_z_extended,"vz_extended",PETSC_TRUE);

#endif




        // (4) compute minimum time step

        PetscScalar *vx_extended_array;
        PetscScalar *vy_extended_array;

#ifdef P4_TO_P8
        PetscScalar *vz_extended_array;
#endif

        this->ierr=VecGetArray(polymer_velocity_x_extended,&vx_extended_array); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(polymer_velocity_y_extended,&vy_extended_array); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecGetArray(polymer_velocity_z_extended,&vz_extended_array); CHKERRXX(this->ierr);
#endif

        double max_norm_u_loc = 0;
        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {
#ifdef P4_TO_P8
            max_norm_u_loc = max(max_norm_u_loc, sqrt(   vx_extended_array[n]*vx_extended_array[n]
                                                         + vy_extended_array[n]*vy_extended_array[n]
                                                         + vz_extended_array[n]*vz_extended_array[n]));
#else
            max_norm_u_loc = max(max_norm_u_loc, sqrt(   vx_extended_array[n]*vx_extended_array[n]
                                                         + vy_extended_array[n]*vy_extended_array[n]
                                                         ));
#endif
        }

        double max_norm_u;
        MPI_Allreduce(&max_norm_u_loc, &max_norm_u, 1, MPI_DOUBLE, MPI_MAX, this->p4est->mpicomm);

        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        double dx_for_dt_min =this->Lx / (double)Nx;

        double dt_for_advection =0;// min(1.,1./max_norm_u) * this->alpha_cfl * dx_for_dt_min;

        //dt_for_advection=0.01;

        this->ierr=VecRestoreArray(polymer_velocity_x_extended,&vx_extended_array);    CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(polymer_velocity_y_extended,&vy_extended_array);    CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(polymer_velocity_z_extended,&vz_extended_array);    CHKERRXX(this->ierr);
#endif

        // (5) Advect the level set in time using the semi lagrangian operator

        // definition of local p4est variables for remeshing up to the advected mask

        // p4est=1
        p4est_t *p4est_np1 = p4est_copy(this->p4est, P4EST_FALSE);
        // ghost=1
        p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        // nodes=1
        p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        // IMPORTANT NOTE: later on override semi lagrangian such that
        // the advection can be done on one level set while the splitting criteria
        // can be done with the mask interface but also together with the statistical fields

        // we will override update_p4est_second_order_uniform_inside_domain
        // and add as arguments wp,wm
        // and follow the comments inside this function

        // however since we advect an adaptive grid and the potentials
        // do not move much in one men field iteration
        // It has to be checked if the grid after the advection stays adaptive and
        // can easily recover to the potential segmentation grid

        // it can be done with a mix of advection for the polymer mask
        // and interpolation for the statistical fields

        // we note the advection has been done previously on semilagrangian
        // and interpolation for statistical fields is in current use on this
        // class


        SemiLagrangian sl(&p4est_np1, &nodes_np1, &ghost_np1, this->brick);

#ifdef P4_TO_P8
        sl.update_p4est_second_order_uniform_inside_domain(polymer_velocity_x_extended,polymer_velocity_y_extended,polymer_velocity_z_extended,
                                                           dt_for_advection,this->polymer_mask);
#else
        sl.update_p4est_second_order_uniform_inside_domain(polymer_velocity_x_extended,polymer_velocity_y_extended,
                                                           dt_for_advection,this->polymer_mask);
#endif

        //Now polymer mask is defined on p4est_np1


        //        / vec=4-1=3
        //        this->ierr = VecDestroy(polymer_velocity_x_extended); CHKERRXX(this->ierr);
        //        this->ierr = VecDestroy(polymer_velocity_y_extended); CHKERRXX(this->ierr);
        //#ifdef P4_TO_P8
        //        this->ierr = VecDestroy(polymer_velocity_z_extended); CHKERRXX(this->ierr);
        //#endif


        // (6) Interpolate w+,w-,polymer_mask (the reinitialyzed signed distance function) on the new grid
        // vec=4-1=3
        this->ierr=VecDestroy(this->phi); CHKERRXX(this->ierr); // memory decrease global to the object 1

        // Either duplicate or create a ghost vector with the advected parallel p4est data structure
        // vec=3+1=4
        this->ierr=VecDuplicate(this->polymer_mask,&this->phi); CHKERRXX(this->ierr); // memory increase global to the object 1


        // Create temporary statistical fields wp,wm
        //   Vec wp_temp,wm_temp;

        // Either duplicate or create ghost vectors with the advected parallel p4est data structure
        // vec=4+2=6

        Vec wp_temp,wm_temp;

        this->ierr=VecDuplicate(this->phi,&wp_temp); CHKERRXX(this->ierr);  // memory increase local to the function 5
        this->ierr=VecDuplicate(this->phi,&wm_temp); CHKERRXX(this->ierr); // memory increase local to the function 6

        //  create the interpolation function with the non-advected parallel p4est data structure

        InterpolatingFunctionNodeBase interp_w(this->p4est,this->nodes,this->ghost,this->brick,this->node_neighbors);

        // buffer the points for the interpolation function for parallel efficiency

        // p4est_topidx *=0+1
        p4est_topidx_t *t2v = p4est_np1->connectivity->tree_to_vertex; // tree to vertex list
        // double *=0+1
        double *t2c = p4est_np1->connectivity->vertices; // coordinates of the vertices of a tree
        for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
        {
            // double *=0+1
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n + nodes_np1->offset_owned_indeps); // no need to be deleted and must not be deleted
            p4est_topidx_t tree_idx = node->p.piggy3.which_tree;

            p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
            double tr_xmin = t2c[3 * tr_mm + 0];
            double tr_ymin = t2c[3 * tr_mm + 1];
#ifdef P4_TO_P8
            double tr_zmin = t2c[3 * tr_mm + 2];
#endif

            double xyz [] =
            {
                node_x_fr_i(node) + tr_xmin,
                node_y_fr_j(node) + tr_ymin
    #ifdef P4_TO_P8
                ,
                node_z_fr_k(node) + tr_zmin
    #endif
            };

            interp_w.add_point_to_buffer(n, xyz);

        }

        // perform the actual interpolation

        interp_w.set_input_parameters(this->myDiffusion->wp,linear);
        interp_w.interpolate(wp_temp);

        // Note consider to substract the spatial average
        // after the interpolation of the pressure potential

        interp_w.set_input_parameters(this->myDiffusion->wm,linear);
        interp_w.interpolate(wm_temp);

        this->ierr=VecGhostUpdateBegin(wp_temp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(wp_temp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(wm_temp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(wm_temp,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);



        // vec=6-2=4
        this->ierr=VecDestroy(this->myDiffusion->wp); CHKERRXX(this->ierr); // memory decrease global to the object 2
        this->ierr=VecDestroy(this->myDiffusion->wm); CHKERRXX(this->ierr); // memory decrease global to the object 3

        // vec=4+2=6
        this->ierr=VecDuplicate(this->phi,&this->myDiffusion->wp); CHKERRXX(this->ierr); // memory increase global to the object 2
        this->ierr=VecDuplicate(this->phi,&this->myDiffusion->wm); CHKERRXX(this->ierr); // memory decrease global to the object 3

        this->ierr= VecCopy(wp_temp,this->myDiffusion->wp);    CHKERRXX(this->ierr);
        this->ierr=VecCopy(wm_temp,this->myDiffusion->wm);     CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD);CHKERRXX(this->ierr);

        // now wp,wm are defined on p4est_np1



        //----------------------------------PREDICT ENERGY CHANGE------------------------------//

        // This is the last opportunity to perform opertaions on the previous geometry
        // data structure.
        // We will compute/ predict what should be the energy change up to the change of level set
        // To test it the scft fields will not be evolved in the next time step to ensure we
        // have computed the right variation.

        // (a) Compute the new_polymer_shape on the old p4est data structure
        // Receives as input the new geometry data structure.
        // We are just before the swap, so while the shape is already on the new data structure
        // we still have as class variables the old one.

        // (b) predict the energy change due to the level set variation
        //     For that compute dphi(x)=phi_new-phi_old

        this->remapFromNewForestToOldForest(p4est_np1, nodes_np1, ghost_np1, this->brick,&this->polymer_shape_stored,
                                            &this->polymer_mask,   &this->new_polymer_shape_on_old_forest);
        my_p4est_level_set PLS(this->node_neighbors);
        PLS.reinitialize_2nd_order(this->new_polymer_shape_on_old_forest);

        this->ierr = VecGhostUpdateBegin(this->new_polymer_shape_on_old_forest, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->new_polymer_shape_on_old_forest, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        this->my_p4est_vtk_write_all_periodic_adapter(&this->new_polymer_shape_on_old_forest,"new_polymer_shape");

        this->predict_energy_change_from_polymer_level_set_change();
        this->ierr=VecDestroy(this->new_polymer_shape_on_old_forest); CHKERRXX(this->ierr);


        //-----------------------------END OF PREDICTION ENERGY CHANGE--------------------//

        // (7) swap forests
        // nodes=1-1=0
        p4est_nodes_destroy(this->nodes);
        // ghost=1-1=0
        p4est_ghost_destroy(this->ghost);
        // p4est=1-1=0
        p4est_destroy(this->p4est);

        // hierarchy=1-1=0
        delete this->hierarchy;
        // node_neighbors=1-1=0;
        delete this->node_neighbors;

        // p4est=0+1=1
        // nodes=0+1=1
        // ghost=0+1=1
        // hierarchy=0+1=1
        // node_neighbors=0+1=1

        this->p4est=p4est_np1;
        this->nodes=nodes_np1;
        this->ghost=ghost_np1;

        my_p4est_hierarchy_t  *hierarchy_temp=new my_p4est_hierarchy_t(this->p4est,this->ghost,this->brick);
        this->hierarchy=hierarchy_temp;

        my_p4est_node_neighbors_t *node_neighbors_temp=new my_p4est_node_neighbors_t(this->hierarchy,this->nodes,this->myMeanFieldPlan->periodic_xyz);
        this->node_neighbors=node_neighbors_temp;

        this->node_neighbors->init_neighbors();


        // plot the remapped fields with the new data tructures
        // using .vtk files on paraview

        // and now wp,wm are defined on p4est


        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wp,"wp_remapped_after_advection",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wm,"wm_remapped_after_advection",PETSC_TRUE);

        //        this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_x,"vx_advected",PETSC_TRUE);
        //        this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_y,"vy_advected",PETSC_TRUE);
        //        this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_velocity_z,"vz_advected",PETSC_TRUE);

        this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"mask",PETSC_TRUE);



        // (8) destroys all the necessarry staff

        // vecs=6-6=0
        this->ierr=VecDestroy(wp_temp);CHKERRXX(this->ierr); // memory decrease local to the function 5
        this->ierr=VecDestroy(wm_temp);CHKERRXX(this->ierr);// memory decrease local to the function 6
        this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr); // memory decrease local to the function 1
        this->ierr=VecDestroy(polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory decrease local to the function 2
        this->ierr=VecDestroy(polymer_velocity_y_extended); CHKERRXX(this->ierr);// memory decrease local to the function 3
#ifdef P4_TO_P8
        this->ierr=VecDestroy(polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory decrease local to the function 4

#endif
    }

    //The velocities are constructed in another function
    this->ierr=VecDestroy(this->polymer_velocity_x); CHKERRXX(this->ierr); // memory decrease global to the object
    this->ierr=VecDestroy(this->polymer_velocity_y); CHKERRXX(this->ierr); // memory decrease global to the object
    this->ierr=VecDestroy(this->polymer_velocity_z); CHKERRXX(this->ierr); // memory decrease global to the object


    return 0;
}


int MeanField::extension_advection_algo_euler_or_gudonov()
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" extension_advection_algo_euler 0"<<std::endl;
    // We do just advect the seed and map it back to the old/current p4est data structure

    int n_games=1;

    for(int i_game=0; i_game<n_games; i_game++)
    {

        this->ierr=VecDuplicate(this->polymer_mask,&this->phi_seed); CHKERRXX(this->ierr);// duplicate +1
        this->ierr=VecCopy(this->polymer_mask,this->phi_seed); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->phi_seed); CHKERRXX(this->ierr);

        if(this->i_mean_field_iteration==this->extension_advection_period)
        {
            this->source_volume_initial=this->compute_volume_source();

        }

        this->volume_phi_seed=this->compute_volume_source();



        // Vecs=0+1=1
        Vec bc_vec_fake;
        this->ierr=VecDuplicate(this->phi_seed,&bc_vec_fake); CHKERRXX(this->ierr); // duplicate +2
        this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" extension_advection_algo_euler 1"<<std::endl;

        this->scatter_petsc_vector(&bc_vec_fake);
        this->interpolate_and_print_vec_to_uniform_grid(&this->phi_seed,"phi_seed",false);
        this->compute_normal_to_the_interface(&this->phi_seed);                  //duplicate +3,4,5
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" extension_advection_algo_euler 2"<<std::endl;

        this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // getArray +1
        this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr);// getArray +2
#ifdef P4_TO_P8
        this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr);// getArray +3
#endif

        // (3) compute minimum time step and normal velocity


        PetscScalar *Vn_local;
        this->ierr=VecDuplicate(this->phi_seed,&this->delta_phi_for_advection); CHKERRXX(this->ierr); // duplicate +6
        this->ierr=VecGetArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr);// getArray +4

        //double max_norm_u_loc = 0;
        double v0=0.5*1.00/128.00;


        Vec wm_extended_in,wp_extended_in,fp_extended_in,fm_extended_in,q_last_extended_in;

        this->ierr=VecDuplicate(this->phi,&wm_extended_in); CHKERRXX(this->ierr);   // duplicate +7
        this->ierr=VecDuplicate(this->phi,&wp_extended_in); CHKERRXX(this->ierr);   // duplicate +8
        this->ierr=VecDuplicate(this->phi,&fm_extended_in); CHKERRXX(this->ierr);   // duplicate +9
        this->ierr=VecDuplicate(this->phi,&fp_extended_in); CHKERRXX(this->ierr);   // duplicate +10
        this->ierr=VecDuplicate(this->phi,&q_last_extended_in); CHKERRXX(this->ierr); // duplicate +11



        this->ierr=VecCopy(this->myDiffusion->wm,wm_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->myDiffusion->wp,wp_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fm_stored,fm_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fp_stored,fp_extended_in); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&wm_extended_in);
        this->scatter_petsc_vector(&wp_extended_in);
        this->scatter_petsc_vector(&fm_extended_in);
        this->scatter_petsc_vector(&fp_extended_in);

        //-------Computation of the velocity for toy 2----------------------------//

        this->ierr=VecCopy(this->last_q_stored,q_last_extended_in); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&q_last_extended_in);
        this->extend_petsc_vector(&this->phi_seed,&q_last_extended_in);
        this->scatter_petsc_vector(&q_last_extended_in);
        this->q_surface_mean_value=this->integrate_over_interface(this->phi_seed,q_last_extended_in)/this->integrate_constant_over_interface(this->phi_seed);
        this->ierr=VecScale(q_last_extended_in,1/this->q_surface_mean_value); CHKERRXX(this->ierr);
        this->ierr=VecShift(q_last_extended_in,-1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&q_last_extended_in);
        this->q_surface_mean_value=this->integrate_over_interface(this->phi_seed,q_last_extended_in)/this->integrate_constant_over_interface(this->phi_seed);

        // ----end of the velocity computation for toy2-------------------//

        bool extend=true;
        if(extend)
        {

            this->extend_petsc_vector(&this->phi_seed,&wm_extended_in);
            if(this->myMeanFieldPlan->get_stride_from_neuman)
                this->extend_petsc_vector_with_stride(&this->phi_seed,&wp_extended_in);
            else
                this->extend_petsc_vector(&this->phi_seed,&wp_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&fm_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&fp_extended_in);
            //this->extend_petsc_vector(&this->phi_seed,&q_last_extended_in);
            this->my_p4est_vtk_write_all_periodic_adapter(&fp_extended_in,"fp_extended_in",PETSC_FALSE);
            this->my_p4est_vtk_write_all_periodic_adapter(&fm_extended_in,"fm_extended_in",PETSC_FALSE);
        }
        else
        {
            this->scatter_petsc_vector(&wm_extended_in);
            this->scatter_petsc_vector(&wp_extended_in);
            this->scatter_petsc_vector(&fm_extended_in);
            this->scatter_petsc_vector(&fp_extended_in);
            // this->scatter_petsc_vector(&q_last_extended_in);

        }
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" extension_advection_algo_euler 3"<<std::endl;

        PetscScalar *wm_local_in;
        PetscScalar *wp_local_in;
        PetscScalar *fp_local_in;
        PetscScalar *fm_local_in;
        PetscScalar *q_last_local_in;


        this->ierr=VecGetArray(wm_extended_in,&wm_local_in); CHKERRXX(this->ierr); // getArray +5
        this->ierr=VecGetArray(wp_extended_in,&wp_local_in); CHKERRXX(this->ierr); // getArray +6
        this->ierr=VecGetArray(fp_extended_in,&fp_local_in); CHKERRXX(this->ierr); // getArray +7
        this->ierr=VecGetArray(fm_extended_in,&fm_local_in); CHKERRXX(this->ierr); // getArray +8
        this->ierr=VecGetArray(q_last_extended_in,&q_last_local_in); CHKERRXX(this->ierr); // getArray +9


        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        double dx_for_dt_min =this->Lx / (double)Nx;

        // v0=0.05;
        PetscScalar *phi_seed_local;
        this->ierr=VecGetArray(this->phi_seed,&phi_seed_local); CHKERRXX(this->ierr); // getArray +10

        //RandomGenerator *my_random_generator=new RandomGenerator(-1.00,1.00,this->nodes->num_owned_indeps);
        // my_random_generator->continue2Generate();
        double abs_grad_phi;

        double tree_xmin,  tree_ymin;

#ifdef P4_TO_P8
        double tree_zmin;
#endif

        double x , y;

#ifdef P4_TO_P8
        double z ;
#endif
        double phi_seed_local_double;
        double denergy_increase_domain;
        double minus_one=-1;

        PetscScalar *kappa_local;
        double kappa_local_n;
        this->ierr=VecDuplicate(this->phi,&this->kappa); CHKERRXX(this->ierr); // duplicate +12
        this->ierr=VecGetArray(this->kappa,&kappa_local); CHKERRXX(this->ierr);// getArray +11





        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {

            phi_seed_local_double=(double)phi_seed_local[n];



            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];


            tree_xmin = this->connectivity->vertices[3*v_mm + 0];
            tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
            tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

            x = node_x_fr_i(node) + tree_xmin;
            y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            z = node_z_fr_k(node) + tree_zmin;
#endif
            abs_grad_phi=this->dmask_dx_local[n]*this->dmask_dx_local[n]+this->dmask_dy_local[n]*this->dmask_dy_local[n];
#ifdef P4_TO_P8
            abs_grad_phi=abs_grad_phi+this->dmask_dz_local[n]*this->dmask_dz_local[n];
#endif
            abs_grad_phi=pow(abs_grad_phi,0.5);

            v0=((4.5/5)*this->Lx/1000.00)*(1.00+MeanField::my_tanh_x(16.00*(y-4.75/5.00*this->Lx)))*this->dmask_dy_local[n]/max(abs_grad_phi,0.01);

            int numberOfBands2ComputeKappa=2;
            if(pow(phi_seed_local_double*phi_seed_local_double,0.5)<dx_for_dt_min*numberOfBands2ComputeKappa)
            {




                kappa_local[n]=this->node_neighbors->neighbors[n].dx_central(this->dmask_dx_local);
                kappa_local[n]+=this->node_neighbors->neighbors[n].dy_central(this->dmask_dy_local);

#ifdef P4_TO_P8
                kappa_local[n]+=this->node_neighbors->neighbors[n].dz_central(this->dmask_dz_local);
#endif


                kappa_local[n]=kappa_local[n]/abs_grad_phi;
            }
            else
            {
                kappa_local[n]=0.01;
            }

            if(this->inverse_design_litography)
            {
                switch (this->my_design_plan->get_my_inverse_litography_velocity())
                {
                case inverse_litography::pressure_velocity:
                {
                    denergy_increase_domain=(1.00)*(wp_local_in[n]);
                    break;
                }
                case inverse_litography::pressure_velocity_with_constraints:
                {
                    switch(this->my_design_plan->my_velocity_stage)
                    {
                    case inverse_litography::optimize_shape_phase:
                        denergy_increase_domain=(1.00)*(wp_local_in[n]);
                        break;
                    case inverse_litography::smooth_shape_phase:
                        denergy_increase_domain=minus_one*this->surface_tension*kappa_local[n];
                        break;
                    }
                    break;
                }
                case inverse_litography::pressure_velocity_with_curvature_constraints:
                {
                    kappa_local_n=kappa_local[n];
                    //if(kappa_local[n]>=0 || wp_local_in[n]>=0)
                    if(ABS(kappa_local_n)<=this->my_design_plan->curvature_barrier
                            || ( kappa_local[n]>=0 && wp_local_in[n]<=0 )
                            || ( kappa_local[n]<=0 && wp_local_in[n]>=0 ) )
                        denergy_increase_domain=(1.00)*(wp_local_in[n]);
                    else
                        denergy_increase_domain=0.00;

                    break;
                }
                case inverse_litography::shape_derivative_velocity:
                {
                    denergy_increase_domain= (minus_one*1/(this->myDiffusion->getV()*this->myDiffusion->getQForward()))
                            *(q_last_local_in[n]);
                    denergy_increase_domain+=  (1/this->myDiffusion->getV())*
                            (-1*wp_local_in[n]+wm_local_in[n]*wm_local_in[n]/this->X_ab);

                    denergy_increase_domain=minus_one*denergy_increase_domain;
                    break;
                }
                }
            }
            else
            {

                switch(this->my_velocity_strategy)
                {
                case MeanField::shape_derivative_velocity:
                {
                    denergy_increase_domain= (minus_one*1/(this->myDiffusion->getV()*this->myDiffusion->getQForward()))
                            *(q_last_local_in[n]);
                    denergy_increase_domain+=  (1/this->myDiffusion->getV())*
                            (-1*wp_local_in[n]+wm_local_in[n]*wm_local_in[n]/this->X_ab);


                    if(!this->conserve_shape_volume)
                    {
                        double v_old=this->myDiffusion->getV();
                        double v2=(v_old*v_old);
                        double e_w_old=this->myDiffusion->getE_w()*this->myDiffusion->getV();

                        denergy_increase_domain+= minus_one*minus_one/v_old;
                        denergy_increase_domain+=minus_one/v2*e_w_old;

                    }
                    denergy_increase_domain=minus_one*denergy_increase_domain;
                    break;
                }
                case MeanField::pressure_velocity:
                {
                    denergy_increase_domain=(1.00)*(wp_local_in[n]);
                    break;
                }
                case MeanField::shape_derivative_with_surface_tension_velocity:
                {
                    denergy_increase_domain= (minus_one*1/(this->myDiffusion->getV()*this->myDiffusion->getQForward()))
                            *(q_last_local_in[n]);
                    denergy_increase_domain+=  (1/this->myDiffusion->getV())*
                            (-1*wp_local_in[n]+wm_local_in[n]*wm_local_in[n]/this->X_ab);

                    if(!this->conserve_shape_volume)
                    {
                        double v_old=this->myDiffusion->getV();
                        double v2=(v_old*v_old);
                        double e_w_old=this->myDiffusion->getE_w()*this->myDiffusion->getV();

                        denergy_increase_domain+= minus_one*minus_one/v_old;
                        denergy_increase_domain+=minus_one/v2*e_w_old;

                    }
                    denergy_increase_domain=minus_one*denergy_increase_domain;
                    denergy_increase_domain+=minus_one*this->surface_tension*kappa_local[n];
                    break;
                }
                case MeanField::pressure_velocity_with_surface_tension_velocity:
                {
                    denergy_increase_domain=(1.00)*(wp_local_in[n]);
                    denergy_increase_domain+=minus_one*this->surface_tension*kappa_local[n];
                    break;
                }
                case MeanField::surface_tension_velocity:
                {
                    denergy_increase_domain=minus_one*this->surface_tension*kappa_local[n];
                    break;
                }
                case MeanField::anti_surface_tension_velocity:
                {
                    denergy_increase_domain=this->surface_tension*kappa_local[n];
                    break;
                }


                }
            }


            if(this->uniform_normal_velocity)
            {
                Vn_local[n]=v0;
            }
            else
            {
                Vn_local[n]=denergy_increase_domain;
            }


        }

        //  delete my_random_generator;
        this->ierr=VecRestoreArray(wm_extended_in,&wm_local_in); CHKERRXX(this->ierr); // restore array +1
        this->ierr=VecRestoreArray(wp_extended_in,&wp_local_in); CHKERRXX(this->ierr);// restore array +2
        this->ierr=VecRestoreArray(fp_extended_in,&fp_local_in); CHKERRXX(this->ierr);// restore array +3
        this->ierr=VecRestoreArray(fm_extended_in,&fm_local_in); CHKERRXX(this->ierr);// restore array +4
        this->ierr=VecRestoreArray(q_last_extended_in,&q_last_local_in); CHKERRXX(this->ierr);// restore array +5
        this->ierr=VecRestoreArray(this->kappa,&kappa_local); CHKERRXX(this->ierr);// restore array +6
        this->ierr=VecRestoreArray(this->phi_seed,&phi_seed_local); CHKERRXX(this->ierr);// restore array +7





        //------------------Set the extended vectors inside wp and wm of the diffusion algo-----------------------------//

        this->ierr=VecCopy(wp_extended_in,this->myDiffusion->wp); CHKERRXX(this->ierr);
        this->ierr=VecCopy(wm_extended_in,this->myDiffusion->wm); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->myDiffusion->wp); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->myDiffusion->wm); CHKERRXX(this->ierr);

        //--------------Finished to det the extended vectors inside wp or wm of the diffusion algo----------------------//



        this->ierr=VecDestroy(wm_extended_in); CHKERRXX(this->ierr);// destroy +1
        this->ierr=VecDestroy(wp_extended_in); CHKERRXX(this->ierr);// destroy +2
        this->ierr=VecDestroy(fp_extended_in); CHKERRXX(this->ierr);// destroy +3
        this->ierr=VecDestroy(fm_extended_in); CHKERRXX(this->ierr);// destroy +4
        this->ierr=VecDestroy(q_last_extended_in); CHKERRXX(this->ierr);// destroy +5

        this->scatter_petsc_vector(&this->kappa);
        this->interpolate_and_print_vec_to_uniform_grid(&this->kappa,"kappa",false);
        this->ierr=VecDestroy(this->kappa); CHKERRXX(this->ierr); // destroy +6




        this->ierr=VecRestoreArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr); // restore array +8
        this->ierr = VecGhostUpdateBegin(this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi_for_advection,"Vn1",PETSC_FALSE);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 4"<<std::endl;

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->delta_phi_for_advection,"delta_phi1",PETSC_FALSE);
        this->compute_lagrange_multiplier_delta_phi();

        if(this->conserve_shape_volume && !this->uniform_normal_velocity)
        {
            this->ierr=VecShift(this->delta_phi_for_advection,-this->lagrange_multiplier); CHKERRXX(this->ierr);

            //            this->ierr=VecRestoreArray(this->delta_phi,&Vn_local); CHKERRXX(this->ierr);
            this->ierr = VecGhostUpdateBegin(this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            //  this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi,"Vn2");
            //this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->delta_phi,"delta_phi2",PETSC_FALSE);
            this->compute_lagrange_multiplier_delta_phi();

        }
        this->ierr=VecGetArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr); // getArray +12

        std::cout<<" start to compute max_norm "<<std::endl;
        double max_norm_u_loc=-10;
        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {
#ifdef P4_TO_P8
            max_norm_u_loc = max(max_norm_u_loc,pow(  Vn_local[n]*  Vn_local[n],0.5));
#else

#endif
        }

        this->ierr=VecRestoreArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr); // restore array +9
        std::cout<<" finish to compute max_norm "<<std::endl;
        double max_norm_u;
        MPI_Allreduce(&max_norm_u_loc, &max_norm_u, 1, MPI_DOUBLE, MPI_MAX, this->p4est->mpicomm);


        double dt_for_advection=1;
        if(!this->uniform_normal_velocity)
            dt_for_advection= min(1.,1./max_norm_u) * this->alpha_cfl * dx_for_dt_min;
        std::cout<<" dt_for_advection "<<dt_for_advection<<std::endl;
        std::cout<<" max dphi [m]"<<dt_for_advection*max_norm_u<<std::endl;
        std::cout<<" max dphi [min_cell]"<<dt_for_advection*max_norm_u/(this->Lx/Nx)<<std::endl;
        this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr);// restore array +10
        this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr);// restore array +11
#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr);// restore array +12
#endif

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" extension_advection_algo_euler 5"<<std::endl;

        this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr); // destroy +7
        this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr);// destroy +8
        this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr); // destroy +9

        // (4) Advect the level set in time using a simple scheme without remeshing
        // Before doing it store phi_seed into phi_seed_old_on_old_p4est

        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr); //duplicate +13
        this->ierr=VecCopy(this->phi_seed,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);




        this->interpolate_and_print_vec_to_uniform_grid(&this->delta_phi_for_advection,"Vn",false);
        switch(this->my_level_set_advection_numerical_scheme)
        {
        case MeanField::euler_advection:
        {
            this->ierr=VecAXPBY(this->phi_seed,-dt_for_advection,1.00,this->delta_phi_for_advection); CHKERRXX(this->ierr);
            break;
        }
        case MeanField::gudonov_advection:
        {
            my_p4est_level_set ls_advecter(this->node_neighbors);
            //            this->ierr=VecScale(this->delta_phi_for_advection,-1.00); CHKERRXX(this->ierr);
            //            this->scatter_petsc_vector(&this->delta_phi_for_advection);
            dt_for_advection=   ls_advecter.advect_in_normal_direction(this->delta_phi_for_advection,this->phi_seed,this->alpha_cfl);
            std::cout<<" dt_for_advection "<<dt_for_advection<<std::endl;
            break;

        }
        case MeanField::semi_lagrangian_advection:
        {
            std::logic_error(" no semilagrangian implemented yet ");//<<std::endl;
            break;
        }

        }
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_after_advection",PETSC_FALSE);


        std::cout<<" initial volume "<<this->source_volume_initial<<" previous volume "<<this->volume_phi_seed<<std::endl;
        std::cout<<" volume source "<<this->compute_volume_source()<<std::endl;



        my_p4est_level_set PLS(this->node_neighbors);
        PLS.reinitialize_2nd_order(this->phi_seed,50);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        //        PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
        //        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        //        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        if(this->conserve_shape_volume)
            this->ensure_volume_conservation();
        double v_source=this->compute_volume_source();
        std::cout<<" volume source "<<v_source<<" volume loss " <<this->source_volume_initial-v_source  << std::endl;



        this->ierr=VecCopy(this->phi_seed,this->polymer_mask); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->polymer_mask); CHKERRXX(this->ierr);


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" extension_advection_algo_euler 6"<<std::endl;



        //----------------------------------PREDICT ENERGY CHANGE------------------------------//

        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed_new_on_old_p4est,"phi_seed_new");
        this->ierr=VecDestroy(this->delta_phi_for_advection); CHKERRXX(this->ierr); // destroy +10



        //if(this->myDiffusion->compute_stress_tensor_bool && this->myDiffusion->my_stress_tensor_computation_mode==StressTensor::shape_derivative)
        {
            this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr); // duplicate +14
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
            this->ierr = VecGhostUpdateBegin(this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed_new_on_old_p4est,"phi_seed_new_after_advection",PETSC_TRUE);


            this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->polymer_mask,"mask_t",PETSC_TRUE);

            this->ierr=VecDuplicate(this->phi_seed_old_on_old_p4est,&this->polymer_shape_stored); CHKERRXX(this->ierr); //duplicate +15
            this->ierr=VecCopy(this->phi_seed_old_on_old_p4est,this->polymer_shape_stored); CHKERRXX(this->ierr);

            this->ierr=VecDuplicate(this->phi_seed,&this->new_polymer_shape_on_old_forest); CHKERRXX(this->ierr); //duplicate +16
            this->ierr=VecCopy(this->phi_seed,this->new_polymer_shape_on_old_forest); CHKERRXX(this->ierr);

            this->scatter_petsc_vector(&this->polymer_shape_stored);
            this->scatter_petsc_vector(&this->new_polymer_shape_on_old_forest);
            this->predict_energy_change_from_polymer_level_set_change();
            this->ierr=VecDestroy(this->polymer_shape_stored); CHKERRXX(this->ierr); //destroy +11
            this->ierr=VecDestroy(this->new_polymer_shape_on_old_forest); CHKERRXX(this->ierr); //destroy +12
            this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr); // destroy +13
        }


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" extension_advection_algo_euler 6"<<std::endl;
        //-----------------------------END OF PREDICTION ENERGY CHANGE--------------------//

        //  destroys all the necessarry staff

        this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);//destroy +14
        this->ierr=VecDestroy(this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr); //destroy +15

        this->ierr=VecDestroy(this->phi_seed); CHKERRXX(this->ierr); //destroy +16


    }
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end seed advection"<<std::endl;



    return 0;
}



int MeanField::seed_advection()
{

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start seed advection"<<std::endl;

    // This algo goes like this
    // Input: phi_seed (phi_seed_old) and velocity on current p4est


    // Output: phi_seed (phi_seed_new) advected on new p4est
    // Data stored for energy changes prediction: phi_seed advected on old p4est


    // We do just advect the seed and map it back to the old/current p4est data structure

    int n_games=1;

    for(int i_game=0; i_game<n_games; i_game++)
    {
        //(1) create level set and parameters for extension

        my_p4est_level_set LS4extension(this->node_neighbors);
        int order_of_extension=2;
        int number_of_bands_to_extend=5;
        int number_of_bands_to_extend_on_all_domain=10;
        // Vecs=0+1=1
        Vec bc_vec_fake;
        this->ierr=VecDuplicate(this->phi_seed,&bc_vec_fake); CHKERRXX(this->ierr); // memory increase local to the function 1
        this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        // (2) Extend cartesian velocities on the current grid from interface to whole domain
        // Vecs=1+3=4
        Vec polymer_velocity_x_extended;
        Vec polymer_velocity_y_extended;
#ifdef P4_TO_P8
        Vec polymer_velocity_z_extended;
#endif

        this->ierr=VecDuplicate(this->phi,&polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory increase local to the function 2
        this->ierr=VecDuplicate(this->phi,&polymer_velocity_y_extended); CHKERRXX(this->ierr); // memory increase local to the function 3

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(this->phi,&polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory increase local to the function 4
#endif

        //        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_x,polymer_velocity_x_extended,number_of_bands_to_extend_on_all_domain);
        //        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_y,polymer_velocity_y_extended,number_of_bands_to_extend_on_all_domain);
        //#ifdef P4_TO_P8
        //        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_z,polymer_velocity_z_extended,number_of_bands_to_extend_on_all_domain);
        //#endif

        this->ierr=VecCopy(this->polymer_velocity_x,polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory increase local to the function 2
        this->ierr=VecCopy(this->polymer_velocity_y,polymer_velocity_y_extended); CHKERRXX(this->ierr); // memory increase local to the function 3

#ifdef P4_TO_P8
        this->ierr=VecCopy(this->polymer_velocity_z,polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory increase local to the function 4
#endif

        this->ierr=VecGhostUpdateBegin(polymer_velocity_x_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(polymer_velocity_x_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(polymer_velocity_y_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(polymer_velocity_y_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_x_extended,"vx_extended",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_y_extended,"vy_extended",PETSC_TRUE);

#ifdef P4_TO_P8
        this->ierr=VecGhostUpdateBegin(polymer_velocity_z_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(polymer_velocity_z_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_z_extended,"vz_extended",PETSC_TRUE);
#endif
        // (3) compute minimum time step

        PetscScalar *vx_extended_array;
        PetscScalar *vy_extended_array;
#ifdef P4_TO_P8
        PetscScalar *vz_extended_array;
#endif
        this->ierr=VecGetArray(polymer_velocity_x_extended,&vx_extended_array); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(polymer_velocity_y_extended,&vy_extended_array); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecGetArray(polymer_velocity_z_extended,&vz_extended_array); CHKERRXX(this->ierr);
#endif
        double max_norm_u_loc = 0;
        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {
#ifdef P4_TO_P8
            max_norm_u_loc = max(max_norm_u_loc, sqrt(   vx_extended_array[n]*vx_extended_array[n]
                                                         + vy_extended_array[n]*vy_extended_array[n]
                                                         + vz_extended_array[n]*vz_extended_array[n]));
#else
            max_norm_u_loc = max(max_norm_u_loc, sqrt(   vx_extended_array[n]*vx_extended_array[n]
                                                         + vy_extended_array[n]*vy_extended_array[n]
                                                         ));

#endif
        }

        double max_norm_u;
        MPI_Allreduce(&max_norm_u_loc, &max_norm_u, 1, MPI_DOUBLE, MPI_MAX, this->p4est->mpicomm);

        double dx_for_dt_min = 1.0 / pow(2.,(double) this->max_level);

        double dt_for_advection = min(1.,1./max_norm_u) * .5 * dx_for_dt_min;

        //dt_for_advection=0.01;

        this->ierr=VecRestoreArray(polymer_velocity_x_extended,&vx_extended_array);    CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(polymer_velocity_y_extended,&vy_extended_array);    CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(polymer_velocity_z_extended,&vz_extended_array);    CHKERRXX(this->ierr);
#endif

        // (4) Advect the level set in time using the semi lagrangian operator

        // definition of local p4est variables for remeshing up to the advected mask

        // p4est=1
        p4est_t *p4est_np1 = p4est_copy(this->p4est, P4EST_FALSE);
        // ghost=1
        p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        // nodes=1
        p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        // IMPORTANT NOTE: later on override semi lagrangian such that
        // the advection can be done on one level set while the splitting criteria
        // can be done with the mask interface but also together with the statistical fields

        // we will override update_p4est_second_order_uniform_inside_domain
        // and add as arguments wp,wm
        // and follow the comments inside this function

        // however since we advect an adaptive grid and the potentials
        // do not move much in one men field iteration
        // It has to be checked if the grid after the advection stays adaptive and
        // can easily recover to the potential segmentation grid

        // it can be done with a mix of advection for the polymer mask
        // and interpolation for the statistical fields

        // we note the advection has been done previously on semilagrangian
        // and interpolation for statistical fields is in current use on this
        // class


        // Before doing it store phi_seed into phi_seed_old_on_old_p4est

        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->phi_seed,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        SemiLagrangian sl(&p4est_np1, &nodes_np1, &ghost_np1, this->brick);

#ifdef P4_TO_P8
        sl.update_p4est_second_order_uniform_inside_domain(polymer_velocity_x_extended,polymer_velocity_y_extended,polymer_velocity_z_extended,
                                                           dt_for_advection,this->phi_seed);
#else
        sl.update_p4est_second_order_uniform_inside_domain(polymer_velocity_x_extended,polymer_velocity_y_extended,
                                                           dt_for_advection,this->phi_seed);
#endif



        this->ierr = VecGhostUpdateBegin(this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);







        //----------------------------------PREDICT ENERGY CHANGE------------------------------//


        // We will compute/ predict what should be the energy change up to the change of level set
        // To test it the scft fields will not be evolved in the next time step to ensure we
        // have computed the right variation.

        // (a) Compute phi_seed on the current p4est data structure
        // Receives as input the new geometry data structure.
        // phi_seed is on the new data structure
        // we still have as class variables the old one.

        // (b) predict the energy change due to the level set variation
        //     For that compute dphi(x)=phi_new-phi_old

        int n_iterations_to_reinitialyze_level_set=100;
        this->remapFromNewForestToOldForest(p4est_np1, nodes_np1, ghost_np1, this->brick,&this->phi_seed_old_on_old_p4est,&this->phi_seed, &this->phi_seed_new_on_old_p4est);
        my_p4est_level_set PLS(this->node_neighbors);
        PLS.reinitialize_2nd_order(this->phi_seed_new_on_old_p4est,n_iterations_to_reinitialyze_level_set);

        this->ierr = VecGhostUpdateBegin(this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed_new_on_old_p4est,"phi_seed_new");

        this->predict_energy_change_from_level_set_change();

        // We destruct non required data
        this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->phi_seed_old_on_old_p4est);


        //-----------------------------END OF PREDICTION ENERGY CHANGE--------------------//

        //--------Remap field of interest from the old data structure on the new data structure-------------//

        // Reamp the polymer mask in the iregular domain case

        if(!this->myMeanFieldPlan->periodic_xyz)
        {

            Vec polymer_mask_temp;
            this->remapFromOldForestToNewForest(p4est_np1,nodes_np1,ghost_np1,this->brick,&this->polymer_mask,&polymer_mask_temp);
            this->ierr=VecDestroy(this->polymer_mask); CHKERRXX(this->ierr);
            this->ierr=VecDuplicate(polymer_mask_temp,&this->polymer_mask); CHKERRXX(this->ierr);
            this->ierr=VecCopy(polymer_mask_temp,this->polymer_mask); CHKERRXX(this->ierr);
            this->ierr = VecGhostUpdateBegin(this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->polymer_mask, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            this->ierr=VecDestroy(polymer_mask_temp); CHKERRXX(this->ierr);

        }


        //-------End of remapping fields----------------------------------------//


        // (7) swap forests
        // nodes=1-1=0
        p4est_nodes_destroy(this->nodes);
        // ghost=1-1=0
        p4est_ghost_destroy(this->ghost);
        // p4est=1-1=0
        p4est_destroy(this->p4est);

        // hierarchy=1-1=0
        delete this->hierarchy;
        // node_neighbors=1-1=0;
        delete this->node_neighbors;

        // p4est=0+1=1
        // nodes=0+1=1
        // ghost=0+1=1
        // hierarchy=0+1=1
        // node_neighbors=0+1=1

        this->p4est=p4est_np1;
        this->nodes=nodes_np1;
        this->ghost=ghost_np1;

        my_p4est_hierarchy_t  *hierarchy_temp=new my_p4est_hierarchy_t(this->p4est,this->ghost,this->brick);
        this->hierarchy=hierarchy_temp;

        my_p4est_node_neighbors_t *node_neighbors_temp=new my_p4est_node_neighbors_t(this->hierarchy,this->nodes,this->myMeanFieldPlan->periodic_xyz);
        this->node_neighbors=node_neighbors_temp;

        this->node_neighbors->init_neighbors();


        // plot the remapped fields with the new data tructures
        // using .vtk files on paraview


        if(!this->myMeanFieldPlan->periodic_xyz)
            this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"mask",PETSC_TRUE);



        // (8) destroys all the necessarry staff

        // vecs=6-6=0
        //this->ierr=VecDestroy(wp_temp);CHKERRXX(this->ierr); // memory decrease local to the function 5
        //this->ierr=VecDestroy(wm_temp);CHKERRXX(this->ierr);// memory decrease local to the function 6
        this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr); // memory decrease local to the function 1
        this->ierr=VecDestroy(polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory decrease local to the function 2
        this->ierr=VecDestroy(polymer_velocity_y_extended); CHKERRXX(this->ierr);// memory decrease local to the function 3

#ifdef P4_TO_P8
        this->ierr=VecDestroy(polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory decrease local to the function 4

#endif
    }

    //The velocities are constructed in another function
    this->ierr=VecDestroy(this->polymer_velocity_x); CHKERRXX(this->ierr); // memory decrease global to the object
    this->ierr=VecDestroy(this->polymer_velocity_y); CHKERRXX(this->ierr); // memory decrease global to the object
    this->ierr=VecDestroy(this->polymer_velocity_z); CHKERRXX(this->ierr); // memory decrease global to the object

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end seed advection"<<std::endl;



    return 0;
}


int MeanField::advect_internal_interface()
{

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start seed advection"<<std::endl;
    // We do just advect the seed and map it back to the old/current p4est data structure

    int n_games=1;

    for(int i_game=0; i_game<n_games; i_game++)
    {
        //(1) create level set and parameters for extension
        my_p4est_level_set LS4extension(this->node_neighbors);
        int order_of_extension=2;
        int number_of_bands_to_extend=5;
        int number_of_bands_to_extend_on_all_domain=10;
        // Vecs=0+1=1
        Vec bc_vec_fake;
        this->ierr=VecDuplicate(this->phi_seed,&bc_vec_fake); CHKERRXX(this->ierr); // memory increase local to the function 1
        this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        // (2) Extend cartesian velocities on the current grid from interface to whole domain
        // Vecs=1+3=4
        Vec polymer_velocity_x_extended;
        Vec polymer_velocity_y_extended;
#ifdef P4_TO_P8
        Vec polymer_velocity_z_extended;
#endif

        this->ierr=VecDuplicate(this->phi,&polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory increase local to the function 2
        this->ierr=VecDuplicate(this->phi,&polymer_velocity_y_extended); CHKERRXX(this->ierr); // memory increase local to the function 3

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(this->phi,&polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory increase local to the function 4
#endif

        //        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_x,polymer_velocity_x_extended,number_of_bands_to_extend_on_all_domain);
        //        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_y,polymer_velocity_y_extended,number_of_bands_to_extend_on_all_domain);
        //#ifdef P4_TO_P8
        //        LS4extension.extend_from_interface_to_whole_domain(this->polymer_mask,this->polymer_velocity_z,polymer_velocity_z_extended,number_of_bands_to_extend_on_all_domain);
        //#endif

        this->ierr=VecCopy(this->polymer_velocity_x,polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory increase local to the function 2
        this->ierr=VecCopy(this->polymer_velocity_y,polymer_velocity_y_extended); CHKERRXX(this->ierr); // memory increase local to the function 3

#ifdef P4_TO_P8
        this->ierr=VecCopy(this->polymer_velocity_z,polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory increase local to the function 4
#endif

        this->ierr=VecGhostUpdateBegin(polymer_velocity_x_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(polymer_velocity_x_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(polymer_velocity_y_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(polymer_velocity_y_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);



        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_x_extended,"vx_extended",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_y_extended,"vy_extended",PETSC_TRUE);


#ifdef P4_TO_P8

        this->my_p4est_vtk_write_all_periodic_adapter(&polymer_velocity_z_extended,"vz_extended",PETSC_TRUE);
        this->ierr=VecGhostUpdateBegin(polymer_velocity_z_extended,INSERT_VALUES,SCATTER_FORWARD);     CHKERRXX(this->ierr);
        this->ierr= VecGhostUpdateEnd(polymer_velocity_z_extended,INSERT_ALL_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
#endif

        this->compute_normal_to_the_interface(&this->phi_seed);

        this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr);
#endif

        // (3) compute minimum time step and normal velocity

        Vec Vn;
        this->ierr=VecDuplicate(this->phi,&Vn); CHKERRXX(this->ierr);
        PetscScalar *Vn_local;
        this->ierr=VecGetArray(Vn,&Vn_local); CHKERRXX(this->ierr);

        PetscScalar *vx_extended_array;
        PetscScalar *vy_extended_array;
#ifdef P4_TO_P8
        PetscScalar *vz_extended_array;
#endif
        this->ierr=VecGetArray(polymer_velocity_x_extended,&vx_extended_array); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(polymer_velocity_y_extended,&vy_extended_array); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecGetArray(polymer_velocity_z_extended,&vz_extended_array); CHKERRXX(this->ierr);
#endif
        double max_norm_u_loc = 0;
        double v0=1;
        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {
#ifdef P4_TO_P8
            //            max_norm_u_loc = max(max_norm_u_loc, sqrt(   vx_extended_array[n]*vx_extended_array[n]
            //                                                         + vy_extended_array[n]*vy_extended_array[n]
            //                                                         + vz_extended_array[n]*vz_extended_array[n]));
            max_norm_u_loc = max(max_norm_u_loc, v0);


            Vn_local[n]=v0;
            /*(vx_extended_array[n]*this->dmask_dx_local[n]
                    + vy_extended_array[n]*this->dmask_dy_local[n]
                    + vz_extended_array[n]*this->dmask_dz_local[n])/
                    (this->dmask_dx_local[n]*this->dmask_dx_local[n]
                       + this->dmask_dy_local[n]*this->dmask_dy_local[n]
                       + this->dmask_dz_local[n]*this->dmask_dz_local[n]);*/
#else

#endif
        }



        this->myDiffusion->printDiffusionArrayFromVector(&Vn,"Vn");

        double max_norm_u;
        MPI_Allreduce(&max_norm_u_loc, &max_norm_u, 1, MPI_DOUBLE, MPI_MAX, this->p4est->mpicomm);

        double dx_for_dt_min = 1.0 / pow(2.,(double) this->max_level);

        double dt_for_advection = min(1.,1./max_norm_u) * .5 * dx_for_dt_min;

        //dt_for_advection=0.01;

        this->ierr=VecRestoreArray(polymer_velocity_x_extended,&vx_extended_array);    CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(polymer_velocity_y_extended,&vy_extended_array);    CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(polymer_velocity_z_extended,&vz_extended_array);    CHKERRXX(this->ierr);
#endif

        this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(this->dmask_dz,&vz_extended_array); CHKERRXX(this->ierr);
#endif

        this->ierr=VecRestoreArray(Vn,&Vn_local); CHKERRXX(this->ierr);

        this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr);

        // (4) Advect the level set in time using a simple scheme without remeshing
        // Before doing it store phi_seed into phi_seed_old_on_old_p4est

        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);

        this->ierr=VecCopy(this->phi_seed,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        double dt_min;
        int n_advections=1;
        for(int t_a=0;t_a<n_advections;t_a++)
        {
            dt_min=LS4extension.advect_in_normal_direction(Vn,this->phi_seed,this->alpha_cfl);

            std::cout<<" dt_min"<<dt_min<<std::endl;
        }
        this->ierr=VecCopy(this->phi_seed,this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        this->ierr = VecGhostUpdateBegin(this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


        //----------------------------------PREDICT ENERGY CHANGE------------------------------//

        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed_new_on_old_p4est,"phi_seed_new");

        this->predict_energy_change_from_level_set_change();

        // We destruct non required data
        this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);

        this->ierr=VecDestroy(Vn); CHKERRXX(this->ierr);

        //-----------------------------END OF PREDICTION ENERGY CHANGE--------------------//





        // (8) destroys all the necessarry staff

        // vecs=6-6=0
        //this->ierr=VecDestroy(wp_temp);CHKERRXX(this->ierr); // memory decrease local to the function 5
        //this->ierr=VecDestroy(wm_temp);CHKERRXX(this->ierr);// memory decrease local to the function 6
        this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr); // memory decrease local to the function 1
        this->ierr=VecDestroy(polymer_velocity_x_extended); CHKERRXX(this->ierr); // memory decrease local to the function 2
        this->ierr=VecDestroy(polymer_velocity_y_extended); CHKERRXX(this->ierr);// memory decrease local to the function 3
#ifdef P4_TO_P8
        this->ierr=VecDestroy(polymer_velocity_z_extended); CHKERRXX(this->ierr);// memory decrease local to the function 4
#endif
    }

    //The velocities are constructed in another function
    this->ierr=VecDestroy(this->polymer_velocity_x); CHKERRXX(this->ierr); // memory decrease global to the object
    this->ierr=VecDestroy(this->polymer_velocity_y); CHKERRXX(this->ierr); // memory decrease global to the object
    this->ierr=VecDestroy(this->polymer_velocity_z); CHKERRXX(this->ierr); // memory decrease global to the object

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end seed advection"<<std::endl;



    return 0;
}


int MeanField::change_internal_interface()
{



    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 0"<<std::endl;
    // We do just advect the seed and map it back to the old/current p4est data structure



    for(int i_game=0; i_game<this->n_change_internal_interfaces; i_game++)
    {
        //(1) create level set and parameters for extension
        my_p4est_level_set LS4extension(this->node_neighbors);
        int order_of_extension=2;
        int number_of_bands_to_extend=5;
        int number_of_bands_to_extend_on_all_domain=10;
        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr); // memory increase local to the function 15 phi_seed_old_on_old_p4est: total 3
        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);// memory increase local to the function 16 phi_seed_new_on_old_p4est: total 4

        this->ierr=VecCopy(this->phi_seed,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        // Vecs=0+1=1
        Vec bc_vec_fake;
        this->ierr=VecDuplicate(this->phi_seed,&bc_vec_fake); CHKERRXX(this->ierr); // memory increase local to the function 1 bc_vec_fake: total 1
        this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 1"<<std::endl;

        this->scatter_petsc_vector(&bc_vec_fake);
        this->compute_normal_to_the_interface(&this->phi_seed);                  // memory increase global to the code 2,3,4 normal: dmask_dx,dmask_dy,dmask_dz: total 4

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 2"<<std::endl;

        this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr);
#endif

        // (3) compute minimum time step and normal velocity


        PetscScalar *Vn_local;
        this->ierr=VecDuplicate(this->phi_seed,&this->delta_phi_for_advection); CHKERRXX(this->ierr); // memory increase local to the function 5 delta phi: total 5
        this->ierr=VecGetArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr);

        //double max_norm_u_loc = 0;
        double v0=0.01;


        Vec wm_extended_in,wp_extended_in,fp_extended_in,fm_extended_in;
        Vec wm_extended_out,wp_extended_out,fp_extended_out,fm_extended_out;

        this->ierr=VecDuplicate(this->phi,&wm_extended_in); CHKERRXX(this->ierr);   // memory increase local to the function 6 wm_extended_in: total 6
        this->ierr=VecDuplicate(this->phi,&wp_extended_in); CHKERRXX(this->ierr);   // memory increase local to the function 7 wp_extended_in: total 7
        this->ierr=VecDuplicate(this->phi,&fm_extended_in); CHKERRXX(this->ierr);   // memory increase local to the function 8 fm_extended_in: total 8
        this->ierr=VecDuplicate(this->phi,&fp_extended_in); CHKERRXX(this->ierr);   // memory increase local to the function 9 fp_extended_in: total 9

        this->ierr=VecCopy(this->myDiffusion->wm,wm_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->myDiffusion->wp,wp_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fm_stored,fm_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fp_stored,fp_extended_in); CHKERRXX(this->ierr);

        this->ierr=VecDuplicate(this->phi,&wm_extended_out); CHKERRXX(this->ierr);  // memory increase local to the function 10 wm_extended_out: total 10
        this->ierr=VecDuplicate(this->phi,&wp_extended_out); CHKERRXX(this->ierr);  // memory increase local to the function 11 wp_extended_out: total 11
        this->ierr=VecDuplicate(this->phi,&fm_extended_out); CHKERRXX(this->ierr);  // memory increase local to the function 12 fm_extended_out: total 12
        this->ierr=VecDuplicate(this->phi,&fp_extended_out); CHKERRXX(this->ierr);  // memory increase local to the function 13 fp_extended_out: total 13

        this->ierr=VecCopy(this->myDiffusion->wm,wm_extended_out); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->myDiffusion->wp,wp_extended_out); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fm_stored,fm_extended_out); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fp_stored,fp_extended_out); CHKERRXX(this->ierr);



        bool extend=true;

        if(this->advance_fields_scft_advance_mask_level_set)
            extend=false;

        if(extend)
        {

            this->extend_petsc_vector(&this->phi_seed,&wm_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&wp_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&fm_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&fp_extended_in);


            this->ierr=VecDuplicate(this->phi_seed,&this->negative_phi_seed); CHKERRXX(this->ierr); // memory increase local to the function 14 negative_phi_seed: total 14

            this->ierr=VecSet(this->negative_phi_seed,0.00); CHKERRXX(this->ierr);
            this->ierr=VecAXPY(this->negative_phi_seed,-1.00,this->phi_seed);   CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateBegin(this->negative_phi_seed,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->negative_phi_seed,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

            this->extend_petsc_vector(&this->negative_phi_seed,&wm_extended_out);
            this->extend_petsc_vector(&this->negative_phi_seed,&wp_extended_out);
            this->extend_petsc_vector(&this->negative_phi_seed,&fm_extended_out);
            this->extend_petsc_vector(&this->negative_phi_seed,&fp_extended_out);

            this->ierr=VecDestroy(this->negative_phi_seed); CHKERRXX(this->ierr); // memory decrease local to the function 14 negative_phi_seed: total 13


            this->my_p4est_vtk_write_all_periodic_adapter(&fp_extended_in,"fp_extended_in",  this->write2VtkShapeOptimization);
            this->my_p4est_vtk_write_all_periodic_adapter(&fp_extended_out,"fp_extended_out",  this->write2VtkShapeOptimization);

            this->my_p4est_vtk_write_all_periodic_adapter(&fm_extended_in,"fm_extended_in",  this->write2VtkShapeOptimization);
            this->my_p4est_vtk_write_all_periodic_adapter(&fm_extended_out,"fm_extended_out",  this->write2VtkShapeOptimization);

        }
        else
        {
            this->scatter_petsc_vector(&wm_extended_in);
            this->scatter_petsc_vector(&wp_extended_in);
            this->scatter_petsc_vector(&fm_extended_in);
            this->scatter_petsc_vector(&fp_extended_in);
            this->scatter_petsc_vector(&wm_extended_out);
            this->scatter_petsc_vector(&wp_extended_out);
            this->scatter_petsc_vector(&fm_extended_out);
            this->scatter_petsc_vector(&fp_extended_out);
        }


        if(this->pressure_gradient_velocity)
        {
            Vec wp_x,wp_y
        #ifdef P4_TO_P8
                    ,wp_z
        #endif
                    ;
            PetscScalar *wp_x_local,*wp_y_local,
        #ifdef P4_TO_P8
                    *wp_z_local,
        #endif
                    *wp_local;


            this->ierr=VecDuplicate(wp_extended_in,&wp_x); CHKERRXX(this->ierr);
            this->ierr=VecDuplicate(wp_extended_in,&wp_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
            this->ierr=VecDuplicate(wp_extended_in,&wp_z); CHKERRXX(this->ierr);
#endif

            this->ierr=VecGetArray(wp_extended_in,&wp_local); CHKERRXX(this->ierr);
            this->ierr=VecGetArray(wp_x,&wp_x_local); CHKERRXX(this->ierr);
            this->ierr=VecGetArray(wp_y,&wp_y_local); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
            this->ierr=VecGetArray(wp_z,&wp_z_local); CHKERRXX(this->ierr);
#endif



            for(int i=0;i<this->nodes->num_owned_indeps;i++)
            {
                wp_x_local[i]=this->node_neighbors->neighbors[i].dx_central(wp_local);
                wp_y_local[i]=this->node_neighbors->neighbors[i].dy_central(wp_local);
#ifdef P4_TO_P8
                wp_z_local[i]=this->node_neighbors->neighbors[i].dz_central(wp_local);
#endif
            }


            this->ierr=VecRestoreArray(wp_extended_in,&wp_local); CHKERRXX(this->ierr);
            this->ierr=VecRestoreArray(wp_x,&wp_x_local); CHKERRXX(this->ierr);
            this->ierr=VecRestoreArray(wp_y,&wp_y_local); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
            this->ierr=VecRestoreArray(wp_z,&wp_z_local); CHKERRXX(this->ierr);
#endif


            this->ierr=VecPointwiseMult(wp_x,this->dmask_dx,wp_x); CHKERRXX(this->ierr);
            this->ierr=VecPointwiseMult(wp_y,this->dmask_dy,wp_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
            this->ierr=VecPointwiseMult(wp_z,this->dmask_dz,wp_z); CHKERRXX(this->ierr);
#endif

            this->ierr=VecAXPY(wp_extended_in,-1.00,wp_x); CHKERRXX(this->ierr);
            this->ierr=VecAXPY(wp_extended_in,-1.00,wp_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
            this->ierr=VecAXPY(wp_extended_in,-1.00,wp_x); CHKERRXX(this->ierr);
#endif

            this->scatter_petsc_vector(&wp_extended_in);

            this->ierr=VecDestroy(wp_x); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(wp_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
            this->ierr=VecDestroy(wp_z); CHKERRXX(this->ierr);
#endif

        }

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 3"<<std::endl;

        PetscScalar *wm_local_in;
        PetscScalar *wp_local_in;
        PetscScalar *fp_local_in;
        PetscScalar *fm_local_in;

        PetscScalar *wm_local_out;
        PetscScalar *wp_local_out;
        PetscScalar *fp_local_out;
        PetscScalar *fm_local_out;



        this->ierr=VecGetArray(wm_extended_in,&wm_local_in); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(wp_extended_in,&wp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fp_extended_in,&fp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fm_extended_in,&fm_local_in); CHKERRXX(this->ierr);


        this->ierr=VecGetArray(wm_extended_out,&wm_local_out); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(wp_extended_out,&wp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fp_extended_out,&fp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fm_extended_out,&fm_local_out); CHKERRXX(this->ierr);



        double acos_argument,denominator,teta;
        double polymer_mask_xc=this->Lx/2.00;
        double polymer_mask_yc=this->Lx/2.00;
        double polymer_mask_zc=this->Lx/2.00;
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        double dx_for_dt_min =this->Lx / (double)Nx;

        // v0=0.05;
        PetscScalar *phi_seed_local;
        this->ierr=VecGetArray(this->phi_seed,&phi_seed_local); CHKERRXX(this->ierr);
        double phi_seed_local_double,denergy_increase_domain;
        double a_p,a_m;

        //a_p=1.00/(0.5*this->X_ab*this->zeta_n_inverse+1.00);
        a_p=(this->zeta_n_inverse*this->interaction_flag*this->xhi_w_p+1.00)/(0.5*this->X_ab*this->zeta_n_inverse+1.00);
        a_m=this->interaction_flag*2*this->xhi_w_m/this->X_ab;

        if(this->terracing)
        {
            a_p=1.00/(0.5*this->X_ab*this->zeta_n_inverse+1.00);
            a_m=0;
        }


        //        double exp_plus_phi;
        //        double exp_minus_phi;
        double dH_dphi_wall;
        double dphi_wall_dphi;
        //            double exp_plus_phi;
        //            double exp_minus_phi;
        double tanh_phi;
        double tanh_phi_1,  tanh_phi_2;
        double tanh_phi_terrace;
        double dphi_trial;
        double phi_seed_local_double_trial;

        double de_p,de_m;
        double phi_w_y;
        double phi_w_1;
        double phi_w_2;
        double phi_w_s_1;
        double phi_w_s_2;


        // This for loop is very innefficient:
        // when time rewrite it in an optimized way:



        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {
            phi_seed_local_double=(double)phi_seed_local[n];
            if(pow(phi_seed_local_double*phi_seed_local_double,0.5)<dx_for_dt_min*2 || !this->velocity_at_interface_only)
            {
                if(this->uniform_normal_velocity)
                {
                    Vn_local[n]=v0;
                }
                else
                {
                    if(this->advance_fields_level_set)
                        denergy_increase_domain=(fp_local_in[n]);

                    if(this->advance_fields_scft_advance_mask_level_set)
                    {


                        phi_seed_local_double=phi_seed_local[n];
                        dH_dphi_wall=wp_local_in[n]*a_p+wm_local_in[n]*a_m;
                        //                        exp_plus_phi=exp(this->alpha_wall*phi_seed_local_double);
                        //                        exp_minus_phi=exp(-this->alpha_wall*phi_seed_local_double);
                        tanh_phi=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
                        tanh_phi=tanh_phi*tanh_phi;
                        if(this->volumetric_derivative)
                            dphi_wall_dphi=0.5*this->alpha_wall*(1.00-tanh_phi);
                        else
                            dphi_wall_dphi=1.00;
                        denergy_increase_domain=dH_dphi_wall*dphi_wall_dphi;

                        if(this->terracing)
                        {

                            PetscBool correcting_option=PETSC_TRUE;

                            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
                            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
                            double tree_ymin = connectivity->vertices[3*v_mm + 1];
                            double y = node_y_fr_j(node) + tree_ymin;

                            tanh_phi_terrace=MeanField::my_tanh_x(this->alpha_wall*(this->y_terrace_floor-y));
                            tanh_phi_1=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
                            phi_w_y=0.5*(1+tanh_phi_terrace);
                            phi_w_s_1=0.5*(1+tanh_phi_1);
                            phi_w_1=MIN(1.00,phi_w_y+phi_w_s_1);



                            if(correcting_option)
                            {
                                if(  tanh_phi_1+tanh_phi_terrace>0)
                                {

                                    // we are either deep in the domain or close to the level set:: the seed one


                                    // if deep outside it doesn't matter
                                    denergy_increase_domain=0;

                                    // if close and we get away it doesn't matter too: thats why we check only decrease of the level set function
                                    // if close and we increase energy by coming it doesn't matter too except for volume constraints
                                    // if close and we decrease energy by coming we want to come

                                    dphi_trial=-dx_for_dt_min*this->alpha_cfl;
                                    phi_seed_local_double_trial=phi_seed_local_double+dphi_trial;
                                    tanh_phi_2=MeanField::my_tanh_x(this->alpha_wall*(phi_seed_local_double_trial));
                                    if(tanh_phi_2+tanh_phi_terrace<0)
                                    {
                                        denergy_increase_domain=0.5*(tanh_phi_2+tanh_phi_terrace)*dH_dphi_wall/dphi_trial;
                                    }
                                }
                            }
                            else
                            {



                                tanh_phi_1=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
                                tanh_phi_terrace=MeanField::my_tanh_x(this->alpha_wall*(this->y_terrace_floor-y));
                                phi_w_y=0.5*(1+tanh_phi_terrace);
                                phi_w_s_1=0.5*(1+tanh_phi_1);
                                phi_w_1=MIN(1.00,phi_w_y+phi_w_s_1);


                                dphi_trial=-dx_for_dt_min*this->alpha_cfl;
                                phi_seed_local_double_trial=phi_seed_local_double+dphi_trial;
                                tanh_phi_2=MeanField::my_tanh_x(this->alpha_wall*(phi_seed_local_double_trial));
                                phi_w_s_2=0.5*(1+tanh_phi_2);
                                phi_w_2=MIN(1.00,phi_w_y+phi_w_s_2);
                                de_m=(phi_w_2-phi_w_1)*dH_dphi_wall;


                                dphi_trial=dx_for_dt_min*this->alpha_cfl;
                                phi_seed_local_double_trial=phi_seed_local_double+dphi_trial;
                                tanh_phi_2=MeanField::my_tanh_x(this->alpha_wall*(phi_seed_local_double_trial));
                                phi_w_s_2=0.5*(1+tanh_phi_2);
                                phi_w_2=MIN(1.00,phi_w_y+phi_w_s_2);
                                de_p=(phi_w_2-phi_w_1)*dH_dphi_wall;


                                //                            if(de_p!=0)
                                //                                std::cout<<"hi";

                                denergy_increase_domain=0;
                                if(de_p<de_m)
                                {
                                    //we want a final positive velocity so a negative velocity now
                                    denergy_increase_domain=-(de_m-de_p)/(2.00*dphi_trial);
                                }
                                if(de_m<de_p)
                                {
                                    //we want a final negative velocity so a positive velocity now
                                    denergy_increase_domain=(de_p-de_m)/(2.00*dphi_trial);
                                }
                            }
                        }

                    }
                    Vn_local[n]=denergy_increase_domain;

                    if(!this->conserve_shape_volume || !this->conserve_reaction_source_volume)
                    {
                        if(this->volumetric_derivative)
                            this->non_conservation_scalar_term=dphi_wall_dphi*log(this->myDiffusion->getQForward());
                        if(!this->volumetric_derivative)
                            this->non_conservation_scalar_term=1*log(this->myDiffusion->getQForward());

                        Vn_local[n]+=this->non_conservation_scalar_term;
                    }

                    if(denergy_increase_domain>0)
                    {
                        double wpl=wp_local_in[n];
                        double wml=wm_local_in[n];
                    }

                    if(denergy_increase_domain<0)
                    {
                        double wpl=wp_local_in[n];
                        double wml=wm_local_in[n];
                    }
                }

            }
            else
            {
                Vn_local[n]=0;
            }
        }

        this->ierr=VecRestoreArray(wm_extended_in,&wm_local_in); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(wp_extended_in,&wp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fp_extended_in,&fp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fm_extended_in,&fm_local_in); CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(wm_extended_out,&wm_local_out); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(wp_extended_out,&wp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fp_extended_out,&fp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fm_extended_out,&fm_local_out); CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(this->phi_seed,&phi_seed_local); CHKERRXX(this->ierr);


        this->ierr=VecDestroy(wm_extended_in); CHKERRXX(this->ierr);// memory decrease local to the function 6 wm_extended_in: total 12
        this->ierr=VecDestroy(wp_extended_in); CHKERRXX(this->ierr);// memory decrease local to the function 7 wp_extended_in: total 11
        this->ierr=VecDestroy(fp_extended_in); CHKERRXX(this->ierr);// memory decrease local to the function 8 fp_extended_in: total 10
        this->ierr=VecDestroy(fm_extended_in); CHKERRXX(this->ierr);// memory decrease local to the function 9 fm_extended_in: total 9


        this->ierr=VecDestroy(wm_extended_out); CHKERRXX(this->ierr);// memory decrease local to the function 10 wm_extended_out: total 8
        this->ierr=VecDestroy(wp_extended_out); CHKERRXX(this->ierr);// memory decrease local to the function 11 wp_extended_out: total 7
        this->ierr=VecDestroy(fp_extended_out); CHKERRXX(this->ierr);// memory decrease local to the function 12 fp_extended_out: total 6
        this->ierr=VecDestroy(fm_extended_out); CHKERRXX(this->ierr);// memory decrease local to the function 13 fm_extended_out: total 5



        this->ierr=VecRestoreArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateBegin(this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);




        int n_global;
        this->ierr=VecGetSize(this->delta_phi_for_advection,&n_global); CHKERRXX(this->ierr);


        //----------------------Compute Energy Change: Should be a huge negative number-------------------------//
        if(this->advance_fields_scft_advance_mask_level_set)
        {
            std::cout<<this->mpi->mpirank<<" Compute Energy Change Before Lagrange : Should be a huge negative number "<<std::endl;
            this->ierr=VecSet(this->phi_seed_new_on_old_p4est,0.00); CHKERRXX(this->ierr);
            this->ierr=VecAXPY(this->phi_seed_new_on_old_p4est,1.00,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);
            this->ierr=VecAXPY(this->phi_seed_new_on_old_p4est,-1.00,this->delta_phi_for_advection); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_new_on_old_p4est);
            this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed_new_on_old_p4est,"phi_seed_new_on_old_p4est");
            this->predict_energy_change_from_level_set_change();
            this->de_predicted_from_level_set_change_current_step_before_lagrange_t=this->de_predicted_from_level_set_change_current_step;
            std::cout<<this->mpi->mpirank<<"Finish Compute Energy Change Before Lagrange : Should be a huge negative number "<<
                       this->de_predicted_from_level_set_change_current_step_before_lagrange_t<<std::endl;

            this->compute_force_delta_phi();
            this->dphi_step_1[mask_counter]=this->shape_force_t;

        }
        //---------------------Finish to Compute Energy Change------------------------------------------------//




        // this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi,"Vn1",PETSC_TRUE);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 4"<<std::endl;

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->delta_phi_for_advection,"DHDPHI",  this->write2VtkShapeOptimization);
        this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi_for_advection,"DHDPHI",PETSC_FALSE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wp,"wp_phi",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"level_set",PETSC_TRUE);
        if(!this->terracing)
            this->compute_lagrange_multiplier_delta_phi();
        else
            this->compute_lagrange_multiplier_delta_phi_terracing();


        if(this->conserve_reaction_source_volume&& mod(this->mask_counter,mask_conservation_period)==0)
        {
            if(!this->terracing )
            {

                if(this->use_petsc_shift)
                {
                    this->ierr=VecShift(this->delta_phi_for_advection,-this->lagrange_multiplier); CHKERRXX(this->ierr);
                }
                else
                {
                    this->ierr=VecGetArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr);
                    this->ierr=VecGetArray(this->phi_seed,&phi_seed_local); CHKERRXX(this->ierr);

                    for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
                    {
                        phi_seed_local_double=(double)phi_seed_local[n];
                        if(pow(phi_seed_local_double*phi_seed_local_double,0.5)<dx_for_dt_min*2 || !this->velocity_at_interface_only)
                        {

                            Vn_local[n]=Vn_local[n]-this->lagrange_multiplier;

                        }
                        else
                        {
                            Vn_local[n]=0;
                        }
                    }

                    this->ierr=VecRestoreArray(this->delta_phi_for_advection,&Vn_local);  CHKERRXX(this->ierr);
                    this->ierr=VecRestoreArray(this->phi_seed,&phi_seed_local); CHKERRXX(this->ierr);
                }
            }
            //            if(this->terracing )
            //            {

            //                this->ierr=VecShift(this->delta_phi_for_advection,-this->lagrange_multiplier); CHKERRXX(this->ierr);
            //            }


            //            this->ierr=VecRestoreArray(this->delta_phi,&Vn_local); CHKERRXX(this->ierr);
            this->ierr = VecGhostUpdateBegin(this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->delta_phi_for_advection, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            //  this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi,"Vn2");
            //this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->delta_phi,"delta_phi2",PETSC_FALSE);


            //----------------------Compute Energy Change After Lagrange : Should be a huge but smaller (abs) negative number-------------------------//
            if(this->advance_fields_scft_advance_mask_level_set)
            {
                std::cout<<this->mpi->mpirank<<" Compute Energy Change After Lagrange : Should be a huge but smaller negative number "<<std::endl;
                this->ierr=VecSet(this->phi_seed_new_on_old_p4est,0.00); CHKERRXX(this->ierr);
                this->ierr=VecAXPY(this->phi_seed_new_on_old_p4est,1.00,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);
                this->ierr=VecAXPY(this->phi_seed_new_on_old_p4est,-1.00,this->delta_phi_for_advection); CHKERRXX(this->ierr);
                this->scatter_petsc_vector(&this->phi_seed_new_on_old_p4est);
                this->predict_energy_change_from_level_set_change();
                this->de_predicted_from_level_set_change_current_step_after_lagrange_t=this->de_predicted_from_level_set_change_current_step;
                std::cout<<this->mpi->mpirank<<"Finish Compute Energy Change After Lagrange : Should be a huge negative number but smaller"<<
                           this->de_predicted_from_level_set_change_current_step_after_lagrange_t<<std::endl;
                this->compute_force_delta_phi();
                this->dphi_step_2[mask_counter]=this->shape_force_t;
            }
            //---------------------Finish to Compute Energy Change------------------------------------------------//


            if(!this->terracing)
                this->compute_lagrange_multiplier_delta_phi();
            else
                this->compute_lagrange_multiplier_delta_phi_terracing();
        }


        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->delta_phi_for_advection,"DHDPHI_2",  this->write2VtkShapeOptimization);
        this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi_for_advection,"DHDPHI_2",PETSC_TRUE);

        //                this->ierr=VecNorm(this->delta_phi_for_advection,NORM_2,&this->shape_force_t); CHKERRXX(this->ierr);
        //                this->shape_force_t=this->shape_force_t/(double)n_global;
        //                this->shape_force[this->mask_counter]=this->shape_force_t;

        //        this->compute_force_delta_phi();

        this->ierr=VecGetArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr);

        std::cout<<" start to compute max_norm "<<std::endl;
        double max_norm_u_loc=-10;
        for(p4est_locidx_t n=0; n<this->nodes->num_owned_indeps; n++)
        {
            max_norm_u_loc = max(max_norm_u_loc,pow(  Vn_local[n]*  Vn_local[n],0.5));

        }
        this->ierr=VecRestoreArray(this->delta_phi_for_advection,&Vn_local); CHKERRXX(this->ierr);
        std::cout<<" finish to compute max_norm "<<std::endl;
        double max_norm_u;
        MPI_Allreduce(&max_norm_u_loc, &max_norm_u, 1, MPI_DOUBLE, MPI_MAX, this->p4est->mpicomm);

        double dt_for_advection=0;
        if(this->i_mean_field_iteration>=this->start_remesh_i)
            dt_for_advection=1.00*this->alpha_cfl;
        else
            dt_for_advection=0.00;
        if(!this->uniform_normal_velocity)
            dt_for_advection= min(1.,1./max_norm_u) * dt_for_advection * dx_for_dt_min;
        std::cout<<" dt_for_advection "<<dt_for_advection<<std::endl;
        std::cout<<" max dphi [m]"<<dt_for_advection*max_norm_u<<std::endl;
        std::cout<<" max dphi [min_cell]"<<dt_for_advection*max_norm_u/(this->Lx/Nx)<<std::endl;
        this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr);
#endif

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 5"<<std::endl;


        // (4) Advect the level set in time using a simple scheme without remeshing
        // Before doing it store phi_seed into phi_seed_old_on_old_p4est

        //        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr); // memory increase local to the function 15 phi_seed_old_on_old_p4est: total 3
        //        this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);// memory increase local to the function 16 phi_seed_new_on_old_p4est: total 4

        //        this->ierr=VecCopy(this->phi_seed,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);

        //        this->ierr=VecGhostUpdateBegin(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        //        this->ierr=VecGhostUpdateEnd(this->phi_seed_old_on_old_p4est,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        // change the level set


        this->myDiffusion->print_vec_by_cells(&this->myDiffusion->wp,"wp_matlab");
        this->myDiffusion->print_vec_by_cells(&this->delta_phi_for_advection,"vn_matlab");
        this->myDiffusion->print_vec_by_cells(&this->myDiffusion->phi_wall,"phi_wall_matlab");


        this->myDiffusion->print_vec_by_cells(&this->phi_seed,"phi_seed_before_advection_matlab");

        switch(this->my_level_set_advection_numerical_scheme)
        {
        case MeanField::euler_advection:
        {
            this->ierr=VecAXPBY(this->phi_seed,-dt_for_advection,1.00,this->delta_phi_for_advection); CHKERRXX(this->ierr);
            break;
        }
        case MeanField::gudonov_advection:
        {
            my_p4est_level_set ls_advecter(this->node_neighbors);
            ls_advecter.advect_in_normal_direction(this->delta_phi_for_advection,this->phi_seed,this->alpha_cfl);
            break;

        }
        case MeanField::semi_lagrangian_advection:
        {
            std::logic_error(" no semilagrangian implemented yet ");//<<std::endl;
            break;
        }

        }

        this->scatter_petsc_vector(&this->phi_seed);


        //----------------------Compute Energy Change: Should be a small negative number since it has been modulated by the max number-------------------------//
        if(this->advance_fields_scft_advance_mask_level_set)
        {
            std::cout<<this->mpi->mpirank<<" Compute Energy Change After CFL : Should be a small negative number "<<
                       std::endl;
            this->ierr=VecSet(this->phi_seed_new_on_old_p4est,0.00); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_new_on_old_p4est);
            this->predict_energy_change_from_level_set_change();
            this->de_predicted_from_level_set_change_current_step_after_cfl_t=this->de_predicted_from_level_set_change_current_step;
            std::cout<<this->mpi->mpirank<<" Compute Energy Change After CFL : Should be a small negative number "<<
                       this->de_predicted_from_level_set_change_current_step_after_cfl_t<<  std::endl;
            double dphi_temp,dphi_temp_0;
            this->compute_delta_phi(dphi_temp,dphi_temp_0);
            this->dphi_step_3[mask_counter]=dphi_temp;
            this->dphi_step_0_3[mask_counter]=dphi_temp_0;
        }
        //---------------------Finish to Compute Energy Change------------------------------------------------//




        double dphi_advection;
        Vec temp_phi;
        this->ierr=VecDuplicate(this->phi_seed_old_on_old_p4est,&temp_phi); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->phi_seed_old_on_old_p4est,temp_phi); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_phi); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(temp_phi,-1.00,this->phi_seed); CHKERRXX(this->ierr);

        this->ierr=VecNorm(temp_phi,NORM_2,&dphi_advection); CHKERRXX(this->ierr);

        dphi_advection=dphi_advection/this->nodes->num_owned_indeps;

        std::cout<<" dphi_advection "<<dphi_advection<<std::endl;
        this->ierr=VecDestroy(temp_phi); CHKERRXX(this->ierr);


        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_after_advection",  this->write2VtkShapeOptimization);
        this->myDiffusion->print_vec_by_cells(&this->phi_seed,"phi_seed_after_advection_matlab");

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_after_advection",  this->write2VtkShapeOptimization);
        std::cout<<" volume source "<<this->compute_volume_source()<<std::endl;



        double dphi_reinitialyzation=0;

        if(this->reinitialize && mod(this->mask_counter,this->mask_conservation_period)==0)
        {

            Vec temp_phi;
            this->ierr=VecDuplicate(this->phi_seed,&temp_phi); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,temp_phi); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&temp_phi); CHKERRXX(this->ierr);

            my_p4est_level_set PLS(this->node_neighbors);
            if(!this->reinitialyze_level_set_with_tolerance)
            {
                PLS.reinitialize_2nd_order(this->phi_seed,50);
            }
            else
            {
                double l1_error,l2_error;
                double ls_tolerance=0.001;
                PLS.reinitialize_2nd_order_with_tolerance(this->phi_seed,ls_tolerance,l1_error,l2_error);

                std::cout<<" l1_error "<<l1_error<<" l2_error "<<l2_error<<std::endl;
            }
            this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            if(this->perturb_level_set_after_reinitialyzation)
            {
                PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
                this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            }


            this->ierr=VecAXPY(temp_phi,-1.00,this->phi_seed); CHKERRXX(this->ierr);

            this->ierr=VecNorm(temp_phi,NORM_2,&dphi_reinitialyzation); CHKERRXX(this->ierr);

            dphi_reinitialyzation=dphi_reinitialyzation/this->nodes->num_owned_indeps;

            std::cout<<" dphi_reinitialyzation "<<dphi_reinitialyzation<<std::endl;

            this->ierr=VecDestroy(temp_phi); CHKERRXX(this->ierr);
        }

        //----------------------Compute Energy Change After Reinitialyzation: Should be a hopefully a small negative number-------------------------//
        if(this->advance_fields_scft_advance_mask_level_set)
        {
            std::cout<<this->mpi->mpirank<<" Compute Energy Change After Reinitialyzation : Should be hopefully a small negative number "<<
                       std::endl;
            this->ierr=VecSet(this->phi_seed_new_on_old_p4est,0.00); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_new_on_old_p4est);
            this->predict_energy_change_from_level_set_change();
            this->de_predicted_from_level_set_change_current_step_after_reinitialyzation_t=this->de_predicted_from_level_set_change_current_step;
            std::cout<<this->mpi->mpirank<<" Compute Energy Change After Reinitialyzation : Should be hopefully a small negative number "<<
                       this->de_predicted_from_level_set_change_current_step_after_reinitialyzation_t<<std::endl;
            double dphi_temp,dphi_temp_0;
            this->compute_delta_phi(dphi_temp,dphi_temp_0);
            this->dphi_step_4[mask_counter]=dphi_temp;
            this->dphi_step_0_4[mask_counter]=dphi_temp_0;
        }
        //---------------------Finish to Compute Energy Change------------------------------------------------//

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_after_reinitializition",  this->write2VtkShapeOptimization);

        if(this->conserve_reaction_source_volume && mod(this->mask_counter,this->mask_conservation_period)==0)
        {
            if(!this->terracing)
                this->ensure_volume_conservation();

            if(terracing)
                this->ensure_volume_conservation_terracing();
        }
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_after_volume_conservation",  this->write2VtkShapeOptimization);

        this->myDiffusion->print_vec_by_cells(&this->phi_seed,"phi_seed_after_volume_conservation");


        //----------------------Compute Energy Change After Volume Conservation: Should be a hopefully a small negative number-------------------------//
        if(this->advance_fields_scft_advance_mask_level_set)
        {
            std::cout<<this->mpi->mpirank<<" Compute Energy Change After Volume Conservation : Should be hopefully a small negative number "<<
                       std::endl;
            this->ierr=VecSet(this->phi_seed_new_on_old_p4est,0.00); CHKERRXX(this->ierr);
            this->ierr=VecCopy(this->phi_seed,this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed_new_on_old_p4est);
            this->predict_energy_change_from_level_set_change();
            this->de_predicted_from_level_set_change_current_step_after_volume_conservation_t=this->de_predicted_from_level_set_change_current_step;
            std::cout<<this->mpi->mpirank<<" Compute Energy Change After Volume Conservation : Should be hopefully a small negative number "<<
                       this->de_predicted_from_level_set_change_current_step_after_volume_conservation_t<<std::endl;

            double dphi_temp,dphi_temp_0;
            this->compute_delta_phi(dphi_temp,dphi_temp_0);
            this->dphi_step_5[mask_counter]=dphi_temp;
            this->dphi_step_0_5[mask_counter]=dphi_temp_0;
        }


        //---------------------Finish to Compute Energy Change------------------------------------------------//


        double dphi_volume_conservation;
        Vec temp_phi2;
        this->ierr=VecDuplicate(this->phi_seed_old_on_old_p4est,&temp_phi2); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->phi_seed_old_on_old_p4est,temp_phi2); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_phi2); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(temp_phi2,-1.00,this->phi_seed); CHKERRXX(this->ierr);

        this->ierr=VecNorm(temp_phi2,NORM_2,&dphi_volume_conservation); CHKERRXX(this->ierr);

        dphi_volume_conservation=dphi_volume_conservation/this->nodes->num_owned_indeps;

        std::cout<<" dphi_volume_conservation "<<dphi_volume_conservation<<std::endl;
        this->ierr=VecDestroy(temp_phi2); CHKERRXX(this->ierr);

        this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr); // memory decrease local to the function 2 dmask_dx: total 4
        this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr); // memory decrease local to the function 3 dmask_dy: total 3

#ifdef P4_TO_P8

        this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr); // memory decrease local to the function 4 dmask_dz: total 2
#endif
        double v_source=this->compute_volume_source();
        std::cout<<" volume source "<<v_source<<" volume loss " <<v_source-this->source_volume_initial<< std::endl;

        this->ierr=VecCopy(this->phi_seed,this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateBegin(this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed_new_on_old_p4est, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 6"<<std::endl;

        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed_new_on_old_p4est,"phi_seed_new_after_advection",PETSC_TRUE);



        //----------------------------------PREDICT ENERGY CHANGE------------------------------//

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed_new_on_old_p4est,"phi_seed_new",  this->write2VtkShapeOptimization);
        this->ierr=VecDestroy(this->delta_phi_for_advection); CHKERRXX(this->ierr); // memory decrease local to the function 5 delta_phi: total 3

        //this->predict_energy_change_from_level_set_change();

        if(this->advance_fields_scft_advance_mask_level_set)
        {

            this->de_predicted_from_level_set_change_current_step_after_t=this->de_predicted_from_level_set_change_current_step;
            this->de_predicted_from_level_set_change_current_step_after[this->mask_counter]=this->de_predicted_from_level_set_change_current_step;
            this->prediction_error_level_set[this->mask_counter]=this->prediction_error_level_set_t;
            this->de_mask[this->mask_counter]=this->de_mask_t;
        }

        // We destruct non required data

        //this->ierr=VecDestroy(this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);

        //        this->ierr=VecDestroy(this->delta_phi); CHKERRXX(this->ierr);
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" change_internal_interface 6"<<std::endl;
        //-----------------------------END OF PREDICTION ENERGY CHANGE--------------------//

        //  destroys all the necessarry staff
        this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);// memory decrease local to the function 1 phi_seed_new_on_old_p4est: total 1
        this->ierr=VecDestroy(this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);
        if(i_game<this->n_change_internal_interfaces-1)
        {
            //will be destructed on the remapping on adaptive grids:  be carefull if ! remesh at this iteration for some reason: must handle the destruction here
            this->ierr=VecDestroy(this->phi_seed_new_on_old_p4est); CHKERRXX(this->ierr); // memory decrease local to the function 15 phi_seed_new_on_old_p4est: total 2

        }
        //this->ierr=VecDestroy(this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);




    }
    if(this->advance_fields_scft_advance_mask_level_set)
    {
        this->printLevelSetEvolution();
        this->mask_counter++;
    }
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end seed advection"<<std::endl;



    return 0;
}

int MeanField::compute_lagrange_multiplier_delta_phi()
{
    this->lagrange_multiplier=0;
    Vec temp_f;
    this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
    this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_f);

    this->lagrange_multiplier=integrate_over_interface(this->phi_seed,this->delta_phi_for_advection);
    this->surface_phi_seed=integrate_over_interface(this->phi_seed,temp_f);

    this->lagrange_multiplier=this->lagrange_multiplier/this->surface_phi_seed;
    std::cout<<" lagrange_multiplier "<<this->lagrange_multiplier<<std::endl;
    std::cout<<" surface_phi_seed  "<<this->surface_phi_seed<<std::endl;

    this->wp_average_on_contour[this->mask_counter]=this->lagrange_multiplier;
    this->ierr=VecSet(temp_f,0.00); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->delta_phi_for_advection,temp_f); CHKERRXX(this->ierr);
    this->ierr=VecShift(temp_f,-this->lagrange_multiplier); CHKERRXX(this->ierr);
    this->ierr=VecPointwiseMult(temp_f,temp_f,temp_f); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_f);

    this->wp_variance_on_contour[this->mask_counter]=integrate_over_interface(this->phi_seed,temp_f)/this->surface_phi_seed;
    this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
}

int MeanField::compute_lagrange_multiplier_delta_phi_volume_integral()
{
    this->lagrange_multiplier=0;
    Vec temp_f;
    this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
    this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_f);



    this->lagrange_multiplier=integrate_over_interface(this->phi_seed,this->delta_phi_for_advection);
    this->surface_phi_seed=integrate_over_interface(this->phi_seed,temp_f);

    this->lagrange_multiplier=this->lagrange_multiplier/this->surface_phi_seed;
    std::cout<<" lagrange_multiplier "<<this->lagrange_multiplier<<std::endl;
    std::cout<<" surface_phi_seed  "<<this->surface_phi_seed<<std::endl;
    this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);


}


int MeanField::compute_lagrange_multiplier_delta_phi_terracing()
{
    this->lagrange_multiplier=0;
    Vec temp_f,temp_delta_phi; PetscScalar *temp_f_local;
    this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(this->phi_seed,&temp_delta_phi); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);
    double tree_ymin; double y;
    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
        tree_ymin = connectivity->vertices[3*v_mm + 1];
        y = node_y_fr_j(node) + tree_ymin;
        if(y-this->y_terrace_floor>=0)
        { temp_f_local[i]=1.00; }
        else
        {temp_f_local[i]=0.00;}
    }
    this->ierr=VecRestoreArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_f);

    this->ierr=VecCopy(this->delta_phi_for_advection,temp_delta_phi); CHKERRXX(this->ierr);
    this->ierr=VecPointwiseMult(temp_delta_phi,temp_f,this->delta_phi_for_advection); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_delta_phi);

    this->lagrange_multiplier=integrate_over_interface(this->phi_seed,temp_delta_phi);
    this->surface_phi_seed=integrate_over_interface(this->phi_seed,temp_f);
    this->lagrange_multiplier=this->lagrange_multiplier/this->surface_phi_seed;
    std::cout<<" lagrange_multiplier "<<this->lagrange_multiplier<<std::endl;
    std::cout<<" surface_phi_seed  "<<this->surface_phi_seed<<std::endl;

    this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(temp_delta_phi); CHKERRXX(this->ierr);

}





int MeanField::compute_force_delta_phi()
{

    Vec temp_f;
    Vec temp_force;

    this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
    this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_f);

    this->ierr=VecDuplicate(this->phi_seed,&temp_force); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->delta_phi_for_advection,temp_force); CHKERRXX(this->ierr);
    this->ierr=VecPointwiseMult(temp_force,temp_force,temp_force); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_force);

    double force_shape_temp=integrate_over_interface(this->phi_seed,temp_force);
    this->surface_phi_seed=integrate_over_interface(this->phi_seed,temp_f);

    this->shape_force_t=force_shape_temp/this->surface_phi_seed;
    std::cout<<" force_shape "<<this->shape_force_t<<std::endl;
    std::cout<<" surface_phi_seed  "<<this->surface_phi_seed<<std::endl;

    this->shape_force[this->mask_counter]=this->shape_force_t;

    this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(temp_force); CHKERRXX(this->ierr);
}


int MeanField::compute_delta_phi(double &delta_phi, double &delta_phi_0)
{

    Vec temp_f;
    Vec temp_force;

    this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
    this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_f);

    this->ierr=VecDuplicate(this->phi_seed,&temp_force); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->phi_seed,temp_force); CHKERRXX(this->ierr);
    this->ierr=VecAXPY(temp_force,-1,this->phi_seed_old_on_old_p4est); CHKERRXX(this->ierr);
    this->ierr=VecPointwiseMult(temp_force,temp_force,temp_force); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_force);

    double force_shape_temp=integrate_over_interface(this->phi_seed,temp_force);
    this->surface_phi_seed=integrate_over_interface(this->phi_seed,temp_f);

    delta_phi=force_shape_temp/this->surface_phi_seed;
    std::cout<<" delta_phi "<<delta_phi<<std::endl;



    this->ierr=VecSet(temp_force,0.00); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->phi_seed,temp_force); CHKERRXX(this->ierr);
    this->ierr=VecAXPY(temp_force,-1,this->phi_seed_initial); CHKERRXX(this->ierr);
    this->ierr=VecPointwiseMult(temp_force,temp_force,temp_force); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&temp_force);

    force_shape_temp=integrate_over_interface(this->phi_seed,temp_force);
    //this->surface_phi_seed=integrate_over_interface(this->phi_seed,temp_f);

    delta_phi_0=force_shape_temp/this->surface_phi_seed;

    this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(temp_force); CHKERRXX(this->ierr);
}


int MeanField::ensure_volume_conservation()
{
    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {

        int i_v_loss=0; int n_v_loss=100;
        double V_loss=1.00;
        double V_loss_normalized=1.00;
        Vec temp_f;
        this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);
        double V_old, V_new, surface_new, surface_old;
        double v_loss_velocity=0;
        V_old=this->source_volume_initial;// integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed_old_on_old_p4est,temp_f);
        surface_old=integrate_over_interface(this->phi_seed_old_on_old_p4est,temp_f);

        if(this->tol_v_loss<0.1)
        {
            V_new=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,temp_f);
            surface_new=integrate_over_interface(this->phi_seed,temp_f);
            V_loss=V_old-V_new;
            V_loss_normalized=ABS(V_loss)/this->source_volume_initial;
        }

        while(i_v_loss<n_v_loss && V_loss_normalized>this->tol_v_loss)
        {

            V_new=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,temp_f);
            surface_new=integrate_over_interface(this->phi_seed,temp_f);
            V_loss=V_old-V_new;
            V_loss_normalized=ABS(V_loss)/this->source_volume_initial;
            v_loss_velocity=V_loss/surface_new;

            std::cout<<this->mpi->mpirank<<" "<< v_loss_velocity<<std::endl;
            this->ierr=VecShift(this->phi_seed,-v_loss_velocity); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed);


            PetscBool no_reinitialyzation_after_volume_conservation=PETSC_TRUE;

            if(!no_reinitialyzation_after_volume_conservation)
            {
                my_p4est_level_set PLS(this->node_neighbors);
                //            PLS.reinitialize_2nd_order(this->phi_seed,50);

                double l1_error,l2_error;
                double ls_tolerance=0.001;
                PLS.reinitialize_2nd_order_with_tolerance(this->phi_seed,ls_tolerance,l1_error,l2_error);

                std::cout<<" l1_error "<<l1_error<<" l2_error "<<l2_error<<std::endl;

                this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
                //PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
                //this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                //this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            }
            i_v_loss++;
            std::cout<<" i "<<i_v_loss<<" vloss "<<V_loss<<" v new "<<V_new<<" v old "<<V_old<<
                       " surface new "<<surface_new<<" surface old  "<<surface_old <<
                       "V Loss Normalized    "<<V_loss_normalized<<std::endl;

        }

        VecDestroy(temp_f);
    }

}

int MeanField::ensure_volume_conservation_terracing()
{
    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        double dx_for_dt_min =this->Lx / (double)Nx;

        int i_v_loss=0; int n_v_loss=100; double tol_v_loss=pow(0.1,2);
        double V_loss=1.00;
        Vec temp_f; PetscScalar *temp_f_local;
        this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);
        double tree_ymin; double y;
        for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
        {
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
            tree_ymin = connectivity->vertices[3*v_mm + 1];
            y = node_y_fr_j(node) + tree_ymin;


            //            if(y-this->y_terrace_floor>=0)
            if(y-this->y_terrace_floor>=0)
            { temp_f_local[i]=1.00; }
            else
            {temp_f_local[i]=0.00;}




        }

        this->ierr=VecRestoreArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);


        this->scatter_petsc_vector(&temp_f);
        double V_old, V_new, surface_new, surface_old;
        double v_loss_velocity=0;
        V_old=this->source_volume_initial;// integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed_old_on_old_p4est,temp_f);
        surface_old=integrate_over_interface(this->phi_seed_old_on_old_p4est,temp_f);
        this->myDiffusion->print_vec_by_cells(&this->phi_seed,"phi_seed_matlab");

        while(i_v_loss<n_v_loss && pow(V_loss*V_loss,0.5)>tol_v_loss)
        {


            V_new=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,temp_f);
            surface_new=integrate_over_interface(this->phi_seed,temp_f);
            V_loss=V_old-V_new;
            v_loss_velocity=V_loss/surface_new;

            this->ierr=VecScale(temp_f,v_loss_velocity); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&temp_f);

            std::cout<<this->mpi->mpirank<<" "<< v_loss_velocity<<std::endl;
            this->ierr=VecAXPY(this->phi_seed,-1,temp_f); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&this->phi_seed);
            this->myDiffusion->print_vec_by_cells(&this->phi_seed,"phi_seed_matlab");

            my_p4est_level_set PLS(this->node_neighbors);
            PLS.reinitialize_2nd_order(this->phi_seed,50);
            this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
            //PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
            //this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            //this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

            this->myDiffusion->print_vec_by_cells(&this->phi_seed,"phi_seed_matlab");

            i_v_loss++;
            std::cout<<" i "<<i_v_loss<<" vloss "<<V_loss<<" v new "<<V_new<<" v old "<<V_old<<
                       " surface new "<<surface_new<<" surface old  "<<surface_old <<std::endl;

            this->ierr=VecScale(temp_f,1.00/v_loss_velocity); CHKERRXX(this->ierr);
            this->scatter_petsc_vector(&temp_f);

        }

        VecDestroy(temp_f);
    }

}



double MeanField::compute_volume_source()
{

    int n_games=1;
    double volume_source_computed=0;
    for(int i_game=0;i_game<n_games;i_game++)
    {
        Vec temp_f;
        this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);

        //double surface_old=integrate_over_interface(this->phi_seed_old_on_old_p4est,temp_f);

        volume_source_computed= integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,temp_f);
        this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    }
    return volume_source_computed;

}

double MeanField::compute_volume_source_remeshed()
{

    int n_games=1;
    double volume_source_computed=0;
    for(int i_game=0;i_game<n_games;i_game++)
    {
        Vec temp_f;
        this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);

        //double surface_old=integrate_over_interface(this->phi_seed_old_on_old_p4est,temp_f);

        volume_source_computed= integrate_over_negative_domain(this->p4est_remeshed,this->nodes_remeshed,this->phi_seed,temp_f);
        this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    }
    return volume_source_computed;

}


double MeanField::compute_volume_source_remeshed_terrace()
{

    int n_games=1;
    double volume_source_computed=0;
    for(int i_game=0;i_game<n_games;i_game++)
    {

        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        double dx_for_dt_min =this->Lx / (double)Nx;
        Vec temp_f; PetscScalar *temp_f_local;
        this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);
        double tree_ymin; double y;
        for (p4est_locidx_t i = 0; i<this->nodes_remeshed->num_owned_indeps; ++i)
        {
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes_remeshed->indep_nodes,
                                                                 i+this->nodes_remeshed->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity_remeshed->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
            tree_ymin = this->connectivity_remeshed->vertices[3*v_mm + 1];
            y = node_y_fr_j(node) + tree_ymin;
            if(this->y_terrace_floor-y>=0)
            { temp_f_local[i]=1.00; }
            else
            {temp_f_local[i]=0.00;}
        }
        this->ierr=VecRestoreArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);
        volume_source_computed= integrate_over_negative_domain(this->p4est_remeshed,this->nodes_remeshed,this->phi_seed,temp_f);
        this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    }
    return volume_source_computed;
}

double MeanField::compute_volume_source_terrace()
{

    int n_games=1;
    double volume_source_computed=0;
    for(int i_game=0;i_game<n_games;i_game++)
    {

        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        double dx_for_dt_min =this->Lx / (double)Nx;
        Vec temp_f; PetscScalar *temp_f_local;
        this->ierr=VecDuplicate(this->phi_seed,&temp_f); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);
        double tree_ymin; double y;
        for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
        {
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
            tree_ymin = connectivity->vertices[3*v_mm + 1];
            y = node_y_fr_j(node) + tree_ymin;
            if(this->y_terrace_floor-y<=0)
            { temp_f_local[i]=1.00; }
            else
            {temp_f_local[i]=0.00;}
        }
        this->ierr=VecRestoreArray(temp_f,&temp_f_local); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);
        volume_source_computed= integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,temp_f);
        this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);
    }
    return volume_source_computed;

}


int MeanField::change_level_sets()
{
    int n_games=1;
    for(int i_game=0; i_game<n_games; i_game++)
    {
        //(1) create level set and parameters for extension

        my_p4est_level_set LS4extension(this->node_neighbors);
        int order_of_extension=2;
        int number_of_bands_to_extend=5;
        int number_of_bands_to_extend_on_all_domain=10;
        // Vecs=0+1=1
        Vec bc_vec_fake;
        this->ierr=VecDuplicate(this->phi,&bc_vec_fake); CHKERRXX(this->ierr);
        this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);
        VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD);
        VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD);


        // (2) Extend statistical potentials on the current grid with Neuman boundary conditions at the interface

        LS4extension.extend_Over_Interface(this->polymer_mask,this->myDiffusion->wp,NEUMANN,bc_vec_fake,order_of_extension,number_of_bands_to_extend);
        LS4extension.extend_Over_Interface(this->polymer_mask,this->myDiffusion->wm,NEUMANN,bc_vec_fake,order_of_extension,number_of_bands_to_extend);

        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wp,"wp_extended",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wm,"wm_extended",PETSC_TRUE);



        //--------------The part where the level set is either changed, swapped, transformed or advected----------------//


        this->ierr=VecDestroy(this->polymer_mask);  CHKERRXX(this->ierr);

        PetscScalar vx,vy,vz;
        PetscScalar a,b,c;

        vx=0.1; vy=0.1; vz=0;



        if(((vx+vy)+vy*vx*this->i_mean_field_iteration)>0)
            vz=-vx*vy/((vx+vy)+vy*vx*this->i_mean_field_iteration);
        else
            vz=0;

        a=1+vx*this->i_mean_field_iteration;
        b=1+vy*this->i_mean_field_iteration;
        c=1+vz*this->i_mean_field_iteration;

        this->create_polymer_mask_moving_ellipse(a,b,c);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->polymer_mask,"mask",PETSC_TRUE);

        // (8) destroys all the necessarry staff
        // vecs=6-6=0

        this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);

    }

    return 0;
}


int MeanField::remapFromOldForestToNewForest()
{

    std::cout<<" start to remap "<<std::endl;
    // Create an interpolating function
    InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, this->brick, this->node_neighbors);

    for (p4est_locidx_t i=0; i<this->nodes_remeshed->num_owned_indeps; i++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes_remeshed->indep_nodes, i+this->nodes_remeshed->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;

        p4est_topidx_t v_mm = this->connectivity_remeshed->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity_remeshed->vertices[3*v_mm + 0];
        double tree_ymin = this->connectivity_remeshed->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = connectivity_remeshed->vertices[3*v_mm + 2];
#endif
        double xyz [P4EST_DIM] =
        {
            node->x == P4EST_ROOT_LEN-1 ? 1.0 + tree_xmin:(double)node->x/(double)P4EST_ROOT_LEN + tree_xmin,
            node->y == P4EST_ROOT_LEN-1 ? 1.0 + tree_ymin:(double)node->y/(double)P4EST_ROOT_LEN + tree_ymin
    #ifdef P4_TO_P8
            ,
            node->z == P4EST_ROOT_LEN-1 ? 1.0 + tree_zmin:(double)node->z/(double)P4EST_ROOT_LEN + tree_zmin
    #endif
        };

        // buffer the point
        w_func.add_point_to_buffer(i, xyz);

        // delete xyz;
    }

    PetscSynchronizedFlush(this->p4est->mpicomm);

    Vec wpRemapped; Vec wmRemapped;

    this->ierr = VecCreateGhostNodes(this->p4est_remeshed,this->nodes_remeshed, &wpRemapped); CHKERRXX(this->ierr);
    this->ierr = VecCreateGhostNodes(this->p4est_remeshed,this->nodes_remeshed, &wmRemapped); CHKERRXX(this->ierr);


    // interpolate
    switch (this->my_interpolation_method)
    {
    case linear:
    {
        w_func.set_input_parameters(this->myDiffusion->wp, linear);
        w_func.interpolate(wpRemapped);
        w_func.set_input_parameters(this->myDiffusion->wm, linear);
        w_func.interpolate(wmRemapped);

        break;
    }
    case quadratic:
    {
        w_func.set_input_parameters(this->myDiffusion->wp, quadratic);
        w_func.interpolate(wpRemapped);
        w_func.set_input_parameters(this->myDiffusion->wm, quadratic);
        w_func.interpolate(wmRemapped);

        break;
    }
    case quadratic_non_oscillatory:
    {
        w_func.set_input_parameters(this->myDiffusion->wp, quadratic_non_oscillatory);
        w_func.interpolate(wpRemapped);
        w_func.set_input_parameters(this->myDiffusion->wm, quadratic_non_oscillatory);
        w_func.interpolate(wmRemapped);

        break;
    }
    default:
        throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }




    this->ierr = VecGhostUpdateBegin(wpRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(wpRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr = VecGhostUpdateBegin(wmRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(wmRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);


    this->ierr=VecDestroy(this->myDiffusion->wp); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->myDiffusion->wm);CHKERRXX(this->ierr);

    this->ierr=VecDuplicate(wpRemapped,&this->myDiffusion->wp);CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(wmRemapped,&this->myDiffusion->wm);CHKERRXX(this->ierr);

    this->ierr=VecCopy(wpRemapped,this->myDiffusion->wp);CHKERRXX(this->ierr);
    this->ierr=VecCopy(wmRemapped,this->myDiffusion->wm);CHKERRXX(this->ierr);

    this->ierr = VecGhostUpdateBegin(this->myDiffusion->wp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(this->myDiffusion->wp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr = VecGhostUpdateBegin(this->myDiffusion->wm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(this->myDiffusion->wm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wp,"wp");
    this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wm,"wm");


    //wp_func.clear();
    //wm_func.clear();

    this->ierr=VecDestroy(wpRemapped);CHKERRXX(this->ierr);
    this->ierr=VecDestroy(wmRemapped);CHKERRXX(this->ierr);

    w_func.clear();



    std::cout<<" end to remap "<<std::endl;
}

int MeanField::remapFromOldForestToNewForest2()
{


    this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wp,"wp_before_remapping");
    this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wm,"wm_before_remapping");

    // Create an interpolating function
    InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, this->brick, this->node_neighbors);



    for (p4est_locidx_t i=0; i<this->nodes_remeshed->num_owned_indeps; i++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes_remeshed->indep_nodes, i+this->nodes_remeshed->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;

        p4est_topidx_t v_mm = this->connectivity_remeshed->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity_remeshed->vertices[3*v_mm + 0];
        double tree_ymin = this->connectivity_remeshed->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = connectivity_remeshed->vertices[3*v_mm + 2];
#endif
        double xyz [P4EST_DIM] =
        {
            node->x == P4EST_ROOT_LEN-1 ? 1.0 + tree_xmin:(double)node->x/(double)P4EST_ROOT_LEN + tree_xmin,
            node->y == P4EST_ROOT_LEN-1 ? 1.0 + tree_ymin:(double)node->y/(double)P4EST_ROOT_LEN + tree_ymin
    #ifdef P4_TO_P8
            ,
            node->z == P4EST_ROOT_LEN-1 ? 1.0 + tree_zmin:(double)node->z/(double)P4EST_ROOT_LEN + tree_zmin
    #endif
        };

        // buffer the point
        w_func.add_point_to_buffer(i, xyz);
    }

    PetscSynchronizedFlush(this->p4est->mpicomm);

    Vec wpRemapped; Vec wmRemapped; Vec maskRemapped;

    this->ierr = VecCreateGhostNodes(this->p4est_remeshed,this->nodes_remeshed, &wpRemapped); CHKERRXX(this->ierr);
    this->ierr = VecCreateGhostNodes(this->p4est_remeshed,this->nodes_remeshed, &wmRemapped); CHKERRXX(this->ierr);
    this->ierr=VecCreateGhostNodes(this->p4est_remeshed,this->nodes_remeshed,&maskRemapped); CHKERRXX(this->ierr);


    // interpolate
    switch (this->my_interpolation_method)
    {
    case linear:
    {
        w_func.set_input_parameters(this->myDiffusion->wp, linear);
        w_func.interpolate(wpRemapped);
        w_func.set_input_parameters(this->myDiffusion->wm, linear);
        w_func.interpolate(wmRemapped);
        w_func.set_input_parameters(this->polymer_mask,linear);
        w_func.interpolate(maskRemapped);
        break;
    }
    case quadratic:
    {
        w_func.set_input_parameters(this->myDiffusion->wp, quadratic);
        w_func.interpolate(wpRemapped);
        w_func.set_input_parameters(this->myDiffusion->wm, quadratic);
        w_func.interpolate(wmRemapped);
        w_func.set_input_parameters(this->polymer_mask,quadratic);
        w_func.interpolate(maskRemapped);
        break;
    }
    case quadratic_non_oscillatory:
    {
        w_func.set_input_parameters(this->myDiffusion->wp, quadratic_non_oscillatory);
        w_func.interpolate(wpRemapped);
        w_func.set_input_parameters(this->myDiffusion->wm, quadratic_non_oscillatory);
        w_func.set_input_parameters(this->polymer_mask,quadratic_non_oscillatory);
        w_func.interpolate(maskRemapped);
        break;
    }
    default:
        throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }



    this->ierr = VecGhostUpdateBegin(wpRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(wpRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr = VecGhostUpdateBegin(wmRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(wmRemapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(maskRemapped,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(maskRemapped,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecDestroy(this->myDiffusion->wp); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->myDiffusion->wm);CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->polymer_mask); CHKERRXX(this->ierr);

    this->ierr=VecDuplicate(wpRemapped,&this->myDiffusion->wp);CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(wmRemapped,&this->myDiffusion->wm);CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(maskRemapped,&this->polymer_mask); CHKERRXX(this->ierr);

    this->ierr=VecCopy(wpRemapped,this->myDiffusion->wp);CHKERRXX(this->ierr);
    this->ierr=VecCopy(wmRemapped,this->myDiffusion->wm);CHKERRXX(this->ierr);
    this->ierr=VecCopy(maskRemapped,this->polymer_mask); CHKERRXX(this->ierr);

    this->ierr = VecGhostUpdateBegin(this->myDiffusion->wp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(this->myDiffusion->wp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr = VecGhostUpdateBegin(this->myDiffusion->wm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(this->myDiffusion->wm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->polymer_mask,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->polymer_mask,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);



    this->ierr=VecDestroy(wpRemapped);CHKERRXX(this->ierr);
    this->ierr=VecDestroy(wmRemapped);CHKERRXX(this->ierr);
    this->ierr=VecDestroy(maskRemapped); CHKERRXX(this->ierr);

    this->myDiffusion->set_polymer_shape(this->polymer_mask);

    this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wp,"wp_after_remapping");
    this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wm,"wm_after_remapping");



}

int MeanField::remapFromNewForestToOldForest(p4est_t *my_p4est_new, p4est_nodes_t *nodes_new, p4est_ghost_t *ghost_new, my_p4est_brick_t *mybrick,
                                             Vec *old_data_structure,Vec *vec_data, Vec  *vec2Remap)
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start remapFromNewForestToOldForest"<<std::endl;


    this->ierr=VecDuplicate(*old_data_structure,vec2Remap); CHKERRXX(this->ierr);
    my_p4est_hierarchy_t  *hierarchy_temp=new my_p4est_hierarchy_t(my_p4est_new,ghost_new,mybrick);
    my_p4est_node_neighbors_t *node_neighbors_temp=new my_p4est_node_neighbors_t(hierarchy_temp,nodes_new,this->myMeanFieldPlan->periodic_xyz);

    // Create an interpolating function
    InterpolatingFunctionNodeBase w_func(my_p4est_new,nodes_new,ghost_new,mybrick, node_neighbors_temp);
    for (p4est_locidx_t i=0; i<this->nodes->num_owned_indeps; i++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, i+this->nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;

        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = this->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif
        double xyz [P4EST_DIM] =
        {
            node->x == P4EST_ROOT_LEN-1 ? 1.0 + tree_xmin:(double)node->x/(double)P4EST_ROOT_LEN + tree_xmin,
            node->y == P4EST_ROOT_LEN-1 ? 1.0 + tree_ymin:(double)node->y/(double)P4EST_ROOT_LEN + tree_ymin
    #ifdef P4_TO_P8
            ,
            node->z == P4EST_ROOT_LEN-1 ? 1.0 + tree_zmin:(double)node->z/(double)P4EST_ROOT_LEN + tree_zmin
    #endif
        };

        // buffer the point
        w_func.add_point_to_buffer(i, xyz);
    }

    PetscSynchronizedFlush(this->p4est->mpicomm);


    // interpolate
    switch (this->my_interpolation_method)
    {
    case linear:
    {

        w_func.set_input_parameters(*vec_data,linear);
        w_func.interpolate(*vec2Remap);
        break;
    }
    case quadratic:
    {

        w_func.set_input_parameters(*vec_data,quadratic);
        w_func.interpolate(*vec2Remap);
        break;
    }
    case quadratic_non_oscillatory:
    {

        w_func.set_input_parameters(*vec_data,quadratic_non_oscillatory);
        w_func.interpolate(*vec2Remap);
        break;
    }
    default:
        throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }

    this->ierr = VecGhostUpdateBegin(*vec2Remap, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(*vec2Remap, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end remapFromNewForestToOldForest"<<std::endl;

}


int MeanField::remapFromOldForestToNewForest(p4est_t *my_p4est_new, p4est_nodes_t *nodes_new, p4est_ghost_t *ghost_new, my_p4est_brick_t *mybrick,
                                             Vec *vec_data_on_old_structure, Vec  *vec2Remap)
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start remapFromOldForestToNewForest overloaded"<<std::endl;


    this->ierr = VecCreateGhostNodes(my_p4est_new,nodes_new, vec2Remap); CHKERRXX(this->ierr);

    // do not forget to destroy these monsters
    // my_p4est_hierarchy_t  *hierarchy_temp=new my_p4est_hierarchy_t(my_p4est_new,ghost_new,mybrick);
    // my_p4est_node_neighbors_t *node_neighbors_temp=new my_p4est_node_neighbors_t(hierarchy_temp,nodes_new,this->periodic_xyz);

    // Create an interpolating function
    InterpolatingFunctionNodeBase w_func(this->p4est,this->nodes,this->ghost,this->brick, this->node_neighbors);
    for (p4est_locidx_t i=0; i<nodes_new->num_owned_indeps; i++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_new->indep_nodes, i+nodes_new->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;

        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = this->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif
        double xyz [P4EST_DIM] =
        {
            node->x == P4EST_ROOT_LEN-1 ? 1.0 + tree_xmin:(double)node->x/(double)P4EST_ROOT_LEN + tree_xmin,
            node->y == P4EST_ROOT_LEN-1 ? 1.0 + tree_ymin:(double)node->y/(double)P4EST_ROOT_LEN + tree_ymin
    #ifdef P4_TO_P8
            ,
            node->z == P4EST_ROOT_LEN-1 ? 1.0 + tree_zmin:(double)node->z/(double)P4EST_ROOT_LEN + tree_zmin
    #endif
        };

        // buffer the point
        w_func.add_point_to_buffer(i, xyz);
    }

    PetscSynchronizedFlush(this->p4est->mpicomm);


    // interpolate
    switch (this->my_interpolation_method)
    {
    case linear:
    {

        w_func.set_input_parameters(*vec_data_on_old_structure,linear);
        w_func.interpolate(*vec2Remap);
        break;
    }
    case quadratic:
    {

        w_func.set_input_parameters(*vec_data_on_old_structure,quadratic);
        w_func.interpolate(*vec2Remap);
        break;
    }
    case quadratic_non_oscillatory:
    {

        w_func.set_input_parameters(*vec_data_on_old_structure,quadratic_non_oscillatory);
        w_func.interpolate(*vec2Remap);
        break;
    }
    default:
        throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }

    this->ierr = VecGhostUpdateBegin(*vec2Remap, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr = VecGhostUpdateEnd(*vec2Remap, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);



    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end remapFromOldForestToNewForest overloaded"<<std::endl;

}


int MeanField::evolve_statistical_fields()
{
    this->evolve_statistical_fields_explicit();
}

int MeanField::evolve_statistical_fields_explicit()
{
    this->myDiffusion->evolve_fields_explicit();
}


int MeanField::set_advected_wall_from_level_set(PetscBool destroy_phi_wall, PetscBool compute_on_remeshed_p4est)
{

    // do not change any value of the level set in this function

    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {
        int n_nodes_local;


        if(destroy_phi_wall)
        {
            this->ierr=VecDestroy(this->myDiffusion->phi_wall); CHKERRXX(this->ierr);
        }

        this->ierr=VecDuplicate(this->phi_seed,&this->myDiffusion->phi_wall); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->myDiffusion->phi_wall,0.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->myDiffusion->phi_wall);


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_advected_fields_from_level_set"<<std::endl;
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start preprocess set_advected_fields_from_level_set"<<std::endl;

        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed",  this->write2VtkShapeOptimization);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed");

        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_before setting_field",PETSC_TRUE);


        if(!compute_on_remeshed_p4est)
        {
            std::cout<<" volume source "<<this->compute_volume_source()<<std::endl;
            n_nodes_local=this->nodes->num_owned_indeps;
        }
        else
        {
            std::cout<<" volume source "<<this->compute_volume_source_remeshed()<<std::endl;
            n_nodes_local=this->nodes_remeshed->num_owned_indeps;
        }

        std::cout<<" n_nodes_local "<<n_nodes_local<<std::endl;

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_for_wall",PETSC_FALSE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector");


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end preprocess set_advected_fields_from_level_set"<<std::endl;


        // step 2
        // seed/ set the statistical fields from the level set function which acts as an Heaviside function

        PetscScalar *phi_wall_local;

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" get array set_advected_fields_from_level_set"<<std::endl;

        this->ierr=VecGetArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);

        PetscScalar *polymer_mask_local;
        this->ierr=VecGetArray(this->phi_seed,&polymer_mask_local); CHKERRXX(this->ierr);
        //        p4est_topidx_t global_node_number=0;
        //        for(int ii=0;ii<this->mpi->mpirank;ii++)
        //            global_node_number+=this->nodes->global_owned_indeps[ii];
        //global_node_number=nodes->offset_owned_indeps;
        double exp_plus_phi;
        double exp_minus_phi;
        double phi_seed_local_double;
        double tanh_phi;


        for (p4est_locidx_t i = 0; i<n_nodes_local; ++i)
        {
            phi_seed_local_double=polymer_mask_local[i];
            //            exp_plus_phi=exp(this->alpha_wall*phi_seed_local_double);
            //            exp_minus_phi=exp(-this->alpha_wall*phi_seed_local_double);
            tanh_phi=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
            phi_wall_local[i]=0.5*(1+tanh_phi);
        }
        this->ierr=VecRestoreArray(this->phi_seed,&polymer_mask_local); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" restores array set_advected_fields_from_level_set"<<std::endl;

        this->ierr=VecRestoreArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&this->myDiffusion->phi_wall);

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->phi_wall,"phi_wall",PETSC_FALSE);


        this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->phi_wall,"phi_wall_after_advection",PETSC_TRUE);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_advected_fields_from_level_set"<<std::endl;
    }
}


int MeanField::correct_advected_wall_from_level_set_terrace(PetscBool destroy_phi_wall, PetscBool compute_on_remeshed_p4est)
{

    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);
    // do not change any value of the level set in this function

    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {

        if(destroy_phi_wall)
        {
            this->ierr=VecDestroy(this->myDiffusion->phi_wall_m); CHKERRXX(this->ierr);
            this->ierr=VecDestroy(this->myDiffusion->phi_wall_p); CHKERRXX(this->ierr);
        }


        this->ierr=VecDuplicate(this->phi_seed,&this->myDiffusion->phi_wall_m); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->myDiffusion->phi_wall_m,0.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->myDiffusion->phi_wall_m);
        this->ierr=VecDuplicate(this->phi_seed,&this->myDiffusion->phi_wall_p); CHKERRXX(this->ierr);
        this->ierr=VecSet(this->myDiffusion->phi_wall_p,0.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->myDiffusion->phi_wall_p);


        int n_nodes_local;
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start correct_advected_fields_from_level_set_terrace"<<std::endl;


        if(!compute_on_remeshed_p4est)
        {
            std::cout<<" volume source "<<this->compute_volume_source_terrace()<<std::endl;
            n_nodes_local=this->nodes->num_owned_indeps;
        }
        else
        {
            std::cout<<" volume source "<<this->compute_volume_source_remeshed_terrace()<<std::endl;
            n_nodes_local=this->nodes_remeshed->num_owned_indeps;
        }

        std::cout<<" n_nodes_local "<<n_nodes_local<<std::endl;


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end preprocess set_advected_fields_from_level_set"<<std::endl;


        // step 2
        // seed/ set the statistical fields from the level set function which acts as an Heaviside function

        PetscScalar *phi_wall_local;
        PetscScalar *phi_wall_local_m;
        PetscScalar *phi_wall_local_p;


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" get array set_advected_fields_from_level_set"<<std::endl;

        this->ierr=VecGetArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->myDiffusion->phi_wall_m,&phi_wall_local_m); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->myDiffusion->phi_wall_p,&phi_wall_local_p); CHKERRXX(this->ierr);

        double tree_xmin;
        double tree_ymin;

#ifdef P4_TO_P8
        double tree_zmin;
#endif

        double x;
        double y;

#ifdef P4_TO_P8
        double z;
#endif



        double tanh_phi;
        double phi_w_y;
        for (p4est_locidx_t i = 0; i<n_nodes_local; ++i)
        {
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;
            p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
            tree_xmin = this->connectivity->vertices[3*v_mm + 0];
            tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
            tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

            x = node_x_fr_i(node) + tree_xmin;
            y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            z = node_z_fr_k(node) + tree_zmin;
#endif

            tanh_phi=MeanField::my_tanh_x(this->alpha_wall*(this->y_terrace_floor-y));
            phi_w_y=0.5*(1+tanh_phi);

            phi_wall_local_m[i]=this->xhi_w_m*phi_w_y;
            phi_wall_local_p[i]=this->xhi_w_p*phi_w_y;

            phi_wall_local[i]=MIN(phi_w_y+phi_wall_local[i],1.00);



        }

        this->ierr=VecRestoreArray(this->myDiffusion->phi_wall,&phi_wall_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->myDiffusion->phi_wall_m,&phi_wall_local_m); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->myDiffusion->phi_wall_p,&phi_wall_local_p); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" restore array set_advected_fields_from_level_set"<<std::endl;
        this->scatter_petsc_vector(&this->myDiffusion->phi_wall);
        this->scatter_petsc_vector(&this->myDiffusion->phi_wall_m);
        this->scatter_petsc_vector(&this->myDiffusion->phi_wall_p);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->phi_wall,"corrected_phi_wall",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->phi_wall_p,"phi_wall_p",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->phi_wall_m,"phi_wall_m",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->phi_wall,"phi_wall_after_correction",PETSC_TRUE);
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_corrected_fields_from_level_set"<<std::endl;
    }
}



int MeanField::set_advected_fields_from_level_set()
{
    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

    // do not change any value of the level set in this function

    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_advected_fields_from_level_set"<<std::endl;
        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start preprocess set_advected_fields_from_level_set"<<std::endl;

        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed");

        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_before setting_field",PETSC_TRUE);


        std::cout<<" volume source "<<this->compute_volume_source()<<std::endl;


        //    my_p4est_level_set PLS(this->node_neighbors);
        //    PLS.reinitialize_2nd_order(this->phi_seed,50);
        //    this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        //    this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        //    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        //    PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double) Nx);

        //    std::cout<<" volume source "<<this->compute_volume_source()<<std::endl;


        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->phi_seed,"phi_seed_reinitialyzed",PETSC_FALSE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector");


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end preprocess set_advected_fields_from_level_set"<<std::endl;


        // step 2
        // seed/ set the statistical fields from the level set function which acts as an Heaviside function

        PetscScalar *wp_local;
        PetscScalar *wm_local;

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" get array set_advected_fields_from_level_set"<<std::endl;

        this->ierr=VecGetArray(this->myDiffusion->wp,&wp_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);




        PetscScalar *polymer_mask_local;
        this->ierr=VecGetArray(this->phi_seed,&polymer_mask_local); CHKERRXX(this->ierr);
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];
        //global_node_number=nodes->offset_owned_indeps;

        for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
        {
            if(polymer_mask_local[i]<0)
            {
                wm_local[i]=0;
                wp_local[i]=-this->X_ab;
            }

            if(polymer_mask_local[i]>=0)
            {
                wm_local[i]=0;
                wp_local[i]=0;
            }


        }
        this->ierr=VecRestoreArray(this->phi_seed,&polymer_mask_local); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" restores array set_advected_fields_from_level_set"<<std::endl;

        this->ierr=VecRestoreArray(this->myDiffusion->wp,&wp_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->myDiffusion->wm,&wm_local); CHKERRXX(this->ierr);



        this->ierr=VecGhostUpdateBegin(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->myDiffusion->wp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        this->ierr=VecGhostUpdateBegin(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->myDiffusion->wm,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->wm,"wm_from_seed",PETSC_FALSE);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->myDiffusion->wp,"wp_from_seed",PETSC_FALSE);

        if(this->i_mean_field_iteration>0)
            this->myDiffusion->printDiffusionArrayFromVector(&this->myDiffusion->wp,"wp_after_advection",PETSC_TRUE);


        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_advected_fields_from_level_set"<<std::endl;
    }
}


int MeanField::set_initial_seed_from_level_set()
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_initial_fields_from_level_set"<<std::endl;


    double polymer_mask_xc=this->Lx/2.00+this->Lx/16;
    double polymer_mask_yc=this->Lx/2.00+this->Lx/16;
    double polymer_mask_zc=this->Lx/2.00+this->Lx/16;

    double ax=1;
    double by=1;
    double cz=1/(ax*by);


    // Step 1: create a level set at the first iteration or change/advect it at the next iterations
    if(this->i_mean_field_iteration==0)
    {

        double polymer_mask_radius=this->Lx/16.00;//+this->dphi;

        ax=ax*polymer_mask_radius;
        by=by*polymer_mask_radius;
        cz=cz*polymer_mask_radius;

        ax=ax*ax;
        by=by*by;
        cz=cz*cz;

        VecDuplicate(this->phi,&this->phi_seed);
        PetscScalar *polymer_mask_local;
        VecGetArray(this->phi_seed,&polymer_mask_local);
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
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
            double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            double z = node_z_fr_k(node) + tree_zmin;
#endif


            double dp=  (x-polymer_mask_xc)  *(x-polymer_mask_xc)/ax+(y-polymer_mask_yc)*(y-polymer_mask_yc)/by;
#ifdef P4_TO_P8
            dp +=(z-polymer_mask_zc)*(z-polymer_mask_zc)/cz ;
#endif

            dp=pow(dp,0.5);

            //            if(dp<polymer_mask_radius)
            //                polymer_mask_local[i]=  dp-polymer_mask_radius; // negative inside the circle
            //            if(dp>polymer_mask_radius)
            //                polymer_mask_local[i]=  dp-polymer_mask_radius; // positive outside the circle
            //            if(dp==0)
            //                polymer_mask_local[i]==0;

            polymer_mask_local[i]=polymer_mask_radius*(dp-1); // negative inside the circle

        }
        VecRestoreArray(this->phi_seed,&polymer_mask_local);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector");
        my_p4est_level_set PLS(this->node_neighbors);
        PLS.reinitialize_2nd_order(this->phi_seed,50);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_reinitialyzed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_reinitialyzed");
    }

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_initial_fields_from_level_set"<<std::endl;

    this->source_volume_initial=   this->compute_volume_source();

}

int MeanField::set_initial_wall_from_level_set()
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_initial_fields_from_level_set"<<std::endl;


    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;

    double ax, by,cz;


    ax=this->a_ellipse;
    by=this->b_ellipse;
    cz=this->c_ellipse;

    // Step 1: create a level set at the first iteration or change/advect it at the next iterations
    if(this->i_mean_field_iteration==0)
    {

        double polymer_mask_radius=this->polymer_mask_radius_for_initial_wall;


        ax=ax*polymer_mask_radius;
        by=by*polymer_mask_radius;
        cz=cz*polymer_mask_radius;

        ax=ax*ax;
        by=by*by;
        cz=cz*cz;

        PetscInt n_local_size_sol;
        VecDuplicate(this->phi,&this->phi_seed);
        this->ierr=VecSet(this->phi_seed,0.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->phi_seed);
        this->ierr=VecGetLocalSize(this->phi,&n_local_size_sol); CHKERRXX(this->ierr);

        std::cout<<"-------------------------------Initialize seed wall------------------------------- "<<std::endl;
        std::cout<<this->mpi->mpirank<<" "<<n_local_size_sol<<" "<<nodes->num_owned_indeps<<std::endl;
        std::cout<<"------------------------------- Initialize seed wall------------------------------- "<<std::endl;


        PetscScalar *polymer_mask_local;
        VecGetArray(this->phi_seed,&polymer_mask_local);

        PetscScalar *wm_local;
        VecGetArray(this->myDiffusion->wm,&wm_local);

        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        //global_node_number=nodes->offset_owned_indeps;

        int N_local=nodes->num_owned_indeps;
        //N_local=this->nodes->indep_nodes.elem_count;

        for (p4est_locidx_t i = 0; i<N_local; ++i)
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
            double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            double z = node_z_fr_k(node) + tree_zmin;
#endif


            double dp=  (x-polymer_mask_xc)  *(x-polymer_mask_xc)/ax+(y-polymer_mask_yc)*(y-polymer_mask_yc)/by;
#ifdef P4_TO_P8
            dp +=(z-polymer_mask_zc)*(z-polymer_mask_zc)/cz ;
#endif

            dp=pow(dp,0.5);

            //            if(dp<polymer_mask_radius)
            //                polymer_mask_local[i]=  dp-polymer_mask_radius; // negative inside the circle
            //            if(dp>polymer_mask_radius)
            //                polymer_mask_local[i]=  dp-polymer_mask_radius; // positive outside the circle
            //            if(dp==0)
            //                polymer_mask_local[i]==0;

            polymer_mask_local[i]=polymer_mask_radius*(dp-1); // negative inside the circle

            if(polymer_mask_local[i]>0 && this->regenerate_potentials_from_mask)
                wm_local[i]=this->X_ab/2;

        }

        PetscInt min_value_i,max_value_i; PetscScalar min_value,max_value;




        VecRestoreArray(this->phi_seed,&polymer_mask_local);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        VecRestoreArray(this->myDiffusion->wm,&wm_local);
        this->ierr = VecGhostUpdateBegin(this->myDiffusion->wm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->myDiffusion->wm, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


        this->ierr=VecMin(this->phi_seed,&min_value_i,&min_value); CHKERRXX(this->ierr);
        this->ierr=VecMax(this->phi_seed,&max_value_i,&max_value); CHKERRXX(this->ierr);


        std::cout<<" min max"<<min_value<<" "<<max_value<<std::endl;

        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_initial",PETSC_TRUE);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->myDiffusion->wm,"wm_initial",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_initial");
        my_p4est_level_set PLS(this->node_neighbors);
        PLS.reinitialize_2nd_order(this->phi_seed,50);
        //        double l1_error,l2_error;
        //        PLS.reinitialize_2nd_order_with_tolerance(this->phi_seed,0.0001,l1_error,l2_error);

        //        std::cout<<" l1_error "<<l1_error<<" l2_error "<<l2_error<<std::endl;

        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_initial_reinitialyzed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_initial_reinitialyzed");
    }

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_initial_fields_from_level_set"<<std::endl;


    this->source_volume_initial=   this->compute_volume_source();
    if(this->terracing)
        this->source_volume_initial=this->compute_volume_source_terrace();
    this->volume_phi_seed=this->source_volume_initial;


    this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_initial); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->phi_seed,this->phi_seed_initial); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&this->phi_seed_initial); CHKERRXX(this->ierr);


}

int MeanField::set_initial_wall_from_level_set_rectangle()
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_initial_fields_from_level_set"<<std::endl;


    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=this->Lx/2.00;
    double polymer_mask_zc=this->Lx/2.00;

    double ax, by,cz;


    ax=this->a_ellipse;
    by=this->b_ellipse;
    cz=this->c_ellipse;

    // Step 1: create a level set at the first iteration or change/advect it at the next iterations
    if(this->i_mean_field_iteration==0)
    {

        //double polymer_mask_radius=this->a_radius*pow(3,0.5)*this->Lx/4.00;
        double polymer_mask_radius=this->a_radius*this->Lx*pow(PI,0.5)/3;


        ax=ax*polymer_mask_radius;
        by=by*polymer_mask_radius;
        cz=cz*polymer_mask_radius;



        PetscInt n_local_size_sol;
        VecDuplicate(this->phi,&this->phi_seed);
        this->ierr=VecSet(this->phi_seed,0.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->phi_seed);
        this->ierr=VecGetLocalSize(this->phi,&n_local_size_sol); CHKERRXX(this->ierr);

        std::cout<<"-------------------------------Initialize seed wall------------------------------- "<<std::endl;
        std::cout<<this->mpi->mpirank<<" "<<n_local_size_sol<<" "<<nodes->num_owned_indeps<<std::endl;
        std::cout<<"------------------------------- Initialize seed wall------------------------------- "<<std::endl;


        PetscScalar *polymer_mask_local;
        VecGetArray(this->phi_seed,&polymer_mask_local);
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        //global_node_number=nodes->offset_owned_indeps;

        int N_local=nodes->num_owned_indeps;
        //N_local=this->nodes->indep_nodes.elem_count;

        for (p4est_locidx_t i = 0; i<N_local; ++i)
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
            double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            double z = node_z_fr_k(node) + tree_zmin;
            double z_0=z-polymer_mask_zc;

#endif


            double x_0=x-polymer_mask_xc;
            double y_0=y-polymer_mask_yc;




            if( (x_0> -ax/2 && x_0<ax/2) && (y_0> -by/2 && y_0<by/2)
        #ifdef P4_TO_P8
                    && (z_0> -cz/2 && z_0<cz/2)
        #endif
                    )
            {
                //                polymer_mask_local[i]=min(-x_0-ax/2, -x_0+ax/2   );
                //                polymer_mask_local[i]=min(polymer_mask_local[i], -y_0-by/2   );
                //                polymer_mask_local[i]=min(polymer_mask_local[i], -y_0+by/2   );

                //                #ifdef P4_TO_P8

                //                polymer_mask_local[i]=min(polymer_mask_local[i], -z_0-cz/2   );
                //                polymer_mask_local[i]=min(polymer_mask_local[i], -z_0+cz/2   );
                //                #endif

                polymer_mask_local[i]=-1;

            }
            else
            {
                polymer_mask_local[i]=1;
            }


        }

        PetscInt min_value_i,max_value_i; PetscScalar min_value,max_value;




        VecRestoreArray(this->phi_seed,&polymer_mask_local);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        this->ierr=VecMin(this->phi_seed,&min_value_i,&min_value); CHKERRXX(this->ierr);
        this->ierr=VecMax(this->phi_seed,&max_value_i,&max_value); CHKERRXX(this->ierr);


        std::cout<<" min max"<<min_value<<" "<<max_value<<std::endl;

        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_initial",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_initial");
        my_p4est_level_set PLS(this->node_neighbors);
        //PLS.reinitialize_2nd_order(this->phi_seed,50);
        double l1_error,l2_error;
        PLS.reinitialize_2nd_order_with_tolerance(this->phi_seed,0.0001,l1_error,l2_error);

        std::cout<<" l1_error "<<l1_error<<" l2_error "<<l2_error<<std::endl;

        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_initial_reinitialyzed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_initial_reinitialyzed");
    }

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_initial_fields_from_level_set"<<std::endl;


    this->source_volume_initial=   this->compute_volume_source();
    if(this->terracing)
        this->source_volume_initial=this->compute_volume_source_terrace();
    this->volume_phi_seed=this->source_volume_initial;


    this->ierr=VecDuplicate(this->phi_seed,&this->phi_seed_initial); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->phi_seed,this->phi_seed_initial); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&this->phi_seed_initial); CHKERRXX(this->ierr);


}


int MeanField::set_initial_wall_from_level_set_terracing()
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_initial_fields_from_level_set"<<std::endl;


    double polymer_mask_xc=this->Lx/2.00;
    double polymer_mask_yc=3*this->Lx/4.00;
    double polymer_mask_zc=this->Lx/2.00;


    double y_front_center=3*this->Lx/4+this->Lx/16;
    double y_front_edges=3*this->Lx/4-this->Lx/16;

    double x_front_left=this->Lx/4.00;
    double x_front_right=3*this->Lx/4.00;

    double ax, by,cz;


    ax=this->a_ellipse;
    by=this->b_ellipse;
    cz=this->c_ellipse;

    // Step 1: create a level set at the first iteration or change/advect it at the next iterations
    if(this->i_mean_field_iteration==0)
    {

        //double polymer_mask_radius=this->a_radius*pow(3,0.5)*this->Lx/4.00;
        double polymer_mask_radius=this->a_radius*this->Lx/3.00;


        ax=ax*polymer_mask_radius;
        by=by*polymer_mask_radius;
        cz=cz*polymer_mask_radius;

        ax=ax*ax;
        by=by*by;
        cz=cz*cz;

        PetscInt n_local_size_sol;
        VecDuplicate(this->phi,&this->phi_seed);
        this->ierr=VecSet(this->phi_seed,0.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&this->phi_seed);
        this->ierr=VecGetLocalSize(this->phi,&n_local_size_sol); CHKERRXX(this->ierr);

        std::cout<<"-------------------------------Initialize seed wall------------------------------- "<<std::endl;
        std::cout<<this->mpi->mpirank<<" "<<n_local_size_sol<<" "<<nodes->num_owned_indeps<<std::endl;
        std::cout<<"------------------------------- Initialize seed wall------------------------------- "<<std::endl;


        PetscScalar *polymer_mask_local;
        VecGetArray(this->phi_seed,&polymer_mask_local);
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        //global_node_number=nodes->offset_owned_indeps;

        int N_local=nodes->num_owned_indeps;
        //N_local=this->nodes->indep_nodes.elem_count;

        for (p4est_locidx_t i = 0; i<N_local; ++i)
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
            double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            double z = node_z_fr_k(node) + tree_zmin;
#endif


            double dp=  (x-polymer_mask_xc)  *(x-polymer_mask_xc)/ax+(y-polymer_mask_yc)*(y-polymer_mask_yc)/by;
#ifdef P4_TO_P8
            dp +=(z-polymer_mask_zc)*(z-polymer_mask_zc)/cz ;
#endif

            dp=pow(dp,0.5);

            //            if(dp<polymer_mask_radius)
            //                polymer_mask_local[i]=  dp-polymer_mask_radius; // negative inside the circle
            //            if(dp>polymer_mask_radius)
            //                polymer_mask_local[i]=  dp-polymer_mask_radius; // positive outside the circle
            //            if(dp==0)
            //                polymer_mask_local[i]==0;



            if(x>x_front_left && x<x_front_right)
                polymer_mask_local[i]=y-y_front_center; //+0.1*this->Lx*sin(4*2*PI*x/this->Lx);
            else
                polymer_mask_local[i]=y-y_front_edges;
        }

        PetscInt min_value_i,max_value_i; PetscScalar min_value,max_value;




        VecRestoreArray(this->phi_seed,&polymer_mask_local);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

        this->ierr=VecMin(this->phi_seed,&min_value_i,&min_value); CHKERRXX(this->ierr);
        this->ierr=VecMax(this->phi_seed,&max_value_i,&max_value); CHKERRXX(this->ierr);


        std::cout<<" min max"<<min_value<<" "<<max_value<<std::endl;

        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_initial",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_initial");
        my_p4est_level_set PLS(this->node_neighbors);
        //PLS.reinitialize_2nd_order(this->phi_seed,50);
        double l1_error,l2_error;
        PLS.reinitialize_2nd_order_with_tolerance(this->phi_seed,0.0001,l1_error,l2_error);

        std::cout<<" l1_error "<<l1_error<<" l2_error "<<l2_error<<std::endl;

        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_initial_reinitialyzed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_initial_reinitialyzed");
    }

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_initial_fields_from_level_set"<<std::endl;


    this->source_volume_initial=   this->compute_volume_source();
    if(this->terracing)
        this->source_volume_initial=this->compute_volume_source_terrace();
    this->volume_phi_seed=this->source_volume_initial;

}


int MeanField::compute_center_of_mass(double &xc, double &yc, double &zc)
{


    double tree_xmin, tree_ymin, tree_zmin ;
    double x,y,z;

    Vec x_vec,y_vec,z_vec;
    PetscScalar *x_local,*y_local,*z_local;

    this->ierr=VecDuplicate(this->phi_seed,&x_vec); CHKERRXX(this->ierr);
    this->ierr=VecSet(x_vec,0.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&x_vec);
    this->ierr=VecGetArray(x_vec,&x_local); CHKERRXX(this->ierr);

    this->ierr=VecDuplicate(this->phi_seed,&y_vec); CHKERRXX(this->ierr);
    this->ierr=VecSet(y_vec,0.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&y_vec);
    this->ierr=VecGetArray(y_vec,&y_local); CHKERRXX(this->ierr);

    this->ierr=VecDuplicate(this->phi_seed,&z_vec); CHKERRXX(this->ierr);
    this->ierr=VecSet(z_vec,0.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&z_vec);
    this->ierr=VecGetArray(z_vec,&z_local); CHKERRXX(this->ierr);



    p4est_topidx_t global_node_number=0;
    for(int ii=0;ii<this->mpi->mpirank;ii++)
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


        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = connectivity->vertices[3*v_mm + 1];

#ifdef P4_TO_P8
        tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        x = node_x_fr_i(node) + tree_xmin;
        y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
        z = node_z_fr_k(node) + tree_zmin;
#endif


        x_local[i]=x;
        y_local[i]=y;
        z_local[i]=z;

    }

    this->ierr=VecRestoreArray(x_vec,&x_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(y_vec,&y_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(z_vec,&x_local); CHKERRXX(this->ierr);

    this->scatter_petsc_vector(&x_vec);
    this->scatter_petsc_vector(&y_vec);
    this->scatter_petsc_vector(&z_vec);

    xc=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,x_vec)/this->volume_phi_seed;
    yc=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,y_vec)/this->volume_phi_seed;
    zc=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed,z_vec)/this->volume_phi_seed;


    this->ierr=VecDestroy(x_vec); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(y_vec); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(z_vec); CHKERRXX(this->ierr);

    std::cout<<" xc "<<xc<<" yc "<<yc<<" zc "<<zc<<std::endl;
}

int MeanField::set_initial_seed_from_level_set_wp()
{
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_initial_fields_from_level_set"<<std::endl;





    // Step 1: create a level set at the first iteration or change/advect it at the next iterations
    if(this->i_mean_field_iteration==0)
    {



        VecDuplicate(this->phi,&this->phi_seed);
        PetscScalar *polymer_mask_local;
        VecGetArray(this->phi_seed,&polymer_mask_local);
        PetscScalar *wp_local;
        VecGetArray(this->myDiffusion->wp,&wp_local);
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
            global_node_number+=this->nodes->global_owned_indeps[ii];

        //global_node_number=nodes->offset_owned_indeps;
        for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
        {
            if(wp_local[i]<0)
                polymer_mask_local[i]=-1; // negative inside the circle
            else
                polymer_mask_local[i]=1;

        }
        VecRestoreArray(this->phi_seed,&polymer_mask_local);
        VecRestoreArray(this->myDiffusion->wp,&wp_local);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector");
        my_p4est_level_set PLS(this->node_neighbors);
        PLS.reinitialize_2nd_order(this->phi_seed,50);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_reinitialyzed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_reinitialyzed");
    }

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_initial_fields_from_level_set"<<std::endl;

    this->source_volume_initial=   this->compute_volume_source();

}


int MeanField::set_initial_seed_from_level_set_helix()
{

#ifdef P4_TO_P8
    double r_helix=this->Lx/32.00;

    double r_mask=this->Lx/6.00;

    double r_sphere=this->Lx/16;

    double t_start=0;
    double t_end=this->Lx;

    double xc=this->Lx/2.00;
    double yc=this->Lx/2.00;


    int tol=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))/4;

    double helix_speed=1;

    int N_t_helix=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));

    double *t_vector=new double[N_t_helix-2*tol];

    double dt_helix=(t_end-t_start)/N_t_helix;
    double *distance_helix=new double[N_t_helix-2*tol];
    double *sint_vector=new double [N_t_helix-2*tol];
    double *cost_vector=new double[N_t_helix-2*tol];

    for(int i=0;i<N_t_helix-2*tol;i++)
    {
        t_vector[i]=dt_helix*(i+tol);
        sint_vector[i]=r_mask*sin(2*PI*helix_speed*t_vector[i])+xc;
        cost_vector[i]=r_mask*cos(2*PI*helix_speed*t_vector[i])+yc;
        std::cout<<i<<" "<<t_vector[i]<<" "
                <<cost_vector[i]<<" "
               << sint_vector[i]<<std::endl;
    }


    //    double x_start=r_mask*cos(t_start+(tol-1)*dt_helix)+xc;
    //    double y_start=r_mask*sin(t_start+(tol-1)*dt_helix)+yc;

    //    double x_end=r_mask*cos(t_end-(tol-1)*dt_helix)+xc;
    //    double y_end=r_mask*sin(t_end-(tol-1)*dt_helix)+yc;




    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start set_initial_fields_from_level_set"<<std::endl;




    // Step 1: create a level set at the first iteration or change/advect it at the next iterations
    if(this->i_mean_field_iteration==0)
    {


        VecDuplicate(this->phi,&this->phi_seed);
        PetscScalar *polymer_mask_local;
        VecGetArray(this->phi_seed,&polymer_mask_local);
        p4est_topidx_t global_node_number=0;
        for(int ii=0;ii<this->mpi->mpirank;ii++)
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

            double x =node_x_fr_i(node) + tree_xmin;
            double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
            double z = node_z_fr_k(node)+ tree_zmin;
#endif

            bool inside_helix=false;


            for(int j=0;j<N_t_helix-2*tol;j++)
            {
                distance_helix[j]=
                        (x-cost_vector[j])*(x-cost_vector[j])
                        +(y-sint_vector[j])*(y-sint_vector[j])
                        +(z-t_vector[j])*(z-t_vector[j]);

                //distance_helix[j]=(z-t_vector[j])*(z-t_vector[j]);
            }

            // this->myDiffusion->printDiffusionArray(distance_helix,N_t_helix,"distance_helix");

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


            //        std::cout<<i<<" "<<x<<" "<<y<<" "<<z<<" "<<minimum_proposed<<" "<<r_helix<<
            //                   " "<<cost_vector[j_min]<<" "<<sint_vector[j_min]<<" "<<t_vector[j_min]<<   std::endl;



            //        double dx= (x-cost_vector[j_min])*(x-cost_vector[j_min]);
            //         double dy=(y-sint_vector[j_min])*(y-sint_vector[j_min]);
            //        double dz=(z-t_vector[j_min])*(z-t_vector[j_min]);

            //       double distance_helix_jmin= (x-cost_vector[j_min])*(x-cost_vector[j_min])+
            //        +(y-sint_vector[j_min])*(y-sint_vector[j_min])
            //        +(z-t_vector[j_min])*(z-t_vector[j_min]);

            //       std::cout<<distance_helix_jmin<<std::endl;

            if(minimum_proposed<r_helix)
                inside_helix=true;
            else
                inside_helix=false;

            //        bool condition1=inside_helix;
            //        bool condition2=true;//(z>t_start && z<t_end);
            //        bool condition3= true;//(x-x_start)*(x-x_start)+(y-y_start)*(y-y_start)+(z-t_start)*(z-t_start)<r_helix*r_helix;
            //        bool condition4=true;//(x-x_end)*(x-x_end)+(y-y_end)*(y-y_end)+(z-t_end)*(z-t_end)<r_helix*r_helix;


            polymer_mask_local[i]=minimum_proposed-r_helix;
            //        if(condition1 && (condition2  ||condition3 ||condition4  ) )
            //            polymer_mask_local[i]=-1;



        }
        VecRestoreArray(this->phi_seed,&polymer_mask_local);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector");
        my_p4est_level_set PLS(this->node_neighbors);
        PLS.reinitialize_2nd_order(this->phi_seed,50);
        this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
        PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_reinitialyzed",PETSC_TRUE);
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_reinitialyzed");
    }

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_initial_fields_from_level_set"<<std::endl;

    this->source_volume_initial=   this->compute_volume_source();


    delete distance_helix;
    delete t_vector;
    delete cost_vector;
    delete sint_vector;
#endif

}

int MeanField::set_initial_seed_from_level_set_bcc()
{

    double ax=1;
    double by=1;
    double cz=1/(ax*by);

    double L=this->Lx;
    double r=this->Lx/8;//*pow(3.00*this->f/(8.00*PI),1.00/3.00);//

    int n_spheres=2;

    ax=ax*r;
    by=by*r;
    cz=cz*r;

    ax=ax*ax;
    by=by*by;
    cz=cz*cz;

    double *xc=new double[9];
    double *yc=new double[9];
    double *zc=new double[9];
    double x0;
    x0=this->Lx/2.00;

    double dL=this->Lx/8;

    xc[0]=x0-dL;yc[0]=x0-dL;zc[0]=x0-dL;
    xc[1]=x0+dL;yc[1]=x0+dL;zc[1]=x0+dL;


    //    xc[0]=x0+this->Lx/4;yc[0]=x0-this->Lx/4;zc[0]=x0-this->Lx/4;
    //    xc[1]=x0-this->Lx/4;yc[1]=x0+this->Lx/4;zc[1]=x0+this->Lx/4;


    // (L,L,L) corner
    //xc[1]=L;yc[1]=L;zc[1]=L;

    //(0,0,0) corner
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

    double *dp=new double[n_spheres];
    double dp_min=this->Lx;


    VecDuplicate(this->phi,&this->phi_seed);
    PetscScalar *polymer_mask_local;
    VecGetArray(this->phi_seed,&polymer_mask_local);

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

        int i_sphere;

        for(int ii=0;ii<2;ii++)
        {
            dp[ii]=  (x-xc[ii])  *(x-xc[ii])/ax+(y-yc[ii])*(y-yc[ii])/by;
#ifdef P4_TO_P8
            dp[ii] +=(z-zc[ii])*(z-zc[ii])/cz ;
#endif

            dp[ii]=pow(dp[ii],0.5);


            if( dp[ii]<1)
            {
                isInsideASphere=true;
                i_sphere=ii;
            }
        }
        if(isInsideASphere)
            polymer_mask_local[i]=r*(dp[i_sphere]-1);
        else
            polymer_mask_local[i]=min(r*(dp[0]-1),r*(dp[1]-1));
    }

    VecRestoreArray(this->phi_seed,&polymer_mask_local);
    this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed",PETSC_TRUE);
    this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector");
    my_p4est_level_set PLS(this->node_neighbors);
    PLS.reinitialize_2nd_order(this->phi_seed,50);
    this->ierr = VecGhostUpdateBegin(this->phi_seed, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd  (this->phi_seed, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
    int Nx=(int)pow(2.00,(double)this->max_level+log(this->nx_trees)/log(2));
    PLS.perturb_level_set_function(&this->phi_seed,this->ls_tolerance*this->Lx/(double)Nx);
    this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed,"phi_seed_reinitialyzed",PETSC_TRUE);
    this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed,"phi_seed_vector_reinitialyzed");
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end set_initial_fields_from_level_set"<<std::endl;
    this->source_volume_initial=   this->compute_volume_source();
    delete dp;
}


int MeanField::predict_energy_change_from_level_set_change()
{

    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_level_set_change"<<std::endl;



        // ----------Simple Count Check------------------------------------------------------//

        PetscScalar *phi_seed_old,*phi_seed_new;
        int n_old=0, n_new=0; int dn_old=0,dn_new=0;
        this->ierr=VecGetArray(this->phi_seed_old_on_old_p4est,&phi_seed_old); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(this->phi_seed_new_on_old_p4est,&phi_seed_new); CHKERRXX(this->ierr);
        for(int i=0;i<this->nodes->num_owned_indeps;i++)
        {
            if(phi_seed_old[i]<0)
                n_old++;
            if(phi_seed_new[i]<0)
                n_new++;
            if(phi_seed_new[i]<0 && phi_seed_old[i]>0)
                dn_new++;
            if(phi_seed_new[i]>0 && phi_seed_old[i]<0)
                dn_old++;
        }

        this->ierr=VecRestoreArray(this->phi_seed_new_on_old_p4est,&phi_seed_new); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(this->phi_seed_old_on_old_p4est,&phi_seed_old); CHKERRXX(this->ierr);


        std::cout<<" n_old "<<n_old<<" n_new "<<n_new<<" dn_old "<<dn_old<<" dn_new "<<dn_new<<std::endl;

        //------------------------------------------------------------------------------------//
        //    Vec normal_norm;
        //    this->ierr=VecDuplicate(this->phi_seed_old_on_old_p4est,&normal_norm); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi_seed_old_on_old_p4est,&this->delta_phi_for_prediction);             CHKERRXX(this->ierr);  // memory increase local to the function delta_phi 1: total 1
        this->ierr=VecDuplicate(this->phi_seed_old_on_old_p4est,&this->negative_phi_seed); CHKERRXX(this->ierr);      // memory increase local to the function negative_phi_seed 2: total 2

        this->ierr=VecSet(this->delta_phi_for_prediction,0.00); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->delta_phi_for_prediction,1.00,this->phi_seed_new_on_old_p4est);   CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->delta_phi_for_prediction,-1.00,this->phi_seed_old_on_old_p4est);  CHKERRXX(this->ierr);

        this->ierr=VecSet(this->negative_phi_seed,0.00); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(this->negative_phi_seed,-1.00,this->phi_seed_old_on_old_p4est);   CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(this->negative_phi_seed,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->negative_phi_seed,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        //scatter
        this->ierr=VecGhostUpdateBegin(this->delta_phi_for_prediction,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->delta_phi_for_prediction,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);



        Vec temp_phi;
        this->ierr=VecDuplicate(this->phi_seed_old_on_old_p4est,&temp_phi); CHKERRXX(this->ierr); // destroy temp_phi
        this->ierr=VecSet(temp_phi,1.00); CHKERRXX(this->ierr);
        //scatter
        this->ierr=VecGhostUpdateBegin(temp_phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(temp_phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        this->volume_phi_seed=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed_old_on_old_p4est,temp_phi);
        double new_volume=integrate_over_negative_domain(this->p4est,this->nodes,this->phi_seed_new_on_old_p4est,temp_phi);

        std::cout<<"old source volume "<<this->volume_phi_seed<<std::endl;
        std::cout<<"new source volume "<<new_volume<<std::endl;



        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed_new_on_old_p4est,"phi_seed_new",PETSC_FALSE);
        this->my_p4est_vtk_write_all_periodic_adapter(&this->phi_seed_old_on_old_p4est,"phi_seed_old",PETSC_FALSE);
        this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->delta_phi_for_prediction,"delta_phi_seed",PETSC_FALSE);

        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed_old_on_old_p4est,"old_seed");
        this->myDiffusion->printDiffusionArrayFromVector(&this->phi_seed_new_on_old_p4est,"new_seed");
        this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi_for_prediction,"delta_phi");


        //    this->compute_normal_to_the_interface(&this->phi_seed);



        //this->print_normal_to_the_interface(&this->dmask_dx,&this->dmask_dy,&this->dmask_dz," normal");

        // Extend Petsc Vectors with level set of choice
        // for periodic domains it will just be a fake domain

        Vec wm_extended_in,wp_extended_in,fp_extended_in,fm_extended_in;
        Vec wm_extended_out,wp_extended_out,fp_extended_out,fm_extended_out;

        this->ierr=VecDuplicate(this->phi,&wm_extended_in); CHKERRXX(this->ierr); // memory increase local to the function wm_extended_in 3: total 3
        this->ierr=VecDuplicate(this->phi,&wp_extended_in); CHKERRXX(this->ierr);// memory increase local to the function wp_extended_in 4: total 4
        this->ierr=VecDuplicate(this->phi,&fm_extended_in); CHKERRXX(this->ierr);// memory increase local to the function fm_extended_in 5: total 5
        this->ierr=VecDuplicate(this->phi,&fp_extended_in); CHKERRXX(this->ierr);// memory increase local to the function fp_extended_in 6: total 6

        this->ierr=VecCopy(this->myDiffusion->wm,wm_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->myDiffusion->wp,wp_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fm_stored,fm_extended_in); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fp_stored,fp_extended_in); CHKERRXX(this->ierr);

        this->ierr=VecDuplicate(this->phi,&wm_extended_out); CHKERRXX(this->ierr); // memory increase local to the function wm_extended_out 7: total 7
        this->ierr=VecDuplicate(this->phi,&wp_extended_out); CHKERRXX(this->ierr);// memory increase local to the function wp_extended_out 8: total 8
        this->ierr=VecDuplicate(this->phi,&fm_extended_out); CHKERRXX(this->ierr);// memory increase local to the function fp_extended_out 9: total 9
        this->ierr=VecDuplicate(this->phi,&fp_extended_out); CHKERRXX(this->ierr);// memory increase local to the function fm_extended_out 10: total 10

        this->ierr=VecCopy(this->myDiffusion->wm,wm_extended_out); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->myDiffusion->wp,wp_extended_out); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fm_stored,fm_extended_out); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->fp_stored,fp_extended_out); CHKERRXX(this->ierr);


        bool extend=true;
        if(this->advance_fields_scft_advance_mask_level_set)
            extend=false;

        if(extend)
        {

            this->extend_petsc_vector(&this->phi_seed,&wm_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&wp_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&fm_extended_in);
            this->extend_petsc_vector(&this->phi_seed,&fp_extended_in);


            this->extend_petsc_vector(&this->negative_phi_seed,&wm_extended_out);
            this->extend_petsc_vector(&this->negative_phi_seed,&wp_extended_out);
            this->extend_petsc_vector(&this->negative_phi_seed,&fm_extended_out);
            this->extend_petsc_vector(&this->negative_phi_seed,&fp_extended_out);
        }
        else
        {

            this->scatter_petsc_vector(&wm_extended_in);
            this->scatter_petsc_vector(&wp_extended_in);
            this->scatter_petsc_vector(&fm_extended_in);
            this->scatter_petsc_vector(&fp_extended_in);
            this->scatter_petsc_vector(&wm_extended_in);
            this->scatter_petsc_vector(&wp_extended_in);
            this->scatter_petsc_vector(&fm_extended_in);
            this->scatter_petsc_vector(&fp_extended_in);

        }
        this->my_p4est_vtk_write_all_periodic_adapter(&fp_extended_in,"fp_extended_in",PETSC_FALSE);
        this->my_p4est_vtk_write_all_periodic_adapter(&fp_extended_out,"fp_extended_out",PETSC_FALSE);

        this->my_p4est_vtk_write_all_periodic_adapter(&fm_extended_in,"fm_extended_in",PETSC_FALSE);
        this->my_p4est_vtk_write_all_periodic_adapter(&fm_extended_out,"fm_extended_out",PETSC_FALSE);


        Vec f2integrate;
        this->ierr=VecDuplicate(this->phi,&f2integrate); CHKERRXX(this->ierr); // memory increase local to the function f2integrate 11: total 11

        PetscScalar *f2integrate_local;

        PetscScalar *wm_local_in;
        PetscScalar *wp_local_in;
        PetscScalar *fp_local_in;
        PetscScalar *fm_local_in;

        PetscScalar *wm_local_out;
        PetscScalar *wp_local_out;
        PetscScalar *fp_local_out;
        PetscScalar *fm_local_out;

        PetscScalar *dphi_local;

        this->ierr=VecGetArray(f2integrate,&f2integrate_local);  CHKERRXX(this->ierr);

        this->ierr=VecGetArray(wm_extended_in,&wm_local_in); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(wp_extended_in,&wp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fp_extended_in,&fp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fm_extended_in,&fm_local_in); CHKERRXX(this->ierr);


        this->ierr=VecGetArray(wm_extended_out,&wm_local_out); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(wp_extended_out,&wp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fp_extended_out,&fp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(fm_extended_out,&fm_local_out); CHKERRXX(this->ierr);


        this->ierr=VecGetArray(this->delta_phi_for_prediction,&dphi_local); CHKERRXX(this->ierr);


        this->myDiffusion->printDiffusionArray(wm_local_in,this->nodes->num_owned_indeps,"wm_local_in");
        this->myDiffusion->printDiffusionArray(wp_local_in,this->nodes->num_owned_indeps,"wp_local_in");
        this->myDiffusion->printDiffusionArray(fm_local_in,this->nodes->num_owned_indeps,"fm_local_in");
        this->myDiffusion->printDiffusionArray(fp_local_in,this->nodes->num_owned_indeps,"fp_local_in");

        this->myDiffusion->printDiffusionArray(wm_local_out,this->nodes->num_owned_indeps,"wm_local_out");
        this->myDiffusion->printDiffusionArray(wp_local_out,this->nodes->num_owned_indeps,"wp_local_out");
        this->myDiffusion->printDiffusionArray(fm_local_out,this->nodes->num_owned_indeps,"fm_local_out");
        this->myDiffusion->printDiffusionArray(fp_local_out,this->nodes->num_owned_indeps,"fp_local_out");


        //this->myDiffusion->printDiffusionArray(normal_norm_local,this->nodes->num_owned_indeps,"normal_norm_local");



        if(this->advance_fields_level_set)
        {

            for(int i=0;i<this->nodes->num_owned_indeps;i++)
            {

                // f2integrate_local[i]=(fp_local[i]*this->X_ab+fm_local[i]*wm_local[i]/normal_norm_local[i])*dphi_local[i];
                // f2integrate_local[i]=(fp_local[i]*this->X_ab/normal_norm_local[i])*dphi_local[i];
                // f2integrate_local[i]=fm_local[i]*this->X_ab/normal_norm_local[i]*dphi_local[i];

                if(dphi_local[i]<0)
                    f2integrate_local[i]=(fp_local_in[i]*this->X_ab)*dphi_local[i];///normal_norm_local[i];

                if(dphi_local[i]>0)
                    f2integrate_local[i]=(fp_local_out[i]*this->X_ab)*dphi_local[i];///normal_norm_local[i];

                if(dphi_local[i]==0)
                    f2integrate_local[i]=0;

            }
        }
        if(this->advance_fields_scft_advance_mask_level_set)
        {

            PetscScalar *phi_seed_local;
            this->ierr=VecGetArray(this->phi_seed_old_on_old_p4est,&phi_seed_local); CHKERRXX(this->ierr);
            double phi_seed_local_double;
            double a_p=(this->zeta_n_inverse*this->interaction_flag*this->xhi_w_p+1.00)/(0.5*this->X_ab*this->zeta_n_inverse+1);
            double a_m=2*this->interaction_flag*this->xhi_w_m/this->X_ab;
            double phi_w_y=0;
            double phi_w_s_1=0;
            double phi_w_s_2=0;
            double phi_w_1=0;
            double phi_w_2=0;

            if(this->terracing)
            {
                a_p=1.00/(0.5*this->X_ab*this->zeta_n_inverse+1.00);
                a_m=0;
            }

            double dH_dphi_wall;
            double dphi_wall_dphi;
            //            double exp_plus_phi;
            //            double exp_minus_phi;
            double tanh_phi;
            double tanh_phi_1,  tanh_phi_2;
            double tanh_phi_terrace;
            double tree_xmin;
            double tree_ymin;

#ifdef P4_TO_P8
            double tree_zmin;
#endif

            double x;
            double y;

#ifdef P4_TO_P8
            double z;
#endif

            for(int i=0;i<this->nodes->num_owned_indeps;i++)
            {
                phi_seed_local_double=phi_seed_local[i];
                dH_dphi_wall=wp_local_in[i]*a_p+wm_local_in[i]*a_m;
                //                exp_plus_phi=exp(this->alpha_wall*phi_seed_local_double);
                //                exp_minus_phi=exp(-this->alpha_wall*phi_seed_local_double);
                tanh_phi=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
                tanh_phi=tanh_phi*tanh_phi;
                dphi_wall_dphi=0.5*this->alpha_wall*(1.00-tanh_phi);
                f2integrate_local[i]=dH_dphi_wall*dphi_wall_dphi*dphi_local[i];

                if(!this->conserve_reaction_source_volume)
                {
                    f2integrate_local[i]+=dphi_wall_dphi*log(this->myDiffusion->getQForward())*dphi_local[i];
                }

                if(this->terracing)
                {
                    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
                    p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                    p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
                    tree_xmin = this->connectivity->vertices[3*v_mm + 0];
                    tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
                    tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

                    x = node_x_fr_i(node) + tree_xmin;
                    y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
                    z = node_z_fr_k(node) + tree_zmin;
#endif
                    tanh_phi_1=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
                    tanh_phi_2=MeanField::my_tanh_x(this->alpha_wall*(phi_seed_local_double+dphi_local[i]));
                    tanh_phi_terrace=MeanField::my_tanh_x(this->alpha_wall*(this->y_terrace_floor-y));
                    phi_w_y=0.5*(1+tanh_phi_terrace);
                    phi_w_s_1=0.5*(1+tanh_phi_1);
                    phi_w_s_2=0.5*(1+tanh_phi_2);
                    phi_w_1=MIN(1.00,phi_w_y+phi_w_s_1);
                    phi_w_2=MIN(1.00,phi_w_y+phi_w_s_2);
                    f2integrate_local[i]=(phi_w_2-phi_w_1)*dH_dphi_wall;
                }
            }

            this->ierr=VecRestoreArray(this->phi_seed_old_on_old_p4est,&phi_seed_local); CHKERRXX(this->ierr);
        }

        this->myDiffusion->printDiffusionArray(f2integrate_local,this->nodes->num_owned_indeps,"f2integrate_local");


        this->ierr=VecRestoreArray(f2integrate,&f2integrate_local);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(wm_extended_in,&wm_local_in); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(wp_extended_in,&wp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fp_extended_in,&fp_local_in); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fm_extended_in,&fm_local_in); CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(wm_extended_out,&wm_local_out); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(wp_extended_out,&wp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fp_extended_out,&fp_local_out); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(fm_extended_out,&fm_local_out); CHKERRXX(this->ierr);


        //this->ierr=VecRestoreArray(normal_norm,&normal_norm_local); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(f2integrate,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(f2integrate,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->de_predicted_from_level_set_change_current_step=this->de_predicted_from_level_set_change_next_step;
        this->de_predicted_from_level_set_change_current_step=this->integrate_over_interface(this->phi_seed,f2integrate);
        int Nx= (int)pow(2,this->max_level+log(this->nx_trees)/log(2));
        double nx_double=(double) Nx;
        this->de_predicted_from_level_set_change_current_step=this->de_predicted_from_level_set_change_current_step/this->myDiffusion->getV();

        if(this->advance_fields_scft_advance_mask_level_set)
        {

            this->de_predicted_from_level_set_change_current_step=this->de_predicted_from_level_set_change_next_step;
            Vec temp_ls;
            VecDuplicate(this->phi_seed,&temp_ls);
            VecSet(temp_ls,-1.00);
            this->scatter_petsc_vector(&temp_ls);
            this->de_predicted_from_level_set_change_current_step=integrate_over_negative_domain(this->p4est,this->nodes,temp_ls,f2integrate);
            VecDestroy(temp_ls);
            this->de_predicted_from_level_set_change_current_step=this->de_predicted_from_level_set_change_current_step/this->myDiffusion->getV();

            double mask_energy_new;
            double mask_energy_old;

            if(!this->terracing)
            {
                this->compute_mask_energy_from_level_set(mask_energy_new,&this->phi_seed_new_on_old_p4est);
                this->compute_mask_energy_from_level_set(mask_energy_old,&this->phi_seed_old_on_old_p4est);
            }
            else
            {
                this->compute_mask_energy_from_level_set_terracing(mask_energy_new,&this->phi_seed_new_on_old_p4est);
                this->compute_mask_energy_from_level_set_terracing(mask_energy_old,&this->phi_seed_old_on_old_p4est);
            }
            std::cout<<" E_new "<<mask_energy_new<<" E_old "<<mask_energy_old<<std::endl;


            this->de_mask_t= mask_energy_new-mask_energy_old;

            std::cout<<" DE "<<this->de_mask_t
                    <<" DE_PREDICTED "<<
                      this->de_predicted_from_level_set_change_current_step <<std::endl;

            this->prediction_error_level_set_t= 100*(ABS(this->de_predicted_from_level_set_change_current_step-(mask_energy_new-mask_energy_old))
                                                     /ABS(mask_energy_new-mask_energy_old));
            std::cout<<" prediction error [%]"<<this->prediction_error_level_set_t
                    <<std::endl;

        }
        this->ierr=VecDestroy(f2integrate); CHKERRXX(this->ierr); // memory decrease local to the function f2integrate 10: total 10

        //this->ierr=VecDestroy(normal_norm); CHKERRXX(this->ierr);

        this->ierr=VecDestroy(wm_extended_in); CHKERRXX(this->ierr); // memory decrease local to the function wm_extended_in 3: total 9
        this->ierr=VecDestroy(wp_extended_in); CHKERRXX(this->ierr);// memory decrease local to the function wp_extended_in 4: total 8
        this->ierr=VecDestroy(fm_extended_in); CHKERRXX(this->ierr);// memory decrease local to the function fm_extended_in 5: total 7
        this->ierr=VecDestroy(fp_extended_in); CHKERRXX(this->ierr);// memory decrease local to the function fp_extended_in 6: total 6

        this->ierr=VecDestroy(wm_extended_out); CHKERRXX(this->ierr); // memory decrease local to the function wm_extended_out 7: total 5
        this->ierr=VecDestroy(wp_extended_out); CHKERRXX(this->ierr);// memory decrease local to the function wp_extended_out 8: total 4
        this->ierr=VecDestroy(fm_extended_out); CHKERRXX(this->ierr);// memory decrease local to the function fm_extended_out 9: total 3
        this->ierr=VecDestroy(fp_extended_out); CHKERRXX(this->ierr);// memory decrease local to the function fp_extended_out 10: total 2


        this->ierr=VecDestroy(this->delta_phi_for_prediction); CHKERRXX(this->ierr); // memory decrease local to the function delta_phi 1: total 1
        //    this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr);
        //    this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr);
        //    this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(this->negative_phi_seed); CHKERRXX(this->ierr); // memory decrease local to the function negative_phi_seed 2: total 0

        this->ierr=VecDestroy(temp_phi); CHKERRXX(this->ierr);// destroy temp_phi

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end predict_energy_change_from_level_set_change"<<std::endl;
    }
}


int MeanField::compute_mask_energy_from_level_set(double &mask_energy, Vec *phi_ls)
{

    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {



        double E_p_compressible, E_p_wall,E_m_wall,E_wm,E_logQ,E_total;

        Vec energy_integrand,phiTemp, phi_wall_temp;

        this->ierr=VecDuplicate(this->phi,&phi_wall_temp); CHKERRXX(this->ierr);

        PetscScalar *polymer_mask_local,*phi_wall_temp_local;
        this->ierr=VecGetArray(*phi_ls,&polymer_mask_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(phi_wall_temp,&phi_wall_temp_local); CHKERRXX(this->ierr);

        double exp_plus_phi;
        double exp_minus_phi;
        double phi_seed_local_double;
        //double phi_wall_local_terrace;
        double tanh_phi;
        //        double tree_xmin;
        //        double tree_ymin;

        //#ifdef P4_TO_P8
        //        double tree_zmin;
        //#endif

        //        double x;
        //        double y;

        //#ifdef P4_TO_P8
        //        double z;
        //#endif

        this->myDiffusion->printDiffusionArray(polymer_mask_local,this->nodes->num_owned_indeps,"polymer_mask_local");

        for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
        {

            phi_seed_local_double=polymer_mask_local[i];


            tanh_phi=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
            // tanh_phi=(exp_plus_phi-exp_minus_phi)/(exp_plus_phi+exp_minus_phi);
            phi_wall_temp_local[i]=0.5*(1+tanh_phi);
        }
        this->ierr=VecRestoreArray(*phi_ls,&polymer_mask_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(phi_wall_temp,&phi_wall_temp_local); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" restores array set_advected_fields_from_level_set"<<std::endl;


        this->scatter_petsc_vector(&phi_wall_temp);
        this->myDiffusion->printDiffusionArrayFromVector(phi_ls,"ls");
        this->myDiffusion->printDiffusionArrayFromVector(&phi_wall_temp,"phi_wall_temp");

        this->ierr=VecDuplicate(this->myDiffusion->wm,&energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->myDiffusion->wm,energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;

        this->ierr=VecGetArray(energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->nodes->num_owned_indeps;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);


        this->ierr=VecDuplicate(this->phi,&phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecSet(phiTemp,-1.00); CHKERRXX(this->ierr);
        std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


        // energy_integrand
        this->ierr=VecGhostUpdateBegin(energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        // phi_temp
        this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


        E_wm=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        std::cout<<"Ewm "<<E_wm<<std::endl;
        E_wm=E_wm/this->myDiffusion->getV();

        std::cout<<"Ewm "<<E_wm<<std::endl;
        std::cout<<"ElogQ"<<this->myDiffusion->get_phi_bar()*(  -log(0.5*this->myDiffusion->getQBackward()+0.5*this->myDiffusion->getQForward()))<<std::endl;
        E_logQ=this->myDiffusion->get_phi_bar()*(-log(0.5*this->myDiffusion->getQBackward()+0.5*this->myDiffusion->getQForward()));
        E_total=E_wm+E_logQ;
        std::cout<<" Ew+ElogQ "<<E_total<<std::endl;


        std::cout<<" start to compute compressibility and wall terms "<<std::endl;
        Vec  wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_1,wall_force_scalar_2;
        PetscScalar minus_one=-1.00;
        PetscScalar minus_half=-0.5;




        // compute scalars
        compressibility_scalar=minus_half*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);
        wall_force_scalar_1=this->interaction_flag*this->xhi_w_p*this->zeta_n_inverse+1.00;
        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure

        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(phi_wall_temp,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,wall_force_scalar_1); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&wall_force);

        this->ierr=VecSet(energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(energy_integrand,this->myDiffusion->wp,this->myDiffusion->wp); CHKERRXX(this->ierr);
        this->ierr=VecScale(energy_integrand,compressibility_scalar); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&energy_integrand); CHKERRXX(this->ierr);

        E_p_compressible=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        E_p_compressible=E_p_compressible/this->myDiffusion->getV();


        this->ierr=VecSet(energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(energy_integrand,this->myDiffusion->wp,wall_force); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&energy_integrand);

        this->myDiffusion->printDiffusionArrayFromVector(&wall_force,"wall_force");
        this->myDiffusion->printDiffusionArrayFromVector(&energy_integrand,"energy_integrand");


        E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        E_p_wall=E_p_wall/this->myDiffusion->getV();


        this->ierr=VecSet(energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(energy_integrand,this->myDiffusion->wm,phi_wall_temp); CHKERRXX(this->ierr);
        this->ierr=VecScale(energy_integrand,this->interaction_flag*2*this->xhi_w_m/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&energy_integrand);

        E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        E_m_wall=E_m_wall/this->myDiffusion->getV();


        std::cout<<" E_p_compressible "<<E_p_compressible<<" E_p_wall "<<E_p_wall<<
                   " E_m_wall "<<E_m_wall<<std::endl;

        E_total=E_wm+E_logQ+E_p_compressible+E_p_wall+E_m_wall;

        this->ierr=VecDestroy(phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(energy_integrand); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(phi_wall_temp); CHKERRXX(this->ierr);

        mask_energy=E_total;
        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<E_total<<std::endl;
    }
}



int MeanField::compute_mask_energy_from_level_set_terracing(double &mask_energy, Vec *phi_ls)
{

    this->xhi_w_p=0.5*(this->xhi_w_a+this->xhi_w_b);
    this->xhi_w_m=0.5*(this->xhi_w_a-this->xhi_w_b);

    int n_games=1;
    for(int i_game=0;i_game<n_games;i_game++)
    {



        double E_p_compressible, E_p_wall,E_m_wall,E_wm,E_logQ,E_total;

        Vec energy_integrand,phiTemp, phi_wall_temp, phi_wall_temp_p, phi_wall_temp_m;

        this->ierr=VecDuplicate(this->phi,&phi_wall_temp); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&phi_wall_temp_p); CHKERRXX(this->ierr);
        this->ierr=VecDuplicate(this->phi,&phi_wall_temp_m); CHKERRXX(this->ierr);

        PetscScalar *polymer_mask_local,*phi_wall_temp_local;
        PetscScalar *phi_wall_temp_local_p;
        PetscScalar *phi_wall_temp_local_m;
        this->ierr=VecGetArray(*phi_ls,&polymer_mask_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(phi_wall_temp,&phi_wall_temp_local); CHKERRXX(this->ierr);

        this->ierr=VecGetArray(phi_wall_temp_p,&phi_wall_temp_local_p); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(phi_wall_temp_m,&phi_wall_temp_local_m); CHKERRXX(this->ierr);


        double exp_plus_phi;
        double exp_minus_phi;
        double phi_seed_local_double;
        double phi_wall_local_terrace;
        double tanh_phi;
        double tree_xmin;
        double tree_ymin;

#ifdef P4_TO_P8
        double tree_zmin;
#endif

        double x;
        double y;

#ifdef P4_TO_P8
        double z;
#endif

        this->myDiffusion->printDiffusionArray(polymer_mask_local,this->nodes->num_owned_indeps,"polymer_mask_local");

        for (p4est_locidx_t i = 0; i<this->nodes->num_owned_indeps; ++i)
        {

            phi_seed_local_double=polymer_mask_local[i];


            tanh_phi=MeanField::my_tanh_x(this->alpha_wall*phi_seed_local_double);
            // tanh_phi=(exp_plus_phi-exp_minus_phi)/(exp_plus_phi+exp_minus_phi);
            phi_wall_temp_local[i]=0.5*(1+tanh_phi);
            if(this->terracing)
            {

                p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
                p4est_topidx_t tree_id = node->p.piggy3.which_tree;
                p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
                tree_xmin = this->connectivity->vertices[3*v_mm + 0];
                tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
                tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

                x = node_x_fr_i(node) + tree_xmin;
                y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
                z = node_z_fr_k(node) + tree_zmin;
#endif
                phi_wall_local_terrace=0.5*(1+MeanField::my_tanh_x(this->alpha_wall*  (this->y_terrace_floor-y)));
                phi_wall_temp_local[i]=MIN(1.00,phi_wall_temp_local[i]+phi_wall_local_terrace);
                phi_wall_temp_local_m[i]=this->xhi_w_m*phi_wall_local_terrace;
                phi_wall_temp_local_p[i]=this->xhi_w_p*phi_wall_local_terrace;

            }
        }
        this->ierr=VecRestoreArray(*phi_ls,&polymer_mask_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(phi_wall_temp,&phi_wall_temp_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(phi_wall_temp_p,&phi_wall_temp_local_p); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(phi_wall_temp_m,&phi_wall_temp_local_m); CHKERRXX(this->ierr);

        std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" restores array set_advected_fields_from_level_set"<<std::endl;


        this->scatter_petsc_vector(&phi_wall_temp);
        this->scatter_petsc_vector(&phi_wall_temp_p);
        this->scatter_petsc_vector(&phi_wall_temp_m);


        this->myDiffusion->printDiffusionArrayFromVector(phi_ls,"ls");
        this->myDiffusion->printDiffusionArrayFromVector(&phi_wall_temp,"phi_wall_temp");

        this->ierr=VecDuplicate(this->myDiffusion->wm,&energy_integrand);  CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->myDiffusion->wm,energy_integrand);        CHKERRXX(this->ierr);
        //this->printDiffusionVector(&this->energy_integrand,"energy_integrand_zero_step");

        PetscScalar *energy_integrand_local;

        this->ierr=VecGetArray(energy_integrand,&energy_integrand_local);  CHKERRXX(this->ierr);
        for(int i=0;i<this->nodes->num_owned_indeps;i++)
        {
            energy_integrand_local[i]=energy_integrand_local[i]*energy_integrand_local[i];
        }

        //this->ierr= VecSetValues(this->energy_integrand,this->n_local_size_sol,this->ix_fromLocal2Global,energy_integrand_local,INSERT_VALUES);  CHKERRXX(this->ierr);

        this->ierr=VecRestoreArray(energy_integrand,&energy_integrand_local); CHKERRXX(this->ierr);
        this->ierr=VecScale(energy_integrand,1.00/this->X_ab); CHKERRXX(this->ierr);


        this->ierr=VecDuplicate(this->phi,&phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecSet(phiTemp,-1.00); CHKERRXX(this->ierr);
        std::cout<<this->mpi->mpirank<<" Energy Computation"<<std::endl;


        // energy_integrand
        this->ierr=VecGhostUpdateBegin(energy_integrand,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(energy_integrand,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

        // phi_temp
        this->ierr=VecGhostUpdateBegin(phiTemp,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(phiTemp,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);


        E_wm=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        std::cout<<"Ewm "<<E_wm<<std::endl;
        E_wm=E_wm/this->myDiffusion->getV();

        std::cout<<"Ewm "<<E_wm<<std::endl;
        std::cout<<"ElogQ"<<this->myDiffusion->get_phi_bar()*(  -log(0.5*this->myDiffusion->getQBackward()+0.5*this->myDiffusion->getQForward()))<<std::endl;
        E_logQ=this->myDiffusion->get_phi_bar()*(-log(0.5*this->myDiffusion->getQBackward()+0.5*this->myDiffusion->getQForward()));
        E_total=E_wm+E_logQ;
        std::cout<<" Ew+ElogQ "<<E_total<<std::endl;


        std::cout<<" start to compute compressibility and wall terms "<<std::endl;
        Vec  wall_force;
        PetscScalar compressibility_scalar;
        PetscScalar wall_force_scalar_2;
        PetscScalar minus_one=-1.00;
        PetscScalar minus_half=-0.5;



        // compute scalars
        compressibility_scalar=minus_half*this->zeta_n_inverse/((0.5*this->X_ab*this->zeta_n_inverse)+1);

        wall_force_scalar_2=0.5*this->X_ab*this->zeta_n_inverse+1.00;


        // Allocate memory using the apropriate petsc vector data structure

        this->ierr=VecDuplicate(this->phi,&wall_force); CHKERRXX(this->ierr);

        this->ierr=VecCopy(phi_wall_temp,wall_force); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(wall_force,this->zeta_n_inverse,phi_wall_temp_p); CHKERRXX(this->ierr);
        this->ierr=VecShift(wall_force,minus_one); CHKERRXX(this->ierr);
        this->ierr=VecScale(wall_force,1.00/wall_force_scalar_2); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&wall_force);

        this->ierr=VecSet(energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(energy_integrand,this->myDiffusion->wp,this->myDiffusion->wp); CHKERRXX(this->ierr);
        this->ierr=VecScale(energy_integrand,compressibility_scalar); CHKERRXX(this->ierr);

        this->scatter_petsc_vector(&energy_integrand); CHKERRXX(this->ierr);

        E_p_compressible=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        E_p_compressible=E_p_compressible/this->myDiffusion->getV();


        this->ierr=VecSet(energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(energy_integrand,this->myDiffusion->wp,wall_force); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&energy_integrand);

        this->myDiffusion->printDiffusionArrayFromVector(&wall_force,"wall_force");
        this->myDiffusion->printDiffusionArrayFromVector(&energy_integrand,"energy_integrand");


        E_p_wall=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        E_p_wall=E_p_wall/this->myDiffusion->getV();


        this->ierr=VecSet(energy_integrand,0.00); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(energy_integrand,this->myDiffusion->wm,phi_wall_temp_m); CHKERRXX(this->ierr);
        this->ierr=VecScale(energy_integrand,2/this->X_ab); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&energy_integrand);

        E_m_wall=integrate_over_negative_domain(this->p4est,this->nodes,phiTemp,energy_integrand);
        E_m_wall=E_m_wall/this->myDiffusion->getV();


        std::cout<<" E_p_compressible "<<E_p_compressible<<" E_p_wall "<<E_p_wall<<
                   " E_m_wall "<<E_m_wall<<std::endl;

        E_total=E_wm+E_logQ+E_p_compressible+E_p_wall+E_m_wall;

        this->ierr=VecDestroy(phiTemp); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(energy_integrand); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(wall_force); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(phi_wall_temp); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(phi_wall_temp_p); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(phi_wall_temp_m); CHKERRXX(this->ierr);

        mask_energy=E_total;
        std::cout<<" Rank Energy Disordered Energy"<<this->mpi->mpirank<<" "<<E_total<<std::endl;
    }
}


int MeanField::predict_energy_change_from_polymer_level_set_change()
{
    //    this->lambda_minus=0,this->lambda_plus=0;
    //    this->myDiffusion->set_lambda_plus_and_lambda_minus(0,0);
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 0"<<std::endl;

    Vec normal_norm;
    this->ierr=VecDuplicate(this->polymer_shape_stored,&normal_norm); CHKERRXX(this->ierr);                    // duplicate +1
    this->ierr=VecDuplicate(this->polymer_shape_stored,&this->delta_phi_for_prediction); CHKERRXX(this->ierr); // duplicate +2

    this->ierr=VecSet(this->delta_phi_for_prediction,0.00); CHKERRXX(this->ierr);
    this->ierr=VecAXPY(this->delta_phi_for_prediction,1.00,this->polymer_shape_stored);              CHKERRXX(this->ierr);
    this->ierr=VecAXPY(this->delta_phi_for_prediction,-1.00,this->new_polymer_shape_on_old_forest);  CHKERRXX(this->ierr);

    //scatter

    this->ierr=VecGhostUpdateBegin(this->delta_phi_for_prediction,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->delta_phi_for_prediction,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->myDiffusion->printDiffusionArrayFromVector(&this->polymer_shape_stored,"old_shape");
    this->myDiffusion->printDiffusionArrayFromVector(&this->new_polymer_shape_on_old_forest,"new_shape");
    this->myDiffusion->printDiffusionArrayFromVector(&this->delta_phi_for_prediction,"delta_phi");

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 1"<<std::endl;


    PetscScalar *normal_norm_local;

    this->ierr=VecGetArray(normal_norm,&normal_norm_local); CHKERRXX(this->ierr);// getArray +1




    this->ierr=VecDuplicate(this->polymer_shape_stored,&this->dmask_dx); CHKERRXX(this->ierr); // duplicate +3
    this->ierr=VecDuplicate(this->polymer_shape_stored,&this->dmask_dy); CHKERRXX(this->ierr);// duplicate +4
    this->ierr=VecDuplicate(this->polymer_shape_stored,&this->dmask_dz); CHKERRXX(this->ierr);// duplicate +5


    PetscScalar *dmask_local;
    this->ierr=VecGetArray(this->polymer_shape_stored,&dmask_local); CHKERRXX(this->ierr); // getArray +2

    this->ierr=VecGetArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // getArray +3
    this->ierr=VecGetArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); // getArray +4
    this->ierr=VecGetArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // getArray +5

    PetscScalar  normalize_normal;
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 2"<<std::endl;

    for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
    {
        this->dmask_dx_local[ix]=this->node_neighbors->neighbors[ix].dx_central(dmask_local);
        this->dmask_dy_local[ix]=this->node_neighbors->neighbors[ix].dy_central(dmask_local);
#ifdef P4_TO_P8
        this->dmask_dz_local[ix]=this->node_neighbors->neighbors[ix].dz_central(dmask_local);
#endif
        normalize_normal=this->dmask_dx_local[ix]*this->dmask_dx_local[ix]+
                this->dmask_dy_local[ix]*this->dmask_dy_local[ix];
#ifdef P4_TO_P8
        normalize_normal+= this->dmask_dz_local[ix]*this->dmask_dz_local[ix];
#endif
        normal_norm_local[ix]=pow(normalize_normal,0.5);
    }

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 3"<<std::endl;

    this->myDiffusion->printDiffusionArray(this->dmask_dx_local,this->nodes->num_owned_indeps,"phi_x");
    this->myDiffusion->printDiffusionArray(this->dmask_dy_local,this->nodes->num_owned_indeps,"phi_y");
    this->myDiffusion->printDiffusionArray(this->dmask_dz_local,this->nodes->num_owned_indeps,"phi_z");


    this->ierr=VecRestoreArray(this->polymer_shape_stored,&dmask_local);CHKERRXX(this->ierr);// restore array +1
    this->ierr=VecRestoreArray(this->dmask_dx,&this->dmask_dx_local); CHKERRXX(this->ierr); // restore array +2
    this->ierr=VecRestoreArray(this->dmask_dy,&this->dmask_dy_local); CHKERRXX(this->ierr); //restore array +3
    this->ierr=VecRestoreArray(this->dmask_dz,&this->dmask_dz_local); CHKERRXX(this->ierr); // restore array +4

    // scatter again

    this->ierr=VecGhostUpdateBegin(this->dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 4"<<std::endl;

    //this->print_normal_to_the_interface(&this->dmask_dx,&this->dmask_dy,&this->dmask_dz," normal");



    // Extend Petsc Vectors with level set of choice
    // for periodic domains it will just be a fake domain
    Vec wm_extended,wp_extended,fp_extended,fm_extended,q_last_extended,snn_extended;

    this->ierr=VecDuplicate(this->polymer_shape_stored,&wm_extended); CHKERRXX(this->ierr); // duplicate +6
    this->ierr=VecDuplicate(this->polymer_shape_stored,&wp_extended); CHKERRXX(this->ierr);// duplicate +7
    this->ierr=VecDuplicate(this->polymer_shape_stored,&fm_extended); CHKERRXX(this->ierr);// duplicate +8
    this->ierr=VecDuplicate(this->polymer_shape_stored,&fp_extended); CHKERRXX(this->ierr);// duplicate +9
    this->ierr=VecDuplicate(this->polymer_shape_stored,&q_last_extended); CHKERRXX(this->ierr);// duplicate +10
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
        this->ierr=VecDuplicate(this->polymer_shape_stored,&snn_extended); CHKERRXX(this->ierr);// duplicate +11


    this->ierr=VecCopy(this->wm_stored,wm_extended); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->wp_stored,wp_extended); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->fm_stored,fm_extended); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->fp_stored,fp_extended); CHKERRXX(this->ierr);
    this->ierr=VecCopy(this->last_q_stored,q_last_extended); CHKERRXX(this->ierr);
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
        this->ierr=VecCopy(this->snn,snn_extended); CHKERRXX(this->ierr);



    PetscBool extend_fields=PETSC_TRUE;

    if(extend_fields)
    {
        this->extend_petsc_vector(&this->polymer_shape_stored,&wm_extended);
        this->extend_petsc_vector(&this->polymer_shape_stored,&wp_extended);
        this->extend_petsc_vector(&this->polymer_shape_stored,&fm_extended);
        this->extend_petsc_vector(&this->polymer_shape_stored,&fp_extended);
        this->extend_petsc_vector(&this->polymer_shape_stored,&q_last_extended);
        if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
            this->extend_petsc_vector(&this->polymer_shape_stored,&snn_extended);
    }
    else
    {
        this->scatter_petsc_vector(&wm_extended);
        this->scatter_petsc_vector(&wp_extended);
        this->scatter_petsc_vector(&fm_extended);
        this->scatter_petsc_vector(&fp_extended);
        this->scatter_petsc_vector(&q_last_extended);
        if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
               this->scatter_petsc_vector(&snn_extended);
    }
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 5"<<std::endl;

    Vec f2integrate;
    this->ierr=VecDuplicate(this->phi,&f2integrate); CHKERRXX(this->ierr); // duplicate +12
    PetscScalar *f2integrate_local;

    Vec f2integrate_osher_and_santosa_term;
    this->ierr=VecDuplicate(this->phi,&f2integrate_osher_and_santosa_term); CHKERRXX(this->ierr);// duplicate +13
    PetscScalar *f2integrate_osher_and_santosa_term_local;

    Vec f2integrate_stress_term;
    if(this->myDiffusion->compute_stress_tensor_bool)
        this->ierr=VecDuplicate(this->phi,&f2integrate_stress_term); CHKERRXX(this->ierr);// duplicate +14
    PetscScalar *f2integrate_stress_term_local;

    Vec f2integrate_simple_term_wm;
    this->ierr=VecDuplicate(this->phi,&f2integrate_simple_term_wm); CHKERRXX(this->ierr);// duplicate +16
    PetscScalar *f2integrate_simple_term_wm_local;

    Vec f2integrate_simple_term_wp;
    this->ierr=VecDuplicate(this->phi,&f2integrate_simple_term_wp); CHKERRXX(this->ierr);// duplicate +17
    PetscScalar *f2integrate_simple_term_wp_local;


    Vec f2integrate_dew_dwp;
    this->ierr=VecDuplicate(this->phi,&f2integrate_dew_dwp); CHKERRXX(this->ierr);// duplicate +18
    PetscScalar *f2integrate_dew_dwp_local;

    Vec f2integrate_dew_dwm;
    this->ierr=VecDuplicate(this->phi,&f2integrate_dew_dwm); CHKERRXX(this->ierr);// duplicate +19
    PetscScalar *f2integrate_dew_dwm_local;

    Vec f2integrate_dlnQ_dwp;
    this->ierr=VecDuplicate(this->phi,&f2integrate_dlnQ_dwp); CHKERRXX(this->ierr);// duplicate +20
    PetscScalar *f2integrate_dlnQ_dwp_local;

    Vec f2integrate_dlnQ_dwm;
    this->ierr=VecDuplicate(this->phi,&f2integrate_dlnQ_dwm); CHKERRXX(this->ierr);// duplicate +21
    PetscScalar *f2integrate_dlnQ_dwm_local;
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 51"<<std::endl;


    this->ierr=VecGetArray(f2integrate_dew_dwp,&f2integrate_dew_dwp_local);  CHKERRXX(this->ierr); // getArray +6
    this->ierr=VecGetArray(f2integrate_dew_dwm,&f2integrate_dew_dwm_local);  CHKERRXX(this->ierr); // getArray +7
    this->ierr=VecGetArray(f2integrate_dlnQ_dwp,&f2integrate_dlnQ_dwp_local);  CHKERRXX(this->ierr);// getArray +8
    this->ierr=VecGetArray(f2integrate_dlnQ_dwm,&f2integrate_dlnQ_dwm_local);  CHKERRXX(this->ierr);// getArray +9



    Vec f2integrate_naive_term;
    this->ierr=VecDuplicate(this->phi,&f2integrate_naive_term); CHKERRXX(this->ierr);// duplicate +222
    PetscScalar *f2integrate_naive_term_local;




    PetscScalar *rho_a_local;
    PetscScalar *rho_b_local;
    PetscScalar *wm_local;
    PetscScalar *wp_local;
    PetscScalar *fp_local;
    PetscScalar *fm_local;
    PetscScalar *dphi_local;
    PetscScalar *q_last_extended_local;
    PetscScalar *snn_local;


    this->ierr=VecGetArray(this->rho_a_stored,&rho_a_local);CHKERRXX(this->ierr);// getArray +10
    this->ierr=VecGetArray(this->rho_b_stored,&rho_b_local);CHKERRXX(this->ierr); // getArray +11



    this->ierr=VecGetArray(f2integrate,&f2integrate_local);  CHKERRXX(this->ierr);// getArray +12
    this->ierr=VecGetArray(f2integrate_naive_term,&f2integrate_naive_term_local);  CHKERRXX(this->ierr);// getArray +13
    this->ierr=VecGetArray(f2integrate_osher_and_santosa_term,&f2integrate_osher_and_santosa_term_local);  CHKERRXX(this->ierr);// getArray +14
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
         this->ierr=VecGetArray(f2integrate_stress_term,&f2integrate_stress_term_local);  CHKERRXX(this->ierr);// getArray +15
    this->ierr=VecGetArray(f2integrate_simple_term_wm,&f2integrate_simple_term_wm_local);  CHKERRXX(this->ierr);// getArray +16
    this->ierr=VecGetArray(f2integrate_simple_term_wp,&f2integrate_simple_term_wp_local);  CHKERRXX(this->ierr);// getArray +17


    this->ierr=VecGetArray(wm_extended,&wm_local); CHKERRXX(this->ierr);// getArray +18
    this->ierr=VecGetArray(wp_extended,&wp_local); CHKERRXX(this->ierr);// getArray +19
    this->ierr=VecGetArray(fp_extended,&fp_local); CHKERRXX(this->ierr);// getArray +20
    this->ierr=VecGetArray(fm_extended,&fm_local); CHKERRXX(this->ierr);// getArray +21
    this->ierr=VecGetArray(this->delta_phi_for_prediction,&dphi_local); CHKERRXX(this->ierr); // getArray +22
    this->ierr=VecGetArray(q_last_extended,&q_last_extended_local); CHKERRXX(this->ierr);// getArray +23
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
         this->ierr=VecGetArray(snn_extended,&snn_local); CHKERRXX(this->ierr);// getArray +24

    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 52"<<std::endl;

    this->myDiffusion->printDiffusionArray(wm_local,this->nodes->num_owned_indeps,"wm_local");
    this->myDiffusion->printDiffusionArray(wp_local,this->nodes->num_owned_indeps,"wp_local");
    this->myDiffusion->printDiffusionArray(fm_local,this->nodes->num_owned_indeps,"fm_local");
    this->myDiffusion->printDiffusionArray(fp_local,this->nodes->num_owned_indeps,"fp_local");
    this->myDiffusion->printDiffusionArray(normal_norm_local,this->nodes->num_owned_indeps,"normal_norm_local");
    this->myDiffusion->printDiffusionArray(dphi_local,this->nodes->num_owned_indeps,"dphi");
    this->myDiffusion->printDiffusionArray(q_last_extended_local,this->nodes->num_owned_indeps,"q_last_extended_local");


    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 6"<<std::endl;



    double minus_one=-1.00;

    double w_av=1.00*this->myDiffusion->get_wp_average();
    for(int i=0;i<this->nodes->num_owned_indeps;i++)
    {


        //%%%%%%%%%%%%%%%%% dphi terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
               f2integrate_stress_term_local[i]=(minus_one*1.00/(this->myDiffusion->getV()*this->myDiffusion->getQForward()))*(snn_local[i])*dphi_local[i];

        f2integrate_osher_and_santosa_term_local[i]=(1.00/this->myDiffusion->getV())*((fp_local[i])*wp_local[i]+fm_local[i]*wm_local[i])*dphi_local[i];
        f2integrate_naive_term_local[i]=(minus_one*1.00/(this->myDiffusion->getV()*this->myDiffusion->getQForward()))*(q_last_extended_local[i])*dphi_local[i];
        f2integrate_simple_term_wm_local[i]=  (1/this->myDiffusion->getV())*(wm_local[i]*wm_local[i]/this->X_ab)*dphi_local[i];
        f2integrate_simple_term_wp_local[i]=  (1/this->myDiffusion->getV())*(-1.00)*(wp_local[i])*dphi_local[i];

        f2integrate_local[i]=f2integrate_naive_term_local[i]+f2integrate_simple_term_wp_local[i]+f2integrate_simple_term_wm_local[i];


        //%%%%%%%%%%%%%%%   dw terms   %%%%%%%%%%%%%%%%%%%%%%%%%%//

        f2integrate_dew_dwm_local[i]=minus_one*2.00*this->lambda_minus*fm_local[i]   *wm_local[i]/(this->myDiffusion->getV()*this->X_ab);
        f2integrate_dew_dwp_local[i]=-1.00*(this->lambda_plus*fp_local[i]-w_av)/(this->myDiffusion->getV());
        f2integrate_dlnQ_dwm_local[i]=minus_one*(rho_b_local[i]-rho_a_local[i])*this->lambda_minus*fm_local[i]/(this->myDiffusion->getV());
        f2integrate_dlnQ_dwp_local[i]=(rho_b_local[i]+rho_a_local[i])*(this->lambda_plus*fp_local[i]-w_av)/(this->myDiffusion->getV());


    }

    this->myDiffusion->printDiffusionArray(f2integrate_local,this->nodes->num_owned_indeps,"f2integrate_local");

    this->ierr=VecRestoreArray(this->rho_a_stored,&rho_a_local); // restore array +5
    this->ierr=VecRestoreArray(this->rho_b_stored,&rho_b_local);// restore array +6

    this->ierr=VecRestoreArray(f2integrate_dew_dwp,&f2integrate_dew_dwp_local);  CHKERRXX(this->ierr);// restore array +7
    this->ierr=VecRestoreArray(f2integrate_dew_dwm,&f2integrate_dew_dwm_local);  CHKERRXX(this->ierr);// restore array +8
    this->ierr=VecRestoreArray(f2integrate_dlnQ_dwp,&f2integrate_dlnQ_dwp_local);  CHKERRXX(this->ierr);// restore array +9
    this->ierr=VecRestoreArray(f2integrate_dlnQ_dwm,&f2integrate_dlnQ_dwm_local);  CHKERRXX(this->ierr);// restore array +10


    this->ierr=VecRestoreArray(f2integrate_simple_term_wm,&f2integrate_simple_term_wm_local); CHKERRXX(this->ierr);// restore array +11
    this->ierr=VecRestoreArray(f2integrate_simple_term_wp,&f2integrate_simple_term_wp_local); CHKERRXX(this->ierr);// restore array +12

    this->ierr=VecRestoreArray(f2integrate,&f2integrate_local);  CHKERRXX(this->ierr);// restore array +13
    this->ierr=VecRestoreArray(f2integrate_osher_and_santosa_term,&f2integrate_osher_and_santosa_term_local);  CHKERRXX(this->ierr); // restore array +14
    this->ierr=VecRestoreArray(f2integrate_naive_term,&f2integrate_naive_term_local);  CHKERRXX(this->ierr);// restore array +15
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
         this->ierr=VecRestoreArray(f2integrate_stress_term,&f2integrate_stress_term_local);  CHKERRXX(this->ierr);// restore array +16

    this->ierr=VecRestoreArray(wm_extended,&wm_local); CHKERRXX(this->ierr);// restore array +17
    this->ierr=VecRestoreArray(wp_extended,&wp_local); CHKERRXX(this->ierr);// restore array +18
    this->ierr=VecRestoreArray(fp_extended,&fp_local); CHKERRXX(this->ierr);// restore array +19
    this->ierr=VecRestoreArray(fm_extended,&fm_local); CHKERRXX(this->ierr);// restore array +20
    this->ierr=VecRestoreArray(normal_norm,&normal_norm_local); CHKERRXX(this->ierr);// restore array +21
    this->ierr=VecRestoreArray(this->delta_phi_for_prediction,&dphi_local); CHKERRXX(this->ierr);// restore array +22
    this->ierr=VecRestoreArray(q_last_extended,&q_last_extended_local); CHKERRXX(this->ierr);// restore array +23

    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
          this->ierr=VecRestoreArray(snn_extended,&snn_local); CHKERRXX(this->ierr);// restore array +24


    //%%%%%%%%%%%%%% dphi terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    this->scatter_petsc_vector(&f2integrate);
    this->scatter_petsc_vector(&f2integrate_osher_and_santosa_term);
    this->scatter_petsc_vector(&f2integrate_naive_term);
    this->scatter_petsc_vector(&f2integrate_simple_term_wm);
    this->scatter_petsc_vector(&f2integrate_simple_term_wp);

    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
            this->scatter_petsc_vector(&f2integrate_stress_term);

    //%%%%%%%%%%%%%% dw terms   %%%%%%%%%%%%%%%%%%%%%%%%%%%//

    this->scatter_petsc_vector(&f2integrate_dew_dwm);
    this->scatter_petsc_vector(&f2integrate_dew_dwp);
    this->scatter_petsc_vector(&f2integrate_dlnQ_dwm);
    this->scatter_petsc_vector(&f2integrate_dlnQ_dwp);


    this->interpolate_and_print_vec_to_uniform_grid(&this->delta_phi_for_prediction,"v_phi",false);
    this->interpolate_and_print_vec_to_uniform_grid(&f2integrate,"dphi",false);
    this->interpolate_and_print_vec_to_uniform_grid(&f2integrate_osher_and_santosa_term,"dphi_os",false);
    this->interpolate_and_print_vec_to_uniform_grid(&f2integrate_naive_term,"dphi_naive",false);
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
        this->interpolate_and_print_vec_to_uniform_grid(&f2integrate_stress_term,"dphi_stress",false);


    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start predict_energy_change_from_polymer_level_set_change 7"<<std::endl;


    this->de_predicted_from_level_set_change_current_step=this->de_predicted_from_level_set_change_next_step;

    this->de_predicted_from_level_set_change_current_step=this->integrate_over_interface(this->polymer_shape_stored,f2integrate);
    this->osher_and_santosa_term=this->integrate_over_interface(this->polymer_shape_stored,f2integrate_osher_and_santosa_term);
    this->naive_term=this->integrate_over_interface(this->polymer_shape_stored,f2integrate_naive_term);
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
         this->stress_term=this->integrate_over_interface(this->polymer_shape_stored,f2integrate_stress_term);
    this->simple_term_wm=this->integrate_over_interface(this->polymer_shape_stored,f2integrate_simple_term_wm);
    this->simple_term_wp=this->integrate_over_interface(this->polymer_shape_stored,f2integrate_simple_term_wp);




    std::cout<< "this->de_predicted_from_level_set_change_current_step "<<this->de_predicted_from_level_set_change_current_step<<std::endl
             <<" osher_and_santosa_term  "<<this->osher_and_santosa_term<<std::endl
            <<" naive_term  "<<this->naive_term<<std::endl
           <<" stress_term "<<this->stress_term<<std::endl
          <<" simple_term wm"<<this->simple_term_wm<<std::endl
         <<" simple_term wp"<<this->simple_term_wp<<std::endl;



    //if(this->conserve_shape_volume)
    //    this->de_predicted_from_level_set_change_current_step/=this->myDiffusion->getV();

    if(!this->conserve_shape_volume )
    {
        std::cout<<" de predicted before correction"<<this->de_predicted_from_level_set_change_current_step<<std::endl;

        //--------correct for log Q-------------------------------//

        Vec temp_f;
        this->ierr=VecDuplicate(this->phi,&temp_f); CHKERRXX(this->ierr); // duplicate +23
        this->ierr=VecSet(temp_f,1.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&temp_f);
        double v_new=integrate_over_negative_domain(this->p4est,this->nodes,this->new_polymer_shape_on_old_forest,temp_f);
        double v_old=this->myDiffusion->getV();
        this->correction_term_for_uncounstrained_volume_log_Q=minus_one*minus_one*(v_new-v_old)/v_old;

        this->ierr=VecDestroy(temp_f); CHKERRXX(this->ierr);// destroy +4
        std::cout<<" correction_term_for_uncounstrained_volume log Q "<<this->correction_term_for_uncounstrained_volume_log_Q<<std::endl;
        this->de_predicted_from_level_set_change_current_step+=this->correction_term_for_uncounstrained_volume_log_Q;


        //------correct for the simple term---------------------//


        double dv=(v_new-v_old);
        double v2=(v_old*v_old);
        double e_w_old=this->myDiffusion->getE_w()*this->myDiffusion->getV();

        this->correction_term_for_uncounstrained_volume_E_w=minus_one*dv/v2*e_w_old;
        std::cout<<" correction_term_for_uncounstrained_volume Ew "<<this->correction_term_for_uncounstrained_volume_E_w<<std::endl;
        this->de_predicted_from_level_set_change_current_step+=this->correction_term_for_uncounstrained_volume_E_w;


    }

    std::cout<<" de predicted "<<this->de_predicted_from_level_set_change_current_step<<std::endl;

    int Nx= (int)pow(2,this->max_level+log(this->nx_trees)/log(2));
    double nx_double=(double) Nx;

    // compute some average values
    this->q_surface_mean_value=this->integrate_over_interface(this->polymer_shape_stored,q_last_extended);
    std::cout<<" q surface mean value "<<this->q_surface_mean_value<<" "
            <<this->integrate_constant_over_interface(this->polymer_shape_stored)<<" "
           <<this->integrate_constant_over_interface(this->phi_seed)<<" "<<std::endl;
    this->q_surface_mean_value=this->q_surface_mean_value/this->integrate_constant_over_interface(this->polymer_shape_stored);

    std::cout<<" q surface mean value "<<this->q_surface_mean_value<<std::endl;

    this->ierr=VecDuplicate(this->last_q_stored,&this->last_q_stored_normalized_surface); CHKERRXX(this->ierr); // duplicate +24
    this->ierr=VecCopy(this->last_q_stored,this->last_q_stored_normalized_surface); CHKERRXX(this->ierr);
    this->ierr=VecShift(this->last_q_stored_normalized_surface,this->q_surface_mean_value); CHKERRXX(this->ierr);
    this->ierr=VecScale(this->last_q_stored_normalized_surface,1.00/this->q_surface_mean_value); CHKERRXX(this->ierr);

    this->scatter_petsc_vector(&this->last_q_stored_normalized_surface); CHKERRXX(this->ierr);

    this->my_p4est_vtk_write_all_periodic_adapter_psdt(&this->last_q_stored_normalized_surface,"last_q_normalized",PETSC_FALSE);


    // dphi terms //
    this->my_delta_H->dEwmdphi=this->simple_term_wm;
    this->my_delta_H->dEwpdphi=this->simple_term_wp;

    this->my_delta_H->dEwdphi=this->my_delta_H->dEwmdphi+this->my_delta_H->dEwpdphi+
            this->correction_term_for_uncounstrained_volume_E_w;
    this->my_delta_H->dlnQdphi_simple_term=this->naive_term+this->correction_term_for_uncounstrained_volume_log_Q;
    this->my_delta_H->dlnQdphi_stress_term=this->stress_term;
    this->my_delta_H->dlnQdphi=-this->my_delta_H->dlnQdphi_simple_term+this->my_delta_H->dlnQdphi_stress_term;


    //combine dphi terms
    this->my_delta_H->dHdphi=this->my_delta_H->dEwdphi+this->my_delta_H->dlnQdphi;

    // dw terms //
    this->my_delta_H->dEwdwm=integrate_over_negative_domain(this->p4est,this->nodes,this->polymer_shape_stored,f2integrate_dew_dwm);
    this->my_delta_H->dEwdwp=integrate_over_negative_domain(this->p4est,this->nodes,this->polymer_shape_stored,f2integrate_dew_dwp);
    this->my_delta_H->dlnQdwm=integrate_over_negative_domain(this->p4est,this->nodes,this->polymer_shape_stored,f2integrate_dlnQ_dwm);
    this->my_delta_H->dlnQdwp=integrate_over_negative_domain(this->p4est,this->nodes,this->polymer_shape_stored,f2integrate_dlnQ_dwp);
    this->my_delta_H->dEwdw=this->my_delta_H->dEwdwm+this->my_delta_H->dEwdwp;
    this->my_delta_H->dlnQdw=this->my_delta_H->dlnQdwm+this->my_delta_H->dlnQdwp;

    //combine dw terms
    this->my_delta_H->dHdw=this->my_delta_H->dEwdw+this->my_delta_H->dlnQdw;


    // combine dphi terms and dw terms
    this->my_delta_H->dEwPredicted=this->my_delta_H->dEwdw+this->my_delta_H->dEwdphi;
    this->my_delta_H->dlnQPredicted=this->my_delta_H->dlnQdphi+this->my_delta_H->dlnQdw;



    // this->de_predicted_from_level_set_change_current_step=this->de_predicted_from_level_set_change_current_step/this->myDiffusion->getV();

    this->ierr=VecDestroy(f2integrate); CHKERRXX(this->ierr);// destroy +5
    this->ierr=VecDestroy(f2integrate_osher_and_santosa_term); CHKERRXX(this->ierr);// destroy +6

    this->ierr=VecDestroy(f2integrate_naive_term); CHKERRXX(this->ierr);// destroy +7

    this->ierr=VecDestroy(f2integrate_simple_term_wp); CHKERRXX(this->ierr);// destroy +8
    this->ierr=VecDestroy(f2integrate_simple_term_wm); CHKERRXX(this->ierr);// destroy +9

    this->ierr=VecDestroy(f2integrate_dew_dwm); CHKERRXX(this->ierr);// destroy +10
    this->ierr=VecDestroy(f2integrate_dew_dwp); CHKERRXX(this->ierr);// destroy +11
    this->ierr=VecDestroy(f2integrate_dlnQ_dwm); CHKERRXX(this->ierr);// destroy +12
    this->ierr=VecDestroy(f2integrate_dlnQ_dwp); CHKERRXX(this->ierr);// destroy +13


    this->ierr=VecDestroy(normal_norm); CHKERRXX(this->ierr);// destroy +14
    this->ierr=VecDestroy(wm_extended); CHKERRXX(this->ierr);// destroy +15
    this->ierr=VecDestroy(wp_extended); CHKERRXX(this->ierr);// destroy +16
    this->ierr=VecDestroy(fm_extended); CHKERRXX(this->ierr);// destroy +17
    this->ierr=VecDestroy(fp_extended); CHKERRXX(this->ierr);// destroy +18

    this->ierr=VecDestroy(q_last_extended); CHKERRXX(this->ierr);// destroy +19
    if(this->myDiffusion->compute_stress_tensor_bool && mod(this->i_mean_field_iteration,this->stress_tensor_computation_period)==0)
     {
        this->ierr=VecDestroy(f2integrate_stress_term); CHKERRXX(this->ierr);// destroy +21
        this->ierr=VecDestroy(this->snn);       CHKERRXX(this->ierr);// destroy +22
        this->ierr=VecDestroy(snn_extended);       CHKERRXX(this->ierr);// destroy +23
    }
    this->ierr=VecDestroy(this->dmask_dx); CHKERRXX(this->ierr); // destroy +1
    this->ierr=VecDestroy(this->dmask_dy); CHKERRXX(this->ierr);// destroy +2
    this->ierr=VecDestroy(this->dmask_dz); CHKERRXX(this->ierr);// destroy +3

    this->ierr=VecDestroy(this->delta_phi_for_prediction); CHKERRXX(this->ierr);// destroy +24
    this->ierr=VecDestroy(this->last_q_stored_normalized_surface); CHKERRXX(this->ierr);// destroy +25
    std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" end predict_energy_change_from_polymer_level_set_change 8"<<std::endl;

}




int MeanField::extend_petsc_vector(Vec *phi_ls,Vec * v2extend, int number_of_bands_extension)
{

    int n_local_phi_ls,n_local_v2extend,n_global_phi_ls,n_global_v2extend;

    this->ierr=VecGetLocalSize(*phi_ls,&n_local_phi_ls); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(*v2extend,&n_local_v2extend); CHKERRXX(this->ierr);

    this->ierr=VecGetSize(*phi_ls,&n_global_phi_ls); CHKERRXX(this->ierr);
    this->ierr=VecGetSize(*v2extend,&n_global_v2extend); CHKERRXX(this->ierr);

    std::cout<<" phi ls"<<n_local_phi_ls<<" "<<n_global_phi_ls<<std::endl;
    std::cout<<" v2extend "<<n_local_v2extend<<" "<<n_global_v2extend<<std::endl;

    int extension_order=0;

    my_p4est_level_set ls(this->node_neighbors);
    Vec bc_vec_fake;
    this->ierr= VecDuplicate(*phi_ls,&bc_vec_fake); CHKERRXX(this->ierr);
    this->ierr=VecSet(bc_vec_fake,0.00);              CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    ls.extend_Over_Interface(*phi_ls,*v2extend,NEUMANN,bc_vec_fake,extension_order,number_of_bands_extension);

    this->ierr=VecGhostUpdateBegin(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);
}

int MeanField::extend_petsc_vector_with_stride(Vec *phi_ls,Vec * v2extend, int number_of_bands_extension)
{

    int n_local_phi_ls,n_local_v2extend,n_global_phi_ls,n_global_v2extend;

    this->ierr=VecGetLocalSize(*phi_ls,&n_local_phi_ls); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(*v2extend,&n_local_v2extend); CHKERRXX(this->ierr);

    this->ierr=VecGetSize(*phi_ls,&n_global_phi_ls); CHKERRXX(this->ierr);
    this->ierr=VecGetSize(*v2extend,&n_global_v2extend); CHKERRXX(this->ierr);

    std::cout<<" phi ls"<<n_local_phi_ls<<" "<<n_global_phi_ls<<std::endl;
    std::cout<<" v2extend "<<n_local_v2extend<<" "<<n_global_v2extend<<std::endl;

    int extension_order=0;

    my_p4est_level_set ls(this->node_neighbors);
    Vec bc_vec_fake;
    this->ierr= VecDuplicate(*phi_ls,&bc_vec_fake); CHKERRXX(this->ierr);
    this->ierr=VecSet(bc_vec_fake,0.00);              CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    ls.extend_Over_Interface_With_Stride(*phi_ls,*v2extend,NEUMANN,this->neuman_stride,extension_order,number_of_bands_extension);

    this->ierr=VecGhostUpdateBegin(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2extend,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);
}


int MeanField::scatter_petsc_vectors(int num,...)
{
    va_list arguments;
    va_start(arguments,num);

    for(int i_arg=0;i_arg<num;i_arg++)
    {
        //        this->ierr=VecGhostUpdateEnd(*va_arg(arguments,*Vec),INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        //        this->ierr=VecGhostUpdateEnd(*va_arg(arguments,*Vec),INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }
    va_end ( arguments );
}

int MeanField::scatter_petsc_vector(Vec *v2scatter)
{
    //std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" start to scatter "<<std::endl;

    this->ierr=VecGhostUpdateBegin(*v2scatter,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2scatter,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    //std::cout<<this->i_mean_field_iteration<<" "<<this->mpi->mpirank<<" finish to scatter "<<std::endl;

}

int MeanField::segmentDiffusionPotentials()
{
    Vec Icell; Vec wremesh;

    VecDuplicate(this->myDiffusion->wp,&wremesh);
    VecCopy(this->myDiffusion->wp,wremesh);
    VecAXPBY(wremesh,1,1,this->myDiffusion->wm);

    this->ierr=VecGhostUpdateBegin(wremesh,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(wremesh,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    //    VecAssemblyBegin(wremesh);
    //    VecAssemblyEnd(wremesh);


    PetscScalar *wremesh_local;
    PetscScalar *Icell_local;
    VecGetArray(wremesh,&wremesh_local);
    PetscInt wremesh_size,w_remesh_local_size;
    VecGetSize(wremesh,&wremesh_size);
    VecGetLocalSize(wremesh,&w_remesh_local_size);

    VecCreateGhostCells(this->p4est,this->ghost,&Icell);
    this->ierr = VecDuplicate(Icell, &this->Ibin); CHKERRXX(this->ierr);

    //this->Icell_local=new PetscScalar[this->myDiffusion->get_n_local_size_sol()];
    ierr=VecGetArray(Icell,&Icell_local); CHKERRXX(ierr);


    PetscInt icell_size,icell_local_size;
    VecGetSize(Icell,&icell_size);
    VecGetLocalSize(Icell,&icell_local_size);


    const p4est_locidx_t *q2n = this->nodes->local_nodes;
    int jj=0;
    for (p4est_locidx_t i = p4est->first_local_tree; i<=p4est->last_local_tree; i++)
    {
        p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];
        std::cout<<"tree2->quadrants.elem_count"<<tree2->quadrants.elem_count<<std::endl;

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


#ifdef P4_TO_P8
            Icell_local[jj]=(1.00/8.00)*
                    (wremesh_local[n0]+wremesh_local[n1]+wremesh_local[n2]+wremesh_local[n3]+wremesh_local[n4]
                     +wremesh_local[n5]+wremesh_local[n6]+wremesh_local[n7]);
#else
            Icell_local[jj]=(1.00/4.00)*
                    (wremesh_local[n0]+wremesh_local[n1]+wremesh_local[n2]+wremesh_local[n3]);
#endif


            jj++;
        }
    }

    ierr = VecGhostUpdateBegin(Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    std::ostringstream oss; oss << "Icell" << p4est->mpisize<<"."<<this->i_mean_field_iteration;
    std::cout<<"_I am here "<<  oss.str()<<std::endl;

    this->myDiffusion->printDiffusionArray(Icell_local,icell_local_size,"icell_local");
    this->myDiffusion->printArrayOnForestOctants2TextFile(Icell_local,wremesh_local,icell_local_size,"icell_nodes_local");

    my_p4est_vtk_write_all(this->p4est_visualization, this->nodes_visualization,this->ghost_visualization    ,
                           P4EST_TRUE, P4EST_TRUE,
                           0, 1, oss.str().c_str(),
                           VTK_CELL_DATA, "I_cell", Icell_local);

    ierr = VecRestoreArray(wremesh, &wremesh_local); CHKERRXX(ierr);
    ierr = VecRestoreArray(Icell,  &Icell_local); CHKERRXX(ierr);


    PetscInt    i_max;
    PetscScalar x_max;

    PetscInt    i_min;
    PetscScalar x_min;

    VecMax(Icell,&i_max,&x_max);
    VecMin(Icell,&i_min,&x_min);

    this->c1_global=x_max;
    this->c2_global=x_min;

    this->e1_global=0;
    this->e2_global=0;

    this->e1_global_2=10;
    this->e2_global_2=10;

    this->computeSegmentationError(&Icell);

    while((this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)>0.01
          ||(this->e1_global_2+this->e2_global_2)<(this->e1_global+this->e2_global) )
    {
        this->e1_global_2=this->e1_global;
        this->e2_global_2=this->e2_global;
        this->computeSegmentationError(&Icell);
        std::cout<<"Kmeans Error";
        std::cout<<(this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)<<std::endl;
    }
    //  this->myDiffusion->printDiffusionVector(&this->Ibin,"IbinAfterSegmentationCell");
    std::ostringstream oss2; oss2 << "Ibin" << p4est->mpisize<<"."<<this->i_mean_field_iteration;
    std::cout<<"I am here "<<  oss2.str()<<std::endl;
    this->ierr= VecGetArray(this->Ibin,&this->Ibin_local); CHKERRXX(this->ierr);
    std::cout<<" write to vtk "<<std::endl;
    if(false)
    {
        if(this->myMeanFieldPlan->periodic_xyz)
            my_p4est_vtk_write_all(this->p4est_visualization, this->nodes_visualization,this->ghost_visualization    ,
                                   P4EST_TRUE, P4EST_TRUE,
                                   0, 1, oss2.str().c_str(),
                                   VTK_CELL_DATA, "Ibin", this->Ibin_local);
        else
            my_p4est_vtk_write_all(this->p4est, this->nodes,this->ghost,
                                   P4EST_TRUE, P4EST_TRUE,
                                   0, 1, oss2.str().c_str(),
                                   VTK_CELL_DATA, "Ibin", this->Ibin_local);
    }
    std::cout<<" finished to write to vtk "<<std::endl;
    VecRestoreArray(this->Ibin,&this->Ibin_local);


}


int MeanField::segmentDiffusionPotentialsW(Vec w2Segment, Vec *IBinSegmented)
{

    Vec wremesh;
    Vec Icell;

    this->ierr=VecDuplicate(w2Segment,&wremesh);CHKERRXX(this->ierr);
    this->ierr=VecCopy(w2Segment,wremesh);CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(wremesh,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(wremesh,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);

    //    this->ierr=VecAssemblyBegin(wremesh);CHKERRXX(this->ierr);
    //    this->ierr=VecAssemblyEnd(wremesh);CHKERRXX(this->ierr);

    PetscScalar *wremesh_local;
    PetscScalar *Icell_local;
    this->ierr=VecGetArray(wremesh,&wremesh_local);CHKERRXX(this->ierr);
    PetscInt wremesh_size,w_remesh_local_size;
    this->ierr=VecGetSize(wremesh,&wremesh_size);CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(wremesh,&w_remesh_local_size);CHKERRXX(this->ierr);

    this->ierr=VecCreateGhostCells(this->p4est,this->ghost,&Icell); CHKERRXX(this->ierr);
    this->ierr = VecDuplicate(Icell, IBinSegmented); CHKERRXX(this->ierr); CHKERRXX(this->ierr);

    //this->Icell_local=new PetscScalar[this->myDiffusion->get_n_local_size_sol()];
    ierr=VecGetArray(Icell,&Icell_local); CHKERRXX(ierr);


    PetscInt icell_size,icell_local_size;
    this->ierr=VecGetSize(Icell,&icell_size); CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(Icell,&icell_local_size); CHKERRXX(this->ierr);


    const p4est_locidx_t *q2n = this->nodes->local_nodes;
    int jj=0;
    for (p4est_locidx_t i = p4est->first_local_tree; i<=p4est->last_local_tree; i++)
    {
        p4est_tree_t *tree2 = (p4est_tree_t*)sc_array_index(p4est->trees, i);
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];
        std::cout<<"tree2->quadrants.elem_count"<<tree2->quadrants.elem_count<<std::endl;

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


#ifdef P4_TO_P8
            Icell_local[jj]=(1.00/8.00)*
                    (wremesh_local[n0]+wremesh_local[n1]+wremesh_local[n2]+wremesh_local[n3]+wremesh_local[n4]
                     +wremesh_local[n5]+wremesh_local[n6]+wremesh_local[n7]);
#else
            Icell_local[jj]=(1.00/4.00)*
                    (wremesh_local[n0]+wremesh_local[n1]+wremesh_local[n2]+wremesh_local[n3]);
#endif


            jj++;
        }
    }

    //    this->ierr = VecGhostUpdateBegin(Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    //    this->ierr= VecGhostUpdateEnd(Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    std::ostringstream oss; oss << "Icell" << p4est->mpirank;//<<"_"<<this->i_mean_field_iteration;
    std::cout<<"_I am here "<<  oss.str()<<std::endl;

    //    this->myDiffusion->printDiffusionArray(this->Icell_local,icell_local_size,"icell_local");
    //    this->myDiffusion->printArrayOnForestOctants2TextFile(this->Icell_local,this->wremesh_local,icell_local_size,"icell_nodes_local");
    if(false)
    {
        if(this->myMeanFieldPlan->periodic_xyz)
            my_p4est_vtk_write_all(this->p4est_visualization, this->nodes_visualization,this->ghost_visualization    ,
                                   P4EST_TRUE, P4EST_TRUE,
                                   0, 1, oss.str().c_str(),
                                   VTK_CELL_DATA, "I_cell", Icell_local);
        else
            my_p4est_vtk_write_all(this->p4est, this->nodes,this->ghost,
                                   P4EST_TRUE, P4EST_TRUE,
                                   0, 1, oss.str().c_str(),
                                   VTK_CELL_DATA, "I_cell", Icell_local);
    }
    this->ierr= VecRestoreArray(wremesh, &wremesh_local); CHKERRXX(ierr);
    this->ierr= VecRestoreArray(Icell, &Icell_local); CHKERRXX(ierr);
    this->ierr= VecGhostUpdateBegin(Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr= VecGhostUpdateEnd(Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    //   this->ierr= VecAssemblyBegin(Icell); CHKERRXX(this->ierr);
    //   this->ierr= VecAssemblyEnd(Icell);CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(wremesh,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(wremesh,INSERT_VALUES,SCATTER_FORWARD);   CHKERRXX(this->ierr);
    //   this->ierr= VecAssemblyBegin(wremesh);CHKERRXX(this->ierr);
    //   this->ierr= VecAssemblyEnd(wremesh);CHKERRXX(this->ierr);

    PetscInt    i_max;
    PetscScalar x_max;

    PetscInt    i_min;
    PetscScalar x_min;

    this->ierr=VecMax(Icell,&i_max,&x_max);CHKERRXX(this->ierr);
    this->ierr=VecMin(Icell,&i_min,&x_min);CHKERRXX(this->ierr);

    this->c1_global=x_max;
    this->c2_global=x_min;

    this->e1_global=0;
    this->e2_global=0;

    this->e1_global_2=10;
    this->e2_global_2=10;

    this->computeSegmentationErrorW(IBinSegmented,&Icell);

    while((this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)>0.01
          ||(this->e1_global_2+this->e2_global_2)<(this->e1_global+this->e2_global) )
    {
        this->e1_global_2=this->e1_global;
        this->e2_global_2=this->e2_global;
        this->computeSegmentationErrorW(IBinSegmented,&Icell);

        std::cout<<"Kmeans Error";
        std::cout<<(this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)<<std::endl;
        std::cout<<" my colours "<<this->c1_global<<" "<<this->c2_global<<std::endl;
        std::cout<<" my areas "<<this->A1_global<<" "<<this->A2_global<<std::endl;
        std::cout<<" my_count "<<this->n_colour1_global<<" "<<this->n_colour2_global<<std::endl;
    }
    //this->myDiffusion->printDiffusionVector(IBinSegmented,"IbinAfterSegmentationCell");
    std::ostringstream oss2; oss2 << "Ibin" << p4est->mpirank;//<<"_"<<this->i_mean_field_iteration;
    std::cout<<"I am here "<<  oss2.str()<<std::endl;

    if(false)
    {
        this->ierr=VecGetArray(*IBinSegmented,&this->Ibin_local);CHKERRXX(this->ierr);

        if(this->myMeanFieldPlan->periodic_xyz)
            my_p4est_vtk_write_all(this->p4est_visualization, this->nodes_visualization,this->ghost_visualization    ,
                                   P4EST_TRUE, P4EST_TRUE,
                                   0, 1, oss2.str().c_str(),
                                   VTK_CELL_DATA, "Ibin", this->Ibin_local);
        else
            my_p4est_vtk_write_all(this->p4est, this->nodes,this->ghost,
                                   P4EST_TRUE, P4EST_TRUE,
                                   0, 1, oss2.str().c_str(),
                                   VTK_CELL_DATA, "Ibin", this->Ibin_local);
        this->ierr= VecRestoreArray(*IBinSegmented,&this->Ibin_local);CHKERRXX(this->ierr);
    }


    this->ierr= VecDestroy(Icell);CHKERRXX(this->ierr);
    this->ierr=VecDestroy(wremesh);CHKERRXX(this->ierr);

}

int MeanField::segmentDiffusionPotentialsWbyNodes(Vec *w2Segment, Vec *IBinSegmented)
{
    std::cout<<" start k means 1"<<std::endl;

    this->ierr=VecDuplicate(*w2Segment,IBinSegmented); CHKERRXX(this->ierr);
    this->ierr=VecSet(*IBinSegmented,0.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(IBinSegmented);

    PetscInt wremesh_size,w_remesh_local_size;
    this->ierr=VecGetSize(*w2Segment,&wremesh_size);CHKERRXX(this->ierr);
    this->ierr=VecGetLocalSize(*w2Segment,&w_remesh_local_size);CHKERRXX(this->ierr);



    PetscInt    i_max;
    PetscScalar x_max;

    PetscInt    i_min;
    PetscScalar x_min;

    this->ierr=VecMax(*w2Segment,&i_max,&x_max);CHKERRXX(this->ierr);
    this->ierr=VecMin(*w2Segment,&i_min,&x_min);CHKERRXX(this->ierr);

    this->c1_global=x_max;
    this->c2_global=x_min;

    this->e1_global=0;
    this->e2_global=0;

    this->e1_global_2=10;
    this->e2_global_2=10;
    std::cout<<" start k means 2"<<std::endl;

    this->computeSegmentationErrorWbyNodes(IBinSegmented,w2Segment);
    std::cout<<" start k means 3"<<std::endl;

    while((this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)>0.01
          ||(this->e1_global_2+this->e2_global_2)<(this->e1_global+this->e2_global) )
    {
        this->e1_global_2=this->e1_global;
        this->e2_global_2=this->e2_global;
        this->computeSegmentationErrorWbyNodes(IBinSegmented,w2Segment);

        std::cout<<"Kmeans Error";
        std::cout<<(this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)<<std::endl;
        std::cout<<" my colours "<<this->c1_global<<" "<<this->c2_global<<std::endl;

        std::cout<<" my_count "<<this->n_colour1_global<<" "<<this->n_colour2_global<<std::endl;
    }
    //this->myDiffusion->printDiffusionVector(IBinSegmented,"IbinAfterSegmentationCell");
    std::ostringstream oss2; oss2 << "Ibin" << p4est->mpirank;//<<"_"<<this->i_mean_field_iteration;
    std::cout<<"I am here "<<  oss2.str()<<std::endl;

    //    if(this->debug_k_means)
    //    {
    //        this->ierr=VecGetArray(*IBinSegmented,&this->Ibin_local);CHKERRXX(this->ierr);

    //        if(this->periodic_xyz)
    //            my_p4est_vtk_write_all(this->p4est_visualization, this->nodes_visualization,this->ghost_visualization    ,
    //                                   P4EST_TRUE, P4EST_TRUE,
    //                                   0, 1, oss2.str().c_str(),
    //                                   VTK_CELL_DATA, "Ibin", this->Ibin_local);
    //        else
    //            my_p4est_vtk_write_all(this->p4est, this->nodes,this->ghost,
    //                                   P4EST_TRUE, P4EST_TRUE,
    //                                   0, 1, oss2.str().c_str(),
    //                                   VTK_CELL_DATA, "Ibin", this->Ibin_local);
    //       this->ierr= VecRestoreArray(*IBinSegmented,&this->Ibin_local);CHKERRXX(this->ierr);
    //    }




}


int MeanField::computeSegmentationError(Vec *Icell)
{
    this->A1=0; this->A2=0;
    this->c1=0; this->c2=0;
    this->e1=0;this->e2=0;

    FILE *outFile;
    if(false)
    {
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("ComputeColoursLog")<<"_"<<this->mpi->mpirank<<".txt";
        mystr2Debug=oss2Debug.str();

        outFile=fopen(mystr2Debug.c_str(),"w");
    }

    p4est_gloidx_t firstProcessorQuadrant=this->p4est->global_first_quadrant[this->mpi->mpirank];
    for(p4est_topidx_t tree_idx=this->p4est->first_local_tree; tree_idx<=this->p4est->last_local_tree;tree_idx++)
    {
        p4est_tree_t *tree=(p4est_tree_t *)sc_array_index(this->p4est->trees,tree_idx);
        int firstTreeQuadrant=   tree->quadrants_offset;
        PetscInt firstQuadrant=(PetscInt)firstProcessorQuadrant+firstTreeQuadrant;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = this->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = this->connectivity->vertices[3*v_mm + 2];
#endif

        for(size_t quad_idx=0;quad_idx<tree->quadrants.elem_count;quad_idx++)
        {
            const p4est_quadrant_t *quad=(const p4est_quadrant_t *)sc_array_index(&tree->quadrants,quad_idx);
            PetscInt *iq2=new PetscInt[1];
            iq2[0]=firstQuadrant+quad_idx;
            const PetscInt *iq=iq2;
            PetscScalar *cellColour=new PetscScalar[1];
            PetscScalar *cellBin=new PetscScalar[1];
            this->ierr=VecGetValues(*Icell,1,iq,cellColour);CHKERRXX(this->ierr);
            PetscInt iq1=iq2[0];
            double A=1./pow(2,quad->level);

#ifdef P4_TO_P8
            A=A*A*A;
#else
            A=A*A;
#endif

            if(ABS(cellColour[0]-this->c1_global)>ABS(cellColour[0]-this->c2_global))
            {
                this->ierr=VecSetValue(this->Ibin,iq1,1,INSERT_VALUES); CHKERRXX(this->ierr);
                this->e2+=pow(cellColour[0]-this->c1_global,2)*A;
                this->A2+=A;
                this->c2+=cellColour[0]*A;
            }
            else
            {
                this->ierr=VecSetValue(this->Ibin,iq1,-1,INSERT_VALUES);CHKERRXX(this->ierr);
                this->e1+=pow(cellColour[0]-this->c2_global,2)*A;
                this->A1+=A;
                this->c1+=cellColour[0]*A;
            }

            this->ierr=VecGetValues(this->Ibin,1,iq,cellBin);CHKERRXX(this->ierr);

            double x = int2double_coordinate_transform(quad->x) + tree_xmin;
            double y = int2double_coordinate_transform(quad->y) + tree_ymin;

#ifdef P4_TO_P8
            double z = int2double_coordinate_transform(quad->z) + tree_zmin;
#endif

            if(false)
            {
                double temp_c1=0,temp_c2=0;
                if(this->A1>0)
                    temp_c1=this->c1/this->A1;
                if(this->A2>0)
                    temp_c2=this->c2/this->A2;
#ifdef P4_TO_P8
                fprintf(outFile,"%d %d %d %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f\n",this->mpi->mpirank, tree_idx,quad_idx,firstProcessorQuadrant,firstTreeQuadrant,firstQuadrant,iq[0],   x,y,z,cellColour[0],cellBin[0],this->c1,this->c2,temp_c1,temp_c2,this->A1,this->A2,this->A1+this->A2);
#else
                fprintf(outFile,"%d %d %d %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f\n",this->mpi->mpirank, tree_idx,quad_idx,firstProcessorQuadrant,firstTreeQuadrant,firstQuadrant,iq[0],   x,y,cellColour[0],cellBin[0],this->c1,this->c2,temp_c1,temp_c2,this->A1,this->A2,this->A1+this->A2);
#endif
            }
        }
    }
    if(false)
        fclose(outFile);
    this->ierr=MPI_Allreduce(&this->e1,&this->e1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->e2,&this->e2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->A1,&this->A1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->A2,&this->A2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->c1,&this->c1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->c2,&this->c2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    if(this->A1_global!=0)
        this->c1_global=this->c1_global/this->A1_global;
    if(this->A2_global!=0)
        this->c2_global=this->c2_global/this->A2_global;


    if(false)
    {
        FILE *outFileKmeans;
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("KmeansLogger")<<"_"<<this->mpi->mpirank<<".txt";
        mystr2Debug=oss2Debug.str();
        if(this->kmeans_iterator==0)
            outFileKmeans=fopen(mystr2Debug.c_str(),"w");
        else
            outFileKmeans=fopen(mystr2Debug.c_str(),"a");
        fprintf(outFileKmeans,"%d %d %f %f %f %f %f %f %f %f\n",this->mpi->mpirank,this->kmeans_iterator,this->c1_global,this->c2_global,this->A1_global,this->A2_global,this->e1_global_2,this->e2_global_2,this->e1_global,this->e2_global);
        fclose(outFileKmeans);
    }
    this->kmeans_iterator++;
}

int MeanField::mapBinaryImageFromCell2Nodes()
{
    VecDuplicate(this->myDiffusion->wp,&this->Ibin_nodes);
    VecGetArray(this->Ibin_nodes,&this->Ibin_nodes_local);

    VecGetArray(this->Ibin,&this->Ibin_local);

    for (p4est_locidx_t i = 0; i<this->myDiffusion->get_n_local_size_sol(); i++)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
               * nonlocal nodes. Number of initial nonlocal nodes is given by
               * nodes->offset_owned_indeps
               */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);

#ifdef P4_TO_P8
        p4est_locidx_t cell_ppm,cell_mmm,cell_pmm,cell_mpm;
        p4est_locidx_t cell_ppp,cell_mmp,cell_pmp,cell_mpp;
#else
        p4est_locidx_t cell_pp,cell_mm,cell_pm,cell_mp;
#endif

        p4est_topidx_t tree_id;
        // find neighboring cell of the nodes
#ifdef P4_TO_P8
        this->node_neighbors->find_neighbor_cell_of_node(node,-1,-1,-1,cell_mmm,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,1,-1,-1,cell_pmm,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,-1,1,-1,cell_mpm,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,1,1,-1,cell_ppm,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,-1,-1,1,cell_mmp,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,1,-1,1,cell_pmp,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,-1,1,1,cell_mpp,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,1,1,1,cell_ppp,tree_id);
#else
        this->node_neighbors->find_neighbor_cell_of_node(node,-1,-1,cell_mm,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,1,-1,cell_pm,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,-1,1,cell_mp,tree_id);
        this->node_neighbors->find_neighbor_cell_of_node(node,1,1,cell_pp,tree_id);
#endif



#ifdef P4_TO_P8

        double ic_mmm= this->Ibin_local[cell_mmm];
        double ic_pmm=this->Ibin_local[cell_pmm];
        double ic_mpm=this->Ibin_local[cell_mpm];
        double ic_ppm=this->Ibin_local[cell_ppm];
        double ic_mmp=this->Ibin_local[cell_mmp];
        double ic_pmp=this->Ibin_local[cell_pmp];
        double ic_mpp=this->Ibin_local[cell_mpp];
        double ic_ppp=this->Ibin_local[cell_ppp] ;
        double ibin_node_local_i=(1.00/8.00)*(ic_mmm+ic_pmm+ic_mpm+ic_ppm+ic_mmp+ic_pmp+ic_mpp+ic_ppp) ;
        this->Ibin_nodes_local[i]=ibin_node_local_i;
#else
        double ic_mm= this->Ibin_local[cell_mm];
        double ic_pm=this->Ibin_local[cell_pm];
        double ic_mp=this->Ibin_local[cell_mp];
        double ic_pp=this->Ibin_local[cell_pp];
        double ibin_node_local_i=(1.00/4.00)*(ic_mm+ic_pm+ic_mp+ic_pp) ;
        this->Ibin_nodes_local[i]=ibin_node_local_i;
#endif


    }
    //if(this->write_to_vtk)
    this->myDiffusion->printDiffusionArrayFromVector(&this->Ibin_nodes,"IbinNodes",PETSC_TRUE);
    this->ierr = VecGhostUpdateBegin(this->Ibin_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    this->ierr = VecGhostUpdateEnd(this->Ibin_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    VecRestoreArray(this->Ibin,&this->Ibin_local);
    VecRestoreArray(this->Ibin_nodes,&this->Ibin_nodes_local);
    std::string file_name="IbinNodesPRV";

    this->my_p4est_vtk_write_all_periodic_adapter(&this->Ibin_nodes,file_name,PETSC_FALSE);

}

int MeanField::computeSegmentationErrorW(Vec *Ibin2, Vec *Icell2)
{
    this->A1=0; this->A2=0;
    this->c1=0; this->c2=0;
    this->e1=0;this->e2=0;
    this->n_colour1_local=0;
    this->n_colour2_local=0;

    FILE *outFile;
    if(false)
    {
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("ComputeColoursLog")<<"_"<<this->mpi->mpirank<<".txt";
        mystr2Debug=oss2Debug.str();

        outFile=fopen(mystr2Debug.c_str(),"w");
    }

    p4est_gloidx_t firstProcessorQuadrant=this->p4est->global_first_quadrant[this->mpi->mpirank];
    for(p4est_topidx_t tree_idx=this->p4est->first_local_tree; tree_idx<=this->p4est->last_local_tree;tree_idx++)
    {
        p4est_tree_t *tree=(p4est_tree_t *)sc_array_index(this->p4est->trees,tree_idx);
        int firstTreeQuadrant=   tree->quadrants_offset;
        PetscInt firstQuadrant=(PetscInt)firstProcessorQuadrant+firstTreeQuadrant;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
        double tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        double tree_ymin = this->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = this->connectivity->vertices[3*v_mm + 2];
#endif

        for(size_t quad_idx=0;quad_idx<tree->quadrants.elem_count;quad_idx++)
        {
            const p4est_quadrant_t *quad=(const p4est_quadrant_t *)sc_array_index(&tree->quadrants,quad_idx);
            PetscInt *iq2=new PetscInt[1];
            iq2[0]=firstQuadrant+quad_idx;
            const PetscInt *iq=iq2;
            PetscScalar *cellColour=new PetscScalar[1];
            PetscScalar *cellBin=new PetscScalar[1];
            VecGetValues(*Icell2,1,iq,cellColour);
            PetscInt iq1=iq2[0];
            double A=1./pow(2,quad->level);

#ifdef P4_TO_P8
            A=A*A*A;
#else
            A=A*A;
#endif

            if(ABS(cellColour[0]-this->c1_global)>ABS(cellColour[0]-this->c2_global))
            {
                VecSetValue(*Ibin2,iq1,1,INSERT_VALUES);
                this->e2+=pow(cellColour[0]-this->c1_global,2)*A;
                this->A2+=A;
                this->n_colour2_local++;
                this->c2+=cellColour[0]*A;
            }
            else
            {
                VecSetValue(*Ibin2,iq1,-1,INSERT_VALUES);
                this->e1+=pow(cellColour[0]-this->c2_global,2)*A;
                this->A1+=A;
                this->n_colour1_local++;
                this->c1+=cellColour[0]*A;
            }

            VecGetValues(*Ibin2,1,iq,cellBin);

            double x = int2double_coordinate_transform(quad->x) + tree_xmin;
            double y = int2double_coordinate_transform(quad->y) + tree_ymin;

#ifdef P4_TO_P8
            double z = int2double_coordinate_transform(quad->z) + tree_zmin;
#endif


            //delete quad;
            delete cellColour;
            delete cellBin;
            delete iq;
            if(false)
            {
                double temp_c1=0,temp_c2=0;
                if(this->A1>0)
                    temp_c1=this->c1/this->A1;
                if(this->A2>0)
                    temp_c2=this->c2/this->A2;
#ifdef P4_TO_P8
                fprintf(outFile,"%d %d %d %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f\n",this->mpi->mpirank, tree_idx,quad_idx,firstProcessorQuadrant,firstTreeQuadrant,firstQuadrant,iq[0],   x,y,z,cellColour[0],cellBin[0],this->c1,this->c2,temp_c1,temp_c2,this->A1,this->A2,this->A1+this->A2);
#else
                fprintf(outFile,"%d %d %d %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f\n",this->mpi->mpirank, tree_idx,quad_idx,firstProcessorQuadrant,firstTreeQuadrant,firstQuadrant,iq[0],   x,y,cellColour[0],cellBin[0],this->c1,this->c2,temp_c1,temp_c2,this->A1,this->A2,this->A1+this->A2);
#endif
            }
        }

        // delete tree;
    }
    if(false)
        fclose(outFile);
    this->ierr=MPI_Allreduce(&this->e1,&this->e1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->e2,&this->e2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->A1,&this->A1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->A2,&this->A2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->c1,&this->c1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->c2,&this->c2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->n_colour1_local,&this->n_colour1_global,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->n_colour2_local,&this->n_colour2_global,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    if(this->A1_global!=0)
        this->c1_global=this->c1_global/this->A1_global;
    if(this->A2_global!=0)
        this->c2_global=this->c2_global/this->A2_global;


    if(false)
    {
        FILE *outFileKmeans;
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("KmeansLogger")<<"_"<<this->mpi->mpirank<<".txt";
        mystr2Debug=oss2Debug.str();
        if(this->kmeans_iterator==0)
            outFileKmeans=fopen(mystr2Debug.c_str(),"w");
        else
            outFileKmeans=fopen(mystr2Debug.c_str(),"a");
        fprintf(outFileKmeans,"%d %d %f %f %f %f %f %f %f %f\n",this->mpi->mpirank,this->kmeans_iterator,this->c1_global,this->c2_global,this->A1_global,this->A2_global,this->e1_global_2,this->e2_global_2,this->e1_global,this->e2_global);
        fclose(outFileKmeans);
    }
    this->kmeans_iterator++;
}

// doesn't work: the mapping is not right anyaway we'll do it node by node
int MeanField::computeSegmentationErrorWbyNodes(Vec *IbinNodes2, Vec *INodes2)
{
    this->A1=0; this->A2=0;
    this->c1=0; this->c2=0;
    this->e1=0;this->e2=0;
    this->n_colour1_local=0;
    this->n_colour2_local=0;

    FILE *outFile;
    if(false)
    {
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("ComputeColoursLog")<<"_"<<this->mpi->mpirank<<".txt";
        mystr2Debug=oss2Debug.str();

        outFile=fopen(mystr2Debug.c_str(),"w");
    }

    p4est_topidx_t global_node_number=0;
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    for(int ii=0;ii<myRank;ii++)
        global_node_number+=this->nodes->global_owned_indeps[ii];

    //global_node_number=nodes->offset_owned_indeps;
    PetscScalar *cellColour;//=new PetscScalar[this->nodes->num_owned_indeps];
    PetscScalar *cellBin;//=new PetscScalar[this->nodes->num_owned_indeps];
    this->ierr=VecGetArray(*INodes2,&cellColour); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(*IbinNodes2,&cellBin); CHKERRXX(this->ierr);

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





        if(ABS(cellColour[i]-this->c1_global)>ABS(cellColour[i]-this->c2_global))
        {
            cellBin[i]=1;
            this->e2+=pow(cellColour[i]-this->c1_global,2);

            this->n_colour2_local++;
            this->c2+=cellColour[i];
        }
        else
        {
            cellBin[i]=-1;
            this->e1+=pow(cellColour[i]-this->c2_global,2);

            this->n_colour1_local++;
            this->c1+=cellColour[i];
        }







        if(false)
        {
            double temp_c1=0,temp_c2=0;
            if(this->n_colour1_local>0)
                temp_c1=this->c1/this->n_colour1_local;
            if(this->n_colour2_local>0)
                temp_c2=this->c2/this->n_colour2_local;
#ifdef P4_TO_P8
            fprintf(outFile,"%d %d %d %f %f %f %f %f %f %f %f %f %f %f %f\n",this->mpi->mpirank, global_node_number,i,   x,y,z,cellColour[i],cellBin[i],this->c1,this->c2,temp_c1,temp_c2,this->n_colour1_local,this->n_colour2_local,this->n_colour1_local+this->n_colour2_local);
#else
            fprintf(outFile,"%d %d %d %f %f %f %f %f %f %f %f %f %f  %f\n",this->mpi->mpirank, global_node_number,i,   x,y,cellColour[i],cellBin[i],this->c1,this->c2,temp_c1,temp_c2,this->n_colour1_local,this->n_colour2_local,this->n_colour1_local+this->n_colour2_local);
#endif
        }


        // delete tree;
    }
    if(false)
        fclose(outFile);
    this->ierr=MPI_Allreduce(&this->e1,&this->e1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->e2,&this->e2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->c1,&this->c1_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->c2,&this->c2_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->n_colour1_local,&this->n_colour1_global,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    this->ierr=MPI_Allreduce(&this->n_colour2_local,&this->n_colour2_global,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);   CHKERRXX(this->ierr);
    if(this->n_colour1_global!=0)
        this->c1_global=this->c1_global/this->n_colour1_global;
    if(this->n_colour2_global!=0)
        this->c2_global=this->c2_global/this->n_colour2_global;


    if(true)
    {
        FILE *outFileKmeans;
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("KmeansLogger")<<"_"<<this->mpi->mpirank<<".txt";
        mystr2Debug=oss2Debug.str();
        if(this->kmeans_iterator==0)
            outFileKmeans=fopen(mystr2Debug.c_str(),"w");
        else
            outFileKmeans=fopen(mystr2Debug.c_str(),"a");
        fprintf(outFileKmeans,"%d %d %f %f %f %f %f %f %f %f\n",this->mpi->mpirank,this->kmeans_iterator,this->c1_global,this->c2_global,this->A1_global,this->A2_global,this->e1_global_2,this->e2_global_2,this->e1_global,this->e2_global);
        fclose(outFileKmeans);
    }
    this->ierr=VecRestoreArray(*INodes2,&cellColour); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(*IbinNodes2,&cellBin); CHKERRXX(this->ierr);

    this->scatter_petsc_vector(INodes2);
    this->scatter_petsc_vector(IbinNodes2);

    this->kmeans_iterator++;
}

// doesn't work: the mapping is not right anyaway we'll do it node by node
int MeanField::mapBinaryImageFromCell2NodesW(Vec *Icell2Map, Vec *InodesMapped)
{


    //    int my_first_number=this->p4est->global_first_quadrant[this->mpi->mpirank];
    //    int global_node_number=0;
    //    int myRank;
    //    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


    //    for(int ii=0;ii<myRank;ii++)
    //        global_node_number+=this->nodes->global_owned_indeps[ii];


    //    std::cout<<" rank global_node_number "<<myRank<<" "<<global_node_number<<std::endl;

    //    this->ierr=VecDuplicate(this->myDiffusion->wp,InodesMapped); CHKERRXX(this->ierr);
    //    this->ierr=VecGetArray(*InodesMapped,&this->Ibin_nodes_local); CHKERRXX(this->ierr);

    //    this->ierr=VecGetArray(*Icell2Map,&this->Ibin_local); CHKERRXX(this->ierr);

    //    for (p4est_locidx_t i = 0; i<this->myDiffusion->get_n_local_size_sol(); i++)
    //    {
    //        /* since we want to access the local nodes, we need to 'jump' over intial
    //               * nonlocal nodes. Number of initial nonlocal nodes is given by
    //               * nodes->offset_owned_indeps
    //               */
    //        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);

    //#ifdef P4_TO_P8
    //        p4est_locidx_t cell_ppm,cell_mmm,cell_pmm,cell_mpm;
    //        p4est_locidx_t cell_ppp,cell_mmp,cell_pmp,cell_mpp;
    //#else
    //        p4est_locidx_t cell_pp,cell_mm,cell_pm,cell_mp;
    //#endif
    //#ifdef P4_TO_P8
    //        p4est_locidx_t tree_id_ppm,tree_id_mmm,tree_id_pmm,tree_id_mpm;
    //        p4est_locidx_t tree_id_ppp,tree_id_mmp,tree_id_pmp,tree_id_mpp;
    //#else
    //        p4est_locidx_t tree_id_pp,tree_id_mm,tree_id_pm,tree_id_mp;
    //#endif

    //#ifdef P4_TO_P8
    //        p4est_locidx_t offset_ppm,offset_mmm,offset_pmm,offset_mpm;
    //        p4est_locidx_t offset_ppp,offset_mmp,offset_pmp,offset_mpp;
    //#else
    //        p4est_locidx_t offset_pp,offset_mm,offset_pm,offset_mp;
    //#endif

    //#ifdef P4_TO_P8
    //        p4est_locidx_t cell_ppm_offset,cell_mmm_offset,cell_pmm_offset,cell_mpm_offset;
    //        p4est_locidx_t cell_ppp_offset,cell_mmp_offset,cell_pmp_offset,cell_mpp_offset;
    //#else
    //        p4est_locidx_t cell_pp_offset,cell_mm_offset,cell_pm_offset,cell_mp_offset;
    //#endif


    //        // find neighboring cell of the nodes
    //#ifdef P4_TO_P8
    //        this->node_neighbors->find_neighbor_cell_of_node(node,-1,-1,-1,cell_mmm,tree_id_mmm);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,1,-1,-1,cell_pmm,tree_id_pmm);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,-1,1,-1,cell_mpm,tree_id_mpm);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,1,1,-1,cell_ppm,tree_id_ppm);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,-1,-1,1,cell_mmp,tree_id_mmp);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,1,-1,1,cell_pmp,tree_id_pmp);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,-1,1,1,cell_mpp,tree_id_mpp);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,1,1,1,cell_ppp,tree_id_ppp);
    //#else
    //        this->node_neighbors->find_neighbor_cell_of_node(node,-1,-1,cell_mm,tree_mm);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,1,-1,cell_pm,tree_pm);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,-1,1,cell_mp,tree_mp);
    //        this->node_neighbors->find_neighbor_cell_of_node(node,1,1,cell_pp,tree_pp);
    //#endif


    //        ///1
    //        ///mmm
    //        p4est_tree_t *tree_mmm = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_mmm);
    //        offset_mmm=tree_mmm->quadrants_offset;
    //        cell_mmm_offset=cell_mmm-offset_mmm;

    //        //2
    //        //pmm
    //        p4est_tree_t *tree_pmm = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_pmm);
    //        offset_pmm=tree_pmm->quadrants_offset;
    //        cell_pmm_offset=cell_pmm-offset_pmm;



    //        //3
    //        //mpm
    //        p4est_tree_t *tree_mpm = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_mpm);
    //        offset_mpm=tree_mpm->quadrants_offset;
    //        cell_mpm_offset=cell_mpm-offset_mpm;


    //        //4
    //        //ppm
    //        p4est_tree_t *tree_ppm = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_ppm);
    //        offset_ppm=tree_ppm->quadrants_offset;
    //        cell_ppm_offset=cell_ppm-offset_ppm;



    //        //5
    //        //mmp
    //        p4est_tree_t *tree_mmp = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_mmp);
    //        offset_mmp=tree_mmp->quadrants_offset;
    //        cell_mmp_offset=cell_ppm-offset_mmp;



    //        //6
    //        //pmp
    //        p4est_tree_t *tree_pmp = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_pmp);
    //        offset_pmp=tree_pmp->quadrants_offset;
    //        cell_pmp_offset=cell_pmp-offset_pmp;


    //        //7
    //        //mpp
    //        p4est_tree_t *tree_mpp = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_mpp);
    //        offset_mpp=tree_mpp->quadrants_offset;
    //        cell_mpp_offset=cell_ppm-offset_mpp;


    //        //8
    //        //ppp
    //        p4est_tree_t *tree_ppp = (p4est_tree_t*)sc_array_index(this->p4est->trees, tree_id_ppp);
    //        offset_ppp=tree_ppp->quadrants_offset;
    //        cell_ppp_offset=cell_ppm-offset_ppp;





    //#ifdef P4_TO_P8

    //        double ic_mmm= this->Ibin_local[cell_mmm_offset];
    //        double ic_pmm=this->Ibin_local[cell_pmm_offset];
    //        double ic_mpm=this->Ibin_local[cell_mpm_offset];
    //        double ic_ppm=this->Ibin_local[cell_ppm_offset];
    //        double ic_mmp=this->Ibin_local[cell_mmp_offset];
    //        double ic_pmp=this->Ibin_local[cell_pmp_offset];
    //        double ic_mpp=this->Ibin_local[cell_mpp_offset];
    //        double ic_ppp=this->Ibin_local[cell_ppp_offset] ;
    //        double ibin_node_local_i=(1.00/8.00)*(ic_mmm+ic_pmm+ic_mpm+ic_ppm+ic_mmp+ic_pmp+ic_mpp+ic_ppp) ;
    //        this->Ibin_nodes_local[i]=ibin_node_local_i;

    //        if( (i)==0 && (global_node_number==0))
    //        {
    //            std::cout<<" check mapping from cell to nodes node zero "<<std::endl;
    //            std::cout<<this->mpi->mpirank<<" "<<i<<" "<<global_node_number<<" "
    //                    <<(i+global_node_number)<<" " <<my_first_number<<std::endl;
    //            std::cout<<" cells "<<cell_ppm<<" "<<cell_mmm<<" "<<cell_pmm<<" "<<cell_mpm<<" "<<cell_ppp<<" "<<cell_mmp<<" "<<cell_pmp<<" "<<cell_mpp<<std::endl;
    //            std::cout<<" offset "<<offset_ppm<<" "<<offset_ppm<<" "<<offset_ppm<<" "<<offset_ppm<<" "<<offset_ppm<<" "<<offset_ppm<<" "<<offset_ppm<<" "<<offset_ppm<<std::endl;
    //            std::cout<<" cell after offset "<<cell_ppm_offset<<" "<<cell_mmm_offset<<" "<<cell_pmm_offset<<" "<<cell_mpm_offset<<" "<<cell_ppp_offset<<" "<<cell_mmp_offset<<" "<<cell_pmp_offset<<" "<<cell_mpp_offset<<std::endl;
    //            std::cout<<" values "<<ic_mmm<<" "<<ic_pmm<<" "<<ic_mpm<<" "<<ic_ppm<<" "<<ic_mmp<<" "<<ic_pmp<<" "<<ic_mpp<<" "<<ic_ppp<<std::endl;
    //            std::cout<<ibin_node_local_i<<std::endl;

    //        }

    //#else
    //        double ic_mm= this->Ibin_local[cell_mm];
    //        double ic_pm=this->Ibin_local[cell_pm];
    //        double ic_mp=this->Ibin_local[cell_mp];
    //        double ic_pp=this->Ibin_local[cell_pp];
    //        double ibin_node_local_i=(1.00/4.00)*(ic_mm+ic_pm+ic_mp+ic_pp) ;
    //        this->Ibin_nodes_local[i]=ibin_node_local_i;
    //#endif


    //    }
    //    //if(!this->minimum_io)
    //    this->myDiffusion->printDiffusionArrayFromVector(InodesMapped,"IbinNodes",PETSC_TRUE);
    //    this->ierr = VecGhostUpdateBegin(*InodesMapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    //    this->ierr = VecGhostUpdateEnd(*InodesMapped, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    //    this->ierr=VecRestoreArray(*Icell2Map,&this->Ibin_local); CHKERRXX(this->ierr);
    //    this->ierr=VecRestoreArray(*InodesMapped,&this->Ibin_nodes_local); CHKERRXX(this->ierr);
    //    std::string file_name="IbinNodesPRV";
    //    this->my_p4est_vtk_write_all_periodic_adapter(InodesMapped,file_name);

}


int MeanField::remeshForestOfTrees(int nx_trees, int ny_trees, int nz_trees)
{

    std::cout<<" Step 1 create a new forest  "<<this->mpi->mpirank<<std::endl;

    // Step 1 create a new forest

    double lip=1.00;
    std::cout<<" Step 2 create a numericalLevelSet   "<<this->mpi->mpirank<<std::endl;
    numericalLevelSet *myLS=new numericalLevelSet(lip,&this->Ibin_nodes,this->p4est,this->nodes,this->ghost,this->brick,this->node_neighbors);

    splitting_criteria_cf_t data(this->min_level, this->max_level,myLS, 1);
    this->nx_trees=nx_trees;
    this->ny_trees=ny_trees;
    this->nz_trees=nz_trees;
    this->brick_remeshed=new my_p4est_brick_t();

    std::cout<<" Step 3 create a connectivity_remeshed  "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_remeshed,1,1,1);
#else
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_remeshed,1,1);
#endif

    // stopwatch
    parStopWatch w1, w2;

    std::cout<<" Step 4 create the actual forest  "<<this->mpi->mpirank<<std::endl;
    /* create the p4est */
    this->p4est_remeshed = p4est_new(this->mpi->mpicomm, this->connectivity_remeshed, 0, NULL, NULL);
    this->p4est_remeshed->user_pointer = (void*)(&data);
    std::cout<<" Step 5 refine the actual forest  "<<this->mpi->mpirank<<std::endl;
    p4est_refine(this->p4est_remeshed, P4EST_TRUE, refine_levelset_cf, NULL);
    std::cout<<" Step 6 partition the forest  "<<this->mpi->mpirank<<std::endl;

    /* partition the p4est */
    p4est_partition(this->p4est_remeshed, NULL);
    std::cout<<" Step 7 create the ghost layer  "<<this->mpi->mpirank<<std::endl;

    /* create the ghost layer */
    this->ghost_remeshed = p4est_ghost_new(this->p4est_remeshed, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    this->nodes_remeshed = my_p4est_nodes_new(this->p4est_remeshed, this->ghost_remeshed);

    std::cout<<" Step 6 initialize the level set which will be used later to duplicate  "<<this->mpi->mpirank<<std::endl;

    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est_remeshed, this->nodes_remeshed, &this->phi); CHKERRXX(ierr);
    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy_remeshed=new  my_p4est_hierarchy_t(this->p4est_remeshed, this->ghost_remeshed, this->brick_remeshed);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors_remeshed=new my_p4est_node_neighbors_t(this->hierarchy_remeshed, this->nodes_remeshed,true);
    this->node_neighbors_remeshed->compute_max_distance();
    this->Lx=this->node_neighbors_remeshed->max_distance;

    return 0;
}

int MeanField::remeshForestOfTrees_parallel(int nx_trees, int ny_trees, int nz_trees)
{



    std::cout<<" Step 1 create a brick remeshed  "<<this->mpi->mpirank<<std::endl;

    this->brick_remeshed=new my_p4est_brick_t();

    std::cout<<" Step 2 create a connectivity_remeshed  "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
#else
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py);
#endif

    // stopwatch
    parStopWatch w1, w2;

    std::cout<<" Step 3 create the actual forest  "<<this->mpi->mpirank<<std::endl;
    /* create the p4est */
    this->p4est_remeshed = p4est_new(this->mpi->mpicomm, this->connectivity_remeshed, 0, NULL, NULL);


    // Remesh using the same algo used in Semi Lagragian.cpp
    // Note the subtility here which comes from parallel computing that
    // Despite that we use values from the previous tree we need to refine using the current tree because
    // We need to use local spatial x y z values
    // Note that we can't use splitting_criteria_cf_t since it can be used only for continuous functions
    //


    std::cout<<" Step 4 refine level by level  "<<this->mpi->mpirank<<std::endl;
    std::vector<double> ibin_nodes_tmp;
    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
    int nb_iter=this->max_level;

    //----------Remeshing level by level algorithm----------------------------//
    // Input:  p4est, Ibin_nodes (parallel vector defined on the nodes of the p4est and acts as a level set).
    // Output: p4est_remeshed
    //
    // Data Structures:
    // p4est:  p4est,p4est_remeshed,p4est_tmp
    // nodes:  nodes,nodes_remeshed,nodes_tmp
    //
    // for (Loop iter=0->nb_iter)
    // {
    //   (1) insert the remeshed forest inside a temporary forest: p4est_tmp=p4est_remeshed
    //   (2) create a temporary nodes structure using p4est_tmp: nodes_tmp=nodes_from_forest(p4est_tmp)
    //   (3) compute Ibin_nodes on an intermediate p4rest: on p4est_tmp.
    //         Computation of i_bin_nodes on an intermediate grid algorithm:
    //          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
    //          Output:   ibin_nodes_local(filled)
    //         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
    //         (b) buffer the local nodes cartesian coordinates
    //         (c) interpolate all in once
    //   (4) create a numerical splitting criteria using p4est_tmp and ibin_nodes_local
    //       splitting_criteria is a data structure with p4est_tmp,ghost_tmp,nodes_tmp with
    //       the classical p4est definitions, and a vector phi_tmp defined locally.
    //       it has only one algorithm associated with this data structure:
    //       given a quad a p4est a tree number, an xyz point defined on a cartesian grid
    //       the value at the vertices of the quad and it will return a linear interpolation.
    //
    //   (5) Refine using the numerical splitting criteria defined on p4est_tmp and not p4est
    //   (6) Partition

    if(!this->myMeanFieldPlan->refine_with_width)
    for(int iter=0;iter<nb_iter;iter++)
    {
        p4est_t *p4est_tmp=p4est_copy(this->p4est_remeshed,P4EST_FALSE);
        p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(p4est_tmp,NULL);

        /* compute  Ibin_nodes on an intermediate grid   */
        ibin_nodes_tmp.resize(nodes_tmp->indep_nodes.elem_count);
        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes,ibin_nodes_tmp.data(),p4est_tmp,nodes_tmp);

        /*refine p4est_remeshed*/

        //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
        splitting_criteria_update_t4MeanField data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp,this->brick,p4est_tmp,NULL,nodes_tmp,this->myMeanFieldPlan->refine_in_minority);
        this->p4est_remeshed->user_pointer=(void*)&data_np1;
        p4est_refine(this->p4est_remeshed,P4EST_FALSE,MeanField::refine_criteria_sl,NULL);
        /* partition the p4est */
        p4est_partition(this->p4est_remeshed, NULL);
        p4est_nodes_destroy(nodes_tmp);
        p4est_destroy(p4est_tmp);
        ibin_nodes_tmp.clear();
    }
    else
    {
        FieldProcessor *width_calculator=new FieldProcessor();
        width_calculator->compute_field_interface_width(&this->myDiffusion->wm, &this->Ibin_nodes,
                                                        this->myMeanFieldPlan->alpha_width, this->X_ab,this->interface_width,
                                                        this->interface_width_negative, this->interface_width_positive);
        for(int iter=0;iter<nb_iter;iter++)
        {
            p4est_t *p4est_tmp=p4est_copy(this->p4est_remeshed,P4EST_FALSE);
            p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(p4est_tmp,NULL);

            /* compute  Ibin_nodes on an intermediate grid   */
            ibin_nodes_tmp.resize(nodes_tmp->indep_nodes.elem_count);
            this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes,ibin_nodes_tmp.data(),p4est_tmp,nodes_tmp);

            /*refine p4est_remeshed*/

            //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
            splitting_criteria_update_t4MeanFieldWithWidth data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,
                                                                    &ibin_nodes_tmp,this->brick,p4est_tmp,NULL,nodes_tmp,
                                                                    this->myMeanFieldPlan->refine_in_minority,this->interface_width,
                                                                    this->interface_width_negative, this->interface_width_positive);
            this->p4est_remeshed->user_pointer=(void*)&data_np1;
            p4est_refine(this->p4est_remeshed,P4EST_FALSE,MeanField::refine_criteria_sl_with_width,NULL);
            /* partition the p4est */
            p4est_partition(this->p4est_remeshed, NULL);
            p4est_nodes_destroy(nodes_tmp);
            p4est_destroy(p4est_tmp);
            ibin_nodes_tmp.clear();
        }
    }
    ibin_nodes_tmp.clear();
    /*restore the user pointer in the p4est*/

    this->p4est_remeshed->user_pointer=this->p4est->user_pointer;

    /*compute new ghost layer*/

    this->ghost_remeshed = p4est_ghost_new(this->p4est_remeshed, P4EST_CONNECT_FULL);


    /*compute the nodes structure*/
    this->nodes_remeshed=my_p4est_nodes_new(this->p4est_remeshed,this->ghost_remeshed);

    VecDestroy(this->phi);
    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est_remeshed, this->nodes_remeshed, &this->phi); CHKERRXX(ierr);
    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy_remeshed=new  my_p4est_hierarchy_t(this->p4est_remeshed, this->ghost_remeshed, this->brick_remeshed);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors_remeshed=new my_p4est_node_neighbors_t(this->hierarchy_remeshed, this->nodes_remeshed,
                                                                this->myMeanFieldPlan->periodic_xyz,
                                                                this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
    this->node_neighbors_remeshed->compute_max_distance();
    this->Lx=this->node_neighbors_remeshed->max_distance;

    return 0;
}

int MeanField::remeshForestOfTrees_parallel_with_interface(int nx_trees, int ny_trees, int nz_trees)
{

    std::cout<<" Step 1 create a brick remeshed  "<<this->mpi->mpirank<<std::endl;

    this->brick_remeshed=new my_p4est_brick_t();

    std::cout<<" Step 2 create a connectivity_remeshed  "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
#else
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py);
#endif

    // stopwatch
    parStopWatch w1, w2;

    std::cout<<" Step 3 create the actual forest  "<<this->mpi->mpirank<<std::endl;
    /* create the p4est */
    this->p4est_remeshed = p4est_new(this->mpi->mpicomm, this->connectivity_remeshed, 0, NULL, NULL);


    // Remesh using the same algo used in Semi Lagragian.cpp
    // Note the subtility here which comes from parallel computing that
    // Despite that we use values from the previous tree we need to refine using the current tree because
    // We need to use local spatial x y z values
    // Note that we can't use splitting_criteria_cf_t since it can be used only for continuous functions
    //


    std::cout<<" Step 4 refine level by level  "<<this->mpi->mpirank<<std::endl;
    std::vector<double> ibin_nodes_tmp;
    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
    int nb_iter=this->max_level;

    //----------Remeshing level by level algorithm----------------------------//
    // Input:  p4est, Ibin_nodes (parallel vector defined on the nodes of the p4est and acts as a level set).
    // Output: p4est_remeshed
    //
    // Data Structures:
    // p4est:  p4est,p4est_remeshed,p4est_tmp
    // nodes:  nodes,nodes_remeshed,nodes_tmp
    //
    // for (Loop iter=0->nb_iter)
    // {
    //   (1) insert the remeshed forest inside a temporary forest: p4est_tmp=p4est_remeshed
    //   (2) create a temporary nodes structure using p4est_tmp: nodes_tmp=nodes_from_forest(p4est_tmp)
    //   (3) compute Ibin_nodes on an intermediate p4rest: on p4est_tmp.
    //         Computation of i_bin_nodes on an intermediate grid algorithm:
    //          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
    //          Output:   ibin_nodes_local(filled)
    //         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
    //         (b) buffer the local nodes cartesian coordinates
    //         (c) interpolate all in once
    //   (4) create a numerical splitting criteria using p4est_tmp and ibin_nodes_local
    //       splitting_criteria is a data structure with p4est_tmp,ghost_tmp,nodes_tmp with
    //       the classical p4est definitions, and a vector phi_tmp defined locally.
    //       it has only one algorithm associated with this data structure:
    //       given a quad a p4est a tree number, an xyz point defined on a cartesian grid
    //       the value at the vertices of the quad and it will return a linear interpolation.
    //
    //   (5) Refine using the numerical splitting criteria defined on p4est_tmp and not p4est
    //   (6) Partition


    for(int iter=0;iter<nb_iter;iter++)
    {
        p4est_t *p4est_tmp=p4est_copy(this->p4est_remeshed,P4EST_FALSE);
        p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(p4est_tmp,NULL);

        /* compute  Ibin_nodes on an intermediate grid   */
        ibin_nodes_tmp.resize(nodes_tmp->indep_nodes.elem_count);
        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes,ibin_nodes_tmp.data(),p4est_tmp,nodes_tmp);

        /*refine p4est_remeshed*/

        //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
        splitting_criteria_update_t4MeanField data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp,this->brick,p4est_tmp,NULL,nodes_tmp,this->myMeanFieldPlan->refine_in_minority);
        this->p4est_remeshed->user_pointer=(void*)&data_np1;
        p4est_refine(this->p4est_remeshed,P4EST_FALSE,MeanField::refine_criteria_sl,NULL);
        /* partition the p4est */
        p4est_partition(this->p4est_remeshed, NULL);
        p4est_nodes_destroy(nodes_tmp);
        p4est_destroy(p4est_tmp);
    }

    /*restore the user pointer in the p4est*/

    this->p4est_remeshed->user_pointer=this->p4est->user_pointer;

    /*compute new ghost layer*/

    this->ghost_remeshed = p4est_ghost_new(this->p4est_remeshed, P4EST_CONNECT_FULL);


    /*compute the nodes structure*/
    this->nodes_remeshed=my_p4est_nodes_new(this->p4est_remeshed,this->ghost_remeshed);


    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est_remeshed, this->nodes_remeshed, &this->phi); CHKERRXX(ierr);
    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy_remeshed=new  my_p4est_hierarchy_t(this->p4est_remeshed, this->ghost_remeshed, this->brick_remeshed);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors_remeshed=new my_p4est_node_neighbors_t(this->hierarchy_remeshed, this->nodes_remeshed,
                                                                this->myMeanFieldPlan->periodic_xyz,this->myMeanFieldPlan->px,
                                                                this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
    this->node_neighbors_remeshed->compute_max_distance();
    this->Lx=this->node_neighbors_remeshed->max_distance;

    return 0;
}



int MeanField::remeshForestOfTreesVisualization_parallel(int nx_trees, int ny_trees, int nz_trees)
{

    if(this->myMeanFieldPlan->periodic_xyz)
    {
        p4est_destroy( p4est_visualization);
        p4est_nodes_destroy(nodes_visualization);
        p4est_ghost_destroy( ghost_visualization);
        p4est_connectivity_destroy( connectivity_visualization);

        delete brick_visualization;
        delete hierarchy_visualization;
        delete node_neighbors_visualization;

        std::cout<<" Step 1 create a brick remeshed  "<<this->mpi->mpirank<<std::endl;

        this->brick_visualization=new my_p4est_brick_t();

        std::cout<<" Step 2 create a connectivity_remeshed  "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
        this->connectivity_visualization = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_visualization,0,0,0);
#else
        this->connectivity_visualization = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_visualization,0,0);
#endif

        // stopwatch
        parStopWatch w1, w2;

        std::cout<<" Step 3 create the actual forest  "<<this->mpi->mpirank<<std::endl;
        /* create the p4est */
        this->p4est_visualization = p4est_new(this->mpi->mpicomm, this->connectivity_visualization, 0, NULL, NULL);


        // Remesh using the same algo used in Semi Lagragian.cpp
        // Note the subtility here which comes from parallel computing that
        // Despite that we use values from the previous tree we need to refine using the current tree because
        // We need to use local spatial x y z values
        // Note that we can't use splitting_criteria_cf_t since it can be used only for continuous functions
        //


        std::cout<<" Step 4 refine level by level  "<<this->mpi->mpirank<<std::endl;
        std::vector<double> ibin_nodes_tmp;
        std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
        int nb_iter=(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl;

        //----------Remeshing level by level algorithm----------------------------//
        // Input:  p4est, Ibin_nodes (parallel vector defined on the nodes of the p4est and acts as a level set).
        // Output: p4est_remeshed
        //
        // Data Structures:
        // p4est:  p4est,p4est_remeshed,p4est_tmp
        // nodes:  nodes,nodes_remeshed,nodes_tmp
        //
        // for (Loop iter=0->nb_iter)
        // {
        //   (1) insert the remeshed forest inside a temporary forest: p4est_tmp=p4est_remeshed
        //   (2) create a temporary nodes structure using p4est_tmp: nodes_tmp=nodes_from_forest(p4est_tmp)
        //   (3) compute Ibin_nodes on an intermediate p4rest: on p4est_tmp.
        //         Computation of i_bin_nodes on an intermediate grid algorithm:
        //          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
        //          Output:   ibin_nodes_local(filled)
        //         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
        //         (b) buffer the local nodes cartesian coordinates
        //         (c) interpolate all in once
        //   (4) create a numerical splitting criteria using p4est_tmp and ibin_nodes_local
        //       splitting_criteria is a data structure with p4est_tmp,ghost_tmp,nodes_tmp with
        //       the classical p4est definitions, and a vector phi_tmp defined locally.
        //       it has only one algorithm associated with this data structure:
        //       given a quad a p4est a tree number, an xyz point defined on a cartesian grid
        //       the value at the vertices of the quad and it will return a linear interpolation.
        //
        //   (5) Refine using the numerical splitting criteria defined on p4est_tmp and not p4est
        //   (6) Partition

if(!this->myMeanFieldPlan->refine_with_width)
        for(int iter=0;iter<nb_iter;iter++)
        {
            p4est_t *p4est_tmp=p4est_copy(this->p4est_visualization,P4EST_FALSE);
            p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(p4est_visualization,NULL);

            /* compute  Ibin_nodes on an intermediate grid   */
            ibin_nodes_tmp.resize(nodes_tmp->indep_nodes.elem_count);
            this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes,ibin_nodes_tmp.data(),p4est_tmp,nodes_tmp);

            /*refine p4est_remeshed*/

            //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
            splitting_criteria_update_t4MeanField data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp,this->brick,p4est_tmp,NULL,nodes_tmp,this->myMeanFieldPlan->refine_in_minority);
            this->p4est_visualization->user_pointer=(void*)&data_np1;
            p4est_refine(this->p4est_visualization,P4EST_FALSE,MeanField::refine_criteria_sl,NULL);
            /* partition the p4est */
            p4est_partition(this->p4est_visualization, NULL);
            p4est_nodes_destroy(nodes_tmp);
            p4est_destroy(p4est_tmp);
            ibin_nodes_tmp.clear();
        }
else
    for(int iter=0;iter<nb_iter;iter++)
    {
        p4est_t *p4est_tmp=p4est_copy(this->p4est_visualization,P4EST_FALSE);
        p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(p4est_visualization,NULL);

        /* compute  Ibin_nodes on an intermediate grid   */
        ibin_nodes_tmp.resize(nodes_tmp->indep_nodes.elem_count);
        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes,ibin_nodes_tmp.data(),p4est_tmp,nodes_tmp);

        /*refine p4est_remeshed*/

        //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
        splitting_criteria_update_t4MeanFieldWithWidth
                data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp,this->brick,p4est_tmp,NULL,nodes_tmp,
                         this->myMeanFieldPlan->refine_in_minority,this->interface_width,this->interface_width_negative,this->interface_width_negative);
        this->p4est_visualization->user_pointer=(void*)&data_np1;
        p4est_refine(this->p4est_visualization,P4EST_FALSE,MeanField::refine_criteria_sl_with_width,NULL);
        /* partition the p4est */
        p4est_partition(this->p4est_visualization, NULL);
        p4est_nodes_destroy(nodes_tmp);
        p4est_destroy(p4est_tmp);
        ibin_nodes_tmp.clear();
    }
        ibin_nodes_tmp.clear();

        /*restore the user pointer in the p4est*/

        this->p4est_visualization->user_pointer=this->p4est->user_pointer;

        /*compute new ghost layer*/

        this->ghost_visualization = p4est_ghost_new(this->p4est_visualization, P4EST_CONNECT_FULL);


        /*compute the nodes structure*/
        this->nodes_visualization=my_p4est_nodes_new(this->p4est_visualization,this->ghost_visualization);


        if(this->i_mean_field_iteration>0)
            this->ierr=VecDestroy(this->phi_vizualization); CHKERRXX(this->ierr);


        /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
        ierr = VecCreateGhostNodes(this->p4est_visualization, this->nodes_visualization, &this->phi_vizualization); CHKERRXX(ierr);
        /* create the hierarchy structure */
        w2.start("construct the hierachy information");
        this->hierarchy_visualization=new  my_p4est_hierarchy_t(this->p4est_visualization, this->ghost_visualization, this->brick_visualization);
        w2.stop(); w2.read_duration();

        /* generate the neighborhood information */
        w2.start("construct the neighborhood information");
        PetscBool periodic_xyz_input=PETSC_FALSE;
        this->node_neighbors_visualization=new my_p4est_node_neighbors_t(this->hierarchy_visualization, this->nodes_visualization,
                                                                         periodic_xyz_input);
        this->node_neighbors_visualization->compute_max_distance();
        w2.stop(); w2.read_duration();
    }
    return 0;
}


int MeanField::remeshForestOfTrees_parallel_two_level_set(int nx_trees, int ny_trees, int nz_trees)
{



    std::cout<<" Step 1 create a brick remeshed  "<<this->mpi->mpirank<<std::endl;

    this->brick_remeshed=new my_p4est_brick_t();

    std::cout<<" Step 2 create a connectivity_remeshed  "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
#else
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py);
#endif

    // stopwatch
    parStopWatch w1, w2;

    std::cout<<" Step 3 create the actual forest  "<<this->mpi->mpirank<<std::endl;
    /* create the p4est */
    this->p4est_remeshed = p4est_new(this->mpi->mpicomm, this->connectivity_remeshed, 0, NULL, NULL);


    // Remesh using the same algo used in Semi Lagragian.cpp
    // Note the subtility here which comes from parallel computing that
    // Despite that we use values from the previous tree we need to refine using the current tree because
    // We need to use local spatial x y z values
    // Note that we can't use splitting_criteria_cf_t since it can be used only for continuous functions
    //


    std::cout<<" Step 4 refine level by level  "<<this->mpi->mpirank<<std::endl;
    std::vector<double> ibin_nodes_tmp_m;
    std::vector<double> ibin_nodes_tmp_p;
    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
    int nb_iter=this->max_level;

    //----------Remeshing level by level algorithm----------------------------//
    // Input:  p4est, Ibin_nodes (parallel vector defined on the nodes of the p4est and acts as a level set).
    // Output: p4est_remeshed
    //
    // Data Structures:
    // p4est:  p4est,p4est_remeshed,p4est_tmp
    // nodes:  nodes,nodes_remeshed,nodes_tmp
    //
    // for (Loop iter=0->nb_iter)
    // {
    //   (1) insert the remeshed forest inside a temporary forest: p4est_tmp=p4est_remeshed
    //   (2) create a temporary nodes structure using p4est_tmp: nodes_tmp=nodes_from_forest(p4est_tmp)
    //   (3) compute Ibin_nodes on an intermediate p4rest: on p4est_tmp.
    //         Computation of i_bin_nodes on an intermediate grid algorithm:
    //          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
    //          Output:   ibin_nodes_local(filled)
    //         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
    //         (b) buffer the local nodes cartesian coordinates
    //         (c) interpolate all in once
    //   (4) create a numerical splitting criteria using p4est_tmp and ibin_nodes_local
    //       splitting_criteria is a data structure with p4est_tmp,ghost_tmp,nodes_tmp with
    //       the classical p4est definitions, and a vector phi_tmp defined locally.
    //       it has only one algorithm associated with this data structure:
    //       given a quad a p4est a tree number, an xyz point defined on a cartesian grid
    //       the value at the vertices of the quad and it will return a linear interpolation.
    //
    //   (5) Refine using the numerical splitting criteria defined on p4est_tmp and not p4est
    //   (6) Partition


    for(int iter=0;iter<nb_iter;iter++)
    {
        std::cout<<" iter "<<iter<<std::endl;
        p4est_t *p4est_tmp=p4est_copy(this->p4est_remeshed,P4EST_FALSE);
        p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(p4est_tmp,NULL);

        /* compute  Ibin_nodes on an intermediate grid   */
        ibin_nodes_tmp_m.resize(nodes_tmp->indep_nodes.elem_count);
        ibin_nodes_tmp_p.resize(nodes_tmp->indep_nodes.elem_count);
        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_m,ibin_nodes_tmp_m.data(),p4est_tmp,nodes_tmp);
        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_p,ibin_nodes_tmp_p.data(),p4est_tmp,nodes_tmp);



        //this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_m.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_m.txt");
        //this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_p.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_p.txt");

        /*refine p4est_remeshed*/

        //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
        splitting_criteria_update_t4MeanFieldComplex data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp_m,&ibin_nodes_tmp_p,this->brick,p4est_tmp,NULL,nodes_tmp,this->myMeanFieldPlan->refine_in_minority);
        this->p4est_remeshed->user_pointer=(void*)&data_np1;
        std::cout<<"refine"<<std::endl;
        p4est_refine(this->p4est_remeshed,P4EST_FALSE,MeanField::refine_criteria_sl_complex,NULL);
        /* partition the p4est */
        p4est_partition(this->p4est_remeshed, NULL);
        p4est_nodes_destroy(nodes_tmp);
        p4est_destroy(p4est_tmp);
        ibin_nodes_tmp_m.clear();
        ibin_nodes_tmp_p.clear();
    }

    ibin_nodes_tmp_m.clear();
    ibin_nodes_tmp_p.clear();

    int N_games=0;


    /*restore the user pointer in the p4est*/

    this->p4est_remeshed->user_pointer=this->p4est->user_pointer;

    /*compute new ghost layer*/

    this->ghost_remeshed = p4est_ghost_new(this->p4est_remeshed, P4EST_CONNECT_FULL);

    for(int n_games=0;n_games<N_games;n_games++)
    {
        std::cout<<n_games<<std::endl;
        p4est_ghost_destroy(this->ghost_remeshed);
        this->ghost_remeshed = p4est_ghost_new(this->p4est_remeshed, P4EST_CONNECT_FULL);
    }


    /*compute the nodes structure*/
    this->nodes_remeshed=my_p4est_nodes_new(this->p4est_remeshed,this->ghost_remeshed);
    for(int n_games=0;n_games<N_games;n_games++)
    {
        std::cout<<n_games<<std::endl;
        p4est_nodes_destroy(this->nodes_remeshed);
        this->nodes_remeshed=my_p4est_nodes_new(this->p4est_remeshed,this->ghost_remeshed);
    }

    //if(this->i_mean_field_iteration)
    VecDestroy(this->phi);

    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est_remeshed, this->nodes_remeshed, &this->phi); CHKERRXX(ierr);
    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy_remeshed=new  my_p4est_hierarchy_t(this->p4est_remeshed, this->ghost_remeshed, this->brick_remeshed);
    for(int n_games=0;n_games<N_games;n_games++)
    {
        std::cout<<n_games<<std::endl;
        delete this->hierarchy_remeshed;
        this->hierarchy_remeshed=new  my_p4est_hierarchy_t(this->p4est_remeshed, this->ghost_remeshed, this->brick_remeshed);
    }


    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors_remeshed=new my_p4est_node_neighbors_t(this->hierarchy_remeshed, this->nodes_remeshed,
                                                                this->myMeanFieldPlan->periodic_xyz,
                                                                this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
    for(int n_games=0;n_games<N_games;n_games++)
    {
        delete this->node_neighbors_remeshed;
        this->node_neighbors_remeshed=new my_p4est_node_neighbors_t(this->hierarchy_remeshed, this->nodes_remeshed,
                                                                    this->myMeanFieldPlan->periodic_xyz,
                                                                    this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
    }
    w2.stop(); w2.read_duration();
    this->node_neighbors_remeshed->compute_max_distance();
    this->Lx=this->node_neighbors_remeshed->max_distance;

    return 0;
}

int MeanField::remeshForestOfTrees_parallel_two_level_set_with_interface(int nx_trees, int ny_trees, int nz_trees)
{

    std::cout<<this->p4est<<" "<<this->nodes<<" "<<this->brick<<" "<<this->node_neighbors<<" "<<this->connectivity<<" "<<
               this->ghost<<std::endl;


    std::cout<<this->polymer_mask<<" "<<this->Ibin_nodes_m<<" "<<this->Ibin_nodes_p<<


               std::cout<<" Step 1 create a brick remeshed  "<<this->mpi->mpirank<<std::endl;

    this->brick_remeshed=new my_p4est_brick_t();

    std::cout<<" Step 2 create a connectivity_remeshed  "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
#else
    this->connectivity_remeshed = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_remeshed,
                                                     this->myMeanFieldPlan->px,this->myMeanFieldPlan->py);
#endif

    // stopwatch
    parStopWatch w1, w2;

    std::cout<<" Step 3 create the actual forest  "<<this->mpi->mpirank<<std::endl;
    /* create the p4est */
    this->p4est_remeshed = p4est_new(this->mpi->mpicomm, this->connectivity_remeshed, 0, NULL, NULL);


    // Remesh using the same algo used in Semi Lagragian.cpp
    // Note the subtility here which comes from parallel computing that
    // Despite that we use values from the previous tree we need to refine using the current tree because
    // We need to use local spatial x y z values
    // Note that we can't use splitting_criteria_cf_t since it can be used only for continuous functions
    //


    std::cout<<" Step 4 refine level by level  "<<this->mpi->mpirank<<std::endl;
    std::vector<double> ibin_nodes_tmp_m;
    std::vector<double> ibin_nodes_tmp_p;
    std::vector<double> ibin_nodes_tmp_mask;

    std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
    int nb_iter=this->max_level;

    //----------Remeshing level by level algorithm----------------------------//
    // Input:  p4est, Ibin_nodes (parallel vector defined on the nodes of the p4est and acts as a level set).
    // Output: p4est_remeshed
    //
    // Data Structures:
    // p4est:  p4est,p4est_remeshed,p4est_tmp
    // nodes:  nodes,nodes_remeshed,nodes_tmp
    //
    // for (Loop iter=0->nb_iter)
    // {
    //   (1) insert the remeshed forest inside a temporary forest: p4est_tmp=p4est_remeshed
    //   (2) create a temporary nodes structure using p4est_tmp: nodes_tmp=nodes_from_forest(p4est_tmp)
    //   (3) compute Ibin_nodes on an intermediate p4rest: on p4est_tmp.
    //         Computation of i_bin_nodes on an intermediate grid algorithm:
    //          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
    //          Output:   ibin_nodes_local(filled)
    //         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
    //         (b) buffer the local nodes cartesian coordinates
    //         (c) interpolate all in once
    //   (4) create a numerical splitting criteria using p4est_tmp and ibin_nodes_local
    //       splitting_criteria is a data structure with p4est_tmp,ghost_tmp,nodes_tmp with
    //       the classical p4est definitions, and a vector phi_tmp defined locally.
    //       it has only one algorithm associated with this data structure:
    //       given a quad a p4est a tree number, an xyz point defined on a cartesian grid
    //       the value at the vertices of the quad and it will return a linear interpolation.
    //
    //   (5) Refine using the numerical splitting criteria defined on p4est_tmp and not p4est
    //   (6) Partition


    for(int iter=0;iter<nb_iter;iter++)
    {
        p4est_t *p4est_tmp=p4est_copy(this->p4est_remeshed,P4EST_FALSE);
        p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(p4est_tmp,NULL);

        /* compute  Ibin_nodes on an intermediate grid   */
        ibin_nodes_tmp_m.resize(nodes_tmp->indep_nodes.elem_count);
        ibin_nodes_tmp_p.resize(nodes_tmp->indep_nodes.elem_count);
        ibin_nodes_tmp_mask.resize(nodes_tmp->indep_nodes.elem_count);

        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_m,ibin_nodes_tmp_m.data(),p4est_tmp,nodes_tmp);
        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_p,ibin_nodes_tmp_p.data(),p4est_tmp,nodes_tmp);
        this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->polymer_mask,ibin_nodes_tmp_mask.data(),p4est_tmp,nodes_tmp);


        if(!this->minimum_io)
        {
            this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_m.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_m.txt");
            this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_p.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_p.txt");
            this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_mask.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_mask.txt");
        }
        /*refine p4est_remeshed*/

        //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;


        if(!this->myMeanFieldPlan->refine_at_interface_more)
        {
            splitting_criteria_update_t4MeanFieldComplexMasked data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp_m,&ibin_nodes_tmp_p,&ibin_nodes_tmp_mask,this->brick,p4est_tmp,NULL,nodes_tmp,this->myMeanFieldPlan->refine_in_minority);
            this->p4est_remeshed->user_pointer=(void*)&data_np1;
            if(!this->uniform_mesh_with_mask)
                p4est_refine(this->p4est_remeshed,P4EST_FALSE,MeanField::refine_criteria_sl_complex_masked,NULL);
            if(this->uniform_mesh_with_mask)
                p4est_refine(this->p4est_remeshed,P4EST_FALSE,MeanField::refine_criteria_sl_complex_masked_uniform,NULL);
        }
        if(this->myMeanFieldPlan->refine_at_interface_more)
        {splitting_criteria_update_t4MeanFieldComplexMaskedThreeLevels
                    data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,this->myMeanFieldPlan->bulk_level,&ibin_nodes_tmp_m,&ibin_nodes_tmp_p,&ibin_nodes_tmp_mask,this->brick,p4est_tmp,NULL,nodes_tmp,this->myMeanFieldPlan->refine_in_minority);
            this->p4est_remeshed->user_pointer=(void*)&data_np1;
            p4est_refine(this->p4est_remeshed,P4EST_FALSE,MeanField::refine_criteria_sl_complex_masked_uniform_three_levels,NULL);

        }
        /* partition the p4est */
        p4est_partition(this->p4est_remeshed, NULL);
        p4est_nodes_destroy(nodes_tmp);
        p4est_destroy(p4est_tmp);
        ibin_nodes_tmp_m.clear();
        ibin_nodes_tmp_p.clear();
        ibin_nodes_tmp_mask.clear();
    }


    ibin_nodes_tmp_m.clear();
    ibin_nodes_tmp_p.clear();
    ibin_nodes_tmp_mask.clear();
    /*restore the user pointer in the p4est*/

    this->p4est_remeshed->user_pointer=this->p4est->user_pointer;

    /*compute new ghost layer*/

    this->ghost_remeshed = p4est_ghost_new(this->p4est_remeshed, P4EST_CONNECT_FULL);


    /*compute the nodes structure*/
    this->nodes_remeshed=my_p4est_nodes_new(this->p4est_remeshed,this->ghost_remeshed);


    VecDestroy(this->phi);

    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est_remeshed, this->nodes_remeshed, &this->phi); CHKERRXX(ierr);


    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy_remeshed=new  my_p4est_hierarchy_t(this->p4est_remeshed, this->ghost_remeshed, this->brick_remeshed);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors_remeshed=new my_p4est_node_neighbors_t(this->hierarchy_remeshed, this->nodes_remeshed,this->myMeanFieldPlan->periodic_xyz,
                                                                this->myMeanFieldPlan->px,this->myMeanFieldPlan->py,this->myMeanFieldPlan->pz);
    this->node_neighbors_remeshed->compute_max_distance();
    this->Lx=this->node_neighbors_remeshed->max_distance;

    return 0;
}



int MeanField::remeshForestOfTreesVisualization_parallel_two_level_set(int nx_trees, int ny_trees, int nz_trees)
{

    int n_games_i=1;

    for(int i=0;i<n_games_i;i++)
    {

        if(this->myMeanFieldPlan->periodic_xyz)
        {

            p4est_nodes_destroy(this->nodes_visualization);
            p4est_ghost_destroy(this->ghost_visualization);
            p4est_destroy(this->p4est_visualization);
            p4est_connectivity_destroy(this->connectivity_visualization);

            delete this->brick_visualization;
            delete this->hierarchy_visualization;
            delete this->node_neighbors_visualization;



            std::cout<<" Step 1 create a brick remeshed vizualiztion "<<this->mpi->mpirank<<std::endl;

            this->brick_visualization=new my_p4est_brick_t();

            std::cout<<" Step 2 create a connectivity_remeshed vizualization "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
            this->connectivity_visualization = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_visualization,0,0,0);
#else
            this->connectivity_visualization = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_visualization,0,0);
#endif

            // stopwatch
            parStopWatch w1, w2;

            std::cout<<" Step 3 create the actual forest visualization "<<this->mpi->mpirank<<std::endl;
            /* create the p4est */
            this->p4est_visualization = p4est_new(this->mpi->mpicomm, this->connectivity_visualization, 0, NULL, NULL);


            // Remesh using the same algo used in Semi Lagragian.cpp
            // Note the subtility here which comes from parallel computing that
            // Despite that we use values from the previous tree we need to refine using the current tree because
            // We need to use local spatial x y z values
            // Note that we can't use splitting_criteria_cf_t since it can be used only for continuous functions
            //


            std::cout<<" Step 4 refine level by level  "<<this->mpi->mpirank<<std::endl;
            std::vector<double> ibin_nodes_tmp_m;
            std::vector<double> ibin_nodes_tmp_p;
            std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
            int nb_iter=(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl;

            //----------Remeshing level by level algorithm----------------------------//
            // Input:  p4est, Ibin_nodes (parallel vector defined on the nodes of the p4est and acts as a level set).
            // Output: p4est_remeshed
            //
            // Data Structures:
            // p4est:  p4est,p4est_remeshed,p4est_tmp
            // nodes:  nodes,nodes_remeshed,nodes_tmp
            //
            // for (Loop iter=0->nb_iter)
            // {
            //   (1) insert the remeshed forest inside a temporary forest: p4est_tmp=p4est_remeshed
            //   (2) create a temporary nodes structure using p4est_tmp: nodes_tmp=nodes_from_forest(p4est_tmp)
            //   (3) compute Ibin_nodes on an intermediate p4rest: on p4est_tmp.
            //         Computation of i_bin_nodes on an intermediate grid algorithm:
            //          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
            //          Output:   ibin_nodes_local(filled)
            //         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
            //         (b) buffer the local nodes cartesian coordinates
            //         (c) interpolate all in once
            //   (4) create a numerical splitting criteria using p4est_tmp and ibin_nodes_local
            //       splitting_criteria is a data structure with p4est_tmp,ghost_tmp,nodes_tmp with
            //       the classical p4est definitions, and a vector phi_tmp defined locally.
            //       it has only one algorithm associated with this data structure:
            //       given a quad a p4est a tree number, an xyz point defined on a cartesian grid
            //       the value at the vertices of the quad and it will return a linear interpolation.
            //
            //   (5) Refine using the numerical splitting criteria defined on p4est_tmp and not p4est
            //   (6) Partition


            for(int iter=0;iter<nb_iter;iter++)
            {
                std::cout<<" iter visualiztion "<<std::endl;

                p4est_t *p4est_tmp=p4est_copy(this->p4est_visualization,P4EST_FALSE);
                p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(this->p4est_visualization,NULL);

                /* compute  Ibin_nodes on an intermediate grid   */
                ibin_nodes_tmp_m.resize(nodes_tmp->indep_nodes.elem_count);
                ibin_nodes_tmp_p.resize(nodes_tmp->indep_nodes.elem_count);
                this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_m,ibin_nodes_tmp_m.data(),p4est_tmp,nodes_tmp);
                this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_p,ibin_nodes_tmp_p.data(),p4est_tmp,nodes_tmp);

                //  this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_m.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_m.txt");
                // this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_p.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_p.txt");

                /*refine p4est_remeshed*/

                //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
                splitting_criteria_update_t4MeanFieldComplex data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp_m,&ibin_nodes_tmp_p,this->brick_visualization,p4est_tmp,NULL,nodes_tmp,this->myMeanFieldPlan->refine_in_minority);
                this->p4est_visualization->user_pointer=(void*)&data_np1;
                p4est_refine(this->p4est_visualization,P4EST_FALSE,MeanField::refine_criteria_sl_complex,NULL);
                /* partition the p4est */
                p4est_partition(this->p4est_visualization, NULL);
                p4est_nodes_destroy(nodes_tmp);
                p4est_destroy(p4est_tmp);
                ibin_nodes_tmp_m.clear();
                ibin_nodes_tmp_p.clear();

            }
            ibin_nodes_tmp_m.clear();
            ibin_nodes_tmp_p.clear();

            /*restore the user pointer in the p4est*/

            std::cout<<" restore pointer visualiztion "<<std::endl;
            this->p4est_visualization->user_pointer=this->p4est->user_pointer;

            /*compute new ghost layer*/

            std::cout<<" compute new ghost layer vizualization"<<std::endl;
            this->ghost_visualization = p4est_ghost_new(this->p4est_visualization, P4EST_CONNECT_FULL);


            /*compute the nodes structure*/
            std::cout<<" compute nodes structure vizualization"<<std::endl;
            this->nodes_visualization=my_p4est_nodes_new(this->p4est_visualization,this->ghost_visualization);


            if(this->i_mean_field_iteration>0 || i>0)
                this->ierr=VecDestroy(this->phi_vizualization); CHKERRXX(this->ierr);

            /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
            ierr = VecCreateGhostNodes(this->p4est_visualization, this->nodes_visualization, &this->phi_vizualization); CHKERRXX(ierr);
            /* create the hierarchy structure */
            w2.start("construct the hierachy information vizu");
            this->hierarchy_visualization=new  my_p4est_hierarchy_t(this->p4est_visualization, this->ghost_visualization, this->brick_visualization);
            w2.stop(); w2.read_duration();

            /* generate the neighborhood information */
            w2.start("construct the neighborhood information vizu");
            PetscBool periodic_xyz_input=PETSC_FALSE;
            this->node_neighbors_visualization=new my_p4est_node_neighbors_t(this->hierarchy_visualization, this->nodes_visualization,periodic_xyz_input);
            this->node_neighbors_visualization->compute_max_distance();
            w2.stop(); w2.read_duration();





        }
    }
    return 0;
}

int MeanField::remeshForestOfTreesVisualization_parallel_two_level_set_with_interface(int nx_trees, int ny_trees, int nz_trees)
{

    int n_games_i=1;

    for(int i=0;i<n_games_i;i++)
    {

        if(this->myMeanFieldPlan->periodic_xyz)
        {

            p4est_nodes_destroy(this->nodes_visualization);
            p4est_ghost_destroy(this->ghost_visualization);
            p4est_destroy(this->p4est_visualization);
            p4est_connectivity_destroy(this->connectivity_visualization);

            delete this->brick_visualization;
            delete this->hierarchy_visualization;
            delete this->node_neighbors_visualization;



            std::cout<<" Step 1 create a brick remeshed vizualiztion "<<this->mpi->mpirank<<std::endl;

            this->brick_visualization=new my_p4est_brick_t();

            std::cout<<" Step 2 create a connectivity_remeshed vizualization "<<this->mpi->mpirank<<std::endl;

#ifdef P4_TO_P8
            this->connectivity_visualization = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_visualization,0,0,0);
#else
            this->connectivity_visualization = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_visualization,0,0);
#endif

            // stopwatch
            parStopWatch w1, w2;

            std::cout<<" Step 3 create the actual forest visualization "<<this->mpi->mpirank<<std::endl;
            /* create the p4est */
            this->p4est_visualization = p4est_new(this->mpi->mpicomm, this->connectivity_visualization, 0, NULL, NULL);


            // Remesh using the same algo used in Semi Lagragian.cpp
            // Note the subtility here which comes from parallel computing that
            // Despite that we use values from the previous tree we need to refine using the current tree because
            // We need to use local spatial x y z values
            // Note that we can't use splitting_criteria_cf_t since it can be used only for continuous functions
            //


            std::cout<<" Step 4 refine level by level  "<<this->mpi->mpirank<<std::endl;
            std::vector<double> ibin_nodes_tmp_m;
            std::vector<double> ibin_nodes_tmp_p;
            std::vector<double> ibin_nodes_tmp_mask;

            std::cout<<(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl<<std::endl;
            int nb_iter=(  (splitting_criteria_t*)(this->p4est->user_pointer))->max_lvl;

            //----------Remeshing level by level algorithm----------------------------//
            // Input:  p4est, Ibin_nodes (parallel vector defined on the nodes of the p4est and acts as a level set).
            // Output: p4est_remeshed
            //
            // Data Structures:
            // p4est:  p4est,p4est_remeshed,p4est_tmp
            // nodes:  nodes,nodes_remeshed,nodes_tmp
            //
            // for (Loop iter=0->nb_iter)
            // {
            //   (1) insert the remeshed forest inside a temporary forest: p4est_tmp=p4est_remeshed
            //   (2) create a temporary nodes structure using p4est_tmp: nodes_tmp=nodes_from_forest(p4est_tmp)
            //   (3) compute Ibin_nodes on an intermediate p4rest: on p4est_tmp.
            //         Computation of i_bin_nodes on an intermediate grid algorithm:
            //          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
            //          Output:   ibin_nodes_local(filled)
            //         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
            //         (b) buffer the local nodes cartesian coordinates
            //         (c) interpolate all in once
            //   (4) create a numerical splitting criteria using p4est_tmp and ibin_nodes_local
            //       splitting_criteria is a data structure with p4est_tmp,ghost_tmp,nodes_tmp with
            //       the classical p4est definitions, and a vector phi_tmp defined locally.
            //       it has only one algorithm associated with this data structure:
            //       given a quad a p4est a tree number, an xyz point defined on a cartesian grid
            //       the value at the vertices of the quad and it will return a linear interpolation.
            //
            //   (5) Refine using the numerical splitting criteria defined on p4est_tmp and not p4est
            //   (6) Partition


            for(int iter=0;iter<nb_iter;iter++)
            {
                std::cout<<" iter visualiztion "<<std::endl;

                p4est_t *p4est_tmp=p4est_copy(this->p4est_visualization,P4EST_FALSE);
                p4est_nodes_t *nodes_tmp=my_p4est_nodes_new(this->p4est_visualization,NULL);

                /* compute  Ibin_nodes on an intermediate grid   */
                ibin_nodes_tmp_m.resize(nodes_tmp->indep_nodes.elem_count);
                ibin_nodes_tmp_p.resize(nodes_tmp->indep_nodes.elem_count);
                ibin_nodes_tmp_mask.resize(nodes_tmp->indep_nodes.elem_count);

                this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_m,ibin_nodes_tmp_m.data(),p4est_tmp,nodes_tmp);
                this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->Ibin_nodes_p,ibin_nodes_tmp_p.data(),p4est_tmp,nodes_tmp);
                this->compute_ibin_nodes_on_intermediate_grids(*this->node_neighbors,this->polymer_mask,ibin_nodes_tmp_mask.data(),p4est_tmp,nodes_tmp);

                //  this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_m.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_m.txt");
                // this->myDiffusion->printDiffusionArray(ibin_nodes_tmp_p.data(),nodes_tmp->indep_nodes.elem_count,"ibin_nodes_tmp_p.txt");

                /*refine p4est_remeshed*/

                //splitting_criteria_t *data=(splitting_criteria_t*) this->p4est->user_pointer;
                splitting_criteria_update_t4MeanFieldComplexMasked data_np1(myMeanFieldPlan->LipshitzConstant,this->min_level,this->max_level,&ibin_nodes_tmp_m,&ibin_nodes_tmp_p,&ibin_nodes_tmp_mask, this->brick_visualization,p4est_tmp,NULL,nodes_tmp,myMeanFieldPlan->refine_in_minority);
                this->p4est_visualization->user_pointer=(void*)&data_np1;
                if(!this->uniform_mesh_with_mask)
                    p4est_refine(this->p4est_visualization,P4EST_FALSE,MeanField::refine_criteria_sl_complex_masked,NULL);
                if(this->uniform_mesh_with_mask)
                    p4est_refine(this->p4est_visualization,P4EST_FALSE,MeanField::refine_criteria_sl_complex_masked_uniform,NULL);
                /* partition the p4est */
                p4est_partition(this->p4est_visualization, NULL);
                p4est_nodes_destroy(nodes_tmp);
                p4est_destroy(p4est_tmp);
                ibin_nodes_tmp_m.clear();
                ibin_nodes_tmp_p.clear();
                ibin_nodes_tmp_mask.clear();

            }
            ibin_nodes_tmp_m.clear();
            ibin_nodes_tmp_p.clear();
            ibin_nodes_tmp_mask.clear();

            /*restore the user pointer in the p4est*/

            std::cout<<" restore pointer visualiztion "<<std::endl;
            this->p4est_visualization->user_pointer=this->p4est->user_pointer;

            /*compute new ghost layer*/

            std::cout<<" compute new ghost layer vizualization"<<std::endl;
            this->ghost_visualization = p4est_ghost_new(this->p4est_visualization, P4EST_CONNECT_FULL);


            /*compute the nodes structure*/
            std::cout<<" compute nodes structure vizualization"<<std::endl;
            this->nodes_visualization=my_p4est_nodes_new(this->p4est_visualization,this->ghost_visualization);


            if(this->i_mean_field_iteration>0 || i>0)
                this->ierr=VecDestroy(this->phi_vizualization); CHKERRXX(this->ierr);

            /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
            ierr = VecCreateGhostNodes(this->p4est_visualization, this->nodes_visualization, &this->phi_vizualization); CHKERRXX(ierr);
            /* create the hierarchy structure */
            w2.start("construct the hierachy information vizu");
            this->hierarchy_visualization=new  my_p4est_hierarchy_t(this->p4est_visualization, this->ghost_visualization, this->brick_visualization);
            w2.stop(); w2.read_duration();

            /* generate the neighborhood information */
            w2.start("construct the neighborhood information vizu");
            PetscBool periodic_xyz_input=PETSC_FALSE;
            this->node_neighbors_visualization=new my_p4est_node_neighbors_t(this->hierarchy_visualization, this->nodes_visualization,periodic_xyz_input);
            this->node_neighbors_visualization->compute_max_distance();
            w2.stop(); w2.read_duration();





        }
    }
    return 0;
}



//         Computation of i_bin_nodes on an intermediate grid algorithm:
//          Input:    ibin_nodes_local(empty),p4est_tmp, Ibin_nodes, p4est
//          Output:   ibin_nodes_local(filled)
//         (a) Creation of an interpolation function ibin_interp using Ibin_nodes and p4est
//         (b) buffer the local nodes cartesian coordinates
//         (c) interpolate all in once

int MeanField::compute_ibin_nodes_on_intermediate_grids( my_p4est_node_neighbors_t &qnnn,Vec phi_n,  double *phi_np1,p4est_t *p4est_np1, p4est_nodes_t *nodes_np1)
{
    InterpolatingFunctionNodeBase interp(this->p4est, this->nodes, this->ghost, this->brick, &qnnn);
#ifdef P4_TO_P8
    interp.set_input_parameters(phi_n, linear);
#else
    interp.set_input_parameters(phi_n,linear);
#endif

    p4est_topidx_t *t2v = p4est_np1->connectivity->tree_to_vertex; // tree to vertex list
    double *t2c = p4est_np1->connectivity->vertices; // coordinates of the vertices of a tree

    p4est_locidx_t ni_begin = 0;
    p4est_locidx_t ni_end   = nodes_np1->indep_nodes.elem_count;

    for (p4est_locidx_t ni = ni_begin; ni < ni_end; ni++)
    {
        //Loop through all nodes of a single processor
        p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
        p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

        p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
        double tr_xmin = t2c[3 * tr_mm + 0];
        double tr_ymin = t2c[3 * tr_mm + 1];
#ifdef P4_TO_P8
        double tr_zmin = t2c[3 * tr_mm + 2];
#endif

        /* Find initial xy points */
        double xyz[] =
        {
            node_x_fr_i(indep_node) + tr_xmin,
            node_y_fr_j(indep_node) + tr_ymin
    #ifdef P4_TO_P8
            ,
            node_z_fr_k(indep_node) + tr_zmin
    #endif
        };
        /* Buffer the point for interpolation */
        interp.add_point_to_buffer(ni, xyz);
    }

    /* interpolate from old vector into our output vector */
    interp.interpolate(phi_np1);

}

int MeanField::remeshForestOfTreesVisualization(int nx_trees, int ny_trees, int nz_trees)
{
    // Step 1 delete previous visualization forest
    delete p4est_visualization;
    delete nodes_visualization;
    delete ghost_visualization;
    delete connectivity_visualization;
    delete brick_visualization;
    delete hierarchy_visualization;

    // node_neighbor_vizualization encapsulates the above fields

    //delete node_neighbors_visualization;


    //Step 2 create a new visualization forest
    double lip=1.00;
    numericalLevelSet *myLS=new numericalLevelSet(lip,&this->Ibin_nodes,this->p4est,this->nodes,this->ghost,this->brick,this->node_neighbors);

#ifdef P4_TO_P8
    std::cout<<myLS->operator ()(0,0,0)<<std::endl;
#else
    std::cout<<myLS->operator ()(0,0)<<std::endl;
#endif



    splitting_criteria_cf_t data(this->min_level+this->nb_splits, this->max_level+this->nb_splits,myLS, 1);
    this->nx_trees=nx_trees;
    this->ny_trees=ny_trees;
    this->nz_trees=nz_trees;
    this->brick_visualization=new my_p4est_brick_t();

#ifdef P4_TO_P8
    this->connectivity_visualization = my_p4est_brick_new(this->nx_trees,this->ny_trees, this->nz_trees, this->brick_remeshed,0,0,0);
#else
    this->connectivity_visualization = my_p4est_brick_new(this->nx_trees, this->ny_trees, this->brick_visualization,0,0);
#endif

    // stopwatch
    parStopWatch w1, w2;

    /* create the p4est */
    this->p4est_visualization = p4est_new(this->mpi->mpicomm, this->connectivity_visualization, 0, NULL, NULL);
    this->p4est_visualization->user_pointer = (void*)(&data);
    p4est_refine(this->p4est_visualization, P4EST_TRUE, refine_levelset_cf, NULL);

    /* partition the p4est */
    p4est_partition(this->p4est_visualization, NULL);

    /* create the ghost layer */
    this->ghost_visualization = p4est_ghost_new(this->p4est_visualization, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    this->nodes_visualization = my_p4est_nodes_new(this->p4est_visualization, this->ghost_visualization);

    //    /* initialize the level set which will be used later to duplicate and get other vectors with the same data structure */
    ierr = VecCreateGhostNodes(this->p4est_visualization, this->nodes_visualization, &this->phi_vizualization); CHKERRXX(ierr);
    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    this->hierarchy_visualization=new  my_p4est_hierarchy_t(this->p4est_visualization, this->ghost_visualization, this->brick_visualization);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    this->node_neighbors_visualization=new my_p4est_node_neighbors_t(this->hierarchy_visualization, this->nodes_visualization,true);
    this->node_neighbors_visualization->compute_max_distance();


    return 0;
}



int MeanField::my_p4est_vtk_write_all_periodic_adapter(Vec *myNodesVector4Paraview,std::string file_name,PetscBool write_anyway)
{
    if(this->i_mean_field_iteration<2 || mod(this->i_mean_field_iteration,this->vtk_period)==0 && myNodesVector4Paraview!=NULL)
    {
        if( (write_anyway) && !this->do_not_write_to_vtk_in_any_case)
        {
            //  std::cout<<" started to write to vtk "<<file_name<<std::endl;

            if(this->myMeanFieldPlan->px ||this->myMeanFieldPlan->py ||this->myMeanFieldPlan->pz)
            {

                //std::cout<<" started to write to vtk periodic "<<std::endl;


                // Transform the nodes based petsc parallel vector
                // to a cell based petsc parallel vector
                // Both of the vectors have the p4est ghost data structure
                // this->ierr = VecGhostUpdateBegin(*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                //this->ierr = VecGhostUpdateEnd  (*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

                Vec myCellsVector4Paraview;
                VecCreateGhostCells(this->p4est,this->ghost,&myCellsVector4Paraview);
                PetscScalar *Cells_Vector_local;
                this->ierr=VecGetArray(myCellsVector4Paraview,&Cells_Vector_local); CHKERRXX(this->ierr);

                PetscInt icell_size,icell_local_size;
                this->ierr=VecGetSize(myCellsVector4Paraview,&icell_size);CHKERRXX(this->ierr);
                this->ierr= VecGetLocalSize(myCellsVector4Paraview,&icell_local_size); CHKERRXX(this->ierr);

                PetscScalar *my_nodes_local;
                this->ierr=VecGetArray(*myNodesVector4Paraview,&my_nodes_local); CHKERRXX(this->ierr);

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


#ifdef P4_TO_P8
                        Cells_Vector_local[jj]=(1.00/8.00)*
                                (my_nodes_local[n0]+my_nodes_local[n1]+my_nodes_local[n2]+my_nodes_local[n3]+my_nodes_local[n4]
                                 +my_nodes_local[n5]+my_nodes_local[n6]+my_nodes_local[n7]);
#else
                        Cells_Vector_local[jj]=(1.00/4.00)*
                                (my_nodes_local[n0]+my_nodes_local[n1]+my_nodes_local[n2]+my_nodes_local[n3]);
#endif
                        jj++;
                    }
                }



                // this->ierr = VecGhostUpdateBegin(myCellsVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
                // this->ierr = VecGhostUpdateEnd(myCellsVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

                //    std::cout<<" started to write to vtk periodic cellwise"<<std::endl;

                my_p4est_vtk_write_all(this->p4est_visualization, this->nodes_visualization,NULL,
                                       P4EST_TRUE, P4EST_TRUE,
                                       0, 1, this->get_full_string(file_name).c_str(),
                                       VTK_CELL_DATA, "I_cell", Cells_Vector_local);
                //    std::cout<<" finished to write to vtk periodic cellwise"<<std::endl;

                this->ierr=VecRestoreArray(myCellsVector4Paraview,&Cells_Vector_local); CHKERRXX(this->ierr);
                this->ierr=VecDestroy(myCellsVector4Paraview); CHKERRXX(this->ierr);
                this->ierr=VecRestoreArray(*myNodesVector4Paraview,&my_nodes_local);   CHKERRXX(this->ierr);
            }
            else
            {

                //     std::cout<<" started to write to vtk not periodic"<<std::endl;

                // Transform the nodes based petsc parallel vector
                // to a cell based petsc parallel vector
                // Both of the vectors have the p4est ghost data structure
                //  this->ierr = VecGhostUpdateBegin(*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                //this->ierr = VecGhostUpdateEnd  (*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


                PetscScalar *my_nodes_local;
                this->ierr=VecGetArray(*myNodesVector4Paraview,&my_nodes_local); CHKERRXX(this->ierr);

                //  std::cout<<" started to write to vtk !periodic nodes"<<std::endl;


                my_p4est_vtk_write_all(this->p4est, this->nodes,this->ghost,
                                       P4EST_TRUE, P4EST_TRUE,
                                       1, 0, this->get_full_string(file_name).c_str(),
                                       VTK_POINT_DATA, "I_nodes", my_nodes_local);

                // std::cout<<" started to write to vtk !periodic nodes"<<std::endl;


                this->ierr=VecRestoreArray(*myNodesVector4Paraview,&my_nodes_local);   CHKERRXX(this->ierr);
            }

            // std::cout<<" finished to write to vtk"<<std::endl;
        }
    }

}

int MeanField::my_p4est_vtk_write_all_periodic_adapter_psdt(Vec *myNodesVector4Paraview,std::string file_name,PetscBool write_anyway)
{
    if(this->i_mean_field_iteration<2 || this->i_mean_field_iteration>this->N_mean_field_iteration-3
            || mod(this->i_mean_field_iteration,this->vtk_period)==0
            && myNodesVector4Paraview!=NULL)
    {
        int vec_size=0;
        this->ierr=VecGetSize(*myNodesVector4Paraview,&vec_size); CHKERRXX(this->ierr);

        //std::cout<<" started to write to vtk "<<file_name<<" vec_size"<<std::endl;


        if(( write_anyway) && !this->do_not_write_to_vtk_in_any_case && vec_size>0 && myNodesVector4Paraview!=NULL)
        {


            if(this->myMeanFieldPlan->px||this->myMeanFieldPlan->py ||this->myMeanFieldPlan->pz)
            {

                std::cout<<" started to write to vtk periodic"<<std::endl;


                // Transform the nodes based petsc parallel vector
                // to a cell based petsc parallel vector
                // Both of the vectors have the p4est ghost data structure
                // this->ierr = VecGhostUpdateBegin(*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                //this->ierr = VecGhostUpdateEnd  (*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

                Vec myCellsVector4Paraview;
                VecCreateGhostCells(this->p4est,this->ghost,&myCellsVector4Paraview);
                PetscScalar *Cells_Vector_local;
                this->ierr=VecGetArray(myCellsVector4Paraview,&Cells_Vector_local); CHKERRXX(this->ierr);

                PetscInt icell_size,icell_local_size;
                this->ierr=VecGetSize(myCellsVector4Paraview,&icell_size);CHKERRXX(this->ierr);
                this->ierr= VecGetLocalSize(myCellsVector4Paraview,&icell_local_size); CHKERRXX(this->ierr);

                PetscScalar *my_nodes_local;
                this->ierr=VecGetArray(*myNodesVector4Paraview,&my_nodes_local); CHKERRXX(this->ierr);

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


#ifdef P4_TO_P8
                        Cells_Vector_local[jj]=(1.00/8.00)*
                                (my_nodes_local[n0]+my_nodes_local[n1]+my_nodes_local[n2]+my_nodes_local[n3]+my_nodes_local[n4]
                                 +my_nodes_local[n5]+my_nodes_local[n6]+my_nodes_local[n7]);
#else
                        Cells_Vector_local[jj]=(1.00/4.00)*
                                (my_nodes_local[n0]+my_nodes_local[n1]+my_nodes_local[n2]+my_nodes_local[n3]);
#endif
                        jj++;
                    }
                }



                // this->ierr = VecGhostUpdateBegin(myCellsVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
                // this->ierr = VecGhostUpdateEnd(myCellsVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);

                std::cout<<" started to write to vtk periodic cellwise"<<std::endl;

                my_p4est_vtk_write_all(this->p4est_visualization, this->nodes_visualization,NULL,
                                       P4EST_TRUE, P4EST_TRUE,
                                       0, 1, this->get_full_string_psdt(file_name).c_str(),
                                       VTK_CELL_DATA, "I_cell", Cells_Vector_local);
                std::cout<<" finished to write to vtk periodic cellwise"<<std::endl;

                this->ierr=VecRestoreArray(myCellsVector4Paraview,&Cells_Vector_local); CHKERRXX(this->ierr);
                this->ierr=VecDestroy(myCellsVector4Paraview); CHKERRXX(this->ierr);
                this->ierr=VecRestoreArray(*myNodesVector4Paraview,&my_nodes_local);   CHKERRXX(this->ierr);
            }
            else
            {

                // std::cout<<" started to write to vtk not periodic"<<std::endl;

                // Transform the nodes based petsc parallel vector
                // to a cell based petsc parallel vector
                // Both of the vectors have the p4est ghost data structure
                //  this->ierr = VecGhostUpdateBegin(*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                //this->ierr = VecGhostUpdateEnd  (*myNodesVector4Paraview, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);


                PetscScalar *my_nodes_local;
                this->ierr=VecGetArray(*myNodesVector4Paraview,&my_nodes_local); CHKERRXX(this->ierr);

                //  std::cout<<" started to write to vtk !periodic nodes"<<std::endl;


                my_p4est_vtk_write_all(this->p4est, this->nodes,this->ghost,
                                       P4EST_TRUE, P4EST_TRUE,
                                       1, 0, this->get_full_string_psdt(file_name).c_str(),
                                       VTK_POINT_DATA, "I_nodes", my_nodes_local);

                //   std::cout<<" started to write to vtk !periodic nodes"<<std::endl;


                this->ierr=VecRestoreArray(*myNodesVector4Paraview,&my_nodes_local);   CHKERRXX(this->ierr);
            }

            //    std::cout<<" finished to write to vtk"<<std::endl;
        }
    }
}

int MeanField::get_field_from_text_file(Vec *v2Fill, string file_name)
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



    fstream my_fstream;

    std::stringstream oss2OpenVec;
    std::string mystr2Open;
    oss2OpenVec <<file_name<<"_"<<this->mpi->mpirank<<"_"<<this->text_file_seed_str<<".txt";
    mystr2Open=oss2OpenVec.str();

    std::cout<<"Started to load uniform parallel vector "<<mystr2Open<<std::endl;


    my_fstream.open(mystr2Open.c_str(),ios_base::in);
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
                my_fstream >>ip >> ix >>iz>> ij>>x_dummy>>y_dummy>>z_dummy>>ls_ij;

                unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
            }
        }
    }
#else
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            my_fstream >>ip >> ix >>ij>>x_dummy>>y_dummy>>ls_ij;
            unf_local_petsc_input[j+i*N1]=ls_ij;
        }
    }
#endif


    my_fstream.close();
    this->ierr=VecRestoreArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);

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


    this->interpolate_and_print_vec_to_uniform_grid(v2Fill,"test",false);
}

int MeanField::get_coarse_field_from_fine_text_file(Vec *v2Fill, string file_name,int file_level)
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
                my_fstream >>ip >> ix >>iz>> ij>>x_dummy>>y_dummy>>z_dummy>>ls_ij;

                unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
            }
        }
    }
#else
    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            my_fstream >>ip >> ix >>ij>>x_dummy>>y_dummy>>ls_ij;
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


    this->interpolate_and_print_vec_to_uniform_grid(v2Fill,"test",false);
}


int MeanField::get_3Dfield_from_2Dtext_file(Vec *v2Fill, string file_name, double delta_lz)
{

    //------------------This function should be used only in 3D---------//


#ifdef P4_TO_P8
    //step 0: declare variables
    IS from_unf;
    IS to_p4est;
    VecScatter  scatter_from_unf_to_p4est;
    Vec         unf_global_petsc_input;
    PetscScalar *unf_local_petsc_input;


    // step 1: create parallel mapping data structure
    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2))

            ,N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))

            ;
    int n_unf=N0*N1;
    n_unf=n_unf*N2;
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

    double k_node,z, offset_tree_z,z_normalized,tree_zmin;


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


        tree_zmin=this->connectivity->vertices[3*v_mm + 2];
        z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
        offset_tree_z=tree_zmin/this->Lx*N0;
        z_normalized=z/normalizer;
        k_node=z_normalized+offset_tree_z;
        k_node=(int)k_node%N2;
        idx_unf_temp=(int)(i_node*((double)N1*N2) + j_node*(double)N2+k_node);
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

    std::stringstream oss2OpenVec;
    std::string mystr2Open;
    oss2OpenVec <<file_name<<"_"<<this->mpi->mpirank<<"_"<<this->text_file_seed_str<<".txt";
    mystr2Open=oss2OpenVec.str();

    my_fstream.open(mystr2Open.c_str(),ios_base::in);
    int ip,ix,ij;
    double x_dummy,y_dummy,ls_ij;


    int iz;
    double z_dummy;

    this->ierr=VecGetArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);

    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            my_fstream >>ip >> ix >>ij>>x_dummy>>y_dummy>>ls_ij;
            for(int k=0; k<N2; k++)
            {
                unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
            }
        }
    }


    my_fstream.close();
    this->ierr=VecRestoreArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);

    // step 3: scatter from uniform grid to parallel AMR vector

    this->ierr=VecScatterBegin(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=ISDestroy(from_unf); CHKERRXX(this->ierr);
    this->ierr=ISDestroy(to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterDestroy(&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);
    delete idx_from_unf_to_p4est_unf_ix;
    delete idx_from_unf_to_p4est_p4est_ix;


    // step 4 correct in order to make a 3D mask


    PetscScalar *v2fill_local;
    this->ierr=VecGetArray(*v2Fill,&v2fill_local); CHKERRXX(this->ierr);

    double v2fill_local_n, dz_up,dz_down;

    for (int n = 0; n < n_local_size_amr; n++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = this->connectivity->vertices[3*v_mm + 1];
        tree_zmin=this->connectivity->vertices[3*v_mm + 2];


        x =  node_x_fr_i(node) + tree_xmin;
        y =  node_y_fr_j(node) + tree_ymin;
        z = node_z_fr_k(node) + tree_zmin;

        v2fill_local_n=v2fill_local[n];

        dz_up=z-(this->Lx-delta_lz);
        dz_down=delta_lz-z;



        //        if(dz_up<=0.00 && dz_down<=0.00 && v2fill_local_n<=0.00)
        //        {
        //            v2fill_local[n]=max(dz_down,max(dz_up,v2fill_local_n));
        //        }

        if( dz_up>=0.00  && v2fill_local_n<=0.00)
        {
            v2fill_local[n]=dz_up;
        }

        if( dz_down>=0.00  && v2fill_local_n<=0.00)
        {
            v2fill_local[n]=dz_down;
        }



    }
    // step 5: interpolate back to uniform an print to matlab
    // (for debugging purposes)
    this->ierr=VecRestoreArray(*v2Fill,&v2fill_local); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(v2Fill);

    this->interpolate_and_print_vec_to_uniform_grid(v2Fill,"test",this->myMeanFieldPlan->compressed_io);


#else
    std::runtime_error("This function cant be used in 2D but 3D only 3D");
#endif

}

int MeanField::get_3DParallelField_from_2DSerialtext_file(Vec *v2Fill, string file_name, double delta_lz)
{

    //------------------This function should be used only in 3D---------//


#ifdef P4_TO_P8
    //step 0: declare variables
    IS from_unf;
    IS to_p4est;
    VecScatter  scatter_from_unf_to_p4est;
    Vec         unf_global_petsc_input;
    PetscScalar *unf_local_petsc_input;


    // step 1: create parallel mapping data structure
    const ptrdiff_t N0 =(int)pow(2,this->max_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,this->max_level+log(this->nx_trees)/log(2))

            ,N2=(int)pow(2,this->max_level+log(this->nx_trees)/log(2))

            ;
    int n_unf=N0*N1;
    n_unf=n_unf*N2;
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

    double k_node,z, offset_tree_z,z_normalized,tree_zmin;


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


        tree_zmin=this->connectivity->vertices[3*v_mm + 2];
        z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
        offset_tree_z=tree_zmin/this->Lx*N0;
        z_normalized=z/normalizer;
        k_node=z_normalized+offset_tree_z;
        k_node=(int)k_node%N2;
        idx_unf_temp=(int)(i_node*((double)N1*N2) + j_node*(double)N2+k_node);
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

    std::stringstream oss2OpenVec;
    std::string mystr2Open;
    oss2OpenVec <<file_name<<"_"<<0<<"_"<<this->text_file_seed_str<<".txt";
    mystr2Open=oss2OpenVec.str();

    my_fstream.open(mystr2Open.c_str(),ios_base::in);
    int ip,ix,ij;
    double x_dummy,y_dummy,ls_ij;




    int iz;
    double z_dummy;
    for(int i=0; i<this->mpi->mpirank*N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            my_fstream >>ip >> ix >>ij>>x_dummy>>y_dummy>>ls_ij;

        }
    }
    this->ierr=VecGetArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);

    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            my_fstream >>ip >> ix >>ij>>x_dummy>>y_dummy>>ls_ij;
            for(int k=0; k<N2; k++)
            {
                unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
            }
        }
    }


    my_fstream.close();
    this->ierr=VecRestoreArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);

    // step 3: scatter from uniform grid to parallel AMR vector

    this->ierr=VecScatterBegin(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecScatterEnd(scatter_from_unf_to_p4est,unf_global_petsc_input,*v2Fill,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=ISDestroy(from_unf); CHKERRXX(this->ierr);
    this->ierr=ISDestroy(to_p4est); CHKERRXX(this->ierr);
    this->ierr=VecScatterDestroy(&scatter_from_unf_to_p4est); CHKERRXX(this->ierr);
    delete idx_from_unf_to_p4est_unf_ix;
    delete idx_from_unf_to_p4est_p4est_ix;


    // step 4 correct in order to make a 3D mask


    PetscScalar *v2fill_local;
    this->ierr=VecGetArray(*v2Fill,&v2fill_local); CHKERRXX(this->ierr);

    double v2fill_local_n, dz_up,dz_down;

    for (int n = 0; n < n_local_size_amr; n++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = this->connectivity->vertices[3*v_mm + 1];
        tree_zmin=this->connectivity->vertices[3*v_mm + 2];


        x =  node_x_fr_i(node) + tree_xmin;
        y =  node_y_fr_j(node) + tree_ymin;
        z = node_z_fr_k(node) + tree_zmin;

        v2fill_local_n=v2fill_local[n];

        dz_up=z-(this->Lx-delta_lz);
        dz_down=delta_lz-z;



        //        if(dz_up<=0.00 && dz_down<=0.00 && v2fill_local_n<=0.00)
        //        {
        //            v2fill_local[n]=max(dz_down,max(dz_up,v2fill_local_n));
        //        }

        if( dz_up>=0.00  && v2fill_local_n<=0.00)
        {
            v2fill_local[n]=dz_up;
        }

        if( dz_down>=0.00  && v2fill_local_n<=0.00)
        {
            v2fill_local[n]=dz_down;
        }



    }
    // step 5: interpolate back to uniform an print to matlab
    // (for debugging purposes)
    this->ierr=VecRestoreArray(*v2Fill,&v2fill_local); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(v2Fill);

    this->interpolate_and_print_vec_to_uniform_grid(v2Fill,"test",this->myMeanFieldPlan->compressed_io);


#else
    std::runtime_error("This function cant be used in 2D but 3D only 3D");
#endif

}


int MeanField::get_3D_coarse_field_from_2D_fine_text_file(Vec *v2Fill, string file_name, double delta_lz,int file_level)
{

    //------------------This function should be used only in 3D and on one processor---------//


#ifdef P4_TO_P8
    //step 0: declare variables
    IS from_unf;
    IS to_p4est;
    VecScatter  scatter_from_unf_to_p4est;
    Vec         unf_global_petsc_input;
    PetscScalar *unf_local_petsc_input;


    // step 1: create parallel mapping data structure
    const ptrdiff_t N0 =(int)pow(2,file_level+log(this->nx_trees)/log(2)),
            N1 = (int)pow(2,file_level+log(this->nx_trees)/log(2))
            ,N2=(int)pow(2,file_level+log(this->nx_trees)/log(2))
            ;
    int n_unf=N0*N1;
    n_unf=n_unf*N2;
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

    double k_node,z, offset_tree_z,z_normalized,tree_zmin;


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


        tree_zmin=this->connectivity->vertices[3*v_mm + 2];
        z=(node->z==P4EST_ROOT_LEN-1? P4EST_ROOT_LEN:node->z);
        offset_tree_z=tree_zmin/this->Lx*N0;
        z_normalized=z/normalizer;
        k_node=z_normalized+offset_tree_z;
        k_node=(int)k_node%N2;
        idx_unf_temp=(int)(i_node*((double)N1*N2) + j_node*(double)N2+k_node);
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


    int iz;
    double z_dummy;

    this->ierr=VecGetArray(unf_global_petsc_input,&unf_local_petsc_input); CHKERRXX(this->ierr);

    for(int i=0;i<this->mpi->mpirank*N0/this->mpi->mpisize;i++)
    {
        for(int j=0;j<this->mpi->mpirank*N0/this->mpi->mpisize;j++)
            my_fstream>>ip>>ix>>ij>>x_dummy>>y_dummy>>ls_ij;
    }

    for(int i=0; i<N0/this->mpi->mpisize; i++)
    {
        for(int j=0; j<N1; j++)
        {
            my_fstream >>ip >> ix >>ij>>x_dummy>>y_dummy>>ls_ij;
            for(int k=0; k<N2; k++)
            {
                unf_local_petsc_input[i*(N1*N2) + j*N2+k]=ls_ij;
            }
        }
    }


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


    // step 4 correct in order to make a 3D mask


    PetscScalar *v2fill_local;
    this->ierr=VecGetArray(*v2Fill,&v2fill_local); CHKERRXX(this->ierr);

    double v2fill_local_n, dz_up,dz_down;

    for (int n = 0; n < n_local_size_amr; n++)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&this->nodes->indep_nodes, n+this->nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;
        p4est_topidx_t v_mm = this->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
        tree_xmin = this->connectivity->vertices[3*v_mm + 0];
        tree_ymin = this->connectivity->vertices[3*v_mm + 1];
        tree_zmin=this->connectivity->vertices[3*v_mm + 2];


        x =  node_x_fr_i(node) + tree_xmin;
        y =  node_y_fr_j(node) + tree_ymin;
        z = node_z_fr_k(node) + tree_zmin;

        v2fill_local_n=v2fill_local[n];

        dz_up=z-(this->Lx-delta_lz);
        dz_down=delta_lz-z;



        //        if(dz_up<=0.00 && dz_down<=0.00 && v2fill_local_n<=0.00)
        //        {
        //            v2fill_local[n]=max(dz_down,max(dz_up,v2fill_local_n));
        //        }

        if( dz_up>=0.00  && v2fill_local_n<=0.00)
        {
            v2fill_local[n]=dz_up;
        }

        if( dz_down>=0.00  && v2fill_local_n<=0.00)
        {
            v2fill_local[n]=dz_down;
        }



    }
    // step 5: interpolate back to uniform an print to matlab
    // (for debugging purposes)
    this->ierr=VecRestoreArray(*v2Fill,&v2fill_local); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(v2Fill);

    this->interpolate_and_print_vec_to_uniform_grid(v2Fill,"test",this->myMeanFieldPlan->compressed_io);


#else
    std::runtime_error("This function cant be used in 2D but 3D only 3D");
#endif

}


int MeanField::interpolate_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid,std::string str_file,bool compressed_io)
{

    if(this->mpi->mpisize>=1 && mod(this->i_mean_field_iteration,this->vtk_period)==0 && !compressed_io)
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
        InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, this->brick, this->node_neighbors);
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
        this->myDiffusion->printDiffusionArrayFromVector(&v_uniform,"v_uniform");


        int mpi_size;
        int mpi_rank;

        MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

        std::stringstream oss2DebugVec;
        std::string mystr2Debug;
        oss2DebugVec << this->convert2FullPath(str_file)<<"_"<<mpi_rank<<"_"<<this->i_mean_field_iteration<<".txt";
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

    }

    int MeanField::interpolate_to_fine_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid,std::string str_file,int file_level)
    {

        if(this->mpi->mpisize>=1 && mod(this->i_mean_field_iteration,this->vtk_period)==0)
        {


            this->scatter_petsc_vector(vec2PrintOnUniformGrid);
            std::cout<<" this->mpi->mpisize "<<this->mpi->mpisize<<std::endl;
            Vec v_uniform;
            std::cout<<this->mpi->mpirank<<" start to interpolate on an uniform grid "<<str_file<<std::endl;
            const ptrdiff_t N0 =(int)pow(2,file_level+log(this->nx_trees)/log(2));
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
            InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, this->brick, this->node_neighbors);
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
            this->myDiffusion->printDiffusionArrayFromVector(&v_uniform,"v_uniform");


            int mpi_size;
            int mpi_rank;

            MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

            std::stringstream oss2DebugVec;
            std::string mystr2Debug;
            if(this->myMeanFieldPlan->write2VtkMovie)
            {
            oss2DebugVec << this->convert2FullPath(str_file)<<"_"<<mpi_rank<<"_"<<this->i_mean_field_iteration<<".txt";
            }
            else
            {
                oss2DebugVec << this->convert2FullPath(str_file)<<"_"<<mpi_rank<<".txt";

            }
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

        }

        double MeanField::integrate_over_interface_in_one_quadrant(  p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
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

        double MeanField::integrate_over_interface( Vec phi, Vec f)
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

        double MeanField::integrate_constant_over_interface_in_one_quadrant(  p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi)
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

        double MeanField::integrate_constant_over_interface(Vec phi)
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

