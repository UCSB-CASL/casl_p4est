
#ifdef P4_TO_P8
#include "stresstensor2.h"
#else
#include "stresstensor.h"
#endif

StressTensor::StressTensor(mpi_context_t *mpi   , p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                           p4est_connectivity_t *connectivity, my_p4est_brick_t *brick, my_p4est_node_neighbors_t *node_neighbors,
                           Vec *phi_data_structure, double *q_forward_hist, double *q_backward_hist, int n_local, int n_local_hist,
                           int N_t, int N_iterations, double dt, Vec *phi_is_all_positive, Vec *is_crossed_neumann,int *ix_fromLocal2Global, double Q, double V,
                           PetscBool periodic_xyz, computation_mode my_computation_mode, PetscBool minimumIO, string IO_path,
                           double Lx,double Lx_physics)
{

    std::cout<<" start stress tensor construction "<<std::endl;
    this->initialyze_default_parameters();

    this->Lx=Lx;
    this->Lx_physics=Lx_physics;
    this->p4est=p4est;
    this->nodes=nodes;
    this->ghost=ghost;
    this->connectivity=connectivity;
    this->brick=brick;
    this->nodes_neighbours=node_neighbors;
    this->phi=phi_data_structure;
    this->mpi=mpi;

    this->phi_is_all_positive=phi_is_all_positive;
    this->is_crossed_neumann=is_crossed_neumann;
    this->ix_fromLocal2Global=ix_fromLocal2Global;
    this->n_local=n_local;
    this->n_local_hist=n_local_hist;
    this->N_t=N_t;
    this->N_iterations=N_iterations;
    this->dt=dt;
    this->Q=Q;
    this->V=V;
    this->LS=new my_p4est_level_set(this->nodes_neighbours);
    //this->LS->reinitialize_2nd_order(*this->phi);
    this->periodic_xyz=periodic_xyz;
    this->my_computation_mode=my_computation_mode;
    this->minimum_IO=minimumIO;
    this->IO_path=IO_path;
    if(!this->test)
    {
        this->q_forward_hist=q_forward_hist;
        this->q_backward_hist=q_backward_hist;
    }
    else
    {
        this->create_test_function();
    }
    std::cout<<" finished stress tensor construction "<<std::endl;
}


StressTensor::~StressTensor()
{
    std::cout<<" start destruct stress tensor "<<std::endl;
    this->cleanStressTensor();
    std::cout<<" finished destruc stress tensor "<<std::endl;
}


int StressTensor::create_test_function()
{
    this->q_forward_hist=new double[this->n_local_hist];
    this->q_backward_hist=new double[this->n_local_hist];

    double tree_xmin;
    double tree_ymin;
#ifdef P4_TO_P8
    double tree_zmin;
#endif
    double x,y,z;
    double r;
    double dt=1/this->N_iterations;
    double t;



    for(int it=0;it<this->N_t;it++)
    {
        t=it*dt;
        for(int i=0;i<this->nodes->num_owned_indeps;i++)
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
            z =node_z_fr_k(node) + tree_zmin;
#endif
            x=x-this->Lx/2.00;
            y=y-this->Lx/2.00;

#ifdef P4_TO_P8
            z =z-this->Lx/2.00;
#endif

            r=x*x+y*y;
#ifdef P4_TO_P8
            r=r+z*z;
#endif

            //r=x+y;
            r=pow(r,0.5);
            //r=r/this->Lx;
            this->q_forward_hist[it*this->n_local+i]=r*r*exp(-t);
            this->q_backward_hist[it*this->n_local+i]=r*r*exp(-t);
        }
    }
}

int StressTensor::cleanStressTensor()
{

    if(this->my_computation_mode!=StressTensor::shape_derivative)
    {
        //---------Needs to clean all the global variables to the class which have not been cleaned yet

        std::cout<<" start clean stress tensor "<<std::endl;

        //--------Clean the spatial stress tensor variables--------------------//
        this->ierr=VecDestroy(this->sxx_global); CHKERRXX(this->ierr); // memory decrease global to the object 10

        if(!this->computeOneComponentOnly)
        {
            this->ierr=VecDestroy(this->syy_global); CHKERRXX(this->ierr); // memory decrease global to the object 11
#ifdef P4_TO_P8
            this->ierr=VecDestroy(this->szz_global); CHKERRXX(this->ierr); // memory decrease global to the object 12
#endif
            this->ierr=VecDestroy(this->sxy_global); CHKERRXX(this->ierr); // memory decrease global to the object 13

#ifdef P4_TO_P8
            this->ierr=VecDestroy(this->sxz_global); CHKERRXX(this->ierr); // memory decrease global to the object 14
            this->ierr=VecDestroy(this->syz_global); CHKERRXX(this->ierr); // memory decrease global to the object 15
#endif
        }
        //-----------Finished to clean the stress tensor variables-----------//


        //-------------Clean the temporal history of the spatial derivatives------------------//


        switch(this->my_computation_mode)
        {
        case StressTensor::qxqcx:
        {
            delete this->qx_forward_local;   // memory derease global to the object 4
            if(!this->computeOneComponentOnly)
            {
                delete this->qy_forward_local;   // memory decrease global to the object 5
#ifdef P4_TO_P8
                delete this->qz_forward_local;   // memory decrease global to the object 6
#endif
            }
            delete this->qx_backward_local; // memory decrease global to the object 7

            if(!this->computeOneComponentOnly)
            {
                delete this->qy_backward_local; // memory decrease global to the object 8
#ifdef P4_TO_P8
                delete this->qz_backward_local; // memory decrease global to the object 9
#endif
            }
            break;
        }

        case StressTensor::qqcxx:
        {

            delete this->qxx_backward_local;   // memory derease global to the object 4

            if(!this->computeOneComponentOnly)
            {
                delete this->qyy_backward_local;   // memory decrease global to the object 5
#ifdef P4_TO_P8
                delete this->qzz_backward_local;   // memory decrease global to the object 6
#endif
            }
            if(!this->computeOneComponentOnly)
            {
                delete this->qxy_backward_local; // memory decrease global to the object 7
#ifdef P4_TO_P8
                delete this->qxz_backward_local; // memory decrease global to the object 8
                delete this->qyz_backward_local; // memory decrease global to the object 9
#endif
            }
        }
        }

    }
    else
    {
        this->cleanShapeDerivative();
    }

    if(this->test)
    {
        delete this->q_forward_hist;
        delete this->q_backward_hist;
    }
    //-------------Finished to clean the temporal history of the spatial derivatives------//
    std::cout<<" finish to cleam stress tensor "<<std::endl;
}





// it is the iteration
// q_hist_local is the input whose dimensions are n_local x N_t
// q_x_local is the output whose dimensions are n_localx Nt
// but at each call of this function I do fill only n_local
// and I call this function N_t times

int StressTensor::extract_and_process_iteration(int it,double *q_hist_local,double *q_x_local,double *q_y_local,double *q_z_local)
{


    Vec q_forward_t;
    Vec bc_vec_fake;
    int order_to_extend=2;
    int number_of_bands_to_extend=5;

    this->ierr= VecDuplicate(*this->phi,&q_forward_t);  CHKERRXX(this->ierr); //memory increase local to the function 1
    this->ierr=VecDuplicate(*this->phi,&bc_vec_fake);   CHKERRXX(this->ierr); //memory increase local to the function 2

    this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);




    //-----------------Fill the input into a local data structure-----------------------------------------//


    PetscScalar *q_forward_local=new double[this->n_local];  // memory increase local to the function 3
    // for each time step get the spatial q from the simulation data
    for(int ix=0;ix<this->n_local;ix++)
        q_forward_local[ix]=q_hist_local[it*this->n_local+ix];

    // set the values into a spatial parallel petsc ghost vector
    this->ierr=VecSetValues(q_forward_t,this->n_local,this->ix_fromLocal2Global,q_forward_local ,INSERT_VALUES); CHKERRXX(this->ierr);

    // Assemble the parallel petsc vector
    this->ierr=VecAssemblyBegin(q_forward_t); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(q_forward_t);   CHKERRXX(this->ierr);

    // scatter forward the parallel petsc vector
    this->ierr=VecGhostUpdateBegin(q_forward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(q_forward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    //------------Finish to load and assemble the local data structure--------------------------------//

    if(!this->periodic_xyz)
    {
        // Extend

        this->LS->extend_Over_Interface(*this->phi,q_forward_t,NEUMANN,bc_vec_fake,order_to_extend,number_of_bands_to_extend);

    }
    // get the vectors on theirs petsc local data structure
    // note:: check it is not the same than the one
    // who previously loaded the data from the transferred historic data base

    PetscScalar *q_forward_local_t;
    this->ierr=VecGetArray(q_forward_t,&q_forward_local_t); CHKERRXX(this->ierr); // memory increase local to the function 4

    this->printDiffusionArray(q_forward_local_t,this->n_local,"q_forward_local_t");


    PetscScalar *q_x_local_f,*q_y_local_f,*q_z_local_f;
    this->ierr=VecGetArray(this->qx_forward,&q_x_local_f); CHKERRXX(this->ierr); // memory increase local to the function 5

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecGetArray(this->qy_forward,&q_y_local_f); CHKERRXX(this->ierr); // memory increase local to the function 6

#ifdef P4_TO_P8
        this->ierr=VecGetArray(this->qz_forward,&q_z_local_f); CHKERRXX(this->ierr); // memory increase local to the function 7
#endif
    }
    for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
    {
        q_x_local_f[ix]=this->nodes_neighbours->neighbors[ix].dx_central(q_forward_local_t);

    }

    if(!this->computeOneComponentOnly)
    {
        for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
        {

            q_y_local_f[ix]=this->nodes_neighbours->neighbors[ix].dy_central(q_forward_local_t);
#ifdef P4_TO_P8
            q_z_local_f[ix]=this->nodes_neighbours->neighbors[ix].dz_central(q_forward_local_t);
#endif
        }
    }
    this->ierr=VecRestoreArray(q_forward_t,&q_forward_local_t);CHKERRXX(this->ierr); // no need to decrease memory local to the function 4
    this->ierr=VecRestoreArray(this->qx_forward,&q_x_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 5

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecRestoreArray(this->qy_forward,&q_y_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 6

#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(this->qz_forward,&q_z_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 7
#endif
        // scatter again
    }
    this->ierr=VecGhostUpdateBegin(this->qx_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qx_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecGhostUpdateBegin(this->qy_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->qy_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecGhostUpdateBegin(this->qz_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->qz_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
#endif
    }
    // Get the arrays again and fill the database
    // note:: we may fill the ghost nodes in each processor which is inefficient
    // to change it in the future and to fill local nodes only

    PetscScalar *q_x_local_f_2,*q_y_local_f_2,*q_z_local_f_2;

    this->ierr=VecGetArray(this->qx_forward,&q_x_local_f_2); CHKERRXX(this->ierr); // memory increase local to the function 8

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecGetArray(this->qy_forward,&q_y_local_f_2); CHKERRXX(this->ierr); // memory increase local to the function 9

#ifdef P4_TO_P8
        this->ierr=VecGetArray(this->qz_forward,&q_z_local_f_2); CHKERRXX(this->ierr); // memory increase local to the function 10
#endif
    }
    //--------------------------------Fill the input for output------------------------------//
    for(int ix=0;ix<this->n_local;ix++)
    {
        q_x_local[it*this->n_local+ix]=q_x_local_f_2[ix];

    }

    if(!this->computeOneComponentOnly)
    {
        for(int ix=0;ix<this->n_local;ix++)
        {

            q_y_local[it*this->n_local+ix]=q_y_local_f_2[ix];
#ifdef P4_TO_P8
            q_z_local[it*this->n_local+ix]=q_z_local_f_2[ix];
#endif
        }
    }
    //--------------------------  Input Filled For Output-----------------------------------//


    //----------------------Print iteration filling-----------------------------//

    this->printDiffusionArray(q_x_local_f_2,this->n_local,"qx_it");

    if(!this->computeOneComponentOnly)
    {
        this->printDiffusionArray(q_y_local_f_2,this->n_local,"qy_it");

#ifdef P4_TO_P8
        this->printDiffusionArray(q_z_local_f_2,this->n_local,"qz_it");
#endif
    }


    //----------------------Finished to print iteration filling---------------//


    this->ierr=VecRestoreArray(this->qx_forward,&q_x_local_f_2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 8

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecRestoreArray(this->qy_forward,&q_y_local_f_2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 9

#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(this->qz_forward,&q_z_local_f_2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 10
#endif
    }


    VecDestroy(q_forward_t); // memory decrease 1
    VecDestroy(bc_vec_fake); // memory decrease 2
    delete q_forward_local; // memory  decrease 3

}

// it is the iteration
// q_hist_local is the input whose dimensions are n_local x N_t
// q_x_local is the output whose dimensions are n_localx Nt
// but at each call of this function I do fill only n_local
// and I call this function N_t times

int StressTensor::extract_and_process_iteration_forward_backward(int it,double *q_hist_local,double *q_x_local,PetscBool forward_solution)
{
    Vec q_forward_t;
    this->ierr= VecDuplicate(*this->phi,&q_forward_t);  CHKERRXX(this->ierr); //memory increase local to the function 1

    //-----------------Fill the input into a local data structure-----------------------------------------//

    PetscScalar *q_forward_local=new double[this->n_local];  // memory increase local to the function 2
    // for each time step get the spatial q from the simulation data
    if(forward_solution)
    {
    for(int ix=0;ix<this->n_local;ix++)
        q_forward_local[ix]=q_hist_local[it*this->n_local+ix];
    }
    else
    {
        for(int ix=0;ix<this->n_local;ix++)
            q_forward_local[ix]=q_hist_local[this->n_local_hist-(it+1)*this->n_local+ix];
    }

    // set the values into a spatial parallel petsc ghost vector
    this->ierr=VecSetValues(q_forward_t,this->n_local,this->ix_fromLocal2Global,q_forward_local ,INSERT_VALUES); CHKERRXX(this->ierr);

    // Assemble the parallel petsc vector
    this->ierr=VecAssemblyBegin(q_forward_t); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(q_forward_t);   CHKERRXX(this->ierr);

    // scatter forward the parallel petsc vector
    this->ierr=VecGhostUpdateBegin(q_forward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(q_forward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    //------------Finish to load and assemble the local data structure--------------------------------//


    // get the vectors on theirs petsc local data structure
    // note:: check it is not the same than the one
    // who previously loaded the data from the transferred historic data base

    PetscScalar *q_forward_local_t;
    this->ierr=VecGetArray(q_forward_t,&q_forward_local_t); CHKERRXX(this->ierr); // memory increase local to the function 3

    this->printDiffusionArray(q_forward_local_t,this->n_local,"q_forward_local_t");


    PetscScalar *q_x_local_f;
    this->ierr=VecGetArray(this->qx_forward,&q_x_local_f); CHKERRXX(this->ierr); // memory increase local to the function 4


    for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
    {
        q_x_local_f[ix]=this->nodes_neighbours->neighbors[ix].dx_central(q_forward_local_t);

    }


    this->ierr=VecRestoreArray(q_forward_t,&q_forward_local_t);CHKERRXX(this->ierr); // no need to decrease memory local to the function 3
    this->ierr=VecRestoreArray(this->qx_forward,&q_x_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 4


    this->ierr=VecGhostUpdateBegin(this->qx_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qx_forward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


    // Get the arrays again and fill the database
    // note:: we may fill the ghost nodes in each processor which is inefficient
    // to change it in the future and to fill local nodes only

    PetscScalar *q_x_local_f_2;

    this->ierr=VecGetArray(this->qx_forward,&q_x_local_f_2); CHKERRXX(this->ierr); // memory increase local to the function 5


    //--------------------------------Fill the input for output------------------------------//
    for(int ix=0;ix<this->n_local;ix++)
    {
        q_x_local[ix]=q_x_local_f_2[ix];

    }

    //----------------------Print iteration filling-----------------------------//

    this->printDiffusionArray(q_x_local_f_2,this->n_local,"qx_it");

    //----------------------Finished to print iteration filling---------------//
    this->ierr=VecRestoreArray(this->qx_forward,&q_x_local_f_2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 5
    this->ierr=VecDestroy(q_forward_t); CHKERRXX(this->ierr); // memory decrease 1
    delete q_forward_local; // memory  decrease 2

}

// NOTE:: this algo is greedy in memory
// but it has many advantages:
// (1) faster than the efficient memory algo
// (2) ability to get the spatial distribution of the stresses

int StressTensor::compute_spatial_integrand()
{

    //------------Compute and fill the history of the spatial derivatives
    // for both q_forward and q_backward. The data structure layout is:
    // it=0:1................N_grid
    // it=1:1................N_grid
    // and so forth

    //compute temporal spatial derivatives on x y and z

    // qx_forward, qy_forward,qx_forward have the size of the grid
    // and are in fact temporary variables where to store the spatial variables for
    // one iteration in time of the diffusion solution
    this->ierr=VecDuplicate(*this->phi,&this->qx_forward); CHKERRXX(this->ierr);  // memory increase global to the object 1
    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecDuplicate(*this->phi,&this->qy_forward); CHKERRXX(this->ierr);  // memory increase global to the object 2
#ifdef P4_TO_P8
        this->ierr=VecDuplicate(*this->phi,&this->qz_forward); CHKERRXX(this->ierr);  // memory increase global to the object 3
#endif
    }

    // very expensive step in terms of memory
    // consider later to do it direction by direction to save memory
    // or on the fly direction by direction and to store only the two last
    // time steps for the integrand which will be integrated by Simpsons.

    this->qx_forward_local=new double [this->n_local_hist];  // memory increase global to the object 4

    if(!this->computeOneComponentOnly)
    {
        this->qy_forward_local=new double [this->n_local_hist];  // memory increase global to the object 5

#ifdef P4_TO_P8
        this->qz_forward_local=new double [this->n_local_hist];  // memory increase global to the object 6
#endif
    }
    this->qx_backward_local=new double[this->n_local_hist];  // memory increase global to the object 7
    if(!this->computeOneComponentOnly)
    {
        this->qy_backward_local=new double[this->n_local_hist];  // memory increase global to the object 8

#ifdef P4_TO_P8
        this->qz_backward_local=new double [this->n_local_hist]; // memory increase global to the object 9
#endif
    }
    // iterate on time
    // Note: consider to not use vecs at all
    for(int it=0;it<this->N_t;it++)
    {
        this->extract_and_process_iteration(it,this->q_forward_hist,this->qx_forward_local,this->qy_forward_local,this->qz_forward_local);
        this->extract_and_process_iteration(it,this->q_backward_hist,this->qx_backward_local,this->qy_backward_local,this->qz_backward_local);
    }


    this->ierr=VecDestroy(this->qx_forward); CHKERRXX(this->ierr); // memory decrease global to the object 1

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecDestroy(this->qy_forward); CHKERRXX(this->ierr); // memory decrease global to the object 2

#ifdef P4_TO_P8
        this->ierr=VecDestroy(this->qz_forward); CHKERRXX(this->ierr); // memory decrease global to the object 3
#endif
    }
    // At the end of this step:
    // qx_backward_local, qy_backward_local, qz_backward_local
    // qx_backward_local, qy_backward_local, qz_backward_local
    // are filled with their correct values

}

// NOTE:: this algo is greedy in memory
// but it has many advantages:
// (1) faster than the efficient memory algo
// (2) ability to get the spatial distribution of the stresses

int StressTensor::fill_snn_hist()
{

    //------------Compute and fill the history of the spatial derivatives
    // for both q_forward and q_backward. The data structure layout is:
    // it=0:1................N_grid
    // it=1:1................N_grid
    // and so forth

    //compute temporal spatial derivatives on x y and z

    // qx_forward, qy_forward,qx_forward have the size of the grid
    // and are in fact temporary variables where to store the spatial variables for
    // one iteration in time of the diffusion solution
    this->ierr=VecDuplicate(*this->phi,&this->qx_forward); CHKERRXX(this->ierr);  // memory increase global to the object 1
    this->ierr=VecDuplicate(*this->phi,&this->qy_forward); CHKERRXX(this->ierr);  // memory increase global to the object 2
#ifdef P4_TO_P8
    this->ierr=VecDuplicate(*this->phi,&this->qz_forward); CHKERRXX(this->ierr);  // memory increase global to the object 3
#endif

    // very expensive step in terms of memory
    // consider later to do it direction by direction to save memory
    // or on the fly direction by direction and to store only the two last
    // time steps for the integrand which will be integrated by Simpsons.

    this->snn_hist=new double [this->n_local_hist];  // memory increase global to the object 4

    // iterate on time
    // Note: consider to not use vecs at all
    for(int it=0;it<this->N_t;it++)
    {
        this->extract_and_process_iteration_fill_snn(it);
    }


    this->ierr=VecDestroy(this->qx_forward); CHKERRXX(this->ierr); // memory decrease global to the object 1
    this->ierr=VecDestroy(this->qy_forward); CHKERRXX(this->ierr); // memory decrease global to the object 2

#ifdef P4_TO_P8
    this->ierr=VecDestroy(this->qz_forward); CHKERRXX(this->ierr); // memory decrease global to the object 3
#endif
    // At the end of this step:
    // qx_backward_local, qy_backward_local, qz_backward_local
    // qx_backward_local, qy_backward_local, qz_backward_local
    // are filled with their correct values

}


int StressTensor::cleanShapeDerivative()
{
    this->ierr=VecDestroy(this->snn_global); CHKERRXX(this->ierr);
}


// it is the iteration
// q_hist_local is the input whose dimensions are n_local x N_t
// q_x_local is the output whose dimensions are n_localx Nt
// but at each call of this function I do fill only n_local
// and I call this function N_t times

int StressTensor::extract_and_process_iteration_irregular(int it)
//,double *q_hist_local,double *q_xx_local,double *q_yy_local,double *q_zz_local,
//                                                                                    double *q_xy_local,double *q_xz_local,double *q_yz_local)
{


    Vec q_backward_t;
    Vec bc_vec_fake;
    int order_to_extend=2;
    int number_of_bands_to_extend=5;

    this->ierr= VecDuplicate(*this->phi,&q_backward_t);  CHKERRXX(this->ierr); //memory increase local to the function 1
    this->ierr=VecDuplicate(*this->phi,&bc_vec_fake);   CHKERRXX(this->ierr); //memory increase local to the function 2

    this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);




    //-----------------Fill the input into a local data structure-----------------------------------------//


    PetscScalar *q_backward_local=new double[this->n_local];  // memory increase local to the function 3
    // for each time step get the spatial q from the simulation data
    for(int ix=0;ix<this->n_local;ix++)
        q_backward_local[ix]=this->q_backward_hist[it*this->n_local+ix];

    // set the values into a spatial parallel petsc ghost vector
    this->ierr=VecSetValues(q_backward_t,this->n_local,this->ix_fromLocal2Global,q_backward_local ,INSERT_VALUES); CHKERRXX(this->ierr);

    // Assemble the parallel petsc vector
    this->ierr=VecAssemblyBegin(q_backward_t); CHKERRXX(this->ierr);
    this->ierr=VecAssemblyEnd(q_backward_t);   CHKERRXX(this->ierr);

    // scatter forward the parallel petsc vector
    this->ierr=VecGhostUpdateBegin(q_backward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(q_backward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    //------------Finish to load and assemble the local data structure--------------------------------//

    if(!this->periodic_xyz)
    {
        // Extend

        this->LS->extend_Over_Interface(*this->phi,q_backward_t,NEUMANN,bc_vec_fake,order_to_extend,number_of_bands_to_extend);
        // scatter forward the parallel petsc vector
        this->ierr=VecGhostUpdateBegin(q_backward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(q_backward_t,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }
    // get the vectors on theirs petsc local data structure
    // note:: check it is not the same than the one
    // who previously loaded the data from the transferred historic data base

    PetscScalar *q_backward_local_t;
    this->ierr=VecGetArray(q_backward_t,&q_backward_local_t); CHKERRXX(this->ierr); // memory increase local to the function 4

    this->printDiffusionArray(q_backward_local_t,this->n_local,"q_forward_local_t");


    PetscScalar *q_xx_local_f,*q_yy_local_f;

#ifdef P4_TO_P8
    PetscScalar *q_zz_local_f;
#endif
    PetscScalar *q_x_local_f,*q_y_local_f;


    this->ierr=VecGetArray(this->qx_backward,&q_x_local_f); CHKERRXX(this->ierr); // memory increase local to the function 11
    this->ierr=VecGetArray(this->qy_backward,&q_y_local_f); CHKERRXX(this->ierr); // memory increase local to the function 12

    this->ierr=VecGetArray(this->qxx_backward,&q_xx_local_f); CHKERRXX(this->ierr); // memory increase local to the function 5
    this->ierr=VecGetArray(this->qyy_backward,&q_yy_local_f); CHKERRXX(this->ierr); // memory increase local to the function 6

#ifdef P4_TO_P8
    this->ierr=VecGetArray(this->qzz_backward,&q_zz_local_f); CHKERRXX(this->ierr); // memory increase local to the function 7
#endif


    for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
    {
        q_xx_local_f[ix]=this->nodes_neighbours->neighbors[ix].dxx_central(q_backward_local_t);
        q_yy_local_f[ix]=this->nodes_neighbours->neighbors[ix].dyy_central(q_backward_local_t);

#ifdef P4_TO_P8
        q_zz_local_f[ix]=this->nodes_neighbours->neighbors[ix].dzz_central(q_backward_local_t);
#endif
    }

    for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
    {
        q_x_local_f[ix]=this->nodes_neighbours->neighbors[ix].dx_central(q_backward_local_t);
        q_y_local_f[ix]=this->nodes_neighbours->neighbors[ix].dy_central(q_backward_local_t);
    }


    this->printDiffusionArray(q_x_local_f,this->n_local,"qx_it");
    this->printDiffusionArray(q_y_local_f,this->n_local,"qy_it");



    this->ierr=VecRestoreArray(q_backward_t,&q_backward_local_t);CHKERRXX(this->ierr); // no need to decrease memory local to the function 4

    this->ierr=VecRestoreArray(this->qxx_backward,&q_xx_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 5
    this->ierr=VecRestoreArray(this->qyy_backward,&q_yy_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 6
    this->ierr=VecRestoreArray(this->qx_backward,&q_x_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 11
    this->ierr=VecRestoreArray(this->qy_backward,&q_y_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 12

#ifdef P4_TO_P8
    this->ierr=VecRestoreArray(this->qzz_backward,&q_zz_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 7
#endif

    // scatter again

    this->ierr=VecGhostUpdateBegin(this->qxx_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qxx_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->qyy_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qyy_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
    this->ierr=VecGhostUpdateBegin(this->qzz_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qzz_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
#endif

    this->ierr=VecGhostUpdateBegin(this->qx_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qx_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->qy_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qy_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);




    //----------------------------Continue to get the mixed derivatives--------------------------------------------------------------//
    //-------------------------------------------------------------------------------------------------------------------------------//

    this->ierr=VecGetArray(this->qx_backward,&q_x_local_f); CHKERRXX(this->ierr); // memory increase local to the function 11
    this->ierr=VecGetArray(this->qy_backward,&q_y_local_f); CHKERRXX(this->ierr); // memory increase local to the function 12


    PetscScalar *q_xy_local_f,*q_xz_local_f;

#ifdef P4_TO_P8
    PetscScalar *q_yz_local_f;
#endif

    this->ierr=VecGetArray(this->qxy_backward,&q_xy_local_f); CHKERRXX(this->ierr); // memory increase local to the function 11
    this->ierr=VecGetArray(this->qxz_backward,&q_xz_local_f); CHKERRXX(this->ierr); // memory increase local to the function 12

#ifdef P4_TO_P8
    this->ierr=VecGetArray(this->qyz_backward,&q_yz_local_f); CHKERRXX(this->ierr); // memory increase local to the function 13
#endif



    for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
    {
        q_xy_local_f[ix]=this->nodes_neighbours->neighbors[ix].dy_central(q_x_local_f);
#ifdef P4_TO_P8
        q_xz_local_f[ix]=this->nodes_neighbours->neighbors[ix].dz_central(q_x_local_f);
        q_yz_local_f[ix]=this->nodes_neighbours->neighbors[ix].dz_central(q_y_local_f);
#endif

    }


    this->ierr=VecRestoreArray(this->qxy_backward,&q_xy_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 11
#ifdef P4_TO_P8
    this->ierr=VecRestoreArray(this->qxz_backward,&q_xz_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 12
    this->ierr=VecRestoreArray(this->qyz_backward,&q_yz_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 13
#endif
    this->ierr=VecRestoreArray(this->qx_backward,&q_x_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 11
    this->ierr=VecRestoreArray(this->qy_backward,&q_y_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 12


    // scatter again

    this->ierr=VecGhostUpdateBegin(this->qxy_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qxy_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
    this->ierr=VecGhostUpdateBegin(this->qxz_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qxz_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(this->qyz_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->qyz_backward,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
#endif

    //-----------------------Finished to get the mixed derivatives------------------------------------------------------------//
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    // Get the arrays again and fill the database
    // note:: we may fill the ghost nodes in each processor which is inefficient
    // to change it in the future and to fill local nodes only

    PetscScalar *q_xx_local_f2,*q_yy_local_f2;
    PetscScalar *q_xy_local_f2;



    this->ierr=VecGetArray(this->qxx_backward,&q_xx_local_f2); CHKERRXX(this->ierr); // memory increase local to the function 14
    this->ierr=VecGetArray(this->qyy_backward,&q_yy_local_f2); CHKERRXX(this->ierr); // memory increase local to the function 15

    this->ierr=VecGetArray(this->qxy_backward,&q_xy_local_f2); CHKERRXX(this->ierr); // memory increase local to the function 17

#ifdef P4_TO_P8
    PetscScalar *q_xz_local_f2,*q_yz_local_f2,*q_zz_local_f2;
    this->ierr=VecGetArray(this->qzz_backward,&q_zz_local_f2); CHKERRXX(this->ierr); // memory increase local to the function 16
    this->ierr=VecGetArray(this->qxz_backward,&q_xz_local_f2); CHKERRXX(this->ierr); // memory increase local to the function 18
    this->ierr=VecGetArray(this->qyz_backward,&q_yz_local_f2); CHKERRXX(this->ierr); // memory increase local to the function 19


#endif

    //--------------------------------Fill the input for output------------------------------//
    for(int ix=0;ix<this->n_local;ix++)
    {
        this->qxx_backward_local[it*this->n_local+ix]=q_xx_local_f2[ix];
        this->qyy_backward_local[it*this->n_local+ix]=q_yy_local_f2[ix];
        this->qxy_backward_local[it*this->n_local+ix]=q_xy_local_f2[ix];

#ifdef P4_TO_P8
        this->qzz_backward_local[it*this->n_local+ix]=q_zz_local_f2[ix];
        this->qxz_backward_local[it*this->n_local+ix]=q_xz_local_f2[ix];
        this->qyz_backward_local[it*this->n_local+ix]=q_yz_local_f2[ix];
#endif
    }
    //--------------------------  Input Filled For Output-----------------------------------//


    //----------------------Print iteration filling-----------------------------//

    this->printDiffusionArray(q_xx_local_f2,this->n_local,"qxx_it");
    this->printDiffusionArray(q_yy_local_f2,this->n_local,"qyy_it");
    this->printDiffusionArray(q_xy_local_f2,this->n_local,"qxy_it");

#ifdef P4_TO_P8
    this->printDiffusionArray(q_xz_local_f2,this->n_local,"qxz_it");
    this->printDiffusionArray(q_yz_local_f2,this->n_local,"qyz_it");
    this->printDiffusionArray(q_zz_local_f2,this->n_local,"qzz_it");
#endif



    //----------------------Finished to print iteration filling---------------//


    this->ierr=VecRestoreArray(this->qxx_backward,&q_xx_local_f2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 14
    this->ierr=VecRestoreArray(this->qyy_backward,&q_yy_local_f2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 15
    this->ierr=VecRestoreArray(this->qxy_backward,&q_xy_local_f2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 17

#ifdef P4_TO_P8
    this->ierr=VecRestoreArray(this->qxz_backward,&q_xz_local_f2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 18
    this->ierr=VecRestoreArray(this->qyz_backward,&q_yz_local_f2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 19
    this->ierr=VecRestoreArray(this->qzz_backward,&q_zz_local_f2); CHKERRXX(this->ierr); // no need to decrease memory local to the function 16
#endif

    this->ierr=VecDestroy(q_backward_t); CHKERRXX(this->ierr);   // memory decrease 1
    this->ierr=VecDestroy(bc_vec_fake); CHKERRXX(this->ierr);  // memory decrease 2
    delete q_backward_local; // memory  decrease 3

}



// it is the iteration
// q_hist_local is the input whose dimensions are n_local x N_t
// q_x_local is the output whose dimensions are n_localx Nt
// but at each call of this function I do fill only n_local
// and I call this function N_t times

int StressTensor::extract_and_process_iteration_fill_snn(int it)
//,double *q_hist_local,double *q_xx_local,double *q_yy_local,double *q_zz_local,
//                                                                                    double *q_xy_local,double *q_xz_local,double *q_yz_local)
{


    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {

        Vec q_backward_t;
        Vec q_forward_t;
        Vec bc_vec_fake;
        int order_to_extend=2;
        int number_of_bands_to_extend=512;

        this->ierr=VecDuplicate(*this->phi,&this->snn_global); CHKERRXX(this->ierr); //memory increase local to the function 0
        this->ierr= VecDuplicate(*this->phi,&q_backward_t);  CHKERRXX(this->ierr); //memory increase local to the function 1
        this->ierr=VecDuplicate(*this->phi,&bc_vec_fake);   CHKERRXX(this->ierr); // memory increase local to the function 2
        this->ierr=VecDuplicate(*this->phi,&q_forward_t); CHKERRXX(this->ierr);   // memory increase local to the function 3


        this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&bc_vec_fake);



        //-----------------Fill the input into a local data structure-----------------------------------------//


        PetscScalar *q_forward_local=new double[this->n_local];  // memory increase local to the function 4
        PetscScalar *q_backward_local=new double[this->n_local];  // memory increase local to the function 5


        // for each time step get the spatial q from the simulation data
        for(int ix=0;ix<this->n_local;ix++)
        {
            q_backward_local[ix]=this->q_backward_hist[this->n_local_hist-(it+1) *this->n_local+ix];
            q_forward_local[ix]=this->q_forward_hist[it *this->n_local+ix];
        }

        // set the values into a spatial parallel petsc ghost vector
        this->ierr=VecSetValues(q_backward_t,this->n_local,this->ix_fromLocal2Global,q_backward_local ,INSERT_VALUES); CHKERRXX(this->ierr);

        // Assemble the parallel petsc vector
        this->ierr=VecAssemblyBegin(q_backward_t); CHKERRXX(this->ierr);
        this->ierr=VecAssemblyEnd(q_backward_t);   CHKERRXX(this->ierr);

        // scatter forward the parallel petsc vector
        this->scatter_petsc_vector(&q_backward_t);

        // set the values into a spatial parallel petsc ghost vector
        this->ierr=VecSetValues(q_forward_t,this->n_local,this->ix_fromLocal2Global,q_forward_local ,INSERT_VALUES); CHKERRXX(this->ierr);

        // Assemble the parallel petsc vector
        this->ierr=VecAssemblyBegin(q_forward_t); CHKERRXX(this->ierr);
        this->ierr=VecAssemblyEnd(q_forward_t);   CHKERRXX(this->ierr);
        // scatter forward the parallel petsc vector
        this->scatter_petsc_vector(&q_forward_t);


        if(it==this->N_t-1)
        {
             this->interpolate_and_print_vec_to_uniform_grid(&q_forward_t,"q_forward_non_extended");
        }
        if(it==0)
        {
             this->interpolate_and_print_vec_to_uniform_grid(&q_backward_t,"q_backward_non_extended");

     }
        //------------Finish to load and assemble the local data structure--------------------------------//


        //------------Extend q and q dagger---------------------------------------------------------------//

//        if(!this->test)
//        {
//            this->LS->extend_Over_Interface_With_Stride(*this->phi,q_backward_t,NEUMANN,*this->phi_is_all_positive,order_to_extend,number_of_bands_to_extend);
//            // scatter forward the parallel petsc vector
//            this->scatter_petsc_vector(&q_backward_t);
//            this->LS->extend_Over_Interface_With_Stride(*this->phi,q_forward_t,NEUMANN,*this->phi_is_all_positive,order_to_extend,number_of_bands_to_extend);
//            // scatter forward the parallel petsc vector
//            this->scatter_petsc_vector(&q_forward_t);
//        }
        delete q_backward_local;  // memory decrease local to the function 4
        delete q_forward_local;    // memory decrease local to the function 5
        //this->printDiffusionArrayFromVector(&q_backward_t,"q_backward_t_extended");
        //this->printDiffusionArrayFromVector(&q_forward_t,"q_forward_t_extended");




        //----------finished to extend q and q dagger----------------------------------------------------------------//

        // get the vectors on theirs petsc local data structure
        // note:: check it is not the same than the one
        // who previously loaded the data from the transferred historic data base

        PetscScalar *q_forward_local_t;
        this->ierr=VecGetArray(q_forward_t,&q_forward_local_t); CHKERRXX(this->ierr);

        this->printDiffusionArray(q_forward_local_t,this->n_local,"q_forward_local_t");



        PetscScalar *q_x_local_f,*q_y_local_f;
#ifdef P4_TO_P8
        PetscScalar *q_z_local_f;
#endif



        this->ierr=VecGetArray(this->qx_forward,&q_x_local_f); CHKERRXX(this->ierr); // memory increase local to the function 11
        this->ierr=VecGetArray(this->qy_forward,&q_y_local_f); CHKERRXX(this->ierr); // memory increase local to the function 12

#ifdef P4_TO_P8
        this->ierr=VecGetArray(this->qz_forward,&q_z_local_f); CHKERRXX(this->ierr); // memory increase local to the function 7
#endif



        for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
        {
            q_x_local_f[ix]=this->nodes_neighbours->neighbors[ix].dx_central(q_forward_local_t);
            q_y_local_f[ix]=this->nodes_neighbours->neighbors[ix].dy_central(q_forward_local_t);
#ifdef P4_TO_P8
            q_z_local_f[ix]=this->nodes_neighbours->neighbors[ix].dz_central(q_forward_local_t);
#endif

        }


        this->printDiffusionArray(q_x_local_f,this->n_local,"qx_it");
        this->printDiffusionArray(q_y_local_f,this->n_local,"qy_it");

#ifdef P4_TO_P8
        this->printDiffusionArray(q_z_local_f,this->n_local,"qz_it");
#endif


        this->ierr=VecRestoreArray(q_forward_t,&q_forward_local_t);CHKERRXX(this->ierr); // no need to decrease memory local to the function 4
        this->ierr=VecRestoreArray(this->qx_forward,&q_x_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 11
        this->ierr=VecRestoreArray(this->qy_forward,&q_y_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 12

#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(this->qz_forward,&q_z_local_f); CHKERRXX(this->ierr); // no need to decrease memory local to the function 7
#endif

        // scatter again

        this->scatter_petsc_vector(&this->qx_forward);
        this->scatter_petsc_vector(&this->qy_forward);
#ifdef P4_TO_P8
        this->scatter_petsc_vector(&this->qz_forward);
#endif


        this->printDiffusionArrayFromVector(&this->qx_forward,"qx_forward");
        this->printDiffusionArrayFromVector(&this->qy_forward,"qy_forward");

#ifdef P4_TO_P8
        this->printDiffusionArrayFromVector(&this->qz_forward,"qz_forward");
#endif


        //----------------------------Second Term--------------------------------------------------------------//
        //-------------------------------------------------------------------------------------------------------------------------------//
        Vec qdq_x_x,qdq_y_y;

#ifdef P4_TO_P8
        Vec qdq_z_z;
#endif

        this->ierr=VecDuplicate(*this->phi,&qdq_x_x); CHKERRXX(this->ierr);  // memory increase local to the function 6
        this->ierr=VecDuplicate(*this->phi,&qdq_y_y); CHKERRXX(this->ierr);   // memory increase local to the function 7

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(*this->phi,&qdq_z_z); CHKERRXX(this->ierr);  // memory increase local to the function 8
#endif


        PetscScalar *qdq_x_x_local,*qdq_y_y_local;

#ifdef P4_TO_P8
        PetscScalar *qdq_z_z_local;
#endif

        Vec qdq_x,qdq_y;

#ifdef P4_TO_P8
        Vec qdq_z;
#endif

        this->ierr=VecDuplicate(*this->phi,&qdq_x); CHKERRXX(this->ierr);  // memory increase local to the function 9
        this->ierr=VecDuplicate(*this->phi,&qdq_y); CHKERRXX(this->ierr);   // memory increase local to the function 10

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(*this->phi,&qdq_z); CHKERRXX(this->ierr);  // memory increase local to the function 11
#endif



        PetscScalar *qdq_x_local,*qdq_y_local;

#ifdef P4_TO_P8
        PetscScalar *qdq_z_local;
#endif

        this->ierr=VecPointwiseMult(qdq_x,q_backward_t,this->qx_forward); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(qdq_y,q_backward_t,this->qy_forward); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecPointwiseMult(qdq_z,q_backward_t,this->qz_forward); CHKERRXX(this->ierr);
#endif

        this->scatter_petsc_vector(&qdq_x);
        this->scatter_petsc_vector(&qdq_y);

#ifdef P4_TO_P8
        this->scatter_petsc_vector(&qdq_z);
#endif


        this->ierr=VecGetArray(qdq_x_x,&qdq_x_x_local); CHKERRXX(this->ierr); // memory increase local to the function 11
        this->ierr=VecGetArray(qdq_y_y,&qdq_y_y_local); CHKERRXX(this->ierr); // memory increase local to the function 12

#ifdef P4_TO_P8
        this->ierr=VecGetArray(qdq_z_z,&qdq_z_z_local); CHKERRXX(this->ierr); // memory increase local to the function 13
#endif

        this->ierr=VecGetArray(qdq_x,&qdq_x_local); CHKERRXX(this->ierr); // memory increase local to the function 11
        this->ierr=VecGetArray(qdq_y,&qdq_y_local); CHKERRXX(this->ierr); // memory increase local to the function 12

#ifdef P4_TO_P8
        this->ierr=VecGetArray(qdq_z,&qdq_z_local); CHKERRXX(this->ierr); // memory increase local to the function 13
#endif

        for(p4est_locidx_t ix=0;ix<this->nodes->num_owned_indeps;ix++)
        {
            qdq_x_x_local[ix]=this->nodes_neighbours->neighbors[ix].dx_central(qdq_x_local);
            qdq_y_y_local[ix]=this->nodes_neighbours->neighbors[ix].dy_central(qdq_y_local);
#ifdef P4_TO_P8
            qdq_z_z_local[ix]=this->nodes_neighbours->neighbors[ix].dz_central(qdq_z_local);
#endif

        }


        this->ierr=VecRestoreArray(qdq_x_x,&qdq_x_x_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(qdq_y_y,&qdq_y_y_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(qdq_z_z,&qdq_z_z_local); CHKERRXX(this->ierr);
#endif

        this->ierr=VecRestoreArray(qdq_x,&qdq_x_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(qdq_y,&qdq_y_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(qdq_z,&qdq_z_local); CHKERRXX(this->ierr);
#endif

        // scatter again

        this->scatter_petsc_vector(&qdq_x_x);
        this->scatter_petsc_vector(&qdq_y_y);

#ifdef P4_TO_P8
        this->scatter_petsc_vector(&qdq_z_z);
#endif


        this->printDiffusionArrayFromVector(&qdq_x_x,"qdq_x_x");
        this->printDiffusionArrayFromVector(&qdq_y_y,"qdq_y_y");

#ifdef P4_TO_P8
        this->printDiffusionArrayFromVector(&qdq_z_z,"qdq_z_z");
#endif

   if(it==this->N_t-1)
   {
        Vec SecondTerm;
        this->ierr=VecDuplicate(qdq_x_x,&SecondTerm); CHKERRXX(this->ierr);
        this->ierr=VecCopy(qdq_x_x,SecondTerm); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(SecondTerm,1.00,qdq_y_y); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(&SecondTerm);
        this->interpolate_and_print_vec_to_uniform_grid(&SecondTerm,"second_term");
        this->ierr=VecDestroy(SecondTerm); CHKERRXX(this->ierr);

        this->interpolate_and_print_vec_to_uniform_grid(&qdq_x_x,"second_term_x");
        this->interpolate_and_print_vec_to_uniform_grid(&qdq_y_y,"second_term_y");
        this->interpolate_and_print_vec_to_uniform_grid(&q_forward_t,"q_forward");
   }
   if(it==0)
        {
        this->interpolate_and_print_vec_to_uniform_grid(&q_backward_t,"q_backward");

}
        //-----------------------Finished to get the second term------------------------------------------------------------//
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        //-----------------------------------First Term-----------------------------------------//

        Vec grad_q_dot_grad_phi;
        Vec grad_grad_q_dot_grad_phi_dot_grad_phi;
        Vec grad_phi_square;
        this->ierr=VecDuplicate(*this->phi,&grad_q_dot_grad_phi);  CHKERRXX(this->ierr);  // memory increase local to the function 14
        this->ierr=VecDuplicate(*this->phi,&grad_grad_q_dot_grad_phi_dot_grad_phi); CHKERRXX(this->ierr);  // memory increase local to the function 15
        this->ierr=VecDuplicate(*this->phi,&grad_phi_square); CHKERRXX(this->ierr);  // memory increase local to the function 16

        this->compute_grad_f1_dot_grad_phi2(&q_forward_t,this->phi,&grad_q_dot_grad_phi);
        this->compute_grad_f1_dot_grad_phi2(&grad_q_dot_grad_phi,this->phi,&grad_grad_q_dot_grad_phi_dot_grad_phi);
        this->compute_grad_f1_dot_grad_phi2(this->phi,this->phi,&grad_phi_square);


        this->ierr=VecPointwiseMult(this->snn_global,q_backward_t,grad_grad_q_dot_grad_phi_dot_grad_phi); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(this->snn_global,this->snn_global,grad_phi_square); CHKERRXX(this->ierr);

           if(it==this->N_t-1)
        this->interpolate_and_print_vec_to_uniform_grid(&this->snn_global,"first_term");


        //---------------------------------End of first term-------------------------------------//





        //------------------Add Second term to first term--------------------------------//


       // this->ierr=VecSet(this->snn_global,0.00); CHKERRXX(this->ierr);

//        this->ierr=VecAXPY(this->snn_global,1.00,qdq_x_x); CHKERRXX(this->ierr);
//        this->ierr=VecAXPY(this->snn_global,1.00,qdq_y_y); CHKERRXX(this->ierr);

//#ifdef P4_TO_P8
//        this->ierr=VecAXPY(this->snn_global,1.00,qdq_z_z); CHKERRXX(this->ierr);
//#endif

        this->scatter_petsc_vector(&this->snn_global);

           ///////////
           //----------------------Print iteration filling-----------------------------//
           double scl=(1.00)*(this->Lx/this->Lx_physics)*(this->Lx/this->Lx_physics);
           this->ierr=VecScale(this->snn_global,scl); CHKERRXX(this->ierr);

           this->scatter_petsc_vector(&this->snn_global);
           //this->printDiffusionArrayFromVector(&this->snn_global,"snn_it");

           if(it==this->n_local_hist)
           this->interpolate_and_print_vec_to_uniform_grid(&this->snn_global,"snn_it");


        //   this->ierr=VecSet(this->snn_global,0.00); CHKERRXX(this->ierr);
          // this->ierr=VecPointwiseMult(this->snn_global,q_forward_t,q_backward_t); CHKERRXX(this->ierr);
          // this->scatter_petsc_vector(&this->snn_global);



        //--------------------------------Fill the input for output------------------------------//


        this->ierr=VecGetArray(this->snn_global,&this->snn_local); CHKERRXX(this->ierr);
        for(int ix=0;ix<this->n_local;ix++)
        {
            this->snn_hist[it*this->n_local+ix]=this->snn_local[ix];

        }
        this->ierr=VecRestoreArray(this->snn_global,&this->snn_local); CHKERRXX(this->ierr);
        //--------------------------  Input Filled For Output-----------------------------------//




        //----------------------Finished to print iteration filling---------------//

        this->ierr= VecDestroy(this->snn_global);  CHKERRXX(this->ierr); //memory decrease local to the function 0
        this->ierr= VecDestroy(q_backward_t);  CHKERRXX(this->ierr); //memory decrease local to the function 1
        this->ierr=VecDestroy(bc_vec_fake);   CHKERRXX(this->ierr); // memory decrease local to the function 2
        this->ierr=VecDestroy(q_forward_t); CHKERRXX(this->ierr);   // memory decrease local to the function 3

        this->ierr=VecDestroy(qdq_x_x); CHKERRXX(this->ierr);  // memory decrease local to the function 6
        this->ierr=VecDestroy(qdq_y_y); CHKERRXX(this->ierr);   // memory decrease local to the function 7

#ifdef P4_TO_P8
        this->ierr=VecDestroy(qdq_z_z); CHKERRXX(this->ierr);  // memory decrease local to the function 8
#endif

        this->ierr=VecDestroy(qdq_x); CHKERRXX(this->ierr);  // memory decrease local to the function 9
        this->ierr=VecDestroy(qdq_y); CHKERRXX(this->ierr);   // memory decrease local to the function 10

#ifdef P4_TO_P8
        this->ierr=VecDestroy(qdq_z); CHKERRXX(this->ierr);  // memory decrease local to the function 11
#endif

        this->ierr=VecDestroy(grad_q_dot_grad_phi);  CHKERRXX(this->ierr);  // memory decrease local to the function 14
        this->ierr=VecDestroy(grad_grad_q_dot_grad_phi_dot_grad_phi); CHKERRXX(this->ierr);  // memory decrease local to the function 15
        this->ierr=VecDestroy(grad_phi_square); CHKERRXX(this->ierr);  // memory decrease local to the function 16



    }

}



int StressTensor::compute_grad_f1_dot_grad_f2(Vec *f1,Vec *f2,Vec *df1_dot_df2)
{
    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {


        PetscScalar *f1_local,*f2_local;
        Vec f1_x,f1_y,f2_x,f2_y;

#ifdef P4_TO_P8
        Vec f1_z,f2_z;
#endif

        PetscScalar *f1_x_local;
        PetscScalar *f1_y_local;
        PetscScalar *f2_x_local;
        PetscScalar *f2_y_local;


#ifdef P4_TO_P8
        PetscScalar *f1_z_local, *f2_z_local;
#endif

        this->ierr=VecDuplicate(*f1,&f1_x); CHKERRXX(this->ierr); // +1
        this->ierr=VecDuplicate(*f1,&f1_y); CHKERRXX(this->ierr); //+2

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(*f1,&f1_z); CHKERRXX(this->ierr); //+3
#endif

        this->ierr=VecDuplicate(*f2,&f2_x); CHKERRXX(this->ierr); //+4
        this->ierr=VecDuplicate(*f2,&f2_y); CHKERRXX(this->ierr); //+5

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(*f2,&f2_z); CHKERRXX(this->ierr); //+6
#endif

        this->ierr=VecGetArray(*f1,&f1_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(*f2,&f2_local); CHKERRXX(this->ierr);


        this->ierr=VecGetArray(f1_x,&f1_x_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f1_y,&f1_y_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f2_x,&f2_x_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f2_y,&f2_y_local); CHKERRXX(this->ierr);


#ifdef P4_TO_P8
        this->ierr=VecGetArray(f1_z,&f1_z_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f2_z,&f2_z_local); CHKERRXX(this->ierr);
#endif


        for(int i=0;i<this->n_local;i++)
        {
            f1_x_local[i]=this->nodes_neighbours->neighbors[i].dx_central(f1_local);
            f1_y_local[i]=this->nodes_neighbours->neighbors[i].dy_central(f1_local);
#ifdef P4_TO_P8
            f1_z_local[i]=this->nodes_neighbours->neighbors[i].dz_central(f1_local);
#endif

            f2_x_local[i]=this->nodes_neighbours->neighbors[i].dx_central(f2_local);
            f2_y_local[i]=this->nodes_neighbours->neighbors[i].dy_central(f2_local);
#ifdef P4_TO_P8
            f2_z_local[i]=this->nodes_neighbours->neighbors[i].dz_central(f2_local);
#endif
        }

        this->ierr=VecRestoreArray(*f1,&f1_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*f2,&f2_local); CHKERRXX(this->ierr);


        this->ierr=VecRestoreArray(f1_x,&f1_x_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f1_y,&f1_y_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f2_x,&f2_x_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f2_y,&f2_y_local); CHKERRXX(this->ierr);


#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(f1_z,&f1_z_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f2_z,&f2_z_local); CHKERRXX(this->ierr);
#endif


        this->ierr=VecPointwiseMult(f1_x,f1_x,f2_x); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(f1_y,f1_y,f2_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecPointwiseMult(f1_z,f1_z,f2_z); CHKERRXX(this->ierr);
#endif

        this->ierr=VecSet(*df1_dot_df2,0.00); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(*df1_dot_df2,1.00,f1_x); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(*df1_dot_df2,1.00,f1_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecAXPY(*df1_dot_df2,1.00,f1_z); CHKERRXX(this->ierr);
#endif

        this->scatter_petsc_vector(df1_dot_df2); CHKERRXX(this->ierr);

        this->ierr=VecDestroy(f1_x); CHKERRXX(this->ierr);// -1
        this->ierr=VecDestroy(f1_y); CHKERRXX(this->ierr);// -2
        this->ierr=VecDestroy(f2_x); CHKERRXX(this->ierr);//-3
        this->ierr=VecDestroy(f2_y); CHKERRXX(this->ierr); //-4

#ifdef P4_TO_P8
        this->ierr=VecDestroy(f2_z); CHKERRXX(this->ierr); //-5
        this->ierr=VecDestroy(f2_z); CHKERRXX(this->ierr);//-6
#endif            
    }
}

int StressTensor::compute_grad_f1_dot_grad_phi2(Vec *f1,Vec *phi2,Vec *df1_dot_dphi2)
{
    int n_games=1;

    for(int i_game=0;i_game<n_games;i_game++)
    {


        double band2Compute=10.00*this->Lx/(pow(2,this->max_level+this->nx_trees));
        PetscScalar *f1_local,*f2_local;
        Vec f1_x,f1_y,f2_x,f2_y;

#ifdef P4_TO_P8
        Vec f1_z,f2_z;
#endif

        PetscScalar *f1_x_local;
        PetscScalar *f1_y_local;
        PetscScalar *f2_x_local;
        PetscScalar *f2_y_local;


#ifdef P4_TO_P8
        PetscScalar *f1_z_local, *f2_z_local;
#endif

        this->ierr=VecDuplicate(*f1,&f1_x); CHKERRXX(this->ierr); // +1
        this->ierr=VecDuplicate(*f1,&f1_y); CHKERRXX(this->ierr); //+2

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(*f1,&f1_z); CHKERRXX(this->ierr); //+3
#endif

        this->ierr=VecDuplicate(*phi2,&f2_x); CHKERRXX(this->ierr); //+4
        this->ierr=VecDuplicate(*phi2,&f2_y); CHKERRXX(this->ierr); //+5

#ifdef P4_TO_P8
        this->ierr=VecDuplicate(*phi2,&f2_z); CHKERRXX(this->ierr); //+6
#endif

        this->ierr=VecGetArray(*f1,&f1_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(*phi2,&f2_local); CHKERRXX(this->ierr);


        this->ierr=VecGetArray(f1_x,&f1_x_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f1_y,&f1_y_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f2_x,&f2_x_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f2_y,&f2_y_local); CHKERRXX(this->ierr);


#ifdef P4_TO_P8
        this->ierr=VecGetArray(f1_z,&f1_z_local); CHKERRXX(this->ierr);
        this->ierr=VecGetArray(f2_z,&f2_z_local); CHKERRXX(this->ierr);
#endif

        PetscScalar *is_crossed_neumann_local;
        this->ierr=VecGetArray(*this->is_crossed_neumann,&is_crossed_neumann_local); CHKERRXX(this->ierr);


        //InterpolatingFunctionNodeBase interp2(this->p4est, this->nodes,this->ghost, this->brick, this->nodes_neighbours);

        InterpolatingFunctionNodeBase interp1(this->p4est,this->nodes,this->ghost, this->brick, this->nodes_neighbours);
        InterpolatingFunctionNodeBase interp2(this->p4est, this->nodes,this->ghost, this->brick, this->nodes_neighbours);

        /* find dx and dy smallest */
        // NOTE: Assuming all trees are of the same size [0, 1]^d
       double mult_diag=4.00;
        double dx = 1.0 / pow(2.,(double) this->max_level+this->nx_trees);
        double dy = dx;
    #ifdef P4_TO_P8
        double dz = dx;
    #endif
        /* NOTE: I don't understand why the this->mult_diag coefficient is needed ... 1 should work, but it doesn't */
    #ifdef P4_TO_P8
        double diag = mult_diag*sqrt(dx*dx + dy*dy + dz*dz);
    #else
        double diag = mult_diag*sqrt(dx*dx + dy*dy);
    #endif


        std::vector<double> q1;
        std::vector<double> q2;


        q1.resize(this->n_local);
        q2.resize(this->n_local);


        // (1) Compute derivatives using black box for the level set function

        for(int i=0;i<this->n_local;i++)
        {
            f2_x_local[i]=this->nodes_neighbours->neighbors[i].dx_central(f2_local);
            f2_y_local[i]=this->nodes_neighbours->neighbors[i].dy_central(f2_local);
#ifdef P4_TO_P8
            f2_z_local[i]=this->nodes_neighbours->neighbors[i].dz_central(f2_local);
#endif
        }

      //(2) Compute second order central differences using black box for internal nodes

        for(int i=0;i<this->n_local;i++)
        {
            if(f2_local[i]<=0 && is_crossed_neumann_local[i]<0.5 && ABS(f2_local[i])>band2Compute)
            {
            f1_x_local[i]=this->nodes_neighbours->neighbors[i].dx_central(f1_local);
            f1_y_local[i]=this->nodes_neighbours->neighbors[i].dy_central(f1_local);
#ifdef P4_TO_P8
            f1_z_local[i]=this->nodes_neighbours->neighbors[i].dz_central(f1_local);
#endif

            }
        }

        //(3) Buffer required points for second order derivatives with first order accuracy.

        for(int i=0;i<this->n_local;i++)
        {
            if( is_crossed_neumann_local[i]>0.5)
            {
#ifdef P4_TO_P8
        Point3 grad_phi(-f2_x_local[i], -f2_y_local[i], -f2_z_local[i]);
#else
        Point2 grad_phi(-f2_x_local[i], -f2_y_local[i]);
#endif

        double local_set_value=f2_local[i];

            grad_phi /= grad_phi.norm_L2();
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
            p4est_topidx_t tree_id = node->p.piggy3.which_tree;

            p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

            double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
            double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
            double tree_zmin = p4est->connectivity->vertices[3*v_mm + 2];
#endif

            double xyz [] =
            {
                node_x_fr_i(node) + tree_xmin,
                node_y_fr_j(node) + tree_ymin
    #ifdef P4_TO_P8
                ,
                node_z_fr_k(node) + tree_zmin
    #endif
            };





                double xyz_ [] =
                {
                    xyz[0] + grad_phi.x * (diag),//+ f2_local[i]),
                    xyz[1] + grad_phi.y * (diag)//+ f2_local[i])
    #ifdef P4_TO_P8
                    ,
                    xyz[2] + grad_phi.z * (diag)//+ f2_local[i])
    #endif
                };
                interp1.add_point_to_buffer(i, xyz_);



                double xyz2_ [] =
                {
                    xyz[0] + grad_phi.x * (2.0*diag),//+ f2_local[i]),
                    xyz[1] + grad_phi.y * (2.0*diag)// + f2_local[i])
    #ifdef P4_TO_P8
                    ,
                    xyz[2] + grad_phi.z * (2.0*diag)//+ f2_local[i])
    #endif
                };
                interp2.add_point_to_buffer(i, xyz2_);
            }

            ierr = PetscLogFlops(26); CHKERRXX(ierr);
        }


        // (4) Interpolate at the required points
        interp1.set_input_parameters(*f1, quadratic);
        interp2.set_input_parameters(*f1, quadratic);

        interp1.interpolate(q1.data());
        interp2.interpolate(q2.data());





        this->ierr=VecRestoreArray(*phi2,&f2_local); CHKERRXX(this->ierr);


        this->ierr=VecRestoreArray(f1_x,&f1_x_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f1_y,&f1_y_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f2_x,&f2_x_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f2_y,&f2_y_local); CHKERRXX(this->ierr);


#ifdef P4_TO_P8
        this->ierr=VecRestoreArray(f1_z,&f1_z_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(f2_z,&f2_z_local); CHKERRXX(this->ierr);
#endif


        this->ierr=VecPointwiseMult(f1_x,f1_x,f2_x); CHKERRXX(this->ierr);
        this->ierr=VecPointwiseMult(f1_y,f1_y,f2_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecPointwiseMult(f1_z,f1_z,f2_z); CHKERRXX(this->ierr);
#endif

        this->ierr=VecSet(*df1_dot_dphi2,0.00); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(*df1_dot_dphi2,1.00,f1_x); CHKERRXX(this->ierr);
        this->ierr=VecAXPY(*df1_dot_dphi2,1.00,f1_y); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
        this->ierr=VecAXPY(*df1_dot_dphi2,1.00,f1_z); CHKERRXX(this->ierr);
#endif

        this->scatter_petsc_vector(df1_dot_dphi2); CHKERRXX(this->ierr);

        // (5) Compute Left sided second order derivatives with first order accuracy

        PetscScalar *df1_dot_dphi2_local;
        this->ierr=VecGetArray(*df1_dot_dphi2,&df1_dot_dphi2_local); CHKERRXX(this->ierr);

        double df1_dot_dphi2_local_i=0;
        double f1_local_i=0;
        double q1_i=0;
        double q2_i=0;

        for(int i=0;i<this->n_local;i++)
        {
            if(is_crossed_neumann_local[i]>0.5)
            {
                f1_local_i=f1_local[i];
                q1_i=q1[i];
                q2_i=q2[i];

                df1_dot_dphi2_local[i]=(3*f1_local[i]-4*q1[i]+q2[i])/(2*mult_diag*diag);
                df1_dot_dphi2_local_i=df1_dot_dphi2_local[i];
            }

        }
        this->ierr=VecRestoreArray(*df1_dot_dphi2,&df1_dot_dphi2_local); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(df1_dot_dphi2); CHKERRXX(this->ierr);


        this->ierr=VecRestoreArray(*f1,&f1_local); CHKERRXX(this->ierr);
        this->ierr=VecRestoreArray(*this->is_crossed_neumann,&is_crossed_neumann_local); CHKERRXX(this->ierr);


        this->ierr=VecDestroy(f1_x); CHKERRXX(this->ierr);// -1
        this->ierr=VecDestroy(f1_y); CHKERRXX(this->ierr);// -2
        this->ierr=VecDestroy(f2_x); CHKERRXX(this->ierr);//-3
        this->ierr=VecDestroy(f2_y); CHKERRXX(this->ierr); //-4

#ifdef P4_TO_P8
        this->ierr=VecDestroy(f2_z); CHKERRXX(this->ierr); //-5
        this->ierr=VecDestroy(f2_z); CHKERRXX(this->ierr);//-6
#endif
    }
}


// NOTE:: this algo is greedy in memory
// but it has many advantages:
// (1) faster than the efficient memory algo
// (2) ability to get the spatial distribution of the stresses
// (3) but to be memory efficient it would be wise in the short future to do it dimension by dimension:
// xx,yy,zz,zy,xz,yz;
// and even here no need to store the hitory forach one of them such that we could loop and sum on the 3 of them
// like doing simpsons after each history extraction

int StressTensor::compute_spatial_integrand_irregular()
{

    //------------Compute and fill the history of the spatial derivatives
    // for both q_forward and q_backward. The data structure layout is:
    // it=0:1................N_grid
    // it=1:1................N_grid
    // and so forth

    //compute temporal spatial derivatives on xx, yy, zz, xy, xz,yz.

    // qx_forward, qy_forward,qx_forward have the size of the grid
    // and are in fact temporary variables where to store the spatial variables for
    // one iteration in time of the diffusion solution
    this->ierr=VecDuplicate(*this->phi,&this->qxx_backward); CHKERRXX(this->ierr);  // memory increase global to the object 1
    this->ierr=VecDuplicate(*this->phi,&this->qyy_backward); CHKERRXX(this->ierr);  // memory increase global to the object 2

#ifdef P4_TO_P8

    this->ierr=VecDuplicate(*this->phi,&this->qzz_backward); CHKERRXX(this->ierr);  // memory increase global to the object 3
#endif

    this->ierr=VecDuplicate(*this->phi,&this->qxy_backward); CHKERRXX(this->ierr);  // memory increase global to the object 4

#ifdef P4_TO_P8
    this->ierr=VecDuplicate(*this->phi,&this->qxz_backward); CHKERRXX(this->ierr);  // memory increase global to the object 5
    this->ierr=VecDuplicate(*this->phi,&this->qyz_backward); CHKERRXX(this->ierr);  // memory increase global to the object 6
#endif

    this->ierr=VecDuplicate(*this->phi,&this->qx_backward); CHKERRXX(this->ierr);
    this->ierr= VecDuplicate(*this->phi,&this->qy_backward); CHKERRXX(this->ierr);


    // very expensive step in terms of memory
    // consider later to do it direction by direction to save memory
    // or on the fly direction by direction and to store only the two last
    // time steps for the integrand which will be integrated by Simpsons.

    this->qxx_backward_local=new double[this->n_local_hist];  // memory increase global to the object 7
    this->qyy_backward_local=new double[this->n_local_hist];  // memory increase global to the object 8

#ifdef P4_TO_P8
    this->qzz_backward_local=new double [this->n_local_hist]; // memory increase global to the object 9
#endif

    this->qxy_backward_local=new double[this->n_local_hist];  // memory increase global to the object 10

 #ifdef P4_TO_P8
    this->qxz_backward_local=new double[this->n_local_hist];  // memory increase global to the object 11
    this->qyz_backward_local=new double [this->n_local_hist]; // memory increase global to the object 12
#endif

    // iterate on time
    // Note: consider to not use vecs at all
    for(int it=0;it<this->N_t;it++)
    {
        this->extract_and_process_iteration_irregular(it);
    }


    this->ierr=VecDestroy(this->qxx_backward); CHKERRXX(this->ierr); // memory decrease global to the object 1
    this->ierr=VecDestroy(this->qyy_backward); CHKERRXX(this->ierr); // memory decrease global to the object 2

#ifdef P4_TO_P8
        this->ierr=VecDestroy(this->qzz_backward); CHKERRXX(this->ierr); // memory decrease global to the object 3
#endif

    this->ierr=VecDestroy(this->qxy_backward); CHKERRXX(this->ierr); // memory decrease global to the object 4

#ifdef P4_TO_P8
    this->ierr=VecDestroy(this->qxz_backward); CHKERRXX(this->ierr); // memory decrease global to the object 5
    this->ierr=VecDestroy(this->qyz_backward); CHKERRXX(this->ierr); // memory decrease global to the object 6
#endif

    this->ierr=VecDestroy(this->qx_backward); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(this->qy_backward); CHKERRXX(this->ierr);


    // At the end of this step:
    // qx_backward_local, qy_backward_local, qz_backward_local
    // qx_backward_local, qy_backward_local, qz_backward_local
    // are filled with their correct values

}

int StressTensor::compute_shape_derivative()
{

    std::cout<<" compute shape derivative "<<std::endl;
    // fill snn_hist
    this->fill_snn_hist();
    PetscScalar *phi_local;
    this->ierr=VecGetArray(*this->phi,&phi_local); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(*this->phi,&this->snn_global); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(this->snn_global,&this->snn_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(*this->phi_is_all_positive,&this->mask_local); CHKERRXX(this->ierr); // memory increase global to the object 22

    this->ds_forward=new double[this->N_t];


    double band2Compute=0.1;

    for(int ix=0;ix<this->n_local;ix++)
    {
        if(this->mask_local[ix]==0 && ((phi_local[ix]*phi_local[ix])<band2Compute*band2Compute) )
        {
            for(int it=0;it<this->N_t;it++)
            {
                this->ds_forward[it]=this->snn_hist[it*this->n_local+ix];
            }
            this->snn_local[ix]=this->compute_time_integral4Shape_derivative();
        }
        else
        {
            this->snn_local[ix]=0;
        }
    }
    this->ierr=VecRestoreArray(this->snn_global,&this->snn_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(*this->phi,&phi_local); CHKERRXX(this->ierr);

    this->scatter_petsc_vector(&this->snn_global);
    this->printDiffusionArrayFromVector(&this->snn_global,"snn_global");
    this->interpolate_and_print_vec_to_uniform_grid(&this->snn_global,"snn_stress_tensor");
    this->ierr=VecRestoreArray(*this->phi_is_all_positive,&this->mask_local); CHKERRXX(this->ierr); // memory increase global to the object 22

    delete this->snn_hist;
    delete this->ds_forward;

    if(this->test)
        this->compute_test_integral();

    std::cout<<" finished to compute shape derivative "<<std::endl;
}

int StressTensor::compute_test_integral()
{
    Vec bc_vec_fake;
    int order_to_extend=2;
    int number_of_bands_to_extend=5;

    this->ierr=VecDuplicate(*this->phi,&bc_vec_fake);   CHKERRXX(this->ierr); // memory increase local to the function 2

    this->ierr=VecSet(bc_vec_fake,0.00); CHKERRXX(this->ierr);
    this->scatter_petsc_vector(&bc_vec_fake);
    this->LS->extend_Over_Interface(*this->phi,this->snn_global,NEUMANN,bc_vec_fake,order_to_extend,number_of_bands_to_extend);
    // scatter forward the parallel petsc vector
    this->scatter_petsc_vector(&this->snn_global);
    this->printDiffusionArrayFromVector(&this->snn_global,"snn_global_test");
    this->DH_test=this->integrate_over_interface(*this->phi,this->snn_global);

    std::cout<<" DH_test "<<this->DH_test<<std::endl;

    double benchmark_value=8*PI*pow(this->Lx/2,4);

    std::cout<<" benchmark value "<<benchmark_value<<std::endl;

    this->ierr=VecDestroy(bc_vec_fake);

}

int StressTensor::Compute_stress()
{

    switch(this->my_computation_mode)
    {
      case StressTensor::qxqcx:
        this->compute_stress();
        break;
    case StressTensor::qqcxx:
        this->compute_stress();
        break;
    case StressTensor::shape_derivative:
        this->compute_shape_derivative();
        break;
    case StressTensor::qxqcx_memory_optimized:
        this->compute_stress_memory_efficient();
        break;

    }
}

int StressTensor::compute_stress()
{
    std::cout<<" compute stresses "<<std::endl;

    switch(this->my_computation_mode)
    {
    case StressTensor::qxqcx:
    {
        this->compute_spatial_integrand();
        break;

    }
    case StressTensor::qqcxx:
    {
        this->compute_spatial_integrand_irregular();
        break;
    }
    }


    this->ierr=VecDuplicate(*this->phi,&this->sxx_global); CHKERRXX(this->ierr); // memory increase global to the object 10

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecDuplicate(*this->phi,&this->syy_global); CHKERRXX(this->ierr); // memory increase global to the object 11
        this->ierr=VecDuplicate(*this->phi,&this->szz_global); CHKERRXX(this->ierr); // memory increase global to the object 12
        this->ierr=VecDuplicate(*this->phi,&this->sxy_global); CHKERRXX(this->ierr); // memory increase global to the object 13
        this->ierr=VecDuplicate(*this->phi,&this->sxz_global); CHKERRXX(this->ierr); // memory increase global to the object 14
        this->ierr=VecDuplicate(*this->phi,&this->syz_global); CHKERRXX(this->ierr); // memory increase global to the object 15
    }
    this->ierr=VecGetArray(this->sxx_global,&this->sxx_local); CHKERRXX(this->ierr); // memory increase global to the object 16
    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecGetArray(this->syy_global,&this->syy_local); CHKERRXX(this->ierr); // memory increase global to the object 17
        this->ierr=VecGetArray(this->szz_global,&this->szz_local); CHKERRXX(this->ierr); // memory increase global to the object 18
        this->ierr=VecGetArray(this->sxy_global,&this->sxy_local); CHKERRXX(this->ierr); // memory increase global to the object 19
        this->ierr=VecGetArray(this->sxz_global,&this->sxz_local); CHKERRXX(this->ierr); // memory increase global to the object 20
        this->ierr=VecGetArray(this->syz_global,&this->syz_local); CHKERRXX(this->ierr); // memory increase global to the object 21
    }

    // time variables point by point direction by direction

    this->ds_forward=new double[this->N_t];    // memory increase local to the function 8
    this->ds_backward=new double[this->N_t];   // memory increase local to the function 9



    this->ierr=VecGetArray(*this->phi_is_all_positive,&this->mask_local); CHKERRXX(this->ierr); // memory increase global to the object 22

    switch(this->my_computation_mode)
    {
    case StressTensor::qxqcx:
    {

        // sxx
        this->printDiffusionArray(this->qx_forward_local,this->n_local_hist,"qx_forward_local");
        this->printDiffusionArray(this->qx_backward_local,this->n_local_hist,"qx_backward_local");

        this->compute_micro_stresses(this->qx_forward_local,this->qx_backward_local,this->sxx_local);
        this->printDiffusionArray(this->sxx_local,this->n_local,"sxx_local");

        if(!this->computeOneComponentOnly)
        {

            // syy
            this->compute_micro_stresses(this->qy_forward_local,this->qy_backward_local,this->syy_local);
            this->printDiffusionArray(this->syy_local,this->n_local,"syy_local");
            //szz
            this->compute_micro_stresses(this->qz_forward_local,this->qz_backward_local,this->szz_local);
            this->printDiffusionArray(this->szz_local,this->n_local,"szz_local");
            //sxy
            this->compute_micro_stresses(this->qx_forward_local,this->qy_backward_local,this->sxy_local);
            this->printDiffusionArray(this->sxy_local,this->n_local,"sxy_local");

            // sxz
            this->compute_micro_stresses(this->qx_forward_local,this->qz_backward_local,this->sxz_local);
            this->printDiffusionArray(this->sxz_local,this->n_local,"sxz_local");

            //syz
            this->compute_micro_stresses(this->qy_forward_local,this->qz_backward_local,this->syz_local);
            this->printDiffusionArray(this->syz_local,this->n_local,"syz_local");
        }
        break;
    }
    case StressTensor::qqcxx:
    {
        // sxx
        this->printDiffusionArray(this->q_forward_hist,this->n_local_hist,"qx_forward_local");
        this->printDiffusionArray(this->qxx_backward_local,this->n_local_hist,"qx_backward_local");
        this->compute_micro_stresses(this->q_forward_hist,this->qxx_backward_local,this->sxx_local);

        if(!this->computeOneComponentOnly)
        {
            this->printDiffusionArray(this->sxx_local,this->n_local,"sxx_local");
            // syy
            this->compute_micro_stresses(this->q_forward_hist,this->qyy_backward_local,this->syy_local);
            this->printDiffusionArray(this->syy_local,this->n_local,"syy_local");
            //szz
            this->compute_micro_stresses(this->q_forward_hist,this->qzz_backward_local,this->szz_local);
            this->printDiffusionArray(this->szz_local,this->n_local,"szz_local");
            //sxy
            this->compute_micro_stresses(this->q_forward_hist,this->qxy_backward_local,this->sxy_local);
            this->printDiffusionArray(this->sxy_local,this->n_local,"sxy_local");

            // sxz
            this->compute_micro_stresses(this->q_forward_hist,this->qxz_backward_local,this->sxz_local);
            this->printDiffusionArray(this->sxz_local,this->n_local,"sxz_local");

            //syz
            this->compute_micro_stresses(this->q_forward_hist,this->qyz_backward_local,this->syz_local);
            this->printDiffusionArray(this->syz_local,this->n_local,"syz_local");
        }

        break;
    }

    }
    this->ierr=VecRestoreArray(*this->phi_is_all_positive,&this->mask_local); // memory decrease global to the object 22


    // set the values to the petsc vectors

    this->ierr=VecRestoreArray(this->sxx_global,&this->sxx_local); CHKERRXX(this->ierr); // memory decrease global to the object 16

    if(!this->computeOneComponentOnly)
    {
        this->ierr=VecRestoreArray(this->syy_global,&this->syy_local); CHKERRXX(this->ierr); // memory decrease global to the object 17
        this->ierr=VecRestoreArray(this->szz_global,&this->szz_local); CHKERRXX(this->ierr); // memory decrease global to the object 18
        this->ierr=VecRestoreArray(this->sxy_global,&this->sxy_local); CHKERRXX(this->ierr); // memory decrease global to the object 19
        this->ierr=VecRestoreArray(this->sxz_global,&this->sxz_local); CHKERRXX(this->ierr); // memory decrease global to the object 20
        this->ierr=VecRestoreArray(this->syz_global,&this->syz_local); CHKERRXX(this->ierr); // memory decrease global to the object 21
    }

    // computes now the macrostresses by integrating the different stresses over the domain


    // first extend the stresses over the interface when we are not in the periodic case

    if(!this->periodic_xyz)
    {

        Vec bc_vec_fake;
        this->ierr=VecDuplicate(*this->phi,&bc_vec_fake); CHKERRXX(this->ierr); // memory increase local to the function 1
        this->ierr=VecSet(bc_vec_fake,0); CHKERRXX(this->ierr);


        this->ierr=VecGhostUpdateBegin(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(bc_vec_fake,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        int order2Extend=0;
        int number_of_bands_to_extend=5;
        Vec sxx_global_extended,syy_global_extended,szz_global_extended;
        Vec sxy_global_extended,sxz_global_extended,syz_global_extended;

        this->ierr=VecDuplicate(this->sxx_global,&sxx_global_extended); CHKERRXX(this->ierr); // memory increase local to the function 2
        this->ierr=VecDuplicate(this->syy_global,&syy_global_extended); CHKERRXX(this->ierr); // memory increase local to the function 3
        this->ierr=VecDuplicate(this->szz_global,&szz_global_extended); CHKERRXX(this->ierr); // memory increase local to the function 4
        this->ierr=VecDuplicate(this->sxy_global,&sxy_global_extended); CHKERRXX(this->ierr); // memory increase local to the function 5
        this->ierr=VecDuplicate(this->sxz_global,&sxz_global_extended); CHKERRXX(this->ierr); // memory increase local to the function 6
        this->ierr=VecDuplicate(this->syz_global,&syz_global_extended); CHKERRXX(this->ierr); // memory increase local to the function 7

        this->ierr=VecCopy(this->sxx_global,sxx_global_extended); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->syy_global,syy_global_extended); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->szz_global,szz_global_extended); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->sxy_global,sxy_global_extended); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->sxz_global,sxz_global_extended); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->syz_global,syz_global_extended); CHKERRXX(this->ierr);



        this->ierr=VecGhostUpdateBegin(sxx_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(sxx_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(syy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(syy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(szz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(szz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(sxy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(sxy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(sxz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(sxz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        this->ierr=VecGhostUpdateBegin(syz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(syz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        this->LS->extend_Over_Interface(*this->phi,sxx_global_extended,NEUMANN,bc_vec_fake,order2Extend,number_of_bands_to_extend);
        this->LS->extend_Over_Interface(*this->phi,syy_global_extended,NEUMANN,bc_vec_fake,order2Extend,number_of_bands_to_extend);
        this->LS->extend_Over_Interface(*this->phi,szz_global_extended,NEUMANN,bc_vec_fake,order2Extend,number_of_bands_to_extend);
        this->LS->extend_Over_Interface(*this->phi,sxy_global_extended,NEUMANN,bc_vec_fake,order2Extend,number_of_bands_to_extend);
        this->LS->extend_Over_Interface(*this->phi,sxz_global_extended,NEUMANN,bc_vec_fake,order2Extend,number_of_bands_to_extend);
        this->LS->extend_Over_Interface(*this->phi,syz_global_extended,NEUMANN,bc_vec_fake,order2Extend,number_of_bands_to_extend);


        this->ierr=VecGhostUpdateBegin(sxx_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(sxx_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(syy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(syy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(szz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(szz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(sxy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(sxy_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        this->ierr=VecGhostUpdateBegin(sxz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(sxz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


        this->ierr=VecGhostUpdateBegin(syz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(syz_global_extended,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);



        this->Sxx=integrate_over_negative_domain(this->p4est,this->nodes,*this->phi,sxx_global_extended);
        this->Syy=integrate_over_negative_domain(this->p4est,this->nodes,*this->phi,syy_global_extended);
        this->Szz=integrate_over_negative_domain(this->p4est,this->nodes,*this->phi,szz_global_extended);
        this->Sxy=integrate_over_negative_domain(this->p4est,this->nodes,*this->phi,sxy_global_extended);
        this->Sxz=integrate_over_negative_domain(this->p4est,this->nodes,*this->phi,sxz_global_extended);
        this->Syz=integrate_over_negative_domain(this->p4est,this->nodes,*this->phi,syz_global_extended);
        // detructs and destroys all the objects no longer needed
        this->ierr=VecDestroy(sxx_global_extended);  CHKERRXX(this->ierr); // memory decrease local to the function 2
        this->ierr=VecDestroy(syy_global_extended);  CHKERRXX(this->ierr); // memory decrease local to the function 3
        this->ierr=VecDestroy(szz_global_extended);  CHKERRXX(this->ierr); // memory decrease local to the function 4
        this->ierr=VecDestroy(sxy_global_extended);  CHKERRXX(this->ierr); // memory decrease local to the function 5
        this->ierr=VecDestroy(sxz_global_extended);  CHKERRXX(this->ierr); // memory decrease local to the function 6
        this->ierr=VecDestroy(syz_global_extended);  CHKERRXX(this->ierr); // memory decrease local to the function 7
        this->ierr=VecDestroy(bc_vec_fake);  CHKERRXX(this->ierr);         // memory decrease local to the function 1

    }
    else
    {
        // we are on the periodic case so there is no need for extension

        this->ierr=VecGhostUpdateBegin(this->sxx_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(this->sxx_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

        if(!this->computeOneComponentOnly)
        {
            this->ierr=VecGhostUpdateBegin(this->syy_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->syy_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

            this->ierr=VecGhostUpdateBegin(this->szz_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->szz_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

            this->ierr=VecGhostUpdateBegin(this->sxy_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->sxy_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

            this->ierr=VecGhostUpdateBegin(this->sxz_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->sxz_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


            this->ierr=VecGhostUpdateBegin(this->syz_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
            this->ierr=VecGhostUpdateEnd(this->syz_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        }

        Vec temp_phi;
        this->ierr=VecDuplicate(*this->phi,&temp_phi); CHKERRXX(this->ierr);
        this->ierr=VecSet(temp_phi,-1.00); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateBegin(temp_phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr=VecGhostUpdateEnd(temp_phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);


            this->Sxx=integrate_over_negative_domain(this->p4est,this->nodes,temp_phi,this->sxx_global);
        if(!this->computeOneComponentOnly)
        {
            this->Syy=integrate_over_negative_domain(this->p4est,this->nodes,temp_phi,this->syy_global);
            this->Szz=integrate_over_negative_domain(this->p4est,this->nodes,temp_phi,this->szz_global);
            this->Sxy=integrate_over_negative_domain(this->p4est,this->nodes,temp_phi,this->sxy_global);
            this->Sxz=integrate_over_negative_domain(this->p4est,this->nodes,temp_phi,this->sxz_global);
            this->Syz=integrate_over_negative_domain(this->p4est,this->nodes,temp_phi,this->syz_global);
        }

        this->ierr=VecDestroy(temp_phi); CHKERRXX(this->ierr);
    }

    this->Sxx=this->Sxx/this->V;
    if(!this->computeOneComponentOnly)
    {
        this->Syy=this->Syy/this->V;
        this->Szz=this->Szz/this->V;

        this->Sxy=this->Sxy/this->V;
        this->Sxz=this->Sxz/this->V;
        this->Syz=this->Syz/this->V;
    }
    else
    {
        this->Syy=0;
        this->Szz=0;
        this->Sxy=0;
        this->Sxz=0;
        this->Syz=0;
    }


    std::cout<<" Stress Tensor "<<std::endl;

    std::cout<<this->Sxx<<" "<<this->Sxy<<" "<<this->Sxz<<std::endl;
    std::cout<<this->Sxy<<" "<<this->Syy<<" "<<this->Syz<<std::endl;
    std::cout<<this->Sxz<<" "<<this->Syz<<" "<<this->Szz<<std::endl;

    //------rescale the stress tensor--------------------------------//

    double rescaler=this->Lx/this->Lx_physics;
    rescaler=rescaler*rescaler;

    this->Sxx=this->Sxx*rescaler;
    this->Syy=this->Syy*rescaler;
    this->Szz=this->Szz*rescaler;

    this->Sxy=this->Sxy*rescaler;
    this->Sxz=this->Sxz*rescaler;
    this->Syz=this->Syz*rescaler;


    std::cout<<" Stress Tensor Rescaled"<<std::endl;

    std::cout<<this->Sxx<<" "<<this->Sxy<<" "<<this->Sxz<<std::endl;
    std::cout<<this->Sxy<<" "<<this->Syy<<" "<<this->Syz<<std::endl;
    std::cout<<this->Sxz<<" "<<this->Syz<<" "<<this->Szz<<std::endl;


    // detructs and destroys all the objects no longer needed
    delete this->ds_forward;   // memory decrease local to the function 8
    delete this->ds_backward;  // memory decrease local to the function 9

    if(this->my_computation_mode==StressTensor::qqcxx)
        this->cleanStressTensor();

}

int StressTensor::compute_stress_memory_efficient()
{
    std::cout<<" compute stresses memory efficient"<<std::endl;
    // Nt must be odd and N_iterations must be even
    // We do have four weights for the (1/3) composite simpsons rule
   double w_0,w_N_t,w_odd,w_even;
   w_0=1.00;
   w_N_t=1.00;
   w_odd=4.00;
   w_even=2.00;
   w_0=w_0*this->dt/3.00;
   w_N_t=w_N_t*this->dt/3.00;
   w_odd=w_odd*this->dt/3.00;
   w_even=w_even*this->dt/3.00;
   double w_t=0;
   this->ierr=VecDuplicate(*this->phi,&this->qx_forward); CHKERRXX(this->ierr);  //memory increase local to the function 1
   this->ierr=VecDuplicate(*this->phi,&this->sxx_global); CHKERRXX(this->ierr); // memory increase global to the object
   this->ierr=VecGetArray(this->sxx_global,&this->sxx_local); CHKERRXX(this->ierr); // memory increase local to the function 3
   double *q_x_forward_local,*q_x_backward_local;
   q_x_forward_local=new double[this->n_local];  // memory increase local to the function 4
   q_x_backward_local=new double[this->n_local]; // memory increase local to the function 5
   PetscBool forward_solution=PETSC_TRUE;
    for(int it=0;it<this->N_t;it++)
    {
        if(it==0)
            w_t=w_0;
        if(it>0 && (it< (this->N_t-1)) && it%2==1)
            w_t=w_odd;
         if(it>0 && (it< (this->N_t-1)) && it%2==0)
             w_t=w_even;
         if(it==this->N_t-1)
             w_t=w_N_t;
         forward_solution=PETSC_TRUE;
         this->extract_and_process_iteration_forward_backward(it,this->q_forward_hist,q_x_forward_local,forward_solution);
         forward_solution=PETSC_FALSE;
         this->extract_and_process_iteration_forward_backward(it,this->q_backward_hist,q_x_backward_local,forward_solution);

         for(int i=0;i<this->nodes->num_owned_indeps;i++)
         {
             this->sxx_local[i]+=w_t*q_x_forward_local[i]*q_x_backward_local[i];
         }


    }
    this->ierr=VecDestroy(this->qx_forward); CHKERRXX(this->ierr); //memory decrease local to the function 1
    delete q_x_forward_local; //memory decrease local to the function 4
    delete q_x_backward_local;//memory decrease local to the function 5

    this->printDiffusionArray(this->sxx_local,this->n_local,"sxx_local");
    // set the values to the petsc vectors
    this->ierr=VecRestoreArray(this->sxx_global,&this->sxx_local); CHKERRXX(this->ierr); // memory decrease local to the function 3

    // computes now the macrostresses by integrating the different stresses over the domain
    // we are on the periodic case so there is no need for extension

    this->ierr=VecGhostUpdateBegin(this->sxx_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(this->sxx_global,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    Vec temp_phi;
    this->ierr=VecDuplicate(*this->phi,&temp_phi); CHKERRXX(this->ierr); //memory increase local to the function 6
    this->ierr=VecSet(temp_phi,-1.00); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateBegin(temp_phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(temp_phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->Sxx=integrate_over_negative_domain(this->p4est,this->nodes,temp_phi,this->sxx_global);
    this->ierr=VecDestroy(temp_phi); CHKERRXX(this->ierr); //memory decrease local to the function 6
    this->Sxx=this->Sxx/this->V;
    if(!this->computeOneComponentOnly)
    {
        this->Syy=this->Syy/this->V;
        this->Szz=this->Szz/this->V;

        this->Sxy=this->Sxy/this->V;
        this->Sxz=this->Sxz/this->V;
        this->Syz=this->Syz/this->V;
    }
    else
    {
        this->Syy=0;
        this->Szz=0;
        this->Sxy=0;
        this->Sxz=0;
        this->Syz=0;
    }

    this->Sxx=this->Sxx/this->Q;
    if(!this->computeOneComponentOnly)
    {
        this->Syy=this->Syy/this->Q;
        this->Szz=this->Szz/this->Q;
        this->Sxy=this->Sxy/this->Q;
        this->Sxz=this->Sxz/this->Q;
        this->Syz=this->Syz/this->Q;
    }
    else
    {
        this->Syy=0;
        this->Szz=0;
        this->Sxy=0;
        this->Sxz=0;
        this->Syz=0;
    }
    std::cout<<" Stress Tensor "<<std::endl;

    std::cout<<this->Sxx<<" "<<this->Sxy<<" "<<this->Sxz<<std::endl;
    std::cout<<this->Sxy<<" "<<this->Syy<<" "<<this->Syz<<std::endl;
    std::cout<<this->Sxz<<" "<<this->Syz<<" "<<this->Szz<<std::endl;

    //------rescale the stress tensor--------------------------------//
    double rescaler=this->Lx/this->Lx_physics;
    rescaler=rescaler*rescaler;
    this->Sxx=this->Sxx*rescaler;
    this->Syy=this->Syy*rescaler;
    this->Szz=this->Szz*rescaler;
    this->Sxy=this->Sxy*rescaler;
    this->Sxz=this->Sxz*rescaler;
    this->Syz=this->Syz*rescaler;
    std::cout<<" Stress Tensor Rescaled"<<std::endl;
    std::cout<<this->Sxx<<" "<<this->Sxy<<" "<<this->Sxz<<std::endl;
    std::cout<<this->Sxy<<" "<<this->Syy<<" "<<this->Syz<<std::endl;
    std::cout<<this->Sxz<<" "<<this->Syz<<" "<<this->Szz<<std::endl;
    // detructs and destroys all the objects no longer needed


}

int StressTensor::compute_micro_stresses(double *q_f, double *q_b, double *s_ii)
{

    // no objects are constructed or destructed in this local function

    for(int ix=0;ix<this->n_local;ix++)
    {
        if(this->mask_local[ix]==0)
        {
            for(int it=0;it<this->N_t;it++)
            {
                //std::cout<<it*this->n_local+ix<<std::endl;
                this->ds_forward[it]=q_f[it*this->n_local+ix];
                //std::cout<<ix<<" "<<it<<" "<<q_f[it*this->n_local+ix]<<std::endl;
                this->ds_backward[it]=q_b[it*this->n_local+ix];
                //std::cout<<ix<<" "<<it<<" "<<q_b[it*this->n_local+ix]<<std::endl;
            }
            s_ii[ix]=this->compute_time_integral4Stresses()/this->Q;
        }
        else
        {
            s_ii[ix]=0;
        }
    }
}


double StressTensor::compute_time_integral4Stresses()
{
    double *realIntegrand=new double[this->N_t]; // memory increase local to the function 1
    for(int it=0;it<this->N_t;it++)
    {
        realIntegrand[it]= this->ds_forward[it]*this->ds_backward[this->N_t-it-1];
    }
    this->printDiffusionArray(this->ds_forward,this->N_t,"ds_forward");
    this->printDiffusionArray(this->ds_backward,this->N_t,"ds_backward");
    this->printDiffusionArray(realIntegrand,this->N_t,"real_integrand");
    double I=0;
    double I1=0;
    double I2=0;
    double I3=0;
    double I4=0;
    double I5=0;

    int n=(int)(1*this->N_iterations);
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
    delete realIntegrand; // memory decrease local to the function 1
    return I;
}


double StressTensor::compute_time_integral4Shape_derivative()
{
    double *realIntegrand=new double[this->N_t]; // memory increase local to the function 1
    for(int it=0;it<this->N_t;it++)
    {
        realIntegrand[it]= this->ds_forward[it];//*this->ds_backward[this->N_t-it-1];
    }
    this->printDiffusionArray(this->ds_forward,this->N_t,"ds_forward");
    // this->printDiffusionArray(this->ds_backward,this->N_t,"ds_backward");
    this->printDiffusionArray(realIntegrand,this->N_t,"real_integrand");
    double I=0;
    double I1=0;
    double I2=0;
    double I3=0;
    double I4=0;
    double I5=0;

    int n=(int)(1*this->N_iterations);
    int NL=n+1;
     double h=this->dt;
    if(n%2==0)
    {

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
    delete realIntegrand; // memory decrease local to the function 1
    return I;
}


int StressTensor::scatter_petsc_vector(Vec *v2scatter)
{
   // std::cout<<this->mpi->mpirank<<" start to scatter "<<std::endl;
    this->ierr=VecGhostUpdateBegin(*v2scatter,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(*v2scatter,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
  //  std::cout<<this->mpi->mpirank<<" finish to scatter "<<std::endl;

}


int StressTensor::interpolate_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid,std::string str_file)
{

    if(!this->minimum_IO)
{
        std::cout<<" this->mpi->mpisize "<<this->mpi->mpisize<<std::endl;
    Vec v_uniform;
    std::cout<<this->mpi->mpirank<<" start to interpolate wt on an uniform grid "<<std::endl;
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



    int i_local_start=this->mpi->mpirank*N0/this->mpi->mpisize;
    // Create an interpolating function
    InterpolatingFunctionNodeBase w_func(this->p4est, this->nodes, this->ghost, this->brick, this->nodes_neighbours);
    int n_p=0;
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
                    (double)(i+i_local_start)*this->Lx/(double)N0,
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

    w_func.set_input_parameters(*vec2PrintOnUniformGrid, linear);
    w_func.interpolate(v_uniform);


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
    std::cout<<this->mpi->mpirank<<" finished to interpolate v on an uniform grid "<<std::endl;

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
                    (double)(i+i_local_start)*this->Lx/(double)N0,
                    (double)j*this->Lx/(double) N0
    #ifdef P4_TO_P8
                    ,(double) k*this->Lx/N0
    #endif
                };


#ifdef P4_TO_P8
                fprintf(outFile,"%d %d %d %d %f %f %f %f \n",mpi_rank, i,j,k,xyz[0],xyz[1],xyz[2],v_local[k+j*N0+i*N0*N0/this->mpi->mpisize]); n_p++;
            }
#else
                fprintf(outFile,"%d %d %d %f %f %f\n",mpi_rank, i,j,xyz[0],xyz[1],v_local[j+i*N0/this->mpi->mpisize]);n_p++;
#endif
            }

        }

        fclose(outFile);
        this->ierr=VecRestoreArray(v_uniform,&v_local); CHKERRXX(this->ierr);
        this->ierr=VecDestroy(v_uniform); CHKERRXX(this->ierr);
    }
    }




int StressTensor::printDiffusionVector(Vec *V2Print, std::string file_name_str)
{

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
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


        PetscViewer lab;
        this->ierr=PetscViewerASCIIOpen(MPI_COMM_WORLD,mystr2DebugVec.c_str(),&lab); CHKERRXX(this->ierr);
        // this->ierr=PetscViewerSetFormat(lab,PETSC_VIEWER_ASCII_INDEX); CHKERRXX(this->ierr);

        this->ierr=VecView(*V2Print,lab); CHKERRXX(this->ierr);
        this->ierr=PetscViewerDestroy(lab);
    }


}

int StressTensor::printDiffusionArray(double *x2Print, int Nx, string file_name_str)
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
    }
    //VecRestoreArray(phiTemp,&phiTempArray);
    //VecDestroy(phiTemp);
    return 0;


}

int StressTensor::printDiffusionArrayFromVector(Vec *v2Print, string file_name_str)
{
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
        oss2DebugVec << this->convert2FullPath(file_name_str)<<"_"<<mpi_rank<<".txt";
        mystr2DebugVec=oss2DebugVec.str();
        FILE *outFileVec;
        outFileVec=fopen(mystr2DebugVec.c_str(),"w");
        for(int i=0;i<Nx;i++)
        {
            PetscScalar temp_x=x2Print[i];
            fprintf(outFileVec,"%d %f \n",i,temp_x);//,phiTempArray[i]);
        }
        fclose(outFileVec);
        VecRestoreArray(*v2Print,&x2Print);
    }


    return 0;
}

double StressTensor::integrate_over_interface_in_one_quadrant(  p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
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

double StressTensor::integrate_over_interface( Vec phi, Vec f)
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

