#include "poisson2d.h"



//poisson2d::poisson2d(int argc, char *argv[])
//{
//    mpi_context_t mpi_context, *mpi = &mpi_context;
//    mpi->mpicomm  = MPI_COMM_WORLD;
//    try{
//      p4est_t            *p4est;
//      p4est_nodes_t      *nodes;
//      PetscErrorCode      ierr;

//      cmdParser cmd;
//      cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
//      cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
//      cmd.add_option("lmin", "the min level of the tree");
//      cmd.add_option("lmax", "the max level of the tree");
//      cmd.add_option("nb_splits", "number of splits to apply to the min and max level");
//      cmd.parse(argc, argv);

//      // decide on the type and value of the boundary conditions
//      BoundaryConditionType bc_wall_type, bc_interface_type;
//      int nb_splits, min_level, max_level;
//      bc_wall_type      = cmd.get("bc_wtype"  , DIRICHLET);
//      bc_interface_type = cmd.get("bc_itype"  , DIRICHLET);
//      nb_splits         = cmd.get("nb_splits" , 0);
//      min_level         = cmd.get("lmin"      ,8);
//      max_level         = cmd.get("lmax"      ,8);

//  #ifdef P4_TO_P8
//      CF_3 *bc_wall_value, *bc_interface_value;
//      WallBC3D *wall_bc;
//  #else
//      CF_2 *bc_wall_value, *bc_interface_value;
//      WallBC2D *wall_bc;
//  #endif

//      switch(bc_interface_type){
//      case DIRICHLET:
//        bc_interface_value = &bc_interface_dirichlet_value;
//        break;
//      case NEUMANN:
//        bc_interface_value = &bc_interface_neumann_value;
//        break;
//      default:
//        throw std::invalid_argument("[ERROR]: Interface bc type can only be 'Dirichlet' or 'Neumann' type");
//      }

//      switch(bc_wall_type){
//      case DIRICHLET:
//        bc_wall_value = &bc_wall_dirichlet_value;
//        wall_bc       = &bc_wall_dirichlet_type;
//        break;
//      case NEUMANN:
//        bc_wall_value = &bc_wall_neumann_value;
//        wall_bc       = &bc_wall_neumann_type;
//        break;
//      default:
//        throw std::invalid_argument("[ERROR]: Wall bc type can only be 'Dirichlet' or 'Neumann' type");
//      }

//  #ifdef P4_TO_P8
//      circle.update(1, 1, 1, .3);
//  #else
//      circle.update(1, 1, .3);
//  #endif
//      splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circle, 1);

//      Session mpi_session;
//      mpi_session.init(argc, argv, mpi->mpicomm);

//      parStopWatch w1, w2;
//      w1.start("total time");

//      MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
//      MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

//      w2.start("initializing the grid");

//      /* create the macro mesh */
//      p4est_connectivity_t *connectivity;
//      my_p4est_brick_t brick;
//  #ifdef P4_TO_P8
//      connectivity = my_p4est_brick_new(2, 2, 2, &brick);
//  #else
//      connectivity = my_p4est_brick_new(2, 2, &brick);
//  #endif

//      /* create the p4est */
//      p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
//      p4est->user_pointer = (void*)(&data);
//      p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

//      /* partition the p4est */
//      p4est_partition(p4est, NULL);

//      /* create the ghost layer */
//      p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

//      /* generate unique node indices */
//      nodes = my_p4est_nodes_new(p4est, ghost);
//      w2.stop(); w2.read_duration();

//      /* initialize the vectors */
//      Vec phi, rhs, uex, sol;
//      ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi, &uex); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

//      sample_cf_on_nodes(p4est, nodes, circle, phi);
//      sample_cf_on_nodes(p4est, nodes, u_ex, uex);
//      sample_cf_on_nodes(p4est, nodes, f_ex, rhs);

//      /* create the hierarchy structure */
//      w2.start("construct the hierachy information");
//      my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
//      w2.stop(); w2.read_duration();

//      /* generate the neighborhood information */
//      w2.start("construct the neighborhood information");
//      my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
//      w2.stop(); w2.read_duration();

//      /* initalize the bc information */
//      Vec interface_value_Vec, wall_value_Vec;
//      ierr = VecDuplicate(phi, &interface_value_Vec); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi, &wall_value_Vec); CHKERRXX(ierr);

//      sample_cf_on_nodes(p4est, nodes, *bc_interface_value, interface_value_Vec);
//      sample_cf_on_nodes(p4est, nodes, *bc_wall_value, wall_value_Vec);

//      InterpolatingFunctionNodeBase interface_interp(p4est, nodes, ghost, &brick, &node_neighbors), wall_interp(p4est, nodes, ghost, &brick, &node_neighbors);
//      interface_interp.set_input_parameters(interface_value_Vec, linear);
//      wall_interp.set_input_parameters(wall_value_Vec, linear);

//      bc_interface_value = &interface_interp;
//      bc_wall_value = &wall_interp;

//  #ifdef P4_TO_P8
//      BoundaryConditions3D bc;
//  #else
//      BoundaryConditions2D bc;
//  #endif
//      bc.setInterfaceType(bc_interface_type);
//      bc.setInterfaceValue(*bc_interface_value);
//      bc.setWallTypes(*wall_bc);
//      bc.setWallValues(*bc_wall_value);

//      /* initialize the poisson solver */
//      w2.start("solve the poisson equation");
//      PoissonSolverNodeBase solver(&node_neighbors);
//      solver.set_phi(phi);
//      solver.set_rhs(rhs);
//      solver.set_bc(bc);

//      /* solve the system */
//      solver.solve(sol);
//      ierr = VecGhostUpdateBegin(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      ierr = VecGhostUpdateEnd  (sol, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
//      w2.stop(); w2.read_duration();

//      /* prepare for output */
//      double *sol_p, *phi_p, *uex_p;
//      ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
//      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//      ierr = VecGetArray(uex, &uex_p); CHKERRXX(ierr);

//      /* compute the error */
//      double err_max = 0;
//      double err[nodes->indep_nodes.elem_count];
//      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//      {
//        if(phi_p[n]<0)
//        {
//          err[n] = fabs(sol_p[n] - uex_p[n]);
//          err_max = max(err_max, err[n]);
//        }
//        else
//          err[n] = 0;
//      }
//      double glob_err_max;
//      MPI_Allreduce(&err_max, &glob_err_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
//      PetscPrintf(p4est->mpicomm, "lvl : %d / %d, L_inf error : %e\n",min_level+nb_splits, max_level+nb_splits, glob_err_max);

//      /* save the vtk file */
//      std::ostringstream oss; oss << P4EST_DIM << "d_solution_" << p4est->mpisize;
//      my_p4est_vtk_write_all(p4est, nodes, ghost,
//                             P4EST_TRUE, P4EST_TRUE,
//                             4, 0, oss.str().c_str(),
//                             VTK_POINT_DATA, "phi", phi_p,
//                             VTK_POINT_DATA, "sol", sol_p,
//                             VTK_POINT_DATA, "uex", uex_p,
//                             VTK_POINT_DATA, "err", err );

//      /* restore internal pointers */
//      ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
//      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
//      ierr = VecRestoreArray(uex, &uex_p); CHKERRXX(ierr);

//      /* destroy allocated vectors */
//      ierr = VecDestroy(phi); CHKERRXX(ierr);
//      ierr = VecDestroy(uex); CHKERRXX(ierr);
//      ierr = VecDestroy(sol); CHKERRXX(ierr);
//      ierr = VecDestroy(rhs); CHKERRXX(ierr);
//      ierr = VecDestroy(wall_value_Vec); CHKERRXX(ierr);
//      ierr = VecDestroy(interface_value_Vec); CHKERRXX(ierr);

//      /* destroy p4est objects */
//      p4est_nodes_destroy (nodes);
//      p4est_ghost_destroy (ghost);
//      p4est_destroy (p4est);
//      my_p4est_brick_destroy(connectivity, &brick);

//      w1.stop(); w1.read_duration();

//    } catch (const std::exception& e) {
//      std::cout << "[" << mpi->mpirank << " -- ERROR]: " << e.what() << std::endl;
//    }

//}


void poisson2d::poisson2d_initialyze_petsc(int argc, char *argv[])
{

    this->mpi = &mpi_context;
    this->mpi->mpicomm  = MPI_COMM_WORLD;
    this->mpi_session=new Session();
    this->mpi_session->init(argc, argv, this->mpi->mpicomm);

}

void poisson2d::poisson2d_finalyze_petsc()
{
    delete this->mpi_session;
}


poisson2d::poisson2d(int argc, char *argv[])
{

    try
    {
        this->cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
        this->cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
        this->cmd.add_option("lmin", "the min level of the tree");
        this->cmd.add_option("lmax", "the max level of the tree");
        this->cmd.add_option("nb_splits", "number of splits to apply to the min and max level");
        this->cmd.parse(argc, argv);

        // decide on the type and value of the boundary conditions

        this->bc_wall_type      = this->cmd.get("bc_wtype"  , DIRICHLET);
        this->bc_interface_type = this->cmd.get("bc_itype"  , DIRICHLET);
        this->nb_splits         = this->cmd.get("nb_splits" , 0);
        this->min_level         = this->cmd.get("lmin"      , 2);
        this->max_level         = this->cmd.get("lmax"      , 2);

        switch(this->bc_interface_type)
        {
        case DIRICHLET:
            this->bc_interface_value = &bc_interface_dirichlet_value;
            break;
        case NEUMANN:
            this->bc_interface_value = &bc_interface_neumann_value;
            break;
        default:
            throw std::invalid_argument("[ERROR]: Interface bc type can only be 'Dirichlet' or 'Neumann' type");
        }

        switch(this->bc_wall_type)
        {
        case DIRICHLET:
        {
            this->bc_wall_value = &bc_wall_dirichlet_value;
            this->wall_bc       = &bc_wall_dirichlet_type;
            std::cout<<"dirichlet"<<std::endl;
        }
            break;
        case NEUMANN:
        {
            this->bc_wall_value = &bc_wall_neumann_value;
            this->wall_bc       = &bc_wall_neumann_type;
            std::cout<<" neuman"<<std::endl;
        }
            break;
        default:
            throw std::invalid_argument("[ERROR]: Wall bc type can only be 'Dirichlet' or 'Neumann' type");
        }

#ifdef P4_TO_P8
        circle.update(1, 1, 1, .3);
#else
        circle.update(1, 1, .3);
#endif
        splitting_criteria_cf_t data(this->min_level+this->nb_splits, this->max_level+this->nb_splits, &circle, 1);
        this->poisson2d_initialyze_petsc(argc,argv);
        parStopWatch w1, w2;
        w1.start("total time");

        MPI_Comm_size (this->mpi->mpicomm, &this->mpi->mpisize);
        MPI_Comm_rank (this->mpi->mpicomm, &this->mpi->mpirank);

        w2.start("initializing the grid");

        /* create the macro mesh */

#ifdef P4_TO_P8
        this->connectivity = my_p4est_brick_new(2, 2, 2, &this->brick);
#else
        this->connectivity = my_p4est_brick_new(2, 2, &this->brick);
#endif

        /* create the p4est */
        this->p4est = p4est_new(this->mpi->mpicomm, this->connectivity, 0, NULL, NULL);
        this->p4est->user_pointer = (void*)(&data);
        p4est_refine(this->p4est, P4EST_TRUE, refine_levelset_cf, NULL);

        /* partition the p4est */
        p4est_partition(this->p4est, NULL);

        /* create the ghost layer */
        this->ghost = p4est_ghost_new(this->p4est, P4EST_CONNECT_FULL);

        /* generate unique node indices */
        this->nodes = my_p4est_nodes_new(this->p4est, this->ghost);
        w2.stop(); w2.read_duration();

        /* initialize the vectors */

        ierr = VecCreateGhostNodes(this->p4est, this->nodes, &this->phi); CHKERRXX(ierr);
        ierr = VecDuplicate(this->phi, &this->rhs); CHKERRXX(ierr);
        ierr = VecDuplicate(this->phi, &this->uex); CHKERRXX(ierr);
        ierr = VecDuplicate(this->phi, &this->sol); CHKERRXX(ierr);

        sample_cf_on_nodes(p4est, this->nodes, circle, this->phi);
        sample_cf_on_nodes(p4est, this->nodes, u_ex, this->uex);
        sample_cf_on_nodes(p4est, this->nodes, f_ex, this->rhs);

        PetscScalar *rhs_value=new PetscScalar[1];
        PetscInt *ix_rhs=new PetscInt[1];
        ix_rhs[0]=0;
//        VecGetValues(this->rhs,1,ix_rhs,rhs_value);
//         std::cout<<"rank rhs "<<this->mpi->mpirank<<" "<<rhs_value[0]<<std::endl;

         delete rhs_value;

         VecGetArray(this->rhs,&rhs_value);
         std::cout<<"rank rhs "<<this->mpi->mpirank<<" "<<rhs_value[0]<<std::endl;

         VecGetArray(this->uex,&rhs_value);
         std::cout<<"rank uex "<<this->mpi->mpirank<<" "<<rhs_value[0]<<std::endl;


        /* create the hierarchy structure */
        w2.start("construct the hierachy information");
        my_p4est_hierarchy_t hierarchy(p4est, this->ghost, &this->brick);
        w2.stop(); w2.read_duration();

        /* generate the neighborhood information */
        w2.start("construct the neighborhood information");
        my_p4est_node_neighbors_t node_neighbors(&hierarchy, this->nodes);
        w2.stop(); w2.read_duration();

        /* initalize the bc information */
        Vec interface_value_Vec, wall_value_Vec;
        ierr = VecDuplicate(this->phi, &interface_value_Vec); CHKERRXX(this->ierr);
        ierr = VecDuplicate(this->phi, &wall_value_Vec); CHKERRXX(this->ierr);

        sample_cf_on_nodes(p4est, this->nodes, *this->bc_interface_value, interface_value_Vec);
        sample_cf_on_nodes(p4est, this->nodes, *this->bc_wall_value, wall_value_Vec);

        PetscScalar *wall_value;
       VecGetArray(wall_value_Vec,&wall_value);

       std::cout<<"rank wv "<<this->mpi->mpirank<<" "<<wall_value[0]<<std::endl;

        InterpolatingFunctionNodeBase interface_interp(p4est, this->nodes, this->ghost, &this->brick, &node_neighbors), wall_interp(p4est, this->nodes, this->ghost, &this->brick, &node_neighbors);
        interface_interp.set_input_parameters(interface_value_Vec, linear);
        wall_interp.set_input_parameters(wall_value_Vec, linear);

        this->bc_interface_value = &interface_interp;
        this->bc_wall_value = &wall_interp;


        this->bc.setInterfaceType(this->bc_interface_type);
        this->bc.setInterfaceValue(*this->bc_interface_value);
        this->bc.setWallTypes(*this->wall_bc);
        this->bc.setWallValues(*this->bc_wall_value);

        /* initialize the poisson solver */
        w2.start("solve the poisson equation");
        this->solver=new PoissonSolverNodeBase(&node_neighbors);
        this->solver->set_phi(this->phi);



        this->solver->set_rhs(this->rhs);
        this->solver->set_bc(this->bc);

        /* solve the system */
        this->solver->solve(this->sol);
        ierr = VecGhostUpdateBegin(this->sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd  (this->sol, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
        w2.stop(); w2.read_duration();

        /* prepare for output */

        ierr = VecGetArray(this->sol, &this->sol_p); CHKERRXX(ierr);
        ierr = VecGetArray(this->phi, &this->phi_p); CHKERRXX(ierr);
        ierr = VecGetArray(this->uex, &this->uex_p); CHKERRXX(ierr);
        ierr = VecGetArray(this->rhs, &this->rhs_p); CHKERRXX(ierr);


        std::cout<<"r rhs "<<this->mpi->mpirank<<this->rhs_p[0]<<std::endl;


        /* compute the error */
        double err_max = 0;
        this->err=new double[this->nodes->indep_nodes.elem_count];
        for(size_t n=0; n<this->nodes->indep_nodes.elem_count; ++n)
        {
            if(this->phi_p[n]<0)
            {
                this->err[n] = fabs(this->sol_p[n] - this->uex_p[n]);
                err_max = max(err_max, this->err[n]);
            }
            else
                this->err[n] = 0;
        }
        double glob_err_max;
        MPI_Allreduce(&err_max, &glob_err_max, 1, MPI_DOUBLE, MPI_MAX, this->p4est->mpicomm);
        PetscPrintf(this->p4est->mpicomm, "lvl : %d / %d, L_inf error : %e\n",min_level+nb_splits, max_level+nb_splits, glob_err_max);



        /*IO: Print To Text File all the relevant information in different data structures*/

        this->printForestNodes2TextFile();
        this->printForestOctants2TextFile();
        this->printGhostCells();


        /* save the vtk file */
        std::ostringstream oss; oss << P4EST_DIM << "d_solution_" << p4est->mpisize;
        my_p4est_vtk_write_all(p4est, this->nodes, this->ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               4, 0, oss.str().c_str(),
                               VTK_POINT_DATA, "phi", this->phi_p,
                               VTK_POINT_DATA, "sol", this->sol_p,
                               VTK_POINT_DATA, "uex", this->uex_p,
                               VTK_POINT_DATA, "err", this->err );

        delete this->err;

        /* restore internal pointers */
        ierr = VecRestoreArray(this->sol, &this->sol_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(this->phi, &this->phi_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(this->uex, &this->uex_p); CHKERRXX(ierr);


        VecDuplicate(this->sol,&this->xx);            VecCopy(this->sol,this->xx);
        VecDuplicate(this->uex,&this->u);             VecCopy(this->uex,this->u);
        VecDuplicate(this->solver->rhs_,&this->b);    VecCopy(this->solver->rhs_,this->b);




        MatDuplicate(this->solver->A,MAT_COPY_VALUES,&this->ADense);
        MatCopy(this->solver->A,this->ADense,SAME_NONZERO_PATTERN);

        this->printDenseLinearAlgebraSolution();

        /* destroy allocated vectors */
        ierr = VecDestroy(this->phi); CHKERRXX(ierr);
        ierr = VecDestroy(this->uex); CHKERRXX(ierr);
        ierr = VecDestroy(this->sol); CHKERRXX(ierr);
        ierr = VecDestroy(this->rhs); CHKERRXX(ierr);
        ierr = VecDestroy(wall_value_Vec); CHKERRXX(ierr);
        ierr = VecDestroy(interface_value_Vec); CHKERRXX(ierr);

        /* destroy p4est objects */
        p4est_nodes_destroy (this->nodes);
        p4est_ghost_destroy (this->ghost);
        p4est_destroy (this->p4est);
        my_p4est_brick_destroy(this->connectivity, &this->brick);

        w1.stop(); w1.read_duration();

    }
    catch (const std::exception& e)
    {
        std::cout << "[" << this->mpi->mpirank << " -- ERROR]: " << e.what() << std::endl;
    }
}



void poisson2d::createDensePetscMatrix(int argc, char *argv[])
{
    this->mpi = &mpi_context;
    this->mpi->mpicomm  = MPI_COMM_WORLD;
    try
    {
        this->mpi_session->init(argc, argv, this->mpi->mpicomm);
        parStopWatch w1, w2;
        w1.start("total time");

        MPI_Comm_size (this->mpi->mpicomm, &this->mpi->mpisize);
        MPI_Comm_rank (this->mpi->mpicomm, &this->mpi->mpirank);

        std::cout<<"rank/size:";
        std::cout<<this->mpi->mpirank<<"/"<<this->mpi->mpisize<<std::endl;


        //MatCreate(MPI_COMM_WORLD, &this->A);
        MatType type=MATMPIAIJ;
        //MatSetType(this->A,type);

        PetscInt M;//=4;   //global number of columns
        PetscInt N;//=4;  //global number of rows
        PetscInt m;//=1;//M/this->mpi->mpisize;  //local number of rows
        PetscInt n;//=1;//N/this->mpi->mpisize;  //local number of columns

        if(false)
        {
            m=1;//M/this->mpi->mpisize;  //local number of rows
            n=1;//N/this->mpi->mpisize;  //local number of columns
            MatCreateDense(MPI_COMM_WORLD,m,n,PETSC_DECIDE,PETSC_DECIDE,PETSC_NULL,&this->A);
        }
        else
        {
            M=4;   //global number of columns
            N=4;  //global number of rows
            MatCreateDense(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,PETSC_NULL,&this->A);
        }


//        std::cout<<"rank/size:";
//        std::cout<<this->mpi->mpirank<<"/"<<this->mpi->mpisize<<"M N m n"<<M<<" "<<N<<" "<<m<<" "<<n<<std::endl;

        PetscInt m_global,n_global;
        PetscInt m_local,n_local;
        PetscInt global_first_row,global_last_row;
        MatGetSize(this->A,&m_global,&n_global);
        MatGetLocalSize(this->A,&m_local,&n_local);
        MatGetOwnershipRange(this->A,&global_first_row,&global_last_row);

        std::cout<<"rank m_global n_global m_local n_local global_first_row global_last_row \n "
                  <<this->mpi->mpirank<<" "<<m_global<<" "<<n_global<<" "<<m_local<<" "<<n_local<<" "
               << global_first_row<<" "<< global_last_row<<" "
                 <<std::endl;



        //This routine inserts a mxn block of values in the matrix.
        PetscInt m_block ;// number of rows
        PetscInt *idxm; // global indexes of rows
        PetscInt   n_block;// number of columns
        PetscInt   *idxn;// global indexes of columns
        PetscScalar  *values;// array containing values to be inserted

//        m_block=  M/this->mpi->mpisize;
//        n_block=N;
//        idxm=new PetscInt[m_block];
//        idxn=new PetscInt[n_block];
//        values=new PetscScalar[m_block*n_block];

//        for(int i=0;i<m_block;i++)
//        {
//            idxm[i]=i+this->mpi->mpirank*M/this->mpi->mpisize;
//        }

//        for(int j=0;j<n_block;j++)
//        {
//            idxn[j]=j;
//        }
//        for(int i=0;i<m_block;i++)
//        {
//            for(int j=0;j<n_block;j++)
//            {
//                values[i*n_block+j]=i*n_block+j;
//            }
//        }



        m_block=  m_global/this->mpi->mpisize;
        n_block=n_global;

        idxm=new PetscInt[m_block*n_block];
        idxn=new PetscInt[m_block*n_block];

        values=new PetscScalar[m_block*n_block];

        int kk=0;
        for(int i=0;i<m_block;i++)
        {

            idxm[i]=this->mpi->mpirank*m_global/this->mpi->mpisize+i;
        }

        kk=0;
        for(int j=0;j<n_block;j++)
        {
            idxn[j]=j;
        }
        for(int i=0;i<m_block;i++)
        {
            for(int j=0;j<n_block;j++)
            {
                values[i*n_block+j]=idxm[i]*n_block+idxn[j];
            }
        }
        std::stringstream oss2Debug;
        std::string mystr2Debug;
        oss2Debug << this->convert2FullPath("Vmatrix")<<"_"<<this->mpi->mpirank<<".txt";
        mystr2Debug=oss2Debug.str();
        FILE *outFile;
        outFile=fopen(mystr2Debug.c_str(),"w");
        for(int i=0;i<m_block;i++)
        {
            for(int j=0;j<n_block;j++)
            {
                fprintf(outFile,"%d %d %d %d %d %f\n",this->mpi->mpirank,i, j, idxm[i], idxn[j],values[i*n_block+j]);
            }
        }
        fclose(outFile);
        MatSetValues(this->A, m_block,idxm,n_block, idxn, values, INSERT_VALUES);
        MatAssemblyBegin(this->A,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(this->A,MAT_FINAL_ASSEMBLY);
    }
    catch (const std::exception& e)
    {
        std::cout << "[" << this->mpi->mpirank << " -- ERROR]: " << e.what() << std::endl;
    }
}

void poisson2d::printDensePetscMatrix()
{

    //http://acts.nersc.gov/petsc/example1/ex2.c.html
    PetscInt m_global,n_global;
    PetscInt m_local,n_local;
    PetscInt global_first_row,global_last_row;
    MatGetSize(this->A,&m_global,&n_global);
    MatGetLocalSize(this->A,&m_local,&n_local);
    MatGetOwnershipRange(this->A,&global_first_row,&global_last_row);

    std::cout<<"rank m_global n_global m_local n_local global_first_row global_last_row  "
              <<this->mpi->mpirank<<" "<<m_global<<" "<<n_global<<" "<<m_local<<" "<<n_local<<" "
           << global_first_row<<" "<< global_last_row<<" "
             <<std::endl;

    PetscInt number_of_rows=m_local;
    PetscInt *global_indices_of_rows=new PetscInt[number_of_rows];
    PetscInt number_of_columns=n_global;
    PetscInt   *global_indices_of_columns=new PetscInt[number_of_columns];
    PetscScalar *store_the_values=new PetscScalar[number_of_rows*number_of_columns];


    for(int i=0;i<number_of_rows;i++)
    {
        global_indices_of_rows[i]=this->mpi->mpirank*m_global/this->mpi->mpisize+i;
    //global_indices_of_rows[i]=i;
    }
    for(int j=0;j<number_of_columns;j++)
    {
        global_indices_of_columns[j]=j;//this->mpi->mpirank*n_global/this->mpi->mpisize+  j;
      //   global_indices_of_columns[j]=  j;

    }


    std::stringstream oss2Debugix;
    std::string mystr2Debugix;
    oss2Debugix << this->convert2FullPath("IXmatrix")<<"_"<<this->mpi->mpirank<<".txt";
    mystr2Debugix=oss2Debugix.str();
    FILE *outFileix;
    outFileix=fopen(mystr2Debugix.c_str(),"w");
    for(int i=0;i<number_of_rows;i++)
    {
        for(int j=0;j<number_of_columns;j++)
        {
            fprintf(outFileix,"%d %d %d %d %d\n",this->mpi->mpirank,i, j, global_indices_of_rows[i], global_indices_of_columns[j]);
        }
    }
    fclose(outFileix);

    MatGetValues(this->A,number_of_rows,global_indices_of_rows,number_of_columns,global_indices_of_columns,store_the_values);

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("Amatrix")<<"_"<<this->mpi->mpirank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");

    for(int i=0;i<number_of_rows;i++)
    {
        for(int j=0;j<number_of_columns;j++)
        {
            fprintf(outFile,"%d %d %d %d %d %f\n",this->mpi->mpirank,i, j, global_indices_of_rows[i], global_indices_of_columns[j],store_the_values[i*number_of_columns+j]);
        }
    }
    fclose(outFile);
}

void poisson2d::printDenseLinearAlgebraSolution()
{
    int mpi_size;
    int mpi_rank;

    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
    //http://acts.nersc.gov/petsc/example1/ex2.c.html
    PetscInt m_global,n_global;
    PetscInt m_local,n_local;
    PetscInt global_first_row,global_last_row;
    MatGetSize(this->ADense,&m_global,&n_global);
    MatGetLocalSize(this->ADense,&m_local,&n_local);
    MatGetOwnershipRange(this->ADense,&global_first_row,&global_last_row);

    std::cout<<"rank m_global n_global m_local n_local global_first_row global_last_row  "
              <<mpi_rank<<" "<<m_global<<" "<<n_global<<" "<<m_local<<" "<<n_local<<" "
           << global_first_row<<" "<< global_last_row<<" "
             <<std::endl;

    PetscInt number_of_rows=m_local;
    PetscInt *global_indices_of_rows=new PetscInt[number_of_rows];
    PetscInt number_of_columns=n_global;
    PetscInt   *global_indices_of_columns=new PetscInt[number_of_columns];
    PetscScalar *store_the_values=new PetscScalar[number_of_rows*number_of_columns];

    for(int i=0;i<number_of_rows;i++)
    {
        global_indices_of_rows[i]=global_first_row+i;
    //global_indices_of_rows[i]=i;
    }
    for(int j=0;j<number_of_columns;j++)
    {
        global_indices_of_columns[j]=j;//this->mpi->mpirank*n_global/this->mpi->mpisize+  j;
      //   global_indices_of_columns[j]=  j;
    }

    std::stringstream oss2Debugix;
    std::string mystr2Debugix;
    oss2Debugix << this->convert2FullPath("IXmatrix")<<"_"<<mpi_rank<<".txt";
    mystr2Debugix=oss2Debugix.str();
    FILE *outFileix;
    outFileix=fopen(mystr2Debugix.c_str(),"w");
    for(int i=0;i<number_of_rows;i++)
    {
        for(int j=0;j<number_of_columns;j++)
        {
            fprintf(outFileix,"%d %d %d %d %d\n",mpi_rank,i, j, global_indices_of_rows[i], global_indices_of_columns[j]);
        }
    }
    fclose(outFileix);
    std::cout<<"rank  "
              <<mpi_rank
             <<std::endl;
    MatGetValues(this->ADense,number_of_rows,global_indices_of_rows,number_of_columns,global_indices_of_columns,store_the_values);
    std::cout<<"rank  "
              <<mpi_rank
             <<std::endl;
    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("Amatrix")<<"_"<<mpi_rank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");

    for(int i=0;i<number_of_rows;i++)
    {
        for(int j=0;j<number_of_columns;j++)
        {
            fprintf(outFile,"%d %d %d %d %d %f\n",mpi_rank,i, j, global_indices_of_rows[i], global_indices_of_columns[j],store_the_values[i*number_of_columns+j]);
        }
    }
    fclose(outFile);


    int n_local_size;
    int n_global_size;
    VecGetLocalSize(this->xx,&n_local_size);
    VecGetLocalSize(this->xx,&n_global_size);

    PetscScalar *local_x=new PetscScalar[n_local_size];
    PetscScalar *local_b=new PetscScalar[n_local_size];
    PetscScalar *local_u=new PetscScalar[n_local_size];

    VecGetArray(this->b,&local_b);
    VecGetArray(this->xx,&local_x);
    VecGetArray(this->u,&local_u);

    std::stringstream oss2DebugVec;
    std::string mystr2DebugVec;
    oss2DebugVec << this->convert2FullPath("BUXvector")<<"_"<<mpi_rank<<".txt";
    mystr2DebugVec=oss2DebugVec.str();
    FILE *outFileVec;
    outFileVec=fopen(mystr2DebugVec.c_str(),"w");

    for(int i=0;i<n_local_size;i++)
    {
        //std::cout<<mpi_rank<<" "<<n_local_size*mpi_rank+i<<" "<<i<<" "<<local_b[i]<<" "<<local_u[i]<<" "<<local_x[i]<<std::endl;
        fprintf(outFileVec,"%d %d %d %f %f %f\n",mpi_rank,n_local_size*mpi_rank+i,i,local_b[i],local_u[i],local_x[i]);
    }
    fclose(outFileVec);

}

int poisson2d::createDenseLinearAlgebraProbelm()
{

    //ierr = PetscOptionsGetInt(NULL,"-n",&this->n,NULL);CHKERRQ(ierr);
    //ierr = PetscOptionsGetBool(NULL,"-nonzero_guess",&this->nonzeroguess,NULL);CHKERRQ(ierr);

    this->nonzeroguess=PETSC_TRUE;

    int n_global_vector=n_petsc;
    int n_global_matrix=n_global_vector;

    this->ierr=VecCreate(PETSC_COMM_WORLD,&this->xx); CHKERRQ(this->ierr);
    this->ierr=PetscObjectSetName((PetscObject)this->xx,"Solution"); CHKERRQ(this->ierr);
    this->ierr=VecSetSizes(this->xx,PETSC_DECIDE,n_global_vector);
    this->ierr=VecSetFromOptions(this->xx); CHKERRQ(this->ierr);

    this->ierr=VecDuplicate(this->xx,&this->b);
    this->ierr=VecDuplicate(this->xx,&this->u);

    this->ierr=MatCreate(PETSC_COMM_WORLD,&this->ADense); CHKERRQ(this->ierr);
    this->ierr=MatSetSizes(this->ADense,PETSC_DECIDE,PETSC_DECIDE,n_global_matrix,n_global_matrix);
    this->ierr=MatSetFromOptions(this->ADense); CHKERRQ(this->ierr);
    this->ierr=MatSetUp(this->ADense); CHKERRQ(this->ierr);


    /*
                 Assemble matrix
              */
    int i;
    this->value[0] = -1.0; this->value[1] = 2.0; this->value[2] = -1.0;
    for (i=1; i<n_global_vector-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(this->ADense,1,&i,3,col,this->value,INSERT_VALUES);CHKERRQ(ierr);
    }
    i    = n_global_vector - 1; this->col[0] = n_global_vector - 2; this->col[1] = n_global_vector - 1;
    ierr = MatSetValues(this->ADense,1,&i,2,this->col,this->value,INSERT_VALUES);CHKERRQ(this->ierr);
    i    = 0; this->col[0] = 0; this->col[1] = 1; this->value[0] = 2.0; this->value[1] = -1.0;
    ierr = MatSetValues(this->ADense,1,&i,2,this->col,this->value,INSERT_VALUES);CHKERRQ(this->ierr);
    ierr = MatAssemblyBegin(this->ADense,MAT_FINAL_ASSEMBLY);CHKERRQ(this->ierr);
    ierr = MatAssemblyEnd(this->ADense,MAT_FINAL_ASSEMBLY);CHKERRQ(this->ierr);

    /*
                 Set exact solution; then compute right-hand-side vector.
              */
    this->ierr = VecSet(this->u,this->one);CHKERRQ(this->ierr);
    this->ierr = MatMult(this->ADense,this->u,this->b);CHKERRQ(this->ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                            Create the linear solver and set various options
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Create linear solver context
              */
    ierr = KSPCreate(PETSC_COMM_WORLD,&this->myKsp);CHKERRQ(ierr);

    /*
                 Set operators. Here the matrix that defines the linear system
                 also serves as the preconditioning matrix.
              */
    ierr = KSPSetOperators(this->myKsp,this->ADense,this->ADense,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    /*
                 Set linear solver defaults for this problem (optional).
                 - By extracting the KSP and PC contexts from the KSP context,
                   we can then directly call any KSP and PC routines to set
                   various options.
                 - The following four statements are optional; all of these
                   parameters could alternatively be specified at runtime via
                   KSPSetFromOptions();
              */
    ierr = KSPGetPC(this->myKsp,&this->pc);CHKERRQ(this->ierr);
    ierr = PCSetType(this->pc,PCJACOBI);CHKERRQ(this->ierr);
    ierr = KSPSetTolerances(this->myKsp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);


    /*
                Set runtime options, e.g.,
                    -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
                These options will override those specified above as long as
                KSPSetFromOptions() is called _after_ any other customization
                routines.
              */
    ierr = KSPSetFromOptions(this->myKsp);CHKERRQ(this->ierr);

    if (nonzeroguess) {
        PetscScalar p = .5;
        ierr = VecSet(this->xx,p);CHKERRQ(this->ierr);
        ierr = KSPSetInitialGuessNonzero(this->myKsp,PETSC_TRUE);CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                  Solve the linear system
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Solve linear system
              */
    ierr = KSPSolve(this->myKsp,this->b,this->xx);CHKERRQ(ierr);

    /*
                 View solver info; we could instead use the option -ksp_view to
                 print this info to the screen at the conclusion of KSPSolve().
              */
    ierr = KSPView(this->myKsp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



    this->printDenseLinearAlgebraSolution();



    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                  Check solution and clean up
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Check the error
              */
    ierr = VecAXPY(this->xx,this->neg_one,this->u);CHKERRQ(this->ierr);
    ierr = VecNorm(this->xx,NORM_2,&norm);CHKERRQ(this->ierr);
    ierr = KSPGetIterationNumber(this->myKsp,&this->its);CHKERRQ(ierr);
    if (norm > tol)
    {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G, Iterations %D\n",norm,its);CHKERRQ(ierr);
    }

    /*
                 Free work space.  All PETSc objects should be destroyed when they
                 are no longer needed.
              */


    /*
                 Always call PetscFinalize() before exiting a program.  This routine
                   - finalizes the PETSc libraries as well as MPI
                   - provides summary and diagnostic information if certain runtime
                     options are chosen (e.g., -log_summary).
              */
  // ierr = PetscFinalize();

    return 0;
}

int poisson2d::createDenseLinearAlgebraProbelmMPI()
{

    //ierr = PetscOptionsGetInt(NULL,"-n",&this->n,NULL);CHKERRQ(ierr);
    //ierr = PetscOptionsGetBool(NULL,"-nonzero_guess",&this->nonzeroguess,NULL);CHKERRQ(ierr);

    this->nonzeroguess=PETSC_TRUE;

    int n_global_vector=this->n_petsc;
    int n_global_matrix=n_global_vector;

    this->ierr=VecCreate(PETSC_COMM_WORLD,&this->xx); CHKERRQ(this->ierr);
    this->ierr=PetscObjectSetName((PetscObject)this->xx,"Solution"); CHKERRQ(this->ierr);
    this->ierr=VecSetSizes(this->xx,PETSC_DECIDE,n_global_vector);
    this->ierr=VecSetFromOptions(this->xx); CHKERRQ(this->ierr);

    this->ierr=VecDuplicate(this->xx,&this->b);
    this->ierr=VecDuplicate(this->xx,&this->u);

    VecGetOwnershipRange(this->xx,&this->rstart,&this->rend);
    VecGetLocalSize(this->xx,&this->nlocal);

    this->ierr=MatCreate(PETSC_COMM_WORLD,&this->ADense); CHKERRQ(this->ierr);
    this->ierr=MatSetSizes(this->ADense,this->nlocal,this->nlocal,n_global_matrix,n_global_matrix);
    this->ierr=MatSetFromOptions(this->ADense); CHKERRQ(this->ierr);
    this->ierr=MatSetUp(this->ADense); CHKERRQ(this->ierr);

    /*
         Assemble matrix.

         The linear system is distributed across the processors by
         chunks of contiguous rows, which correspond to contiguous
         sections of the mesh on which the problem is discretized.
         For matrix assembly, each processor contributes entries for
         the part that it owns locally.
      */

    PetscInt i;

    if (!this->rstart)
      {
        this->rstart = 1;
        i      = 0;
        this->col[0] = 0;
        this->col[1] = 1;
        this->value[0] = 2.0;
        this->value[1] = -1.0;
        this->ierr   = MatSetValues(this->ADense,1,&i,2,this->col,this->value,INSERT_VALUES);
        CHKERRQ(this->ierr);
      }
      if (this->rend == this->n_petsc)
      {
        this->rend = this->n_petsc-1;
        i    = this->n_petsc-1; col[0] = this->n_petsc-2; col[1] = this->n_petsc-1; value[0] = -1.0; value[1] = 2.0;
        ierr = MatSetValues(this->ADense,1,&i,2,this->col,this->value,INSERT_VALUES);CHKERRQ(this->ierr);
      }

      /* Set entries corresponding to the mesh interior */
      this->value[0] = -1.0; this->value[1] = 2.0; this->value[2] = -1.0;
      for (i=this->rstart; i<this->rend; i++)
      {
        this->col[0] = i-1; this->col[1] = i; this->col[2] = i+1;
        ierr   = MatSetValues(this->ADense,1,&i,3,this->col,this->value,INSERT_VALUES);CHKERRQ(this->ierr);
      }

      /* Assemble the matrix */
      ierr = MatAssemblyBegin(this->ADense,MAT_FINAL_ASSEMBLY);CHKERRQ(this->ierr);
      ierr = MatAssemblyEnd(this->ADense,MAT_FINAL_ASSEMBLY);CHKERRQ(this->ierr);

      /*
                 Set exact solution; then compute right-hand-side vector.
              */
    this->ierr = VecSet(this->u,this->one);CHKERRQ(this->ierr);
    this->ierr = MatMult(this->ADense,this->u,this->b);CHKERRQ(this->ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                            Create the linear solver and set various options
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Create linear solver context
              */
    ierr = KSPCreate(PETSC_COMM_WORLD,&this->myKsp);CHKERRQ(ierr);

    /*
                 Set operators. Here the matrix that defines the linear system
                 also serves as the preconditioning matrix.
              */
    ierr = KSPSetOperators(this->myKsp,this->ADense,this->ADense,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    /*
                 Set linear solver defaults for this problem (optional).
                 - By extracting the KSP and PC contexts from the KSP context,
                   we can then directly call any KSP and PC routines to set
                   various options.
                 - The following four statements are optional; all of these
                   parameters could alternatively be specified at runtime via
                   KSPSetFromOptions();
              */
    ierr = KSPGetPC(this->myKsp,&this->pc);CHKERRQ(this->ierr);
    ierr = PCSetType(this->pc,PCJACOBI);CHKERRQ(this->ierr);
    ierr = KSPSetTolerances(this->myKsp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);


    /*
                Set runtime options, e.g.,
                    -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
                These options will override those specified above as long as
                KSPSetFromOptions() is called _after_ any other customization
                routines.
              */
    ierr = KSPSetFromOptions(this->myKsp);CHKERRQ(this->ierr);


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                  Solve the linear system
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Solve linear system
              */
    ierr = KSPSolve(this->myKsp,this->b,this->xx);CHKERRQ(ierr);

    /*
                 View solver info; we could instead use the option -ksp_view to
                 print this info to the screen at the conclusion of KSPSolve().
              */
    ierr = KSPView(this->myKsp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



    this->printDenseLinearAlgebraSolution();



    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                  Check solution and clean up
                 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
                 Check the error
              */
    ierr = VecAXPY(this->xx,this->neg_one,this->u);CHKERRQ(this->ierr);
    ierr = VecNorm(this->xx,NORM_2,&this->norm);CHKERRQ(this->ierr);
    ierr = KSPGetIterationNumber(this->myKsp,&this->its);CHKERRQ(ierr);
    if (this->norm >this->tol)
    {
        this->ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G, Iterations %D\n",this->norm,this->its);CHKERRQ(this->ierr);
    }

    /*
                 Free work space.  All PETSc objects should be destroyed when they
                 are no longer needed.
              */


    /*
                 Always call PetscFinalize() before exiting a program.  This routine
                   - finalizes the PETSc libraries as well as MPI
                   - provides summary and diagnostic information if certain runtime
                     options are chosen (e.g., -log_summary).
              */
  // ierr = PetscFinalize();

    return 0;
}


int poisson2d::destructDenseLinearAlgebraProblem()
{
    this->ierr = VecDestroy(this->xx);CHKERRQ(this->ierr);
    ierr = VecDestroy(this->u);CHKERRQ(this->ierr);
    ierr = VecDestroy(this->b);CHKERRQ(this->ierr);
    ierr = MatDestroy(this->ADense);CHKERRQ(this->ierr);
    ierr = KSPDestroy(this->myKsp);CHKERRQ(this->ierr);

}

void poisson2d::createSparsePetscMatrix(int argc, char *argv[])
{
    this->mpi = &mpi_context;
    this->mpi->mpicomm  = MPI_COMM_WORLD;
    try
    {
        this->mpi_session->init(argc, argv, this->mpi->mpicomm);
        parStopWatch w1, w2;
        w1.start("total time");

        MPI_Comm_size (this->mpi->mpicomm, &this->mpi->mpisize);
        MPI_Comm_rank (this->mpi->mpicomm, &this->mpi->mpirank);

        std::cout<<"rank/size:";
        std::cout<<this->mpi->mpirank<<"/"<<this->mpi->mpisize<<std::endl;


        //MatCreate(MPI_COMM_WORLD, &this->A);
        MatType type=MATMPIAIJ;
        //MatSetType(this->A,type);
        PetscInt M=4;   //global number of columns
        PetscInt N=4;  //global number of rows
       // PetscInt m=M/this->mpi->mpisize;  //local number of rows
        //PetscInt n=N/this->mpi->mpisize;  //local number of columns

        //MatSetSizes(this->A, m, n,M, N);

        MatCreateDense(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,PETSC_NULL,&this->A);

//        std::cout<<"rank/size:";
//        std::cout<<this->mpi->mpirank<<"/"<<this->mpi->mpisize<<"M N m n"<<M<<" "<<N<<" "<<m<<" "<<n<<std::endl;

        PetscInt m_global,n_global;
        PetscInt m_local,n_local;
        PetscInt global_first_row,global_last_row;
        MatGetSize(this->A,&m_global,&n_global);
        MatGetLocalSize(this->A,&m_local,&n_local);
        MatGetOwnershipRange(this->A,&global_first_row,&global_last_row);

        std::cout<<"rank m_global n_global m_local n_local global_first_row global_last_row \n "
                  <<this->mpi->mpirank<<" "<<m_global<<" "<<n_global<<" "<<m_local<<" "<<n_local<<" "
               << global_first_row<<" "<< global_last_row<<" "
                 <<std::endl;



        //This routine inserts a mxn block of values in the matrix.
        PetscInt m_block ;// number of rows
        PetscInt *idxm; // global indexes of rows
        PetscInt   n_block;// number of columns
        PetscInt   *idxn;// global indexes of columns
        PetscScalar  *values;// array containing values to be inserted

        m_block=M/this->mpi->mpisize;
        n_block=N;
        idxm=new PetscInt[m_block];
        idxn=new PetscInt[n_block];
        values=new PetscScalar[m_block*n_block];

        for(int i=0;i<m_block;i++)
        {
            idxm[i]=i+this->mpi->mpirank*M/this->mpi->mpisize;
        }

        for(int j=0;j<n_block;j++)
        {
            idxn[j]=j;
        }
        for(int i=0;i<m_block;i++)
        {
            for(int j=0;j<n_block;j++)
            {
                values[i*n_block+j]=i*n_block+j;
            }
        }

        MatSetValues(this->A, m_block,idxm,n_block, idxn, values, INSERT_VALUES);
        MatAssemblyBegin(this->A,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(this->A,MAT_FINAL_ASSEMBLY);
    }
    catch (const std::exception& e)
    {
        std::cout << "[" << this->mpi->mpirank << " -- ERROR]: " << e.what() << std::endl;
    }
}
void poisson2d::printSparsePetscMatrix()
{

    //http://acts.nersc.gov/petsc/example1/ex2.c.html
    PetscInt m_global,n_global;
    PetscInt m_local,n_local;
    PetscInt global_first_row,global_last_row;
    MatGetSize(this->A,&m_global,&n_global);
    MatGetLocalSize(this->A,&m_local,&n_local);
    MatGetOwnershipRange(this->A,&global_first_row,&global_last_row);

    std::cout<<"rank m_global n_global m_local n_local global_first_row global_last_row  "
              <<this->mpi->mpirank<<" "<<m_global<<" "<<n_global<<" "<<m_local<<" "<<n_local<<" "
           << global_first_row<<" "<< global_last_row<<" "
             <<std::endl;

    PetscInt number_of_rows=m_local;
    PetscInt *global_indices_of_rows=new PetscInt[number_of_rows];
    PetscInt number_of_columns=n_local;
    PetscInt   *global_indices_of_columns=new PetscInt[number_of_columns];
    PetscScalar *store_the_values=new PetscScalar[number_of_rows*number_of_columns];


    for(int i=0;i<number_of_rows;i++)
    {
        global_indices_of_rows[i]=this->mpi->mpirank*m_global/this->mpi->mpisize+i;
   // global_indices_of_rows[i]=i;
    }
    for(int j=0;j<number_of_columns;j++)
    {
        global_indices_of_columns[j]=this->mpi->mpirank*n_global/this->mpi->mpisize+  j;
       //  global_indices_of_columns[j]=  j;

    }

    MatGetValues(this->A,number_of_rows,global_indices_of_rows,number_of_columns,global_indices_of_columns,store_the_values);

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("Amatrix")<<"_"<<this->mpi->mpirank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");

    for(int i=0;i<number_of_rows;i++)
    {
        for(int j=0;j<number_of_columns;j++)
        {
            fprintf(outFile,"%d %d %d %f\n",this->mpi->mpirank,i, j,store_the_values[i*number_of_columns+j]);
        }
    }
    fclose(outFile);
}



void poisson2d::printForestNodes2TextFile()
{

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    std::stringstream oss2Debug;
    std::string mystr2Debug;
    oss2Debug << this->convert2FullPath("forest_of_independent_nodes")<<"_"<<myRank<<".txt";
    mystr2Debug=oss2Debug.str();
    FILE *outFile;
    outFile=fopen(mystr2Debug.c_str(),"w");

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


#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,this->sol_p[i],this->uex_p[i],this->err[i]);
#else
        fprintf(outFile,"%d %d %d %d %d %f %f %f %f %f %f\n",isNodeWall,  myRank,tree_id,i,global_node_number+i,  x,y,this->sol_p[i],this->uex_p[i],this->rhs_p[i],this->err[i]);
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


void poisson2d::printForestQNodes2TextFile()
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

            int quad_idx=j;

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


void poisson2d::printForestOctants2TextFile()
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
            fprintf(outFile,"%d %d %d %d %d %d %f %f %f %f\n",this->mpi->mpirank, i,j,my_local_num,my_first_number,my_tree_first_number,   x,y,z,this->sol_p_cell[j]);
#else
            fprintf(outFile,"%d %d %d %d %d %d %f %f %f\n",this->mpi->mpirank, i,j,my_local_num,my_first_number,my_tree_first_number,   x,y,this->sol_p_cell[j]);
#endif

        }
    }
    fclose(outFile);
}


void poisson2d::printGhostCells()
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

void poisson2d::printGhostNodes()
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






