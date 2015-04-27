
#ifdef P4_TO_P8
#include "my_p8est_kmeans.h"
#else
#include "my_p4est_kmeans.h"
#endif


my_p4est_kmeans::my_p4est_kmeans(int argc, char* argv[])
{
    std::cout<<" p4estLocal "<<std::endl;
    this->mpi= &mpi_context;
    //----------------MPI Stuff-----------------------------------

    this->mpi->mpicomm  = MPI_COMM_WORLD;
    //------------------------------------------------------------

    //---------p4est fields declaration----------------

    //------------------------------------------------

    //----------Refinement functions---------------------


    //#ifdef P4_TO_P8
    //  circle circ(1, 1, 1, .1);
    //#else
    //  circle circ(1, 1, .1);
    //#endif

    //#ifdef P4_TO_P8
    //  ellipse ella(1, 1, 1,0.2,2,3, 1);
    //#else
    //  ellipse ella(1, 1,1,1 .1);
    //#endif

#ifdef P4_TO_P8
    BCCGenerator2 mySplitter(0.1,2.00);
#else
    BCCGenerator2 mySplitter(0.2,2);
#endif

    int min_level=2; int max_level=2;

    this->c1=10; this->c2=-10;

    splitting_criteria_cf_t data(min_level, max_level, &mySplitter, 1);

    //---------------------------------------------------------------------//


    //---------------------Initialyze MPI--------------------------------//

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);
    parStopWatch w1, w2;
    w1.start("total time");
    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    //--------------------------------------------------------------//


    // Create the connectivity object which connects between the partitionned quadrants or octants

    w2.start("connectivity");


#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(2, 2, 2, &brick,0,0,0);
#else
    connectivity = my_p4est_brick_new(2, 2, &brick,0,0);
#endif
    w2.stop(); w2.read_duration();


    w2.start("p4est generation");
    /* Create a new forest.
   * The new forest consists of equi-partitioned root quadrants.
   * When there are more processors than trees, some processors are empty.
   *
   * \param [in] mpicomm       A valid MPI communicator.
   * \param [in] connectivity  This is the connectivity information that
   *                           the forest is built with.  Note the p4est
   *                           does not take ownership of the memory.
   * \param [in] data_size     This is the size of data for each quadrant which
   *                           can be zero.  Then user_data_pool is set to NULL.
   * \param [in] init_fn       Callback function to initialize the user_data
   *                           which is already allocated automatically.
   * \param [in] user_pointer  Assign to the user_pointer member of the p4est
   *                           before init_fn is called the first time.
   *
   * \return This returns a valid forest.
   *
   * \note The connectivity structure must not be destroyed
   *       during the lifetime of this forest.
   */


    //  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

    p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL,(void*)(&data));

    w2.stop(); w2.read_duration();

    // Now refine the tree
    w2.start("refine");
    //p4est->user_pointer = (void*)(&data);
    p4est_refine(this->p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    p4est_partition(this->p4est, NULL);
    w2.stop(); w2.read_duration();

    // generate the ghost data-structure
    w2.start("generating ghost data structure");
    this->ghost = p4est_ghost_new(this->p4est, P4EST_CONNECT_FULL);
    w2.stop(); w2.read_duration();

    // generate the node data structure
    w2.start("creating nodes data structure");
    this->nodes = my_p4est_nodes_new(this->p4est, this->ghost);
    w2.stop(); w2.read_duration();

    /* Parallel vector:
   * To save the levelset function, we need a parallel vector. We do this by
   * using PETSc. Here, we just need PETSc's Vec object which is parallel vector
   * To create it, just call 'VecCreateGhostNodes' and pass in p4est, nodes, and the
   * vec object as arguments. Here we call our vector 'phi_global' to emphasize
   * that it lives across multiple processors.
   */
    w2.start("creating Ghosted vector");


    ierr = VecCreateGhostNodes(this->p4est, this->nodes, &this->phi_global); CHKERRXX(ierr);

    w2.stop(); w2.read_duration();

    /* Computing parallel levelset
   * As the first example, we need to compute the levelset function on the
   * nodes. Now, PETSc is written in pure C so you cannot just do phi_global[i]
   * because C does not understand [] for non-pointer objects. To fix this, we
   * ask PETSc to return a pointer to the actual data. This is done by calling
   * 'VecGetArray' and passing the Vec object and double* pointer.
   *
   * BE CAREFUL: the pointer is literally pointing to the actual data so if you
   * do something silly with it, like change the values it points to by mistake
   * or call free() on it or else, the compiler is not going stop you!
   *
   * NOTE: PETSc will take care of memory management. DO *NOT* FREE THE POINTER
   */

    ierr = VecGetArray(this->phi_global, &this->phi); CHKERRXX(ierr);

    /* Actuall loop:
   * Now that we have the pointer, we need to loop over nodes and compute the
   * levelset. You have two options:
   * 1) You loop over ALL nodes including both local and ghost. This is done as
   * shown below. This generally won't work since you don't know what to do with
   * ghost points ... here, however, its OK since we are just calling a function
   * circle that can be evaluated ANYWHERE. Also, note that if you go with this
   * method, you need to convert the index from p4est to PETSc. This is done by
   * calling p4est2petsc_local_numbering. I'll change it to a map later down the
   * road.
   * 2) The other option you have is to only compute the level set on local
   * nodes and ask PETSc to do the communication for you to find the values of
   * ghost points. This is done for the second method shown below.
   */
    w2.start("setting phi values");

    w2.stop(); w2.read_duration();

    /* Second method:
   * In the second method, we only compute the levelset on local points and then
   * ask PETSc to communicate among processors to update the ghost values. Note
   * that this is really not required here and is redundant, but is shown just
   * to teach you how to do the update when you will need it later on.
   *
   * To do this we first create a duplicate of the old Vec. Note that this does
   * NOT copy data in the old Vec.
   */

    // do the loop. Note how we only loop over LOCAL nodes
    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
        /* since we want to access the local nodes, we need to 'jump' over intial
     * nonlocal nodes. Number of initial nonlocal nodes is given by
     * nodes->offset_owned_indeps
     */
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;

        p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = connectivity->vertices[3*v_mm + 0];
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
        phi[i] = mySplitter(x,y,z);
#else
        phi[i] = mySplitter(x,y);
#endif
    }





    /* Update ghost points from local:
   * Now that we have calculated the levelset from local nodes, we can ask PETSc
   * to update ghosts from local. This is done by calling 'VecGhostUpdateBegin'
   * and 'VecGhostUpdateEnd' function pairs. You need to pass the Vec object
   * (here phi_global_copy) and two flags. The first one indicates if you want
   * to either add the new values to the old one, or just replace them (or other
   * stuff I do not talk about here). We just need to replace them so we pass
   * INSERT_VALUES.
   * The second flag, asks if you want to update ghost values from local or the
   * reverse process (i.e. update local values from ghost). We want the fist one
   * i.e. we want each processor send its local valid info to other processors
   * sso that they can update their ghost values. For this you use the flag
   * SCATTER_FORWARD. If you want the reverse, meaning you want each processor
   * to update its local values from other's ghosts (used lesss often) you use
   * SCATTER_REVERSE instead.
   */
    ierr = VecGhostUpdateBegin(this->phi_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(this->phi_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    VecCreateGhostCells(this->p4est,this->ghost,&this->Icell);

    ierr = VecDuplicate(this->Icell, &this->Ibin); CHKERRXX(ierr);

    ierr=VecGetArray(this->Icell,&this->phi_cell); CHKERRXX(ierr);

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
            phi_cell[jj]=(1.00/8.00)*
                    (this->phi[n0]+phi[n1]+phi[n2]+phi[n3]+phi[n4]+phi[n5]+phi[n6]+phi[n7]);
#else
            this->phi_cell[jj]=(1.00/4.00)*
                    (phi[n0]+phi[n1]+phi[n2]+phi[n3]);
#endif


            jj++;
        }

    }

    this->printForestNodes2TextFile();
    this->printForestQNodes2TextFile();
    this->printForestOctants2TextFile();
    this->printGhostCells();
    this->printGhostNodes();

    /* OK, now that we are done with the levelsets, we need to tell that to PETSc
   * so that it can mark its internal data structre. This is good because if
   * after this point you access the pointers by mistake, PETSc is going to
   * throw errors which will be helpful in debugging
   */

    ierr = VecGhostUpdateBegin(this->Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(this->Icell, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    // done. lets write both levelset. they MUST be identical when you open them.
    // done. lets write both levelset. they MUST be identical when you open them.
    std::ostringstream oss; oss << P4EST_DIM << "d_partition_" << p4est->mpisize;
    std::cout<<"I am here "<<  oss.str()<<std::endl;

    //     my_p4est_vtk_write_all(p4est, this->nodes,this->ghost    ,
    //                            P4EST_TRUE, P4EST_TRUE,
    //                            1, 0, oss.str().c_str(),
    //                            VTK_POINT_DATA, "phi", this->phi);

    my_p4est_vtk_write_all(p4est, this->nodes,this->ghost    ,
                           P4EST_TRUE, P4EST_TRUE,
                           0, 1, oss.str().c_str(),
                           VTK_CELL_DATA, "phi_cell", this->phi_cell);


    ierr = VecRestoreArray(this->phi_global, &this->phi); CHKERRXX(ierr);
    ierr = VecRestoreArray(this->Icell, &this->phi_cell); CHKERRXX(ierr);


    PetscInt    i_max;
    PetscScalar x_max;

    PetscInt    i_min;
    PetscScalar x_min;

    VecMax(this->Icell,&i_max,&x_max);
    VecMin(this->Icell,&i_min,&x_min);

    this->c1_global=x_max;
    this->c2_global=x_min;

    this->e1_global=0;
    this->e2_global=0;

    this->e1_global_2=10;
    this->e2_global_2=10;

    this->computeSegmentationError();

    while((this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)>0.01
          ||(this->e1_global_2+this->e2_global_2)<(this->e1_global+this->e2_global) )
    {
        this->e1_global_2=this->e1_global;
        this->e2_global_2=this->e2_global;
        this->computeSegmentationError();


        std::cout<<"Diff Error";
        std::cout<<(this->e1_global_2+this->e2_global_2)-(this->e1_global+this->e2_global)<<std::endl;


    }

    // finally, delete PETSc Vecs by calling 'VecDestroy' function
    ierr = VecDestroy(this->phi_global); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (this->nodes);
    p4est_ghost_destroy (this->ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(this->connectivity, &this->brick);

#ifdef P4_TO_P8
    std::cout<<" my parallel octree"<<std::endl;
#else
    std::cout<<" my parallel quadtree"<<std::endl;
#endif

    w1.stop(); w1.read_duration();

}


void my_p4est_kmeans::printForestNodes2TextFile()
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
        BCCGenerator2 mySplitter(0.1,2.00);
#else
        BCCGenerator2 mySplitter(0.1,2);
#endif

#ifdef P4_TO_P8
        phi_copy[i] = mySplitter(x,y,z);
#else
        phi_copy[i] = mySplitter(x,y);
#endif

#ifdef P4_TO_P8
        fprintf(outFile,"%d %d %d %d %f %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,z,phi_copy[i]);
#else
        fprintf(outFile,"%d %d %d %d %f %f %f\n",myRank,tree_id,i,global_node_number+i,  x,y,phi_copy[i]);
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


void my_p4est_kmeans::printForestQNodes2TextFile()
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


void my_p4est_kmeans::printForestOctants2TextFile()
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
            BCCGenerator2 mySplitter(0.1,2.00);
#else
            BCCGenerator2 mySplitter(0.1,2);
#endif



#ifdef P4_TO_P8
            fprintf(outFile,"%d %d %d %d %d %d %f %f %f %f\n",this->mpi->mpirank, i,j,my_local_num,my_first_number,my_tree_first_number,   x,y,z,this->phi_cell[j]);
#else
            fprintf(outFile,"%d %d %d %d %d %d %f %f %f\n",this->mpi->mpirank, i,j,my_local_num,my_first_number,my_tree_first_number,   x,y,this->phi_cell[j]);
#endif

        }
    }
    fclose(outFile);
}


void my_p4est_kmeans::printGhostCells()
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

void my_p4est_kmeans::printGhostNodes()
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



void my_p4est_kmeans::segmentKmeans()
{

    //    this->I1=v1;

    //    VecCreateMPI(MPI_COMM_WORLD,this->p4est->local_num_quadrants,this->p4est->global_num_quadrants,&this->Icell);
    //    VecCreateMPI(MPI_COMM_WORLD,this->p4est->local_num_quadrants,this->p4est->global_num_quadrants,&this->Ibin);

    ////    int i1=this->p4est->global_first_quadrant;
    ////    int i2=this->p4est->global_first_quadrant+this->p4est->global_num_quadrants;


    ////    double e_total=0;
    ////    double e_total_fiif=0;
    ////    double e1=0;
    ////    double e2=0;


    ////    for(int i=0;i<this->p4est->local_num_quadrants;i++)
    ////    {
    ////      VecSetValue(this->Icell,i1+i,this->I1
    ////    }



}


void my_p4est_kmeans::computeSegmentationError()
{
    this->A1=0; this->A2=0;
    this->c1=0; this->c2=0;
    this->e1=0;this->e2=0;

    FILE *outFile;
    if(this->debug)
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
        p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
        double tree_xmin = connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
        double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

        for(size_t quad_idx=0;quad_idx<tree->quadrants.elem_count;quad_idx++)
        {
            const p4est_quadrant_t *quad=(const p4est_quadrant_t *)sc_array_index(&tree->quadrants,quad_idx);
            PetscInt *iq2=new PetscInt[1];
            iq2[0]=firstQuadrant+quad_idx;
            const PetscInt *iq=iq2;
            PetscScalar *cellColour=new PetscScalar[1];
            PetscScalar *cellBin=new PetscScalar[1];
            VecGetValues(this->Icell,1,iq,cellColour);


            PetscInt iq1=iq2[0];
            double A=1./pow(2,quad->level);

#ifdef P4_TO_P8
            A=A*A*A;
#else
            A=A*A;
#endif

            if(ABS(cellColour[0]-this->c1_global)>ABS(cellColour[0]-this->c2_global))
            {
                VecSetValue(this->Ibin,iq1,1,INSERT_VALUES);
                this->e2+=pow(cellColour[0]-this->c1_global,2)*A;
                this->A2+=A;
                this->c2+=cellColour[0]*A;
            }
            else
            {
                VecSetValue(this->Ibin,iq1,-1,INSERT_VALUES);
                this->e1+=pow(cellColour[0]-this->c2_global,2)*A;
                this->A1+=A;
                this->c1+=cellColour[0]*A;
            }

            VecGetValues(this->Ibin,1,iq,cellBin);

            double x = int2double_coordinate_transform(quad->x) + tree_xmin;
            double y = int2double_coordinate_transform(quad->y) + tree_ymin;

#ifdef P4_TO_P8
            double z = int2double_coordinate_transform(quad->z) + tree_zmin;
#endif

            if(this->debug)
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

    if(this->debug)
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


    if(this->debug)
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


void my_p4est_kmeans::petscGames()
{

}
