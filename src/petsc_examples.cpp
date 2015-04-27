#include "petsc_examples.h"
static char help[] = "Reads U and V matrices from a file and performs y = V*U'*x.\n\
-f <input_file> : file to load \n\n";

petsc_examples::petsc_examples()
{
}

PetscErrorCode petsc_examples::petscVecExample(int argc, char *args[])
{
     //   http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatLoad.html
    Mat            U,V;                       /* matrix */
      PetscViewer    fd;                      /* viewer */
      char           file[PETSC_MAX_PATH_LEN];            /* input file name */
      PetscErrorCode ierr;
      PetscBool      flg;
      Vec            x,y,work1,work2;
      PetscInt       i,N,n,M,m;
      PetscScalar    *xx;




        std::cout<<U->data<<std::endl;

      PetscInitialize(&argc,&args,(char*)0,help);

      /*
         Determine file from which we read the matrix

      */
      ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);
        std::cout<<"1"<<std::endl;
        CHKERRQ(ierr);
      if (!flg)
        SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
    std::cout<<"2"<<std::endl;

      /*
         Open binary file.  Note that we use FILE_MODE_READ to indicate
         reading from this file.
      */
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

        std::cout<<"3"<<std::endl;

      /*
        Load the matrix; then destroy the viewer.
        Note both U and V are stored as tall skinny matrices
      */
      ierr = MatCreate(PETSC_COMM_WORLD,&U);
      CHKERRQ(ierr);
      ierr = MatSetType(U,MATMPIDENSE);
      CHKERRQ(ierr);
      ierr = MatLoad(U,fd);
      CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD,&V);
      CHKERRQ(ierr);
      ierr = MatSetType(V,MATMPIDENSE);
      CHKERRQ(ierr);
      ierr = MatLoad(V,fd);
      CHKERRQ(ierr);
      ierr = PetscViewerDestroy(fd);
      CHKERRQ(ierr);

      ierr = MatGetLocalSize(U,&N,&n);
      CHKERRQ(ierr);
      ierr = MatGetLocalSize(V,&M,&m);
      CHKERRQ(ierr);
      if (N != M) SETERRQ2(PETSC_COMM_SELF,1,"U and V matrices must have same number of local rows %D %D",N,M);
      if (n != m) SETERRQ2(PETSC_COMM_SELF,1,"U and V matrices must have same number of local columns %D %D",n,m);

      ierr = VecCreateMPI(PETSC_COMM_WORLD,N,PETSC_DETERMINE,&x);
      CHKERRQ(ierr);
      ierr = VecDuplicate(x,&y);
      CHKERRQ(ierr);

      ierr = MatGetSize(U,0,&n);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,n,&work1);CHKERRQ(ierr);
      ierr = VecDuplicate(work1,&work2);CHKERRQ(ierr);

      /* put some initial values into x for testing */
      ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
      for (i=0; i<N; i++) xx[i] = i;
      ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
      ierr = LowRankUpdate(U,V,x,y,work1,work2,n);CHKERRQ(ierr);
      ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

      /*
         Free work space.  All PETSc objects should be destroyed when they
         are no longer needed.
      */


      ierr = MatDestroy(U);
      CHKERRQ(ierr);
      ierr = MatDestroy(V);
      CHKERRQ(ierr);
      ierr = VecDestroy(x);
      CHKERRQ(ierr);
      ierr = VecDestroy(y);
      CHKERRQ(ierr);
      ierr = VecDestroy(work1);
      CHKERRQ(ierr);
      ierr = VecDestroy(work2);
      CHKERRQ(ierr);

      ierr = PetscFinalize();

}
PetscErrorCode petsc_examples::LowRankUpdate(Mat U,Mat V,Vec x,Vec y,Vec work1,Vec work2,PetscInt nwork)
{

  Mat            Ulocal = ((Mat_MPIDense*)U->data)->A;
  Mat            Vlocal = ((Mat_MPIDense*)V->data)->A;
  PetscInt       Nsave  = x->map->N;
  PetscErrorCode ierr;
  PetscScalar    *w1,*w2;

  PetscFunctionBegin;
  /* First multiply the local part of U with the local part of x */
  x->map->N = x->map->n; /* this tricks the silly error checking in MatMultTranspose();*/
  ierr      = MatMultTranspose(Ulocal,x,work1);CHKERRQ(ierr); /* note in this call x is treated as a sequential vector  */
  x->map->N = Nsave;

  /* Form the sum of all the local multiplies : this is work2 = U'*x = sum_{all processors} work1 */
  ierr = VecGetArray(work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(work2,&w2);CHKERRQ(ierr);
  ierr = MPI_Allreduce(w1,w2,nwork,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = VecRestoreArray(work1,&w1);CHKERRQ(ierr);
  ierr = VecRestoreArray(work2,&w2);CHKERRQ(ierr);

  /* multiply y = V*work2 */
  y->map->N = y->map->n; /* this tricks the silly error checking in MatMult() */
  ierr      = MatMult(Vlocal,work2,y);CHKERRQ(ierr); /* note in this call y is treated as a sequential vector  */
  y->map->N = Nsave;
  PetscFunctionReturn(0);
}


void petsc_examples::petscVecAssemblyExample(int argc, char *args[])
{

    PetscMPIInt    rank;
    PetscInt       i,N;

    Vec            x;
    PetscInitialize(&argc,&args,(char*)0,help);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    PetscScalar    one = 5.0*rank;
    std::cout<<rank<<" "<<one<<std::endl;



  /*
     32:      Create a parallel vector.
     33:       - In this case, we specify the size of each processor's local
     34:         portion, and PETSc computes the global size.  Alternatively,
     35:         if we pass the global size and use PETSC_DECIDE for the
     36:         local size PETSc will choose a reasonable partition trying
     37:         to put nearly an equal number of elements on each processor.
     38:   */

    VecCreate(PETSC_COMM_WORLD,&x);
    VecSetSizes(x,10,PETSC_DECIDE);
    VecSetFromOptions(x);
    VecGetSize(x,&N);
    VecSet(x,one);

        /*
     46:      Set the vector elements.
     47:       - Always specify global locations of vector entries.
     48:       - Each processor can contribute any vector entries,
     49:         regardless of which processor "owns" them; any nonlocal
     50:         contributions will be transferred to the appropriate processor
     51:         during the assembly process.
     52:       - In this example, the flag ADD_VALUES indicates that all
     53:         contributions will be added together.
     54:   */


    for (i=0; i<N-rank; i++)
    {
        VecSetValues(x,1,&i,&one,ADD_VALUES);
    }

        /*
     60:      Assemble vector, using the 2-step process:
     61:        VecAssemblyBegin(), VecAssemblyEnd()
     62:      Computations can be done while messages are in transition
     63:      by placing code between these two statements.
     64:   */
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);

    PetscInt low,high;

    VecGetOwnershipRange(x,&low,&high);

    std::cout<<"rank low high"<<std::endl;
    std::cout<<rank<<" "<<low<<" "<<high<<std::endl;

    /*
     * View the vector; then destroy it.
     */

    PetscInt *ir=new PetscInt[1];
    ir[0]=rank;
    PetscScalar *sc=new PetscScalar[1];
    VecGetValues(x,1,ir,sc);

    std::cout<<" rank first member"<<std::endl;
    std::cout<<rank<<" "<<sc[0]<<endl;

     VecView(x,PETSC_VIEWER_STDOUT_WORLD);
     VecDestroy(x);
     PetscFinalize();
}

int petsc_examples::petscVecGhostExample(int argc, char *args[])
{

     PetscMPIInt    rank,size;
     PetscInt       nlocal = 6,nghost = 2,ifrom[3],i,rstart,rend;
     PetscErrorCode ierr;
     PetscBool      flg,flg2;
     PetscScalar    value,*array,*tarray=0;
     Vec            lx,gx,gxs;

     PetscInitialize(&argc,&args,(char*)0,help);
     ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
     ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
     if (size != 2) SETERRQ(PETSC_COMM_SELF,1,"Must run example with two processors\n");


     std::cout<<" rank "<<rank<<" size "<<size<<std::endl;

     /*
        Construct a two dimensional graph connecting nlocal degrees of
        freedom per processor. From this we will generate the global
        indices of needed ghost values

        For simplicity we generate the entire graph on each processor:
        in real application the graph would stored in parallel, but this
        example is only to demonstrate the management of ghost padding
        with VecCreateGhost().

        In this example we consider the vector as representing
        degrees of freedom in a one dimensional grid with periodic
        boundary conditions.

           ----Processor  1---------  ----Processor 2 --------
            0    1   2   3   4    5    6    7   8   9   10   11
                                  |----|
            |-------------------------------------------------|

     */

     if (!rank)
     {
         ifrom[0] = 11; ifrom[1] = 6; ifrom[2]=7;
     }
     else
     {
         ifrom[0] = 0;  ifrom[1] = 5; ifrom[2]=1;
     }

     /*
        Create the vector with two slots for ghost points. Note that both
        the local vector (lx) and the global vector (gx) share the same
        array for storing vector values.
     */

     ierr = PetscOptionsHasName(NULL,"-allocate",&flg);CHKERRQ(ierr);
     ierr = PetscOptionsHasName(NULL,"-vecmpisetghost",&flg2);CHKERRQ(ierr);
     if (flg)
     {
         std::cout<<" option 1 rank"<<rank<<std::endl;

       ierr = PetscMalloc((nlocal+nghost)*sizeof(PetscScalar),&tarray);CHKERRQ(ierr);
       ierr = VecCreateGhostWithArray(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,tarray,&gxs);CHKERRQ(ierr);
     }
     else if (flg2)
     {
       std::cout<<" option 2 rank"<<rank<<std::endl;
       ierr = VecCreate(PETSC_COMM_WORLD,&gxs);CHKERRQ(ierr);
       ierr = VecSetType(gxs,VECMPI);CHKERRQ(ierr);
       ierr = VecSetSizes(gxs,nlocal,PETSC_DECIDE);CHKERRQ(ierr);
       ierr = VecMPISetGhost(gxs,nghost,ifrom);CHKERRQ(ierr);
     }
     else
     {
       std::cout<<" option 3 rank"<<rank<<std::endl;
       ierr = VecCreateGhost(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,&gxs);CHKERRQ(ierr);
     }


     /*
      * View the vector;
      */

     std::cout<<" rank "<<rank<<" start viewing the vector gxs "<<std::endl;

     PetscInt low,high;
     VecGetOwnershipRange(gxs,&low,&high);
     std::cout<<"rank low high"<<std::endl;
     std::cout<<rank<<" "<<low<<" "<<high<<std::endl;

     int size_vec=high-low;

     PetscScalar *y=new PetscScalar[size_vec];
     PetscInt *ix=new PetscInt[size_vec];

     for(int i=0;i<size_vec;i++)
         ix[i]=low+i;

     VecGetValues(gxs,size_vec,ix,y);

     for(int i=0;i<size_vec*size;i++)
         PetscPrintf(PETSC_COMM_WORLD,"%d %d %f\n",rank,i,y[i]);

      VecView(gxs,PETSC_VIEWER_STDOUT_WORLD);
      std::cout<<" End Viewing the vector gxs"<<std::endl;
      /*
       * End Viewing the vector;
       */


     /*
         Test VecDuplicate()
     */
     ierr = VecDuplicate(gxs,&gx);CHKERRQ(ierr);
     ierr = VecDestroy(gxs);CHKERRQ(ierr);


     /*
      * View the vector;
      */

     std::cout<<" rank "<<rank<<" start viewing the vector gx "<<std::endl;


     VecGetOwnershipRange(gx,&low,&high);
     std::cout<<"rank low high"<<std::endl;
     std::cout<<rank<<" "<<low<<" "<<high<<std::endl;

    size_vec=high-low;

    delete y; delete ix;

     y=new PetscScalar[size_vec];
     ix=new PetscInt[size_vec];

     for(int i=0;i<size_vec;i++)
         ix[i]=low+i;

     VecGetValues(gx,size_vec,ix,y);

     for(int i=0;i<size_vec*size;i++)
         PetscPrintf(PETSC_COMM_WORLD,"%d %d %f\n",rank,i,y[i]);

      VecView(gx,PETSC_VIEWER_STDOUT_WORLD);
      std::cout<<" End Viewing the vector gx"<<std::endl;
      /*
       * End Viewing the vector;
       */


     /*
        Access the local representation
     */

      ierr = VecGhostGetLocalForm(gx,&lx);CHKERRQ(ierr);


     /*
        Set the values from 0 to 12 into the "global" vector
     */
     ierr = VecGetOwnershipRange(gx,&rstart,&rend);CHKERRQ(ierr);
     for (i=rstart; i<rend; i++)
     {
       value = (PetscScalar) 2*i;
       ierr  = VecSetValues(gx,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
     }
     ierr = VecAssemblyBegin(gx);CHKERRQ(ierr);
     ierr = VecAssemblyEnd(gx);CHKERRQ(ierr);

     ierr = VecGhostUpdateBegin(gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
     ierr = VecGhostUpdateEnd(gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

     /*
        Print out each vector, including the ghost padding region.
     */
     ierr = VecGetArray(lx,&array);CHKERRQ(ierr);
     for (i=0; i<nlocal+nghost; i++)
     {
       ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%d %D %G\n",rank,i,PetscRealPart(array[i]));CHKERRQ(ierr);
     }
     ierr = VecRestoreArray(lx,&array);CHKERRQ(ierr);
     ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

     ierr = VecGhostRestoreLocalForm(gx,&lx);CHKERRQ(ierr);
     ierr = VecDestroy(gx);CHKERRQ(ierr);
     if (flg)
     {
         ierr = PetscFree(tarray);CHKERRQ(ierr);
     }
     ierr = PetscFinalize();
     return 0;

}


int petsc_examples::petscVecRestoreExample(int argc, char *args[])
{
PetscErrorCode ierr;
      PetscMPIInt    rank,nproc;
      PetscInt       rstart,rend,i,k,N,numPoints=1000000;
      PetscScalar    dummy,result=0,h=1.0/numPoints,*xarray;
      Vec            x,xend;

      PetscInitialize(&argc,&args,(char*)0,help);
      ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
      ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);

      /*
         Create a parallel vector.
           Here we set up our x vector which will be given values below.
           The xend vector is a dummy vector to find the value of the
             elements at the endpoints for use in the trapezoid rule.
      */
      ierr   = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
      ierr   = VecSetSizes(x,PETSC_DECIDE,numPoints);CHKERRQ(ierr);
      ierr   = VecSetFromOptions(x);CHKERRQ(ierr);
      ierr   = VecGetSize(x,&N);CHKERRQ(ierr);
      ierr   = VecSet(x,result);CHKERRQ(ierr);
      ierr   = VecDuplicate(x,&xend);CHKERRQ(ierr);
      result = 0.5;
      if (!rank) {
        i    = 0;
        ierr = VecSetValues(xend,1,&i,&result,INSERT_VALUES);CHKERRQ(ierr);
      } else if (rank == nproc) {
        i    = N-1;
        ierr = VecSetValues(xend,1,&i,&result,INSERT_VALUES);CHKERRQ(ierr);
      }
      /*
         Assemble vector, using the 2-step process:
           VecAssemblyBegin(), VecAssemblyEnd()
         Computations can be done while messages are in transition
         by placing code between these two statements.
      */
      ierr = VecAssemblyBegin(xend);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(xend);CHKERRQ(ierr);

      /*
         Set the x vector elements.
          i*h will return 0 for i=0 and 1 for i=N-1.
          The function evaluated (2x/(1+x^2)) is defined above.
          Each evaluation is put into the local array of the vector without message passing.
      */
      ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
      ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
      k    = 0;
      for (i=rstart; i<rend; i++) {
        xarray[k] = i*h;
        xarray[k] = func2(xarray[k]);
        k++;
      }
      ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);


      /*
         Evaluates the integral.  First the sum of all the points is taken.
         That result is multiplied by the step size for the trapezoid rule.
         Then half the value at each endpoint is subtracted,
         this is part of the composite trapezoid rule.
      */
      ierr   = VecSum(x,&result);CHKERRQ(ierr);
      result = result*h;
      ierr   = VecDot(x,xend,&dummy);CHKERRQ(ierr);
      result = result-h*dummy;

      /*
          Return the value of the integral.
      */
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ln(2) is %G\n",result);CHKERRQ(ierr);
      ierr = VecDestroy(x);CHKERRQ(ierr);
      ierr = VecDestroy(xend);CHKERRQ(ierr);

      ierr = PetscFinalize();
      return 0;


}
