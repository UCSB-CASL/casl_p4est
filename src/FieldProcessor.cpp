#include "FieldProcessor.h"




int FieldProcessor::compute_field_interface_width(Vec *w2process, Vec *phi2Process,
                                                  double alpha, double Xab,double &width,
                                                  double &phi_negative, double &phi_positive)
{

    PetscScalar *w2process_local;
    PetscScalar *phi2Process_local;
    PetscInt n_local;
    this->ierr=VecGetLocalSize(*w2process,&n_local);
    this->ierr=VecGetArray(*w2process,&w2process_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(*phi2Process,&phi2Process_local); CHKERRXX(this->ierr);
    double phi_negative_loc=-100;
    double phi_positive_loc=100;
    PetscScalar max_effective=alpha*Xab/2;
    VecMax(*w2process,NULL,&max_effective); max_effective  =alpha*max_effective;


    PetscScalar min_effective=-alpha*Xab/2;
    VecMin(*w2process,NULL,&min_effective); min_effective=alpha*min_effective;
    for(int i=0;i<n_local;i++)
    {

        if(w2process_local[i]>max_effective && phi2Process_local[i]>phi_negative_loc)
        {

            phi_negative_loc=phi2Process_local[i];
            continue;
        }


        if(w2process_local[i]<min_effective && phi2Process_local[i]<phi_positive_loc)
        {
            phi_positive_loc=phi2Process_local[i];
            continue;
        }

    }

    this->ierr=VecRestoreArray(*w2process,&w2process_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(*phi2Process,&phi2Process_local); CHKERRXX(this->ierr);


    MPI_Allreduce(&phi_negative_loc, &phi_negative, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&phi_positive_loc, &phi_positive, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);



    width=phi_positive-phi_negative;
    phi_negative=fabs(phi_negative);

    return 0;
}


int FieldProcessor::compute_coarse_gradients(Vec *w2process, Vec *phi2Process,
                                             double width,
                                             double &coarse_gradient_inside_the_domain,
                                             double &coarse_gradient_outside_the_domain,
                                             double &max_phi_negative,
                                             double &max_phi_positive,
                                             double &min_phi_negative,
                                             double &min_phi_positive)
{
//    PetscScalar *w2process_local;
//    PetscScalar *phi2Process_local;

//    PetscInt n_local;
//    this->ierr=VecGetLocalSize(*w2process,&n_local);


//    this->ierr=VecGetArray(*w2process,&w2process_local); CHKERRXX(this->ierr);
//    this->ierr=VecGetArray(*phi2Process,&phi2Process_local); CHKERRXX(this->ierr);

//    max_phi_negative=-100;
//    max_phi_positive=100;

//    for(int i=0;i<n_local;i++)
//    {

//        if(w2process_local[i]>alpha*Xab/2 && phi2Process_local[i]>max_phi_negative)
//        {

//            phi_negative=phi2Process_local[i];
//            continue;
//        }


//        if(w2process_local[i]<-alpha*Xab/2 && phi2Process_local[i]>min_phi_positive)
//        {
//            phi_positive=phi2Process_local[i];
//            continue;
//        }

//    }

//    this->ierr=VecRestoreArray(*w2process,&w2process_local); CHKERRXX(this->ierr);
//    this->ierr=VecRestoreArray(*phi2Process,&phi2Process_local); CHKERRXX(this->ierr);

//    width=max_phi_positive-min_phi_negative;

    return 0;
}



int FieldProcessor::smooth_level_set(my_p4est_node_neighbors_t *node_neighbors, Vec *phi,int n_local,int n_smoothies,PetscScalar band2computeKappa)
{



    // Compute first order derivatives
    Vec dmask_dx,dmask_dy
        #ifdef P4_TO_P8
            ,dmask_dz
        #endif
            ;
    PetscScalar *dmask_dx_local,*dmask_dy_local
        #ifdef P4_TO_P8
            ,*dmask_dz_local
        #endif
            ;

    this->ierr=VecDuplicate(*phi,&dmask_dx); CHKERRXX(this->ierr);
    this->ierr=VecDuplicate(*phi,&dmask_dy); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
    this->ierr=VecDuplicate(*phi,&dmask_dz); CHKERRXX(this->ierr);
#endif

    PetscScalar *dmask_local;
    this->ierr=VecGetArray(*phi,&dmask_local); CHKERRXX(this->ierr);

    this->ierr=VecGetArray(dmask_dx,&dmask_dx_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(dmask_dy,&dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
    this->ierr=VecGetArray(dmask_dz,&dmask_dz_local); CHKERRXX(this->ierr);
#endif
    PetscScalar normalize_normal;
    for(p4est_locidx_t ix=0;ix<n_local;ix++)
    {
        dmask_dx_local[ix]=node_neighbors->neighbors[ix].dx_central(dmask_local);
        dmask_dy_local[ix]=node_neighbors->neighbors[ix].dy_central(dmask_local);

#ifdef P4_TO_P8
        dmask_dz_local[ix]=node_neighbors->neighbors[ix].dz_central(dmask_local);
#endif
        normalize_normal=dmask_dx_local[ix]*dmask_dx_local[ix]+dmask_dy_local[ix]*dmask_dy_local[ix];

#ifdef P4_TO_P8
        normalize_normal+=dmask_dz_local[ix]*dmask_dz_local[ix];
#endif


    }
    //
    this->ierr=VecRestoreArray(dmask_dx,&dmask_dx_local); CHKERRXX(this->ierr);
   this->ierr=VecRestoreArray(dmask_dy,&dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
   this->ierr=VecRestoreArray(dmask_dz,&dmask_dz_local); CHKERRXX(this->ierr);
#endif

   // scatter again

   this->ierr=VecGhostUpdateBegin(dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
   this->ierr=VecGhostUpdateEnd(dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

   this->ierr=VecGhostUpdateBegin(dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
   this->ierr=VecGhostUpdateEnd(dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
   this->ierr=VecGhostUpdateBegin(dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
   this->ierr=VecGhostUpdateEnd(dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
#endif
    this->ierr=VecGetArray(dmask_dx,&dmask_dx_local); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(dmask_dy,&dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
    this->ierr=VecGetArray(dmask_dz,&dmask_dz_local); CHKERRXX(this->ierr);
#endif
    // Compute Second Order Derivatives

    Vec kappa;
    PetscScalar *kappa_local;
    this->ierr=VecDuplicate(*phi,&kappa); CHKERRXX(this->ierr);
    this->ierr=VecGetArray(kappa,&kappa_local); CHKERRXX(this->ierr);


    double abs_grad_phi;
double dmask_local_i;

    for (int i=0; i<n_local;i++)
    {
       dmask_local_i= dmask_local[i];
        abs_grad_phi=dmask_dx_local[i]*dmask_dx_local[i]+dmask_dy_local[i]*dmask_dy_local[i];
#ifdef P4_TO_P8
        abs_grad_phi=abs_grad_phi+dmask_dz_local[i]*dmask_dz_local[i];
#endif
        abs_grad_phi=pow(abs_grad_phi,0.5);
        if(dmask_local_i*dmask_local_i<band2computeKappa*band2computeKappa)
        {
            kappa_local[i]=node_neighbors->neighbors[i].dx_central(dmask_dx_local);
            kappa_local[i]+=node_neighbors->neighbors[i].dy_central(dmask_dy_local);
#ifdef P4_TO_P8
            kappa_local[i]+=node_neighbors->neighbors[i].dz_central(dmask_dz_local);
#endif

            kappa_local[i]=kappa_local[i]/abs_grad_phi;
            std::cout<<kappa_local[i]<<" "<<std::endl;
        }
        else
        {
            kappa_local[i]=0.01;
        }


    }



    this->ierr=VecRestoreArray(kappa,&kappa_local); CHKERRXX(this->ierr);
    this->ierr=VecScale(kappa,-1.00); CHKERRXX(this->ierr);

    this->ierr=VecGhostUpdateBegin(kappa,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
    this->ierr=VecGhostUpdateEnd(kappa,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    this->ierr=VecRestoreArray(*phi,&dmask_local);CHKERRXX(this->ierr);

    this->ierr=VecRestoreArray(dmask_dx,&dmask_dx_local); CHKERRXX(this->ierr);
    this->ierr=VecRestoreArray(dmask_dy,&dmask_dy_local); CHKERRXX(this->ierr);
#ifdef P4_TO_P8
   this->ierr=VecRestoreArray(dmask_dz,&dmask_dz_local); CHKERRXX(this->ierr);
#endif

   // scatter again

   this->ierr=VecGhostUpdateBegin(dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
   this->ierr=VecGhostUpdateEnd(dmask_dx,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

   this->ierr=VecGhostUpdateBegin(dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
   this->ierr=VecGhostUpdateEnd(dmask_dy,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
   this->ierr=VecGhostUpdateBegin(dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
   this->ierr=VecGhostUpdateEnd(dmask_dz,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
#endif
    // Motion by curvature

    my_p4est_level_set ls(node_neighbors);
    double alpha_cfl=0.5;
    double dt_get;

    for(int i=0;i<n_smoothies;i++)
    {
      dt_get=ls.advect_in_normal_direction(kappa,*phi,alpha_cfl);
      std::cout<<" dt_get smoother  "<<dt_get<<" "<<n_smoothies<<std::endl;
      this->ierr=VecGhostUpdateBegin(*phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);
      this->ierr=VecGhostUpdateEnd(*phi,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(this->ierr);

    }

    // destroy the petscvectors


    this->ierr=VecDestroy(dmask_dx); CHKERRXX(this->ierr);
    this->ierr=VecDestroy(dmask_dy); CHKERRXX(this->ierr);

#ifdef P4_TO_P8
    this->ierr=VecDestroy(dmask_dz); CHKERRXX(this->ierr);
#endif

    this->ierr=VecDestroy(kappa); CHKERRXX(this->ierr);


    return 0;
}
