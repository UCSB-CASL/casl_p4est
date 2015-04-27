#ifndef INVERSE_LITOGRAPHY_H
#define INVERSE_LITOGRAPHY_H

#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <cstdarg>
#include <src/CASL_math.h>




using namespace std;

class inverse_litography
{
public:
    int n2Check;
    double Lx,r_mask,r_spot;
    double *xc,*yc,*zc;
    double ax,by,cz;
    double d_x;
    double d_y;
    double V0,f0;
    double Vmask;
    double cut_r_spot;
    double Vspot;
    bool reseed;
double curvature_barrier;
bool dynamic_f_A;
double wp_tolerance;

public:
    enum inverse_litography_stage
    {
        move_pressure_field,
        move_pressure_and_level_set_field,
        move_level_set_field,
        move_exchange_field,
        move_pressure_and_exchange_field
    };

    enum mask_target
    {
        pyramid,
        L_shape,
        super_pyramid,
        super_L_shape,
        eyes,
        customized,
        array_of_spots
    };

    mask_target my_mask_target;
    enum inverse_litography_strategy
    {
        wp_phi_wmwp,
        wpphi1_wpwm2,
        wp_av_min
    };

    enum inverse_litography_velocity
    {
        pressure_velocity,
        pressure_velocity_with_curvature_constraints,
        pressure_velocity_with_constraints,
        shape_derivative_velocity
    };

    enum velocity_stage
    {
        optimize_shape_phase,
        smooth_shape_phase
    };

    velocity_stage my_velocity_stage;

    inline inverse_litography_velocity get_my_inverse_litography_velocity(){return this->my_inverse_litography_velocity;}



private:

    double f_a;


    int smooth_counter;
    int max_smooth_counters;

    inverse_litography_strategy my_inverse_litography_strategy;
    inverse_litography_velocity my_inverse_litography_velocity;



    void (inverse_litography::*my_algo_decision)
    (int &start_remesh, int &stop_remesh, double &lambda_m, double lambda_in, double &lambda_p, double V0, double *Vt,
     double wp_av,double wp_std);//=NULL;

    inverse_litography::inverse_litography_stage my_inverse_litography_stage;
    double lambda_plus;
    double lambda_minus;
    int n_pressure,
    n_pressure_and_level_set,
    n_level_set,n_exchange,n_scft;
    int i_mean_field;
    int i_mean_field_in_current_cycle;
    int n_mean_field_steps_per_cycle;
    int total_number_of_mean_field_steps;
    int number_of_cycles;
    int cycle_iteration;
    int start_remesh;
    int stop_remesh;

    void initialyze_default_parameters()
    {
        this->reseed=true;
        this->cut_r_spot=2.00;
        this->number_of_cycles=1;
        this->n_mean_field_steps_per_cycle=this->total_number_of_mean_field_steps;
        this->cycle_iteration=0;
        this->n_pressure=0;
        this->n_exchange=0;
        this->n_level_set=0;
        this->n_scft=this->total_number_of_mean_field_steps;
        this->start_remesh=this->total_number_of_mean_field_steps;
        this->stop_remesh=0;
        this->my_inverse_litography_velocity=inverse_litography::pressure_velocity;
        this->my_velocity_stage=inverse_litography::optimize_shape_phase;
        this->smooth_counter=0;
        this->max_smooth_counters=10;
        this->ax=1.00,this->by=1.00,this->cz=1.00;
        this->curvature_barrier=0.00;
        this->dynamic_f_A=true;
        this->wp_tolerance=0.001;
    }
    void compute_i_cycle();



public:

    double compute_f_initial_from_design()
    {
        this->Vmask=PI*this->r_spot*this->r_spot*this->n2Check;
        this->V0=PI*this->r_mask*this->r_mask;
        this->Vspot=PI*this->r_spot*this->r_spot*this->n2Check/this->cut_r_spot;
        this->f0=this->Vmask/this->V0;
        this->f0=round(100*this->f0)/100;
        return this->f0;

    }

    inverse_litography_stage get_my_inverse_litography_stage()
    {
        return this->my_inverse_litography_stage;
    }
    inverse_litography(int n_mean_field_steps,double f_a);
    void initialyze_inverse_litography(int n_cycles,int n_pressure,int n_pressure_and_level_set,int n_level_set,int n_exchange,
                                       int n_scft,inverse_litography_strategy my_inverse_litography_strategy,inverse_litography_velocity my_inverse_litography_velocity)
    {
        this->number_of_cycles=n_cycles;
        this->n_mean_field_steps_per_cycle=this->total_number_of_mean_field_steps/this->number_of_cycles;
        this->n_pressure=n_pressure;
        this->n_pressure_and_level_set=n_pressure_and_level_set;
        this->n_level_set=n_level_set;
        this->n_exchange=n_exchange;
        this->n_scft=n_scft;
        // check consistency
//        if(this->n_mean_field_steps_per_cycle!=(this->n_pressure+this->n_pressure_and_level_set+this->n_level_set+this->n_exchange+this->n_scft))
//            throw std::runtime_error("not consistent numbers for the inverse litography plan");

        this->my_inverse_litography_strategy=my_inverse_litography_strategy;
        this->my_inverse_litography_velocity=my_inverse_litography_velocity;
        switch(this->my_inverse_litography_strategy)
        {
        case inverse_litography::wp_phi_wmwp:
            this->my_algo_decision=& inverse_litography::step_strategy;
            break;
        case inverse_litography::wpphi1_wpwm2:
            this->my_algo_decision=&inverse_litography::two_step_strategy;
            this->my_inverse_litography_stage=inverse_litography::move_pressure_and_level_set_field;
            break;
        case inverse_litography::wp_av_min:
            this->my_algo_decision=&inverse_litography::wp_av_strategy;
            this->my_inverse_litography_stage=inverse_litography::move_pressure_and_level_set_field;
            break;
        }

    }

    void initialyze_mask_design(int n2Check,double L,double r_mask,double r_spot,double d_x,double d_y,double *xc,double *yc,double *zc,inverse_litography::mask_target my_mask_target,double cut_r_spot)
    {
        this->cut_r_spot=cut_r_spot;
        this->n2Check=n2Check;
        this->Lx=L;
        double x0=this->Lx/2;
        this->r_mask=r_mask;
        this->r_spot=r_spot;
        this->d_x=d_x;
        this->d_y=d_y;
        this->xc=xc;
        this->yc=yc;
        this->zc=zc;
        this->my_mask_target=my_mask_target;
        switch (this->my_mask_target)
        {

        case inverse_litography::eyes:
        {
            this->xc[0]=x0-this->d_x;
            this->yc[0]=x0;
            this->zc[0]=x0;

            this->xc[1]=x0+this->d_x;
            this->yc[1]=x0;
            this->zc[1]=x0;


            break;
        }
        case inverse_litography::pyramid:
        {
            this->xc[0]=x0-this->d_x;
            this->yc[0]=x0-this->d_y;
            this->zc[0]=x0;

            this->xc[1]=x0+this->d_x;
            this->yc[1]=x0-this->d_y;
            this->zc[1]=x0;

            this->xc[2]=x0;
            this->yc[2]=x0-this->d_y;
            this->zc[2]=x0;

            this->xc[3]=x0+0.75*this->d_x;
            this->yc[3]=x0+this->d_y;
            this->zc[3]=x0;

            this->xc[4]=x0-0.75*this->d_x;
            this->yc[4]=x0+this->d_y;
            this->zc[4]=x0;
            break;
        }
        case inverse_litography::L_shape:
        {
            this->xc[0]=x0-this->d_x;
            this->yc[0]=x0-this->d_y;
            this->zc[0]=x0;

            this->xc[1]=x0-this->d_x;
            this->yc[1]=x0;
            this->zc[1]=x0;

            this->xc[2]=x0-this->d_x;
            this->yc[2]=x0+this->d_y;
            this->zc[2]=x0;

            this->xc[3]=x0;
            this->yc[3]=x0+this->d_y;
            this->zc[3]=x0;

            this->xc[4]=x0+this->d_x;
            this->yc[4]=x0+this->d_y;
            this->zc[4]=x0;
            break;
        }
        case inverse_litography::super_pyramid:
        {
            int i_cercle=0;
            this->xc[i_cercle]=x0-2*this->d_x;
            this->yc[i_cercle]=x0-1.5*this->d_y;
            this->zc[i_cercle]=x0; i_cercle++;

            //          this->xc[i_cercle]=x0-0.5*this->d_x;
            //          this->yc[i_cercle]=x0-1.5*this->d_y;
            //          this->zc[i_cercle]=x0;i_cercle++;

            //          this->xc[i_cercle]=x0+0.5*this->d_x;
            //          this->yc[i_cercle]=x0-1.5*this->d_y;
            //          this->zc[i_cercle]=x0;i_cercle++;

            this->xc[i_cercle]=x0+2*this->d_x;
            this->yc[i_cercle]=x0-1.5*this->d_y;
            this->zc[i_cercle]=x0;i_cercle++;

            this->xc[i_cercle]=x0-1.33*this->d_x;
            this->yc[i_cercle]=x0-0.5*this->d_y;
            this->zc[i_cercle]=x0;i_cercle++;


            //            this->xc[i_cercle]=x0;
            //            this->yc[i_cercle]=x0-0.5*this->d_y;
            //            this->zc[i_cercle]=x0;i_cercle++;

            this->xc[i_cercle]=x0+1.33*this->d_x;
            this->yc[i_cercle]=x0-0.5*this->d_y;
            this->zc[i_cercle]=x0;i_cercle++;

            this->xc[i_cercle]=x0-0.66*this->d_x;
            this->yc[i_cercle]=x0+0.5*this->d_y;
            this->zc[i_cercle]=x0;i_cercle++;

            this->xc[i_cercle]=x0+0.66*this->d_x;
            this->yc[i_cercle]=x0+0.5*this->d_y;
            this->zc[i_cercle]=x0;i_cercle++;

            //            this->xc[i_cercle]=x0;
            //            this->yc[i_cercle]=x0+1.5*this->d_y;
            //            this->zc[i_cercle]=x0;i_cercle++;
            break;
        }
        case inverse_litography::super_L_shape:
        {
            this->xc[0]=x0-1.5*this->d_x;
            this->yc[0]=x0-1.5*this->d_y;
            this->zc[0]=x0;

            this->xc[1]=x0-1.5*this->d_x;
            this->yc[1]=x0-0.5*this->d_y;
            this->zc[1]=x0;

            this->xc[2]=x0-1.5*this->d_x;
            this->yc[2]=x0+0.5*this->d_y;
            this->zc[2]=x0;

            this->xc[3]=x0-1.5*this->d_x;;
            this->yc[3]=x0+1.5*this->d_y;
            this->zc[3]=x0;

            this->xc[4]=x0-0.5*this->d_x;
            this->yc[4]=x0+1.5*this->d_y;
            this->zc[4]=x0;

            this->xc[5]=x0+0.5*this->d_x;
            this->yc[5]=x0+1.5*this->d_y;
            this->zc[5]=x0;

            this->xc[6]=x0+1.5*this->d_x;
            this->yc[6]=x0+1.5*this->d_y;
            this->zc[6]=x0;
            break;
        }








        }
    }

    void compute_litography_stage(int i_mean_field_step, double lambda_in, double &lambda_m, double &lambda_p,
                                  int &start_remesh, int &stop_remesh,double V0,double *Vt,double wp_av,double wp_std);


    void step_strategy(int &start_remesh, int &stop_remesh, double &lambda_m, double lambda_in, double &lambda_p, double V0, double *Vt, double wp_av, double wp_std);

    void two_step_strategy(int &start_remesh, int &stop_remesh, double &lambda_m, double lambda_in, double &lambda_p,double V0, double *Vt, double wp_av, double wp_std);

    void wp_av_strategy(int &start_remesh, int &stop_remesh, double &lambda_m, double lambda_in, double &lambda_p,double V0, double *Vt, double wp_av, double wp_std);


};

#endif // INVERSE_LITOGRAPHY_H
