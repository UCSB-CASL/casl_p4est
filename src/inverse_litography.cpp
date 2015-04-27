#include "inverse_litography.h"

inverse_litography::inverse_litography(int n_mean_field_steps, double f_a)
{

    this->f_a=f_a;
    this->total_number_of_mean_field_steps=n_mean_field_steps;

    this->initialyze_default_parameters();
    std::cout<<" Inverse Litography Constructor "<<std::endl;

}

void inverse_litography::compute_i_cycle()
{
    std::cout<<" compute i_cycle "<<std::endl;

    this->i_mean_field_in_current_cycle=this->i_mean_field%this->n_mean_field_steps_per_cycle;
    this->cycle_iteration=(this->i_mean_field-this->i_mean_field_in_current_cycle)/this->n_mean_field_steps_per_cycle;
    std::cout<<" i_mean_field_in_current_cycle "<<this->i_mean_field_in_current_cycle<<
               " cycle_iteration "<<this->cycle_iteration<<
               std::endl;

}

void inverse_litography::step_strategy(int &start_remesh, int &stop_remesh, double &lambda_m,
                                       double lambda_in, double &lambda_p,double V0,double *Vt, double wp_av, double wp_std)
{

 this->compute_i_cycle();
    if(this->i_mean_field_in_current_cycle<this->n_pressure)
        this->my_inverse_litography_stage=inverse_litography::move_pressure_field;


    if(this->i_mean_field_in_current_cycle>=this->n_pressure
            && this->i_mean_field_in_current_cycle<(this->n_pressure+this->n_pressure_and_level_set))
        this->my_inverse_litography_stage=inverse_litography::move_pressure_and_level_set_field;


    if(this->i_mean_field_in_current_cycle>=this->n_pressure+this->n_pressure_and_level_set
            && this->i_mean_field_in_current_cycle<(this->n_pressure+this->n_pressure_and_level_set+this->n_level_set))
        this->my_inverse_litography_stage=inverse_litography::move_level_set_field;



    if(this->i_mean_field_in_current_cycle>=(this->n_pressure+this->n_pressure_and_level_set+this->n_level_set)
            && this->i_mean_field_in_current_cycle<(this->n_pressure+this->n_pressure_and_level_set+this->n_level_set+this->n_exchange))
        this->my_inverse_litography_stage=inverse_litography::move_exchange_field;


    if(this->i_mean_field_in_current_cycle>=(this->n_pressure+this->n_pressure_and_level_set+this->n_level_set+this->n_exchange)
            && this->i_mean_field_in_current_cycle<(this->n_pressure+this->n_pressure_and_level_set+this->n_level_set+this->n_exchange+this->n_scft) )
        this->my_inverse_litography_stage=inverse_litography::move_pressure_and_exchange_field;



    this->start_remesh=this->cycle_iteration*this->n_mean_field_steps_per_cycle+this->n_pressure;
    this->stop_remesh=this->cycle_iteration*this->n_mean_field_steps_per_cycle+this->n_pressure+this->n_pressure_and_level_set+this->n_level_set;

    start_remesh=this->start_remesh;
    stop_remesh=this->stop_remesh;



    switch(this->my_inverse_litography_stage)
    {
    case inverse_litography::move_pressure_field:
    {
        this->lambda_minus=0.00;
        this->lambda_plus=1.00;
        break;
    }
    case inverse_litography::move_pressure_and_level_set_field:
    {
        this->lambda_minus=0.00;
        this->lambda_plus=1.00;
        break;
    }
    case inverse_litography::move_level_set_field:
    {
        this->lambda_minus=0.00;
        this->lambda_plus=0.00;
        break;
    }
    case inverse_litography::move_exchange_field:
    {
        this->lambda_minus=1.00;
        this->lambda_plus=0.00;
        break;
    }
    case inverse_litography::move_pressure_and_exchange_field:
    {
        this->lambda_minus=1.00;
        this->lambda_plus=1.00;
        break;
    }
    }

    lambda_p=this->lambda_plus*lambda_in;
    lambda_m=this->lambda_minus*lambda_in;
}

void inverse_litography::two_step_strategy(int &start_remesh, int &stop_remesh, double &lambda_m,
                                           double lambda_in, double &lambda_p, double V0, double *Vt, double wp_av, double wp_std)
{
 this->compute_i_cycle();
    int i_2Check=max(this->i_mean_field-1,0);
    double cut_off=V0/Vt[i_2Check];

    int i_mean_field_4_stability=100;
    int i_shift=10;
    double tolerance_stability=0.0001;
    double check_stability=2.00;
    std::cout<<" check_stability "<<check_stability<<std::endl;
    if(i_2Check>(i_shift+i_mean_field_4_stability))
    {
        check_stability=Vt[i_2Check]/Vt[i_2Check-i_shift];

        std::cout<<" Vt[i_2Check] Vt[i_2Check-i_shift]"<< Vt[i_2Check]<<" "<<Vt[i_2Check-i_shift]<<std::endl;;

        check_stability=ABS(check_stability-1);

    }





    std::cout<<" check_stability "<<check_stability<<std::endl;
    int min_mean_field_for_stopping_remesh=3;

    if( (cut_off<this->cut_r_spot && check_stability>tolerance_stability
         &&  this->my_inverse_litography_stage==inverse_litography::move_pressure_and_level_set_field&& this->smooth_counter==0)||
            this->i_mean_field_in_current_cycle<min_mean_field_for_stopping_remesh )
        //    if( (check_stability>tolerance_stability
        //         &&  this->my_inverse_litography_stage==inverse_litography::move_pressure_field)||
        //            this->i_mean_field<min_mean_field_for_stopping_remesh)
    {
        this->my_inverse_litography_stage=inverse_litography::move_pressure_and_level_set_field;
        this->i_mean_field_in_current_cycle=0;
        this->lambda_minus=0.00;
        this->lambda_plus=1.00;
        this->start_remesh=0;
        this->stop_remesh=this->total_number_of_mean_field_steps;
        this->my_velocity_stage=inverse_litography::optimize_shape_phase;
    }
    else
    {
        switch(this->my_inverse_litography_velocity)
        {
        case inverse_litography::pressure_velocity:
        {
            this->my_inverse_litography_stage=inverse_litography::move_pressure_and_exchange_field;

            this->start_remesh=this->total_number_of_mean_field_steps;
            this->stop_remesh=0;
            this->lambda_minus=1.00;
            this->lambda_plus=1.00;
            break;
        }
        case inverse_litography::pressure_velocity_with_curvature_constraints:
        {
            if(this->i_mean_field_in_current_cycle<this->n_scft)
            {
            this->my_inverse_litography_stage=inverse_litography::move_pressure_and_exchange_field;
            this->start_remesh=this->total_number_of_mean_field_steps;
            this->stop_remesh=0;
            this->lambda_minus=1.00;
            this->lambda_plus=1.00;
            this->i_mean_field_in_current_cycle++;
            }
            else
            {
                this->my_inverse_litography_stage=inverse_litography::move_pressure_and_level_set_field;
                this->i_mean_field_in_current_cycle=0;
                this->lambda_minus=0.00;
                this->lambda_plus=1.00;
                this->start_remesh=0;
               // this->dynamic_f_A=true;
                this->stop_remesh=this->total_number_of_mean_field_steps;
                this->my_velocity_stage=inverse_litography::optimize_shape_phase;
            }
            break;
        }
        case inverse_litography::pressure_velocity_with_constraints:
        {
            if(this->smooth_counter>=this->max_smooth_counters)
            {
                this->my_inverse_litography_stage=inverse_litography::move_pressure_and_exchange_field;
                this->start_remesh=this->total_number_of_mean_field_steps;
                this->stop_remesh=0;
                this->lambda_minus=1.00;
                this->lambda_plus=1.00;

            }
            else
            {
                this->my_inverse_litography_stage=inverse_litography::move_level_set_field;
                this->start_remesh=0;
                this->stop_remesh=this->total_number_of_mean_field_steps;;
                this->lambda_minus=0.00;
                this->lambda_plus=0.00;
                this->my_velocity_stage=inverse_litography::smooth_shape_phase;
                this->smooth_counter++;
            }
            break;
        }

        }
    }
    lambda_p=this->lambda_plus*lambda_in;
    lambda_m=this->lambda_minus*lambda_in;
    start_remesh=this->start_remesh;
    stop_remesh=this->stop_remesh;
}


void inverse_litography::wp_av_strategy(int &start_remesh, int &stop_remesh, double &lambda_m,
                                           double lambda_in, double &lambda_p, double V0, double *Vt, double wp_av, double wp_std)
{
    int i_2Check=max(this->i_mean_field-1,0);
    double cut_off=V0/Vt[i_2Check];

    int i_mean_field_4_stability=100;
    int i_shift=10;
    //double tolerance_stability=0.0001;
    double check_stability=2.00;
    std::cout<<" check_stability "<<check_stability<<std::endl;
    if(i_2Check>(i_shift+i_mean_field_4_stability))
    {
        check_stability=Vt[i_2Check]/Vt[i_2Check-i_shift];

        std::cout<<" Vt[i_2Check] Vt[i_2Check-i_shift]"<< Vt[i_2Check]<<" "<<Vt[i_2Check-i_shift]<<std::endl;;

        check_stability=ABS(check_stability-1);

    }


    double wp_average_t=ABS(wp_av);



    std::cout<<" check_stability "<<check_stability<<std::endl;
    int min_mean_field_for_stopping_remesh=3;

    if( (cut_off<this->cut_r_spot && wp_average_t>this->wp_tolerance
         &&  this->my_inverse_litography_stage==inverse_litography::move_pressure_and_level_set_field&& this->smooth_counter==0)||
            this->i_mean_field<min_mean_field_for_stopping_remesh )

    {
        this->my_inverse_litography_stage=inverse_litography::move_pressure_and_level_set_field;
        this->i_mean_field_in_current_cycle=0;
        this->lambda_minus=0.00;
        this->lambda_plus=1.00;
        this->start_remesh=0;
        this->stop_remesh=this->total_number_of_mean_field_steps;
        this->my_velocity_stage=inverse_litography::optimize_shape_phase;
    }
    else
    {
        switch(this->my_inverse_litography_velocity)
        {
        case inverse_litography::pressure_velocity:
        {
            this->my_inverse_litography_stage=inverse_litography::move_pressure_and_exchange_field;

            this->start_remesh=this->total_number_of_mean_field_steps;
            this->stop_remesh=0;
            this->lambda_minus=1.00;
            this->lambda_plus=1.00;
            break;
        }
        case inverse_litography::pressure_velocity_with_curvature_constraints:
        {
            if(this->i_mean_field_in_current_cycle<this->n_scft)
            {
            this->my_inverse_litography_stage=inverse_litography::move_pressure_and_exchange_field;
            this->start_remesh=this->total_number_of_mean_field_steps;
            this->stop_remesh=0;
            this->lambda_minus=1.00;
            this->lambda_plus=1.00;
            this->i_mean_field_in_current_cycle++;
            }
            else
            {
                this->my_inverse_litography_stage=inverse_litography::move_pressure_and_level_set_field;
                this->i_mean_field_in_current_cycle=0;
                this->lambda_minus=0.00;
                this->lambda_plus=1.00;
                this->start_remesh=0;
               // this->dynamic_f_A=true;
                this->wp_tolerance=10*this->wp_tolerance;
                this->stop_remesh=this->total_number_of_mean_field_steps;
                this->my_velocity_stage=inverse_litography::optimize_shape_phase;
            }
            break;
        }
        case inverse_litography::pressure_velocity_with_constraints:
        {
            if(this->smooth_counter>=this->max_smooth_counters)
            {
                this->my_inverse_litography_stage=inverse_litography::move_pressure_and_exchange_field;
                this->start_remesh=this->total_number_of_mean_field_steps;
                this->stop_remesh=0;
                this->lambda_minus=1.00;
                this->lambda_plus=1.00;

            }
            else
            {
                this->my_inverse_litography_stage=inverse_litography::move_level_set_field;
                this->start_remesh=0;
                this->stop_remesh=this->total_number_of_mean_field_steps;;
                this->lambda_minus=0.00;
                this->lambda_plus=0.00;
                this->my_velocity_stage=inverse_litography::smooth_shape_phase;
                this->smooth_counter++;
            }
            break;
        }

        }
    }
    lambda_p=this->lambda_plus*lambda_in;
    lambda_m=this->lambda_minus*lambda_in;
    start_remesh=this->start_remesh;
    stop_remesh=this->stop_remesh;
}



void inverse_litography::compute_litography_stage(int i_mean_field_step, double lambda_in,
                                                  double &lambda_m, double &lambda_p, int &start_remesh, int &stop_remesh,
                                                  double V0, double *Vt,
                                                  double wp_av, double wp_std)
{
this->i_mean_field=i_mean_field_step;
    (this->*my_algo_decision)(start_remesh, stop_remesh, lambda_m, lambda_in, lambda_p,V0,Vt,wp_av,wp_std);
}

