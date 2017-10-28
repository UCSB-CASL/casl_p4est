/* 
 * Title: electroporation
 * Description:
 * Author:
 * Date Created: 09-22-2016
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#include <src/my_p4est_electroporation_solve.h>
#include <src/voronoi2D.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
#endif

#include <src/Parser.h>
#include <src/math.h>

using namespace std;



int test = 5;

/* 0 or 1 */
int implicit = 1;
/* order 1, 2 or 3. If choosing 3, implicit only */
int order = 3;

/* number of cells in x and y dimensions */
int x_cells = 1;
int y_cells = 1;

/* cell radius */
double r0 = 50e-6;
double xmin = test<6 ? -2*x_cells*r0 : -1e-3;
double xmax = test<6 ?  2*x_cells*r0 :  1e-3;
double ymin = test<6 ? -2*y_cells*r0 : -1e-3;
double ymax = test<6 ?  2*y_cells*r0 :  1e-3;

int lmin = 2;
int lmax = 5;
int nb_splits = 1;

double dt_scale = 40;

double tn;
double tf = 1e-6;
double dt; // = 20e-9;

double E_unscaled = 40; /* kv */
double E = E_unscaled * 1e3 * (xmax-xmin);
//double E = 40e3 * (xmax-xmin);

double sigma_c = 1;
double sigma_e = 15;

double Cm = test<=2 ? 0 : 9.5e-3;

double SL = 1.9;
double S0 = 1.1e6;
double S1 = 1e4;
double X_0 = 0;
double X_1 = 0;

double Vep = 258e-3;
double Xep = 0.5;

double tau_ep   = 1e-6;
double tau_perm = 1e-6;
double tau_res  = 60;

double R1 = .25*MIN(xmax-xmin, ymax-ymin);
double R2 =  3*MAX(xmax-xmin, ymax-ymin);

bool save_vtk = true;
bool save_error = true;
int save_every_n = 1;
bool check_partition = false;
bool save_voro = true;
bool save_stats = true;



class LevelSet : public CF_2
{
public:
    LevelSet(){lip=1.2;}
    double operator()(double x, double y) const
    {
        double d = DBL_MAX;

        double xm, ym, zm; xm=ym=zm=-.5e-3;
        double dx = 1e-3/(x_cells+1);
        double dy = 1e-3/(y_cells+1);

        switch(test)
        {

        case 0: return sqrt(SQR(x) + SQR(y)) - r0;
        case 1: return sqrt(SQR(x) + SQR(y)) - R1;
        case 2: return sqrt(SQR(x) + SQR(y)) - r0;
        case 3: return sqrt(SQR(x) + SQR(y)) - r0;
        case 4: return sqrt(SQR(x) + SQR(y)) - R1;
        case 5:
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    d = MIN(d, sqrt(SQR(x-(xmin+i*4*r0+2*r0)) + SQR(y-(ymin+j*4*r0+2*r0))) - r0);
            return d;
        case 6:
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    d = MIN(d, sqrt(SQR(x-(xm+(i+1)*dx)) + SQR(y-(ym+(j+1)*dy))) - r0);
            return d;
        default: throw std::invalid_argument("Choose a valid test.");
        }
    }
} level_set;

double u_exact(double x, double y, double t)
{
    double r = sqrt(x*x + y*y);
    double theta = atan2(y,x);
    double g = E*R2;
    if(test==1)
    {
        double alpha_c = g/((sigma_c/(SL*R1)+1+sigma_c/sigma_e)*R2 + (sigma_c/(SL*R1)+1-sigma_c/sigma_e)*R1*R1/R2);
        double alpha_e = .5*(sigma_c/(SL*R1)+1+sigma_c/sigma_e)*alpha_c;
        double beta_e = .5*(sigma_c/(SL*R1)+1-sigma_c/sigma_e)*alpha_c*R1*R1;
        return (level_set(x,y)>0 ? (alpha_e*r+beta_e/r)*cos(theta) : alpha_c*r*cos(theta));
    }
    else if(test==4)
    {
        double K = -sigma_e/(R1*R1*(sigma_e-sigma_c)+R2*R2*(sigma_e+sigma_c));

        double A = -2*sigma_c*R2*K;
        double B = sigma_c*(R1*R1+R2*R2)/R1*K;
        double vv = A/(SL-B)*g*(1-exp((B-SL)/Cm*t));

        double alpha_e = -R2*(sigma_c+sigma_e)/sigma_e*K*g + R1*sigma_c/sigma_e*K*vv;
        double beta_e = R1*R1*R2*(sigma_c-sigma_e)/sigma_e*K*g - R1*R2*R2*sigma_c/sigma_e*K*vv;
        double alpha_c = -2*R2*K*g + (R1*R1+R2*R2)/R1*K*vv;

        return level_set(x,y)>0 ? (alpha_e*r + beta_e/r)*cos(theta) : alpha_c*r*cos(theta);
    }
    else
        return 0;
}

double v_exact(double x, double y, double tn)
{
    double theta = atan2(y,x);
    double g = E*R2;
    /* static case coefficient, test 1 */
    double alpha_c = g/((sigma_c/(SL*R1)+1+sigma_c/sigma_e)*R2 + (sigma_c/(SL*R1)+1-sigma_c/sigma_e)*R1*R1/R2);
    /* dynamic case coefficient, test 4 */
    double K = -sigma_e/(R1*R1*(sigma_e-sigma_c)+R2*R2*(sigma_e+sigma_c));
    double A = -2*sigma_c*R2*K;
    double B = sigma_c*(R1*R1+R2*R2)/R1*K;

    if(test==1)
    {
        return sigma_c/SL * alpha_c*cos(theta);
    }
    else if(test==4)
    {
        return A/(SL-B)*g*(1-exp(-(SL-B)/Cm*tn))*cos(theta);
    }
    else
        return 0;
}
struct BCWALLTYPE : WallBC2D
{
    BoundaryConditionType operator()(double x, double y) const
    {
        switch(test)
        {

        case 1: return DIRICHLET;
        case 2:
            if(ABS(x-xmin)<EPS || ABS(x-xmax)<EPS) return DIRICHLET;
            else                                           return NEUMANN;
        case 3:
            if(ABS(x-xmin)<EPS || ABS(x-xmax)<EPS) return DIRICHLET;
            else                                           return NEUMANN;
        case 4: return DIRICHLET;
        case 5:
            if(ABS(x-xmin)<EPS || ABS(x-xmax)<EPS) return DIRICHLET;
            else                                           return NEUMANN;
        case 6:
            if(ABS(x-xmin)<EPS || ABS(x-xmax)<EPS) return DIRICHLET;
            else                                           return NEUMANN;
        default: throw std::invalid_argument("Choose a valid test.");
        }
    }
} bc_wall_type_p;

struct BCWALLVALUE : CF_2
{
    double operator()(double x, double y) const
    {
        switch(test)
        {

        case 1: return u_exact(x,y,0);
        case 2:
            if(ABS(x-xmin)<EPS) return E;
            if(ABS(x-xmax)<EPS) return 0;
            return 0;
        case 3:
            if(ABS(x-xmin)<EPS) return E;
            if(ABS(x-xmax)<EPS) return 0;
            return 0;
        case 4: return u_exact(x,y,t);
        case 5:
            if(ABS(x-xmin)<EPS) return E;
            if(ABS(x-xmax)<EPS) return 0;
            return 0;
        case 6:
            if(ABS(x-xmin)<EPS) return E;
            if(ABS(x-xmax)<EPS) return 0;
            return 0;
        default: throw std::invalid_argument("Choose a valid test.");
        }
    }
} bc_wall_value_p;

double sigma(double x, double y)
{
    return level_set(x,y)<0 ? sigma_c : sigma_e;
}

class SIGMA : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return level_set(x,y)<=0 ? sigma_c : sigma_e;
    }
}sigma_in;

class BETA_0 : public CF_1
{
public:
    double operator ()(double lambda) const
    {
        return exp(-Vep*Vep/(lambda*lambda));
    }
}beta_0_in;

class BETA_1 : public CF_1
{
public:
    double operator ()(double lambda) const
    {
        return exp(-Xep*Xep/(lambda*lambda));
    }
}beta_1_in;




class Initial_U : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return 0;
    }
} initial_u;

class Initial_Vnm1 : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        if (test==4)
        {

            double K = -sigma_e/(R1*R1*(sigma_e-sigma_c)+R2*R2*(sigma_e+sigma_c));
            double A = -2*sigma_c*R2*K;
            double B = sigma_c*(R1*R1+R2*R2)/R1*K;
            double g = E*R2;
            double theta = atan2(y,x);

            if(order>1) return A/(SL-B)*g*(1-exp(-(SL-B)/Cm*(-1*dt)))*cos(theta);
            else        return 0;
        }
        else        return 0;
    }
} initial_vnm1;


class Initial_Vnm2 : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        if (test==4)
        {

            double K = -sigma_e/(R1*R1*(sigma_e-sigma_c)+R2*R2*(sigma_e+sigma_c));
            double A = -2*sigma_c*R2*K;
            double B = sigma_c*(R1*R1+R2*R2)/R1*K;
            double g = E*R2;
            double theta = atan2(y,x);

            if(order>2) return A/(SL-B)*g*(1-exp(-(SL-B)/Cm*(-2*dt)))*cos(theta);
            else        return 0;
        }
        else        return 0;
    }
} initial_vnm2;

class Initial_Vn : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return 0;
    }
} initial_vn;


class Initial_X0 : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return X_0;
    }
} initial_x0;

class Initial_X1 : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return X_1;
    }
} initial_x1;

class Initial_Sm : public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return SL;
    }
} initial_sm;











class MU_M: public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return sigma_c;
    }
} mu_m;

class MU_P: public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return sigma_e;
    }
} mu_p;

struct U_M : CF_2
{
    double operator()(double x, double y) const
    {
        return 0;
    }
} u_m;

struct U_P : CF_2
{
    double operator()(double x, double y) const
    {
        return 0;
    }
} u_p;

class U_JUMP: public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return u_p(x,y) - u_m(x,y);
    }
} u_jump;



double grad_u_m(double x, double y)
{
    return 0;
}

double grad_u_p(double x, double y)
{
    return 0;
}


class MU_GRAD_U_JUMP: public CF_2
{
public:
    double operator()(double x, double y) const
    {
        return mu_p(x,y)*grad_u_p(x,y) - mu_m(x,y)*grad_u_m(x,y);
    }
} mu_grad_u_jump;










void solve_Poisson_Jump( p4est_t *p4est, p4est_nodes_t *nodes,
                         my_p4est_node_neighbors_t *ngbd_n, my_p4est_cell_neighbors_t *ngbd_c,
                         Vec phi, Vec sol, double dt, Vec X0, Vec X1, Vec Sm, Vec vn, my_p4est_level_set_t ls, double tn, Vec vnm1, Vec vnm2)
{
    PetscErrorCode ierr;

    Vec rhs_m, rhs_p;
    Vec mu_m_, mu_p_;
    Vec u_jump_;
    Vec mu_grad_u_jump_;
    ierr = VecDuplicate(phi, &rhs_m); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs_p); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_m_); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_p_); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &u_jump_); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_grad_u_jump_); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mu_m, mu_m_);
    sample_cf_on_nodes(p4est, nodes, mu_p, mu_p_);
    sample_cf_on_nodes(p4est, nodes, u_jump, u_jump_);
    sample_cf_on_nodes(p4est, nodes, mu_grad_u_jump, mu_grad_u_jump_);




    Vec u;
    ierr = VecDuplicate(phi, &u); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, initial_u, u);




    double *rhs_m_p, *rhs_p_p;
    ierr = VecGetArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        rhs_m_p[n] = 0;
        rhs_p_p[n] = 0;

    }
    ierr = VecRestoreArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);



#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif
    bc_wall_value_p.t = tn+dt;
    bc.setWallTypes(bc_wall_type_p);
    bc.setWallValues(bc_wall_value_p);

    my_p4est_electroporation_solve_t solver(ngbd_n, ngbd_c);







    solver.set_beta0(beta_0_in);
    solver.set_beta1(beta_1_in);
    solver.set_Sm(Sm);
    solver.set_X0(X0);
    solver.set_X1(X1);
    solver.set_parameters(implicit, order, dt, test, SL, tau_ep, tau_res, tau_perm, S0, S1, tn);
    solver.t = tn+dt;
    solver.dt = dt;
    solver.Cm = Cm;
    solver.set_vnm1(vnm1);
    solver.set_vn(vn);
    solver.set_vnm2(vnm2);
    solver.set_phi(phi);
    solver.set_bc(bc);
    solver.set_mu(mu_m_, mu_p_);
    //    solver.set_u_jump(vn);
    //    solver.set_u_jump(u_jump_);
    solver.set_mu_grad_u_jump(mu_grad_u_jump_);
    solver.set_rhs(rhs_m, rhs_p);

    Vec X_0_v, X_1_v, l, l0, l1;
    ierr = VecDuplicate(phi, &X_0_v); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &X_1_v); CHKERRXX(ierr);

    ierr = VecGhostGetLocalForm(X_0_v, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(X0, &l0); CHKERRXX(ierr);
    ierr = VecCopy(l0, l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X_0_v, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X0, &l0); CHKERRXX(ierr);

    ierr = VecGhostGetLocalForm(X_1_v, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(X1, &l1); CHKERRXX(ierr);
    ierr = VecCopy(l1, l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X_1_v, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X1, &l1); CHKERRXX(ierr);
    double convergence_Sm;
    convergence_Sm = 0;
    do
    {

        solver.solve(sol);

        if(order>2)
        {
            Vec vnm2_l, vnm1_l;
            VecGhostGetLocalForm(vnm1, &vnm1_l);
            VecGhostGetLocalForm(vnm2, &vnm2_l);
            ierr = VecCopy(vnm1_l, vnm2_l); CHKERRXX(ierr);
            VecGhostRestoreLocalForm(vnm1, &vnm1_l);
            VecGhostRestoreLocalForm(vnm2, &vnm2_l);
        }
        if(order>1)
        {
            Vec vn_l, vnm1_l;
            VecGhostGetLocalForm(vnm1, &vnm1_l);
            VecGhostGetLocalForm(vn, &vn_l);
            ierr = VecCopy(vn_l, vnm1_l); CHKERRXX(ierr);
            VecGhostRestoreLocalForm(vnm1, &vnm1_l);
            VecGhostRestoreLocalForm(vn, &vn_l);
        }

        // compute jump
        Vec u_plus_ext, u_minus_ext, u_plus_ext_l, u_minus_ext_l, sol_l;
        ierr = VecDuplicate(sol, &u_plus_ext); CHKERRXX(ierr);
        ierr = VecDuplicate(sol, &u_minus_ext); CHKERRXX(ierr);
        VecGhostGetLocalForm(sol, &sol_l);
        VecGhostGetLocalForm(u_plus_ext, &u_plus_ext_l);
        VecGhostGetLocalForm(u_minus_ext, &u_minus_ext_l);
        ierr = VecCopy(sol_l, u_plus_ext_l); CHKERRXX(ierr);
        ierr = VecCopy(sol_l, u_minus_ext_l); CHKERRXX(ierr);
        VecGhostRestoreLocalForm(sol, &sol_l);
        VecGhostRestoreLocalForm(u_plus_ext, &u_plus_ext_l);
        VecGhostRestoreLocalForm(u_minus_ext, &u_minus_ext_l);


        ls.extend_Over_Interface(phi, u_plus_ext, 2, 1);
        Vec phi_l;
        VecGhostGetLocalForm(phi, &phi_l);
        ierr = VecScale(phi_l, -1);CHKERRXX(ierr);
        ls.extend_Over_Interface(phi, u_minus_ext, 2, 1);
        ierr = VecScale(phi_l, -1);CHKERRXX(ierr);
        VecGhostRestoreLocalForm(phi, &phi_l);
        Vec vn_l;
        VecGhostGetLocalForm(u_minus_ext, &u_minus_ext_l);
        VecGhostGetLocalForm(u_plus_ext, &u_plus_ext_l);
        ierr = VecAXPY(u_minus_ext_l, -1.0, u_plus_ext_l); CHKERRXX(ierr);
        VecGhostRestoreLocalForm(u_minus_ext, &u_minus_ext_l);
        VecGhostRestoreLocalForm(u_plus_ext, &u_plus_ext_l);
        VecGhostGetLocalForm(u_minus_ext, &u_minus_ext_l);
        VecGhostGetLocalForm(vn, &vn_l);
        ierr = VecCopy(u_minus_ext_l, vn_l); CHKERRXX(ierr);
        VecGhostRestoreLocalForm(u_minus_ext, &u_minus_ext_l);
        VecGhostRestoreLocalForm(vn, &vn_l);

//                ls.extend_Over_Interface(phi, vn, 1, 1);
//        ls.average_Onto_Interface(phi, sol, u_plus_ext, u_minus_ext, vn, 1);






        // compute X and Sm
        if(test==1 || test==2 || test==4)
        {
            Vec l;
            ierr = VecGhostGetLocalForm(Sm, &l); CHKERRXX(ierr);
            ierr = VecSet(l, SL); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(Sm, &l); CHKERRXX(ierr);

            ierr = VecGhostGetLocalForm(X0, &l); CHKERRXX(ierr);
            ierr = VecSet(l, 0); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X0, &l); CHKERRXX(ierr);

            ierr = VecGhostGetLocalForm(X1, &l); CHKERRXX(ierr);
            ierr = VecSet(l, 0); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X1, &l); CHKERRXX(ierr);
            convergence_Sm = 1e-4;
        }
        else if(test==3)
        {
            for(unsigned int n=0; n<nodes->num_owned_indeps; ++n)
            {
                //        convergence_Sm = MAX(convergence_Sm, ABS(Sm[n] - (SL + (Sir-SL)*beta(vn_n_p[n])))/Sm[n] );
                //        Sm[n] = SL + (Sir-SL)*beta(vn_n_p[n]);
            }
            convergence_Sm = 1e-4;
        }
        else
        {

            double *vn_n_p, *Sm_n_p, *X0_np1, *X1_np1;
            ierr = VecGetArray(Sm, &Sm_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(vn, &vn_n_p); CHKERRXX(ierr);

            ierr = VecGetArray(X0, &X0_np1); CHKERRXX(ierr);
            ierr = VecGetArray(X1, &X1_np1); CHKERRXX(ierr);





            double *X_0_v_p, *X_1_v_p;
            ierr = VecGetArray(X_0_v, &X_0_v_p); CHKERRXX(ierr);
            ierr = VecGetArray(X_1_v, &X_1_v_p); CHKERRXX(ierr);
            for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
            {
                X_0_v_p[n] = X0_np1[n] + dt*((beta_0_in(vn_n_p[n]) - X_0_v_p[n])/tau_ep);
                X_1_v_p[n] = X1_np1[n] + dt*MAX( (beta_1_in(X_0_v_p[n])-X_1_v_p[n])/tau_perm, (beta_1_in(X_0_v_p[n])-X_1_v_p[n])/tau_res );


                convergence_Sm = MAX(convergence_Sm, ABS(Sm_n_p[n] - (SL + S0*X_0_v_p[n] + S1*X_1_v_p[n])) );

                Sm_n_p[n] = SL + S0*X_0_v_p[n] + S1*X_1_v_p[n];

            }



            ierr = VecRestoreArray(Sm, &Sm_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X0,&X0_np1); CHKERRXX(ierr);
            ierr = VecRestoreArray(X1,&X1_np1); CHKERRXX(ierr);
            ierr = VecRestoreArray(vn, &vn_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X_0_v, &X_0_v_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X_1_v, &X_1_v_p); CHKERRXX(ierr);

            ierr = VecGhostGetLocalForm(X_0_v, &l); CHKERRXX(ierr);
            ierr = VecGhostGetLocalForm(X0, &l0); CHKERRXX(ierr);
            ierr = VecCopy(l, l0); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X_0_v, &l); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X0, &l0); CHKERRXX(ierr);

            ierr = VecGhostGetLocalForm(X_1_v, &l); CHKERRXX(ierr);
            ierr = VecGhostGetLocalForm(X1, &l1); CHKERRXX(ierr);
            ierr = VecCopy(l, l1); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X_1_v, &l); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X1, &l1); CHKERRXX(ierr);

        }
    }while(0 && convergence_Sm>1e-3);








    if(check_partition)
        solver.check_voronoi_partition();

    char out_path[1000];
    char *out_dir = NULL;
    out_dir = getenv("OUT_DIR");
    if(out_dir==NULL)
    {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save stats\n"); CHKERRXX(ierr);
    }
    else
    {
        if(save_stats)
        {
            sprintf(out_path, "%s/stats.dat", out_dir);
            solver.write_stats(out_path);
        }
        if(save_voro)
        {
            snprintf(out_path,1000, "%s/voronoi", out_dir);
            solver.print_voronoi_VTK(out_path);
            PetscPrintf(p4est->mpicomm, "HERE!\n");
        }
    }


    ierr = VecDestroy(rhs_m); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_p); CHKERRXX(ierr);
    ierr = VecDestroy(mu_m_); CHKERRXX(ierr);
    ierr = VecDestroy(mu_p_); CHKERRXX(ierr);
    ierr = VecDestroy(u_jump_); CHKERRXX(ierr);
    ierr = VecDestroy(mu_grad_u_jump_); CHKERRXX(ierr);
}



void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err,
              int compt, Vec X0, Vec X1, Vec Sm, Vec vn)
{
    PetscErrorCode ierr;
    char *out_dir = NULL;
    out_dir = getenv("OUT_DIR");
    if(out_dir==NULL)
    {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save visuals\n"); CHKERRXX(ierr);
        return;
    }

    std::ostringstream oss;

    oss << out_dir << "/jump_"
        << p4est->mpisize << "_"
        << brick->nxyztrees[0] << "x"
        << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
           "x" << brick->nxyztrees[2] <<
       #endif
           "." << compt;

    double *phi_p, *sol_p, *err_p, *X0_p, *X1_p, *Sm_p, *vn_p;
    ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecGetArray(X0, &X0_p); CHKERRXX(ierr);
    ierr = VecGetArray(X1, &X1_p); CHKERRXX(ierr);
    ierr = VecGetArray(Sm, &Sm_p); CHKERRXX(ierr);
    Vec mu;
    ierr = VecDuplicate(phi, &mu); CHKERRXX(ierr);
    double *mu_p_;
    ierr = VecGetArray(mu, &mu_p_); CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        mu_p_[n] = phi_p[n]<0 ? mu_m(x,y,z) : mu_p(x,y,z);
#else
        mu_p_[n] = phi_p[n]<0 ? mu_m(x,y) : mu_p(x,y);
#endif
    }
    /* save the size of the leaves */
    Vec leaf_level;
    ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
    double *l_p;
    ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
            const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
            l_p[tree->quadrants_offset+q] = quad->level;
        }
    }

    for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
    {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        l_p[p4est->local_num_quadrants+q] = quad->level;
    }
    PetscReal val;
    ierr = VecMax(vn,NULL,&val); CHKERRXX(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "maximum jump is: %g\n", (double)val); CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           8, 1, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "mu", mu_p_,
                           VTK_POINT_DATA, "sol", sol_p,
                           VTK_POINT_DATA, "X0", X0_p,
                           VTK_POINT_DATA, "X1", X1_p,
                           VTK_POINT_DATA, "vn", vn_p,
                           VTK_POINT_DATA, "Sm", Sm_p,
                           VTK_POINT_DATA, "err", err_p,
                           VTK_CELL_DATA , "leaf_level", l_p);

    ierr = VecRestoreArray(mu, &mu_p_); CHKERRXX(ierr);
    ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
    ierr = VecDestroy(mu); CHKERRXX(ierr);
    ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(Sm, &Sm_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(X0, &X0_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(X1, &X1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(vn, &vn_p); CHKERRXX(ierr);
    PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



int main(int argc, char** argv) {
    PetscErrorCode ierr;
    // prepare parallel enviroment
    mpi_environment_t mpi;
    mpi.init(argc, argv);

    // stopwatch
    parStopWatch w;
    w.start("Running example: electroporation");

    // p4est variables
    p4est_t*              p4est;
    p4est_nodes_t*        nodes;
    p4est_ghost_t*        ghost;
    p4est_connectivity_t* conn;
    my_p4est_brick_t      brick;

    // domain size information
    const int n_xyz []      = {1, 1, 1};
    const double xyz_min [] = {xmin, ymin};
    const double xyz_max [] = {xmax, ymax};
    int periodic[] = {0, 0, 0};
    conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

    for(int iter=0; iter<nb_splits; ++iter)
    {
        ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);

        // create the forest
        p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

        // refine based on distance to a level-set
        splitting_criteria_cf_t sp(lmin+iter, lmax+iter, &level_set, 1.2);
        p4est->user_pointer = &sp;
        for(int i=0; i<lmax+iter; ++i)
        {
            my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
            my_p4est_partition(p4est, P4EST_FALSE, NULL);
        }

        /* create the initial forest at time nm1 */
        p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
        my_p4est_partition(p4est, P4EST_FALSE, NULL);

        // create ghost layer at time nm1
        ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
        my_p4est_ghost_expand(p4est, ghost);
        // create node structure at time nm1
        nodes = my_p4est_nodes_new(p4est, ghost);

        if(p4est->mpirank==0)
        {
            p4est_gloidx_t nb_nodes = 0;
            for(int r=0; r<p4est->mpisize; ++r)
                nb_nodes += nodes->global_owned_indeps[r];
            ierr = PetscPrintf(p4est->mpicomm, "number of nodes : %d\n", nb_nodes); CHKERRXX(ierr);
        }

        my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);

        my_p4est_cell_neighbors_t ngbd_c(&hierarchy);
        my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);





        /* initialize the variables */

        Vec phi, X0, X1, Sm, vn;
        ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, level_set, phi);


        ierr = VecDuplicate(phi, &X0); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &X1); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &Sm); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, initial_x0, X0);
        sample_cf_on_nodes(p4est, nodes, initial_x1, X1);
        sample_cf_on_nodes(p4est, nodes, initial_sm, Sm);
        ierr = VecDuplicate(phi, &vn); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, initial_vn, vn);

        Vec vnm1, vnm2;
        ierr = VecDuplicate(phi, &vnm1); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &vnm2); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, initial_vnm1, vnm1);
        sample_cf_on_nodes(p4est, nodes, initial_vnm2, vnm2);


        /* perturb level set */
        my_p4est_level_set_t ls(&ngbd_n);
        ls.perturb_level_set_function(phi, EPS);



        /* set initial time step */
        p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
        p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
        double xminn = p4est->connectivity->vertices[3*vm + 0];
        double yminn = p4est->connectivity->vertices[3*vm + 1];
        double xmaxx = p4est->connectivity->vertices[3*vp + 0];
        double ymaxx = p4est->connectivity->vertices[3*vp + 1];
        double dx = (xmaxx-xminn) / pow(2., (double) sp.max_lvl);
        double dy = (ymaxx-yminn) / pow(2., (double) sp.max_lvl);
#ifdef P4_TO_P8
        double zminn = p4est->connectivity->vertices[3*vm + 2];
        double zmaxx = p4est->connectivity->vertices[3*vp + 2];
        double dz = (zmaxx-zminn) / pow(2.,(double) sp.max_lvl);
#endif

#ifdef P4_TO_P8
        double dt = MIN(dx,dy,dz)/dt_scale;
#else
        double dt = MIN(dx,dy)/dt_scale;

#endif

        //                dt = 1e-6;
        printf("initial dt=%g \n", dt);


        // loop over time
        int iteration = 0;


        FILE *fp;
        FILE *fp_err;
        char name[10000];
        char name_err[1000];
        if(save_error)
        {
            sprintf(name, "/home/pouria/Work/Electroporation_Output/2d/data.dat");
            ierr = PetscFOpen(mpi.comm(), name, "w", &fp); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(), fp, "%% time | avg S | avg poration | avg permeabilization | v_pole...\n"); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(), fp,  "%e", 0.); CHKERRXX(ierr);
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    ierr = PetscFPrintf(mpi.comm(), fp,  "\t %e\t %e\t %e\t %e", 0., 0., 0., 0.); CHKERRXX(ierr);

            ierr = PetscFClose(mpi.comm(), fp); CHKERRXX(ierr);

            if(test==4)
            {
                sprintf(name_err, "/home/pouria/Work/Electroporation_Output/2d/err_%d.dat", nb_splits);
                ierr = PetscFOpen(mpi.comm(), name_err, "w", &fp_err); CHKERRXX(ierr);
                ierr = PetscFClose(mpi.comm(), fp_err); CHKERRXX(ierr);
                ierr = PetscPrintf(mpi.comm(), "Saving error in %s\n", name_err); CHKERRXX(ierr);
            }
            ierr = PetscPrintf(mpi.comm(), "Saving data in %s\n", name); CHKERRXX(ierr);
        }


        double tn = 0;
        double err_n   = 0;
        double err_nm1 = 0;

        Vec sol;
        while(tn<tf)
        {
            ierr = PetscPrintf(mpi.comm(), "Iteration %d, time %e\n", iteration, tn); CHKERRXX(ierr);
            ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);
            solve_Poisson_Jump(p4est, nodes, &ngbd_n, &ngbd_c, phi, sol, dt, X0, X1, Sm, vn, ls,tn, vnm1, vnm2);
            PetscPrintf(p4est->mpicomm, "solved!\n");
            /* compute the error on the tree*/
            Vec err;
            ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
            double *err_p, *sol_p;
            ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
            ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
            err_nm1 = err_n;
            err_n = 0;
            for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
            {
                double x = node_x_fr_n(n, p4est, nodes);
                double y = node_y_fr_n(n, p4est, nodes);
                err_p[n] = fabs(u_exact(x,y,tn+dt) - sol_p[n]);
                err_n = max(err_n, err_p[n]);
            }

            MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
            PetscPrintf(p4est->mpicomm, "Iter %d : %g, \t order : %g\n", iteration, err_n, log(err_nm1/err_n)/log(2));
            //            PetscPrintf(p4est->mpicomm, "error at %g, %g, %g, qh = %g, dist_interface = %g\n", x_err, y_err, z_err, (double) 2/pow(2,lmax+iteration+1), fabs(sqrt(SQR(x_err-1) + SQR(y_err-1) + SQR(z_err-1))-r0));
            ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
            if(save_vtk && iteration%save_every_n == 0)
                save_VTK(p4est, ghost, nodes, &brick, phi, sol, err, iteration, X0, X1, Sm, vn);
            tn += dt;
            iteration++;
        }

        // save the grid into vtk
        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               0, 0, "electroporation");

        // destroy the structures
        p4est_nodes_destroy(nodes);
        p4est_ghost_destroy(ghost);
        p4est_destroy      (p4est);
    }
    my_p4est_brick_destroy(conn, &brick);

    w.stop(); w.read_duration();

}

