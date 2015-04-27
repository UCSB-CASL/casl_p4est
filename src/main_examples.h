#ifndef MAIN_EXAMPLES_H
#define MAIN_EXAMPLES_H


// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_utils.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_utils.h>
#endif

#include <src/CASL_math.h>
#include <src/petsc_compatibility.h>




class main_examples
{
public:

    /*
      Include "petscmat.h" so that we can use matrices.
      automatically includes:
         petscsys.h       - base PETSc routines   petscvec.h    - vectors
         petscmat.h    - matrices
         petscis.h     - index sets            petscviewer.h - viewers
    */
    main_examples();
    void parallel_vtk(int argc, char* argv[]);
    void simple_example(int argc, char* argv[]);


};

#ifdef P4_TO_P8
struct circle:CF_3{
  circle(double x0_, double y0_, double z0_, double r_)
    : x0(x0_), y0(y0_), z0(z0_), r(r_)
  {}
  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
private:
  double x0, y0, z0, r;
};
#else
struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};
#endif

#ifdef P4_TO_P8
struct simple_generator:CF_3{
  simple_generator()

  {}
  double operator()(double x, double y, double z) const {
    return 0;
  }

};
#else
struct simple_generator:CF_2{
  simple_generator()  {}
  double operator()(double x, double y) const {
    return 0;
  }

};
#endif

#ifdef P4_TO_P8
struct ellipse:CF_3{
  ellipse(double x0_, double y0_, double z0_, double r_,double a_,double b_,double c_)
      : x0(x0_), y0(y0_), z0(z0_), r(r_),a(a_),b(b_),c(c_)
  {}
  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR((x-x0)/a) + SQR((y-y0)/b) + SQR((z-z0)/c) );
  }
private:
  double x0, y0, z0, r,a,b,c;
};
#else
struct ellipse:CF_2
{
    ellipse(double x0_, double y0_, double r_,double a_,double b_): x0(x0_), y0(y0_), r(r_),a(a_),b(b_)
    {}
    double operator()(double x, double y) const
    {
        return r - sqrt( SQR((x-x0)/a) + SQR((y-y0)/b) );
    }
private:
  double x0, y0, r,a,b;
};
#endif



#ifdef P4_TO_P8
struct BCCGenerator:CF_3{
  BCCGenerator(double r,double L)
      : r(r), L(L)
  {}
  double operator()(double x, double y, double z) const {

      double *xc=new double[9];
      double *yc=new double[9];
      double *zc=new double[9];


      xc[0]=this->L/2.00;yc[0]=this->L/2.00;zc[0]=this->L/2.00;

      // (L,L,L) corner
      xc[1]=this->L;yc[1]=L;zc[1]=L;

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

      bool isInsideASphere=false;

      int i_sphere=-1;
      double min_distance=this->L*3;
      int im=-1;

      for(int ii=0;ii<9;ii++)
      {
          double d=pow(x-xc[ii],2)+pow(y-yc[ii],2)+pow(z-zc[ii],2);
          if( d<pow(this->r,2))
          {
              isInsideASphere=true;
              i_sphere=ii;
          }

          if(d<min_distance)
          {
              im=ii;
              min_distance=d;
          }
      }

      if(isInsideASphere)
          return this->r-pow(min_distance,0.5);
      else
          return -(pow(min_distance,0.5)-this->r);

  }
private:
  double r,L;
};
#else
struct BCCGenerator:CF_2
{
  BCCGenerator(double r,double L):r(r), L(L)
    {}
    double operator()(double x, double y) const
    {
        double *xc=new double[5];
        double *yc=new double[5];



        xc[0]=this->L/2.00;yc[0]=this->L/2.00;

        // (L,L) corner
        xc[1]=this->L;yc[1]=L;
        // (0,0) corner
        xc[2]=0;yc[2]=0;
        //(0,L) corner
        xc[3]=0;yc[3]=L;
        //(L,0)
        xc[4]=L;yc[4]=0;

        bool isInsideASphere=false;

        int i_sphere=-1;
        double min_distance=this->L*3;
        int im=-1;

        for(int ii=0;ii<5;ii++)
        {
            double d=pow(x-xc[ii],2)+pow(y-yc[ii],2);
            if( d<pow(this->r,2))
            {
                isInsideASphere=true;
                i_sphere=ii;
            }

            if(d<min_distance)
            {
                im=ii;
                min_distance=d;
            }
        }

        if(isInsideASphere)
            return this->r-pow(min_distance,0.5);
        else
            return -(pow(min_distance,0.5)-this->r);

    }
private:
  double r,L;
};
#endif


static int
simple_refine (p4est_t * p4est, p4est_topidx_t which_tree,
        p4est_quadrant_t * quadrant)
{
        return which_tree == 0 && quadrant->level < 2;
}



#endif // MAIN_EXAMPLES_H
