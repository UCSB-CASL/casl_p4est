#ifndef MY_P4EST_KMEANS_H
#define MY_P4EST_KMEANS_H


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

#ifdef P4_TO_P8
struct BCCGenerator2:CF_3{
  BCCGenerator2(double r,double L)
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
          return this->r-pow(min_distance,0.5);

  }
private:
  double r,L;
};
#else
struct BCCGenerator2:CF_2
{
  BCCGenerator2(double r,double L):r(r), L(L)
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
        {
            return 1;//this->r-pow(min_distance,0.5);
        }
        else
        {
            return -1;//-(pow(min_distance,0.5)-this->r);
        }

    }
private:
  double r,L;
};
#endif


class my_p4est_kmeans
{

private:
    Vec Ibin;
    Vec ix1;
    Vec ix2;
    Vec Icell;
    double *phi,*phi_cell;
    Vec phi_global;
    Vec *I1;
    double c1;
    double c2;


    double c1_global;
    double c2_global;

    double A1,A2;
    double A1_global,A2_global;
    p4est_t            *p4est; // the forest itself
    p4est_nodes_t      *nodes; // the nodes
    PetscErrorCode      ierr;//
    p4est_connectivity_t *connectivity;//
    my_p4est_brick_t brick;//
    p4est_ghost_t* ghost;
    mpi_context_t mpi_context, *mpi;
    bool debug=true;
     double e1=0; double e2=0;


     double e1_global,e2_global;
     double e1_global_2,e2_global_2;
     int kmeans_iterator=0;

public:
    my_p4est_kmeans(int argc, char* argv[]);


    void segmentKmeans();
    void computeSegmentationError();



    std::string IO_path="/Users/gaddielouaknin/p4estLocal/";
    inline std::string convert2FullPath(std::string file_name)
    {
      std::stringstream oss;
      std::string mystr;
      oss <<this->IO_path <<file_name;
      mystr=oss.str();
      return mystr;
    }
    void printForestNodes2TextFile();
    void printForestOctants2TextFile();    
    void printForestQNodes2TextFile();
    void printGhostNodes();
    void printGhostCells();


    void petscGames();
};

#endif // MY_P4EST_KMEANS_H


