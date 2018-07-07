#ifndef CASL_VORONOI3D_H
#define CASL_VORONOI3D_H

#include <float.h>
#include <fstream>
#include <vector>

#include <src/my_p4est_utils.h>
#include <src/math.h>
#include <src/point3.h>

#include <voro++.hh>

//#include "libqhullcpp/Qhull.h"
//#include "libqhullcpp/QhullError.h"
//#include "libqhullcpp/RboxPoints.h"
//#include "libqhullcpp/QhullFacetList.h"


//#include "libqhullcpp/QhullQh.h"
//#include "libqhullcpp/QhullFacet.h"
//#include "libqhull_r/libqhull_r.h"




using std::vector;

struct VoroNgbd {
  voro::container* voronoi;
  voro::particle_order* po;
  VoroNgbd() : voronoi(NULL), po(NULL) {}
  ~VoroNgbd()
  {
    if(voronoi!=NULL) delete voronoi;
    if(po!=NULL) delete po;
  }
};

struct Voronoi3DPoint
{
  /*!
     * \brief the index of the point
     */
  int n;

  /*!
     * \brief the coordinates of the point
     */
  Point3 p;

  /*!
   * \brief the surface of the face separating the center point from this neighbor
   */
  double s;

  void operator=(Voronoi3DPoint v)
  {
    n = v.n; p = v.p; s=v.s;
  }
};

/*!
 * \brief The Voronoi3D class construct a Voronoi partition for a point given its surrounding points.
 * This class also provides functions to compute the volume of the Voronoi partition around the point.
 */
class Voronoi3D
{
private:
  Point3 pc;
  int nc;
  vector<Voronoi3DPoint> points;
  double volume;
  int *adj, *adji; /*facets and facet index */
  int nadj, nadjpos; /*number of facets */
public:
  /*!
     * \brief default constructor for the Voronoi2D class
     */
  Voronoi3D() { pc.x=DBL_MAX; pc.y=DBL_MAX; pc.z=DBL_MAX; }

  /*!
     * \brief reset the voronoi partition
     */
  void clear();

  /*!
     * \brief get the partition after it has been built using construct_Partition
     * \param points the list of neighbor points used to create the partition
     * \param partition the list of vertices that define the cell around the center point
     *
     * The vertices of the voronoi partition associated with the point number m are m and m+1, i.e. points(m) corresponds to vertices partition(m) and partition(m+1).
     */
  void get_Points( const vector<Voronoi3DPoint>*& points) const;

  /*!
   * \brief set the voronoi cell with precomputed values
   * \param points the list of voronoi neighbors with their properties
   * \param the volume of the voronoi cell
   */
  void set_Points( vector<Voronoi3DPoint> &points, double volume );

  /*!
     * \brief set the level-set values at the vertices of the voronoi partition
     * \param phi_values the list of the level-set values
     */
  void set_Level_Set_Values( vector<double>& phi_values );

  /*!
     * \brief set the point at the center of the partition
     * \param pc the coordinates of the point
     */
  void set_Center_Point( int nc, Point3 &pc);

  /*!
     * \brief set the coordinates of the point at the center of the partition
     * \param x the first coordinate of the point
     * \param y the second coordinate of the point
     */
  void set_Center_Point( int nc, double x, double y, double z);

  /*!
     * \brief get the point at the center of the partition
     * \param pc the coordinates of the point
     */
  inline const Point3& get_Center_Point() const { return pc; }

  /*!
     * \brief add a point to the list of collocation points, making sure there is no repetition
     * \param n the index of the point to add
     * \param x the first coordinate of the point to add
     * \param y the second coordinate of the point to add
     */
  void push( int n, double x, double y, double z);
  void push( int n, Point3 &pt);

  /*!
     * \brief construct the voronoi parition around point pc using the neighborhood given in "points"
     */
  void construct_Partition(const double *xyz_min, const double *xyz_max, const bool *periodic);

 // int qh_eachvoronoi_mod (qhT * qh, FILE * fp, printvridgeT printvridge,
 //                         vertexT * atvertex, boolT visitall, qh_RIDGE innerouter,
 //                         boolT inorder, int*adj, int*adji, int nadjpos, int nadj);

  inline double get_volume() const { return this->volume; }

  /*!
     * \brief save the voronoi partition in the .vtk format
     * \param voro the list of voronoi partitions to save
     * \param file_name the file in which the voronoi partition is to be saved
     */
  static void print_VTK_Format(const std::vector<Voronoi3D>& voro, const char* file_name,
                                const double *xyz_min, const double *xyz_max, const bool *periodic, double *vn_p, double *sol_voro_p);

  static void print_VTK_Format(const std::vector<Voronoi3D>& voro, const char* file_name,
                                const double *xyz_min, const double *xyz_max, const bool *periodic);
};

#endif // CASL_VORONOI3D_H
