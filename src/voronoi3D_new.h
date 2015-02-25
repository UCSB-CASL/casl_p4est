#ifndef CASL_VORONOI3D_NEW_H
#define CASL_VORONOI3D_NEW_H

#include <float.h>
#include <fstream>
#include <vector>

#include <src/my_p8est_utils.h>
#include <src/CASL_math.h>
#include <src/point3.h>

using std::vector;

struct Voronoi3D_NEWPoint
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

  void operator=(Voronoi3D_NEWPoint v)
  {
    n = v.n; p = v.p; s=v.s;
  }
};


typedef struct candidate
{
  int n;
  Point3 p;
  Point3 norm;

  /* basis for the plane */
  Point3 u;
  Point3 v;

  vector<Point3> polygon;

} candidate_t;


/*!
 * \brief The Voronoi3D class construct a Voronoi partition for a point given its surrounding points.
 * This class also provides functions to compute the volume of the Voronoi partition around the point.
 */
class Voronoi3D_NEW
{
private:
  double big;

  Point3 pc;
  int nc;
  vector<Voronoi3D_NEWPoint> points;
  vector<candidate_t> partitions;
  double volume_;

  double xmin, ymin, zmin;
  double xmax, ymax, zmax;

  void init_polygon(int n);

  /* return true if n was entirely deleted, false otherwise */
  bool cut_polygon(int n, int c);

public:
  static int cpt_restart;
  /*!
     * \brief default constructor for the Voronoi2D class
     */
  Voronoi3D_NEW(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
  {
    pc.x=DBL_MAX; pc.y=DBL_MAX; pc.z=DBL_MAX;
    this->xmin=xmin; this->ymin=ymin; this->zmin=zmin;
    this->xmax=xmax; this->ymax=ymax; this->zmax=zmax;
    big = 2*MAX(xmin-xmax, ymin-ymax, zmin-zmax);
  }

  void get_partitions(vector<candidate_t> &part)
  {
    part = partitions;
  }

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
  void get_Points( const vector<Voronoi3D_NEWPoint>*& points) const;

  /*!
     * \brief set the level-set values at the vertices of the voronoi partition
     * \param phi_values the list of the level-set values
     */
  void set_Level_Set_Values( vector<double>& phi_values );

  /*!
     * \brief set the point at the center of the partition
     * \param pc the coordinates of the point
     */
  void set_Center_Point( int nc, Point3 &pc );

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
  void construct_Partition();

  inline double volume() const { return this->volume_; }

  /*!
   * \brief check that the voronoi partition is valid, i.e. if k is a neighbor of n, then n
   *   is also a neighbor of k, and the area connecting them is the same from both perspectives
   * \param voro the voronoi partition. Note that you must call "construct_Partition" for all elements first
   * \return true is the partition is valid, false otherwise
   */
  static bool check_Partition(const vector<Voronoi3D_NEW>& voro);

  /*!
     * \brief save the voronoi partition in the .vtk format
     * \param voro the list of voronoi partitions to save
     * \param file_name the file in which the voronoi partition is to be saved
     */
  static void print_VTK_Format( vector<Voronoi3D_NEW>& voro, const char* file_name );
};

#endif /* CASL_VORONOI3D_H */
