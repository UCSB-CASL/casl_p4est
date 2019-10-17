#ifndef VORONOI2D_H
#define VORONOI2D_H

#include <mpi.h>
#include <float.h>
#include <fstream>
#include <vector>

#include <src/my_p4est_utils.h>
#include <src/casl_math.h>
#include <src/point2.h>

using std::vector;

struct Voronoi2DPoint
{
  /*!
     * \brief the index of the point
     */
  int n;

  /*!
     * \brief the coordinates of the point
     */
  Point2 p;

  /*!
     * \brief the angle theta with the reference point
     */
  double theta;

  void operator=(Voronoi2DPoint v)
  {
    n = v.n; p = v.p; theta = v.theta;
  }
};

/*!
 * \brief The Voronoi2D class construct a Voronoi partition for a point given its surrounding points.
 * This class also provides functions to compute the volume of the Voronoi partition around the point.
 */
class Voronoi2D
{
private:
  Point2 pc;
  vector<Voronoi2DPoint> points;
  vector<Point2> partition;
  vector<double> phi_values;
  double phi_c;
  double volume;

public:
  /*!
     * \brief default constructor for the Voronoi2D class
     */
  Voronoi2D() { pc.x=DBL_MAX; pc.y=DBL_MAX; }

  /*!
     * \brief reset the voronoi partition
     */
  void clear();

  /*!
   * \brief operator = copy an existing Voronoi2D
   * \param voro the existing Voronoi2D to be copied
   */
  void operator=( const Voronoi2D& voro );

  /*!
     * \brief get the partition after it has been built using construct_Partition
     * \param points the list of neighbor points used to create the partition
     * \param partition the list of vertices that define the cell around the center point
     *
     * The vertices of the voronoi partition associated with the point number m are m and m+1, i.e. points(m) corresponds to vertices partition(m) and partition(m+1).
     */
  void get_Points( const vector<Voronoi2DPoint> *&points) const;
  void get_Partition( const vector<Point2> *&partition ) const;
  void get_Points( vector<Voronoi2DPoint> *&points);
  void get_Partition( vector<Point2> *&partition );

  /*!
     * \brief update the partition
     * \param partition the new partition
     */
  void set_Partition( vector<Point2>& partition );

  /*!
     * \brief set the precomputed voronoi cell
     * \param points the voronoi neighbors
     * \param partition the new partition
     * \param volume the volume/area of the partition
     */
  void set_Points_And_Partition( vector<Voronoi2DPoint>& points, vector<Point2>& partition, double volume );

  /*!
     * \brief set the level-set values at the vertices of the voronoi partition
     * \param ls a continuous description of the level-set function
     */
  void set_Level_Set_Values( const CF_2& ls );

  /*!
     * \brief set the level-set values at the vertices of the voronoi partition
     * \param phi_values the list of the level-set values
     * \param phi_c the value of the level-set at the center point
     */
  void set_Level_Set_Values( vector<double>& phi_values, double phi_c );

  /*!
     * \brief set the point at the center of the partition
     * \param pc the coordinates of the point
     */
  void set_Center_Point( Point2 pc );

  /*!
     * \brief set the coordinates of the point at the center of the partition
     * \param x the first coordinate of the point
     * \param y the second coordinate of the point
     */
  void set_Center_Point( double x, double y );

  /*!
     * \brief get the point at the center of the partition
     * \param pc the coordinates of the point
     */
  inline const Point2& get_Center_Point() const { return pc; }

  /*!
     * \brief add a point to the list of collocation points, making sure there is no repetition
     * \param n the index of the point to add
     * \param x the first coordinate of the point to add
     * \param y the second coordinate of the point to add
     */
  void push( int n, double x, double y );

  /*!
     * \brief modify the coordinates of the points to take in account the periodicity
     * \param p_x periodic in x
     * \param p_y periodic in y
     * \param xmin
     * \param xmax
     * \param ymin
     * \param ymax
     * example: if the center point is (0,1) for a domain [0 1]x[0 2] periodic in x, with a neighbor point
     *  that has coordinates (0.9,1), the point should be transformed to (-0.1,1)
     */
  void enforce_Periodicity( bool p_x, bool p_y, double xmin, double xmax, double ymin, double ymax );

  /*!
     * \brief construct the voronoi parition around point pc using the neighborhood given in "points"
     */
  void construct_Partition();

  /*!
     * \brief clip the voronoi partition to exclude the points in the positive domain
     *
     * This function remove the points located outside of the negative domain and clips the partition
     * to the negative domain. It also updates the level-set values at the vertices of the partition
     */
  void clip_Interface( const CF_2& ls );

  /*!
     * \brief clip the voronoi partition to exclude the points in the positive domain
     *
     * This function remove the points located outside of the negative domain and clips the partition
     * to the negative domain. It also updates the level-set values at the vertices of the partition
     */
  void clip_Interface();

  /*!
   * \brief Check if the voronoi cell is crossed by the irregular interface
   * \return true if the cell is crossed by the interface, false otherwise
   */
  bool is_Interface() const;

  /*!
     * \brief compute the volume enclosed by the voronoi partition
     */
  void compute_volume();

  /*!
     * \brief get the area inside the voronoi partition
     * \return the area of the voronoi partition containing pc and built using the provided points
     */
  inline double get_volume() const { return volume; }

  /*!
   * \brief is_Wall
   * \return true if the voronoi cell is in contact with a wall, false otherwise
   */
  bool is_Wall() const;

  /*!
   * \brief area_In_Negative_Domain
   * \param ls the level-set function
   * \return the area of the voronoi partition in the negative domain
   */
  double area_In_Negative_Domain( const CF_2& ls ) const;

  /*!
   * \brief area_In_Negative_Domain you must have set a level-set function prior to calling this routine
   * \return the area of the voronoi partition in the negative domain
   */
  double area_In_Negative_Domain() const;

  /*!
   * \brief integral of a field f over the negative domain
   * \param ls the level-set function
   * \param fc the value of f at the center point of the voronoi partition
   * \param f the values of f at the vertices of the voronoi partition
   * \return the integral of the quantity f over the negative domain defined by the level-set function
   */
  double integral( const CF_2& ls, double fc, vector<double> &f ) const;

  /*!
   * \brief integral you must have set a level-set function prior to calling this routine
   * \param fc the value of f at the center point of the voronoi partition
   * \param f the values of f at the vertices of the voronoi partition
   * \return the integral of the quantity f over the negative domain defined by the level-set function
   */
  double integral( double fc, vector<double> &f ) const;

  /*!
   * \brief integrate_Over_Interface you must have set a level-set function prior to calling this routine
   * \param fc the value of f at the center point of the voronoi partition
   * \param f the values of f at the vertices of the voronoi partition
   * \return the integral of the quantity f over the interface
   */
  double integrate_Over_Interface( double fc, vector<double> &f ) const;

  /*!
   * \brief integrate_Over_Interface you must have set a level-set function prior to calling this routine
   * \param f the continuous function describing f
   * \return the integral of the quantity f over the interface
   */
  double integrate_Over_Interface( const CF_2& f ) const;

  /*!
     * \brief save the voronoi partition in the .vtk format
     * \param voro the list of voronoi partitions to save
     * \param file_name the file in which the voronoi partition is to be saved
     */
  static void print_VTK_Format( const vector<Voronoi2D>& voro, std::string file_name );

  static void print_VTK_Format( const vector<Voronoi2D> &voro, const vector<double> &f, std::string data_name, std::string file_name );

  static void print_VTK_Format( const vector<Voronoi2D> &voro, const vector<double> &u, const vector<double> &v, std::string data_name, std::string file_name );

  /*!
     * \brief overload the << operator for Voronoi2D
     */
  friend std::ostream& operator<<(std::ostream& os, const Voronoi2D& v);
};

#endif /* VORONOI2D_H */
