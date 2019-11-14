#ifndef VORONOI2D_H
#define VORONOI2D_H

#include <fstream>
#include <vector>

#include <src/my_p4est_utils.h>
#include <src/casl_math.h>
#include <src/point2.h>

#ifdef Voronoi_DIM
#undef Voronoi_DIM
#endif
#define Voronoi_DIM Voronoi2D

using std::vector;

struct ngbd2Dseed
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

  /*!
   * \brief the distance from this neighbor Voronoi seed to the cell center seed
   */
  double dist;

  void operator=(ngbd2Dseed v)
  {
    n = v.n; p = v.p; theta = v.theta; dist = v.dist;
  }
  inline bool operator<(const ngbd2Dseed& v) const
  {
    return (((this->theta <= 2.0*PI) && (this->theta >=0.0) && (v.theta <= 2.0*PI) && (v.theta >=0.0) && (fabs(this->theta - v.theta) > 2.0*PI*EPS))? (this->theta < v.theta):(this->dist < v.dist));
  }
};


/*!
 * \brief The Voronoi2D class construct a Voronoi partition for a point given its surrounding points.
 * This class also provides functions to compute the volume of the Voronoi partition around the point.
 */
class Voronoi2D
{
private:
  Point2 center_seed;
  vector<ngbd2Dseed> nb_seeds;
  vector<Point2> partition;
  vector<double> phi_values;
  double phi_c;
  double volume;

public:
  /*!
     * \brief default constructor for the Voronoi2D class
     */
  Voronoi2D() { center_seed.x=DBL_MAX; center_seed.y=DBL_MAX; }

//  bool comparison (ngbd2Dseed nb_left, ngbd2Dseed nb_right) {return (nb_left.dist < nb_right.dist);}

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
  void get_neighbor_seeds( const vector<ngbd2Dseed> *&neighbors) const;
  void get_partition( const vector<Point2> *&partition_ ) const;
  void get_neighbor_seeds(vector<ngbd2Dseed> *&neighbors);
  void get_partition( vector<Point2> *&partition_ );

  /*!
     * \brief update the partition
     * \param partition the new partition
     */
  void set_partition( vector<Point2>& partition );

  /*!
     * \brief set the precomputed voronoi cell
     * \param neighbors_ the voronoi neighbors
     * \param partition_ the new partition
     * \param volume_ the volume/area of the partition
     */
  void set_neighbors_and_partition( vector<ngbd2Dseed>& neighbors_, vector<Point2>& partition_, double volume_ );

  /*!
     * \brief set the level-set values at the vertices of the voronoi partition
     * \param ls a continuous description of the level-set function
     */
  void set_level_set_values( const CF_2& ls );

  /*!
     * \brief set the level-set values at the vertices of the voronoi partition
     * \param phi_values the list of the level-set values
     * \param phi_c the value of the level-set at the center point
     */
  void set_level_set_values( vector<double>& phi_values, double phi_c );

  /*!
     * \brief set the point at the center of the partition
     * \param center_seed_ the coordinates of the point
     */
  void set_center_point( Point2 center_seed_ );

  /*!
     * \brief set the coordinates of the point at the center of the partition
     * \param x the first coordinate of the point
     * \param y the second coordinate of the point
     */
  void set_center_point( double x, double y );

  /*!
     * \brief get the point at the center of the partition
     * \return center_seed the coordinates of the point
     */
  inline const Point2& get_center_point() const { return center_seed; }

  /*!
     * \brief add a point to the list of collocation points, making sure there is no repetition
     * \param n the index of the point to add
     * \param x the first coordinate of the point to add
     * \param y the second coordinate of the point to add
     */
  void push( int n, double x, double y, const bool* periodicity, const double* xyz_min, const double* xyz_max);

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
  void enforce_periodicity( bool p_x, bool p_y, double xmin, double xmax, double ymin, double ymax );

  /*!
     * \brief construct the voronoi partition around point center_seed using the neighborhood given in nb_seeds
     */
  void construct_partition();

  /*!
     * \brief clip the voronoi partition to exclude the points in the positive domain
     *
     * This function remove the points located outside of the negative domain and clips the partition
     * to the negative domain. It also updates the level-set values at the vertices of the partition
     */
  void clip_interface( const CF_2& ls );

  /*!
     * \brief clip the voronoi partition to exclude the points in the positive domain
     *
     * This function remove the points located outside of the negative domain and clips the partition
     * to the negative domain. It also updates the level-set values at the vertices of the partition
     */
  void clip_interface();

  /*!
   * \brief Check if the voronoi cell is crossed by the irregular interface
   * \return true if the cell is crossed by the interface, false otherwise
   */
  bool is_interface() const;

  /*!
     * \brief compute the volume enclosed by the voronoi partition
     */
  void compute_volume();

  /*!
     * \brief get the area inside the voronoi partition
     * \return the area of the voronoi partition containing center_seed and built using the provided points
     */
  inline double get_volume() const { return volume; }

  /*!
   * \brief is_Wall
   * \return true if the voronoi cell is in contact with a wall, false otherwise
   */
  bool is_wall() const;

  /*!
   * \brief area_In_Negative_Domain
   * \param ls the level-set function
   * \return the area of the voronoi partition in the negative domain
   */
  double area_in_negative_domain( const CF_2& ls ) const;

  /*!
   * \brief area_In_Negative_Domain you must have set a level-set function prior to calling this routine
   * \return the area of the voronoi partition in the negative domain
   */
  double area_in_negative_domain() const;

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
  double integrate_over_interface( double fc, vector<double> &f ) const;

  /*!
   * \brief integrate_Over_Interface you must have set a level-set function prior to calling this routine
   * \param f the continuous function describing fngbd2Dseed
   * \return the integral of the quantity f over the interface
   */
  double integrate_over_interface( const CF_2& f ) const;

  /*!
     * \brief save the voronoi partition in the .vtk format
     * \param voro the list of voronoi partitions to save
     * \param file_name the file in which the voronoi partition is to be saved
     */
  static void print_VTK_format( const vector<Voronoi2D>& voro, std::string file_name );

  static void print_VTK_format( const vector<Voronoi2D> &voro, const vector<double> &f, std::string data_name, std::string file_name );

  static void print_VTK_format( const vector<Voronoi2D> &voro, const vector<double> &u, const vector<double> &v, std::string data_name, std::string file_name );

  /*!
     * \brief overload the << operator for Voronoi2D
     */
  friend std::ostream& operator<<(std::ostream& os, const Voronoi2D& v);
};

#endif /* VORONOI2D_H */
