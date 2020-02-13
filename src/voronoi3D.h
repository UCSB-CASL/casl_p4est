#ifndef CASL_VORONOI3D_H
#define CASL_VORONOI3D_H
#include <float.h>
#include <fstream>
#include <vector>

// We are in 3D (you, idiot!), so include p8est headers immediately
#include <src/my_p8est_utils.h>
//#include <src/my_p8est_faces.h>
#include <src/casl_math.h>
#include <src/point3.h>

#include <voro++.hh>

#ifdef Voronoi_DIM
#undef Voronoi_DIM
#endif
#ifdef ngbdDIMseed
#undef ngbdDIMseed
#endif
#define Voronoi_DIM Voronoi3D
#define ngbdDIMseed ngbd3Dseed

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

struct ngbd3Dseed
{
  /*!
     * \brief the index of the neighbor seed
     */
  int n;

  /*!
     * \brief the coordinates of the neighbor seed
     */
  Point3 p;

  /*!
   * \brief the surface of the face separating the center seed from this neighbor seed
   */
  double s;

  /*!
   * \brief dist distance from the center seed to this neighbor seed
   */
  double dist;

  void operator=(ngbd3Dseed v)
  {
    n = v.n; p = v.p; s=v.s; dist = v.dist;
  }
  inline bool operator<(const ngbd3Dseed& v) const
  {
    return (this->dist < v.dist);
  }
};

/*!
 * \brief The Voronoi3D class construct a Voronoi partition for a point (the seed) given its surrounding
 * points (neighbor seeds).
 * This class also provides functions to compute the volume of the Voronoi partition around the point.
 */
class Voronoi3D
{
private:
  Point3 center_seed;
  int idx_center_seed;
  vector<ngbd3Dseed> nb_seeds;
  double volume;

  /*!
   * \brief add a neighbor seed, WITHOUT making sure there is no repetition
   * \param n the index of the point to add
   * \param pt coordinates of the candidate neighbor seed to add
   * \param periodicity the periodicity flag for the computational domain
   * \param xyz_min the coordinates of the lower left corner of the computational domain
   * \param xyz_min the coordinates of the upper right corner of the computational domain
   */
  void add_point( int n, Point3 &pt, const bool* periodicity, const double* xyz_min, const double* xyz_max);
  inline void add_point( int n, double x, double y, double z, const bool* periodicity, const double* xyz_min, const double* xyz_max)
  {
    Point3 pt(x, y, z);
    add_point(n, pt, periodicity, xyz_min, xyz_max);
  }

public:
  /*!
     * \brief default constructor for the Voronoi3D class
     */
  Voronoi3D() { center_seed.x=DBL_MAX; center_seed.y=DBL_MAX; center_seed.z=DBL_MAX; }

  /*!
     * \brief reset the Voronoi partition
     */
  void clear() { nb_seeds.resize(0); }

  /*!
     * \brief get the partition after it has been built using construct_partition
     * \param neighbors the list of neighbor points used to create the partition (actual direct neighbors)
     */
  void get_neighbor_seeds( const vector<ngbd3Dseed>*& neighbor_seeds) const { neighbor_seeds = &this->nb_seeds; }

  /*!
   * \brief set the voronoi cell with precomputed values
   * \param neighbors the list of neighbor seeds with their properties
   * \param the volume of the Voronoi cell
   */
  void set_cell( vector<ngbd3Dseed> &neighbors, double volume );

  /*!
     * \brief set the center seed of the partition
     * \param center_seed_ the coordinates of the center seed
     */
  void set_center_point( int idx_center_seed_, Point3 &center_seed_);
  // overloading
  void set_center_point( int idx_center_seed_, double x, double y, double z) { Point3 tmp(x, y, z); set_center_point(idx_center_seed_, tmp); }
  void set_center_point( int idx_center_seed_, const double* xyz) { set_center_point(idx_center_seed_, xyz[0], xyz[1], xyz[2]); }

  /*!
     * \brief get the center seed of the partition
     * \param return the coordinates of the point
     */
  inline const Point3& get_center_point() const { return center_seed; }

  inline void get_center_point(double *xyz) const { xyz[0] = center_seed.x; xyz[1] = center_seed.y; xyz[2] = center_seed.z; }

  /*!
   * \brief add a potential neighbor seed candidate, after making sure there is no repetition
   * \param n the index of the point to add
   * \param pt coordinates of the candidate neighbor seed to add
   * \param periodicity the periodicity flag for the computational domain
   * \param xyz_min the coordinates of the lower left corner of the computational domain
   * \param xyz_min the coordinates of the upper right corner of the computational domain
   */
  void push( int n, Point3 &pt, const bool* periodicity, const double* xyz_min, const double* xyz_max);
  // overloading
  void push( int n, double x, double y, double z, const bool* periodicity, const double* xyz_min, const double* xyz_max) { Point3 tmp(x, y, z); push(n, tmp, periodicity, xyz_min, xyz_max); }


  void assemble_from_set_of_faces(const std::set<indexed_and_located_face>& set_of_neighbor_faces, const bool* periodicity, const double* xyz_min, const double* xyz_max);

  /*!
   * \brief construct_partition constructs the voronoi cell around point pc using the neighborhood given in nb_seeds
   * \param [in] xyz_min:   minimal bounds of the domain
   * \param [in] xyz_max:   maximal bounds of the domain
   * \param [in] periodic:  periodicity flags for each cartesian direction
   * \return a flag that is true iff the constructed cell has a wall neighbor.
   */
  bool construct_partition(const double *xyz_min, const double *xyz_max, const bool *periodic);

  inline double get_volume() const { return this->volume; }

  /*!
     * \brief save the voronoi partition in the .vtk format
     * \param voro the list of voronoi partitions to save
     * \param file_name the file in which the voronoi partition is to be saved
     */
  static void print_VTK_format( const std::vector<Voronoi3D>& voro, const char* file_name,
                                const double *xyz_min, const double *xyz_max, const bool *periodic);
};

#endif // CASL_VORONOI3D_H
