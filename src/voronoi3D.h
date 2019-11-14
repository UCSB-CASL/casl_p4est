#ifndef CASL_VORONOI3D_H
#define CASL_VORONOI3D_H

#include <float.h>
#include <fstream>
#include <vector>

#include <src/my_p4est_utils.h>
#include <src/casl_math.h>
#include <src/point3.h>

#include <voro++.hh>

#ifdef Voronoi_DIM
#undef Voronoi_DIM
#endif
#define Voronoi_DIM Voronoi3D

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

  /*!
     * \brief get the center seed of the partition
     * \param return the coordinates of the point
     */
  inline const Point3& get_center_point() const { return center_seed; }

  /*!
   * \brief add a potential neighbor seed candidate, making sure there is no repetition
   * \param n the index of the point to add
   * \param pt coordinates of the candidate neighbor seed to add
   */
  void push( int n, Point3 &pt, const bool* periodicity, const double* xyz_min, const double* xyz_max);
  // overloading
  void push( int n, double x, double y, double z, const bool* periodicity, const double* xyz_min, const double* xyz_max) { Point3 tmp(x, y, z); push(n, tmp, periodicity, xyz_min, xyz_max); }

  /*!
     * \brief construct the voronoi parition around point pc using the neighborhood given in nb_seeds
     */
  void construct_partition(const double *xyz_min, const double *xyz_max, const bool *periodic);

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
