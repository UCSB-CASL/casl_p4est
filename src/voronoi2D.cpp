#include <src/voronoi2D.h>
#include <src/simplex2.h>
#include <algorithm>

void Voronoi2D::clear()
{
  nb_seeds.resize(0);
  partition.resize(0);
  phi_values.resize(0);
}

void Voronoi2D::operator=( const Voronoi2D& voro )
{
  center_seed = voro.center_seed;
  nb_seeds = voro.nb_seeds;
  partition = voro.partition;
  phi_values = voro.phi_values;
  phi_c = voro.phi_c;
}

void Voronoi2D::get_neighbor_seeds(const vector<ngbd2Dseed>*& neighbors) const
{
  neighbors = &this->nb_seeds;
}

void Voronoi2D::get_partition(const vector<Point2> *&partition_ ) const
{
  partition_ = &(this->partition);
}

void Voronoi2D::get_neighbor_seeds(vector<ngbd2Dseed>*& neighbors)
{
  neighbors = &this->nb_seeds;
}

void Voronoi2D::get_partition( vector<Point2> *&partition_ )
{
  partition_ = &(this->partition);
}

void Voronoi2D::set_partition( vector<Point2>& partition_ )
{
  this->partition = partition_;
}

void Voronoi2D::set_neighbors_and_partition(vector<ngbd2Dseed>& neighbors_, vector<Point2>& partition_, double volume_)
{
  this->nb_seeds = neighbors_;
  this->partition = partition_;
  this->volume = volume_;
}

void Voronoi2D::set_level_set_values(const CF_2 &ls )
{
  phi_c = ls(center_seed.x, center_seed.y);
  phi_values.resize(partition.size());
  for(unsigned  m=0; m<partition.size(); ++m)
    phi_values[m] = ls(partition[m].x, partition[m].y);
}

void Voronoi2D::set_level_set_values( vector<double>& phi_values, double phi_c )
{
  this->phi_c = phi_c;
  this->phi_values = phi_values;
}

void Voronoi2D::push( int n, double x, double y, const bool* periodicity, const double* xyz_min, const double* xyz_max)
{
  for(unsigned int m=0; m<nb_seeds.size(); m++)
    if(nb_seeds[m].n == n)
      return;
  add_point(n, x, y, periodicity, xyz_min, xyz_max);
}

void Voronoi2D::assemble_from_set_of_faces(const std::set<indexed_and_located_face>& set_of_neighbor_faces, const bool* periodicity, const double* xyz_min, const double* xyz_max)
{
  nb_seeds.clear();
  for (std::set<indexed_and_located_face>::const_iterator got_it= set_of_neighbor_faces.begin(); got_it != set_of_neighbor_faces.end(); ++got_it) {
    P4EST_ASSERT((*got_it).face_idx >= 0);
    add_point((*got_it).face_idx, (*got_it).xyz_face[0], (*got_it).xyz_face[1], periodicity, xyz_min, xyz_max); // no need to check for duplicates by definition of std::set
  }
}


void Voronoi2D::add_point( int n, double x, double y, const bool* periodicity, const double* xyz_min, const double* xyz_max)
{
  ngbd2Dseed p;
  p.n     = n;
  p.p.x   = x;
  p.p.y   = y;
  p.dist  = (p.p - center_seed).norm_L2();
  p.theta = DBL_MAX;
  nb_seeds.push_back(p);
  if(periodicity[0] || periodicity[1]) // some periodicity ?
  {
    const double domain_diag = sqrt(SQR(xyz_max[0] - xyz_min[0]) + SQR(xyz_max[1] - xyz_min[1]));
    if(periodicity[0]) // x periodic
    {
      // we use 0.49 instead of 0.5 to ensure everything goes fine even for a 1/1 grid
      int x_coeff = (fabs(x-center_seed.x) > 0.49*(xyz_max[0] - xyz_min[0]))? ((x<center_seed.x)?+1:-1): 0;
      if(x_coeff != 0)
      {
        ngbd2Dseed x_wrapped_neighbor;
        x_wrapped_neighbor.n      = n;
        x_wrapped_neighbor.p.x    = x + ((double) x_coeff)*(xyz_max[0] - xyz_min[0]);
        x_wrapped_neighbor.p.y    = y;
        x_wrapped_neighbor.dist   = (x_wrapped_neighbor.p - center_seed).norm_L2();
        x_wrapped_neighbor.theta  = DBL_MAX;
        if(x_wrapped_neighbor.dist < 0.51*domain_diag)
          nb_seeds.push_back(x_wrapped_neighbor);
      }
      if(periodicity[1]) // x periodic AND y periodic
      {
        int y_coeff = (fabs(y-center_seed.y) > 0.49*(xyz_max[1] - xyz_min[1]))? ((y<center_seed.y)?+1:-1): 0;
        // first add the y-wrapped if needed
        if(y_coeff != 0)
        {
          ngbd2Dseed y_wrapped_neighbor;
          y_wrapped_neighbor.n      = n;
          y_wrapped_neighbor.p.x    = x;
          y_wrapped_neighbor.p.y    = y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
          y_wrapped_neighbor.dist   = (y_wrapped_neighbor.p - center_seed).norm_L2();
          y_wrapped_neighbor.theta  = DBL_MAX;
          if(y_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(y_wrapped_neighbor);
        }
        // then add the xy-wrapped if need
        if(x_coeff != 0)
        {
          ngbd2Dseed xy_wrapped_neighbor;
          xy_wrapped_neighbor.n     = n;
          xy_wrapped_neighbor.p.x   = x + ((double) x_coeff)*(xyz_max[0] - xyz_min[0]);
          xy_wrapped_neighbor.p.y   = y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
          xy_wrapped_neighbor.dist  = (xy_wrapped_neighbor.p - center_seed).norm_L2();
          xy_wrapped_neighbor.theta = DBL_MAX;
          if(xy_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(xy_wrapped_neighbor);
        }
      }
    }
    else // only y-periodic
    {
      int y_coeff = (fabs(y-center_seed.y) > 0.49*(xyz_max[1] - xyz_min[1]))? ((y<center_seed.y)?+1:-1): 0;
      if(y_coeff != 0)
      {
        ngbd2Dseed y_wrapped_neighbor;
        y_wrapped_neighbor.n      = n;
        y_wrapped_neighbor.p.x    = x;
        y_wrapped_neighbor.p.y    = y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
        y_wrapped_neighbor.dist   = (y_wrapped_neighbor.p - center_seed).norm_L2();
        y_wrapped_neighbor.theta  = DBL_MAX;
        if(y_wrapped_neighbor.dist < 0.51*domain_diag)
          nb_seeds.push_back(y_wrapped_neighbor);
      }
    }
  }
}

void Voronoi2D::set_center_point(Point2 center_seed_ )
{
  this->center_seed = center_seed_;
}

void Voronoi2D::set_center_point(double x, double y)
{
  center_seed.x = x;
  center_seed.y = y;
}

void Voronoi2D::construct_partition()
{
#ifdef CASL_THROWS
  if(center_seed.x==DBL_MAX || center_seed.y==DBL_MAX) throw std::invalid_argument("[CASL_ERROR]: Voronoi2D: invalid center point to build the voronoi partition.");
  if(nb_seeds.size()<3) throw std::runtime_error("[CASL_ERROR]: Voronoi2D: not enough points to build the voronoi partition.");
#endif

  // angles are not set yet so sort by increasing distance from the seed
  std::sort(nb_seeds.begin(), nb_seeds.end());

  // scale it to a domain-independent geometry (closest neighbor at distance 1.0)
  // compute the angles with the reference point on-the-fly
  // scaling information
  /*  -------------- Feel free to change the following parameter to any other reasonable value ---------------      */
  const double closest_distance = 1.0;
  const double scaling_length = nb_seeds[0].dist/closest_distance;
  P4EST_ASSERT(scaling_length>0.0 && scaling_length > (nb_seeds.back()).dist*EPS);
  // center the seed to (0.0, 0.0)
  Point2 center_seed_saved = center_seed; center_seed.x = 0.0; center_seed.y = 0.0;
  double angle;
  for (size_t m = 0; m < nb_seeds.size(); ++m) {
    nb_seeds[m].p     = (nb_seeds[m].p - center_seed_saved)/scaling_length;
    nb_seeds[m].dist  /= scaling_length;
    if(m == 0)
      angle = 0.0;
    else
    {
      angle = acos(MAX(-1., MIN(1., (nb_seeds[0].p).dot(nb_seeds[m].p)/(closest_distance*(nb_seeds[m].p).norm_L2()))));
      if((nb_seeds[0].p).cross(nb_seeds[m].p) < 0.0)
        angle = 2.0*PI - angle;
    }
    nb_seeds[m].theta = angle;
  }

  // sort the list with increasing theta angles
  // although the first element in the list should remain first on paper,
  // we do not include it in the list to be sorted to ensure robust behavior...
  std::sort(nb_seeds.begin()+1, nb_seeds.end());


  // construct the vertices of the voronoi partition
  vector<double> theta_vertices(nb_seeds.size());
  partition.resize(nb_seeds.size());
  for(size_t m=0; m<nb_seeds.size(); ++m)
  {
    size_t k = mod(m+1, nb_seeds.size());
    // find unit director vector of bisector planes m or k by (normed) cross
    // product between e_z and nb_seeds[m].p or nb_seeds[k].p, where e_z it the
    // out-of-plane unit vector
    Point2 bisector_dir_m(-nb_seeds[m].p.y, nb_seeds[m].p.x); bisector_dir_m /= nb_seeds[m].dist;
    Point2 bisector_dir_k(-nb_seeds[k].p.y, nb_seeds[k].p.x); bisector_dir_k /= nb_seeds[k].dist;
    double denom = bisector_dir_m.cross(bisector_dir_k); // also the sine of the angle between the bisector cuts

    // if the points are aligned, keep the point that is the closest to center_seed
    if(denom < EPS)
    {
      if( (nb_seeds[m].p-center_seed).norm_L2() > (nb_seeds[k].p-center_seed).norm_L2() )
        k = m;
      nb_seeds.erase(nb_seeds.begin() + k);
      partition.erase(partition.begin() + k);
      theta_vertices.erase(theta_vertices.begin() + k);
      m -= m==k? 2:1;
      continue;
    }

    // law of sines:
    double lambda = 0.5*((nb_seeds[k].p - nb_seeds[m].p).cross(bisector_dir_k))/denom;
    partition[m] = nb_seeds[m].p*0.5 + bisector_dir_m*lambda;

    // compute the angle between the new vertex point and the reference point
    angle = acos((nb_seeds[0].p).dot(partition[m])/(closest_distance*partition[m].norm_L2()));

    if((nb_seeds[0].p).cross(partition[m]) < 0)
      angle = 2.*PI-angle;
    theta_vertices[m] = angle;

    // check if the new vertex point is indeed a vertex of the voronoi partition
    if(m!=0)
    {
      k = mod(m-1, nb_seeds.size());

      // if the new vertex is before the previous one, in trigonometric order
      // or check for a double vertex, which means the new point [m] leads to an edge of length zero
      if( partition[k].cross(partition[m]) < 0.0 || (partition[m] - partition[k]).norm_L2() < EPS*partition[k].norm_L2())
      {
        nb_seeds.erase(nb_seeds.begin() + m);
        partition.erase(partition.begin() + m);
        theta_vertices.erase(theta_vertices.begin() + m);
        m-=2;
      }
    }
  }

  P4EST_ASSERT(partition.size() == nb_seeds.size());
  center_seed = center_seed_saved;
  for (size_t m = 0; m < partition.size(); ++m) {
    nb_seeds[m].p     = center_seed + (nb_seeds[m].p)*scaling_length;
    nb_seeds[m].dist  *= scaling_length;
    partition[m]      = center_seed + (partition[m])*scaling_length;
  }

  compute_volume();

  return;
}



void Voronoi2D::clip_interface( const CF_2& ls )
{
  set_level_set_values(ls);
  clip_interface();
}


void Voronoi2D::clip_interface()
{
#ifdef CASL_THROWS
  if(phi_values.size() != nb_seeds.size() || phi_values.size() != partition.size())
    throw std::invalid_argument("[CASL_THROWS]: Voronoi2D: the lists of points, vertices and/or level-set values do not have the same length.");
#endif

  /* the threshold used to tell if a point is in the negative domain
   * a point is in the negative domain if phi(n)<thresh.
   * note that using epsilon instead of zero eliminate the problem of cells
   * with very small areas
   */
  double thresh = -EPS/2;

  /* find a vertex that is in the negative domain */
  size_t m0 = 0;
  for(; m0 < partition.size(); ++m0)
    if(phi_values[m0] <= thresh)
      break;

  /* the partition is entirely in the positive domain */
  if(m0 >= partition.size())
  {
    nb_seeds.resize(0);
    partition.resize(0);
    phi_values.resize(0);
    volume = 0;
    return;
  }

  /* otherwise clip the partition */
  unsigned int m = m0;
  unsigned int k = mod(m + 1, partition.size());
  do
  {
    /* the next vertex needs to be clipped */
    if(phi_values[k] > thresh)
    {
      Point2 dir;
      double theta;

      /* find the vertex between m and k */
      dir = partition[k] - partition[m];
      theta = fraction_Interval_Covered_By_Irregular_Domain(phi_values[m], phi_values[k], dir.norm_L2(), dir.norm_L2());
      Point2 pmk = partition[m] + dir*theta;

      /* find the following vertex */
      unsigned int l = k;
      unsigned int r = mod(l + 1, partition.size());
      while(phi_values[r] > thresh)
      {
        l = r;
        r = mod(l + 1, partition.size());
      }

      /* remove intermediate vertices if necessary */
      if(k != l)
      {
        unsigned int h = mod(k + 1, partition.size());
        while(h != l)
        {
          nb_seeds.erase(nb_seeds.begin() + h);
          partition.erase(partition.begin() + h);
          phi_values.erase(phi_values.begin() + h);

          if(h < m0) m0--;
          if(h < m)  m--;
          if(h < k)  k--;
          if(h < l)  l--;
          if(h < r)  r--;

          h = mod(k + 1, partition.size());
        }
      }

      /* find the vertex between l and r */
      dir = partition[r] - partition[l];
      theta = fraction_Interval_Covered_By_Irregular_Domain(phi_values[r], phi_values[l], dir.norm_L2(), dir.norm_L2());
      Point2 plr = partition[r] - dir*theta;

      /* add the vertices to the partition and create the ghost points */
      Point2 u = plr - pmk;

      /* the new points are on top of each other ... which can only mean that the value at point k=l is exactly 0 */
      if(u.norm_L2() < EPS)
      {
        m = r;
        continue;
      }

      u /= u.norm_L2(); // --> norm of u is 1.0
      ngbd2Dseed tmp;
      tmp.n = INTERFACE;
      const Point2 center_to_pmk = pmk - center_seed;
      tmp.p = center_seed + (center_to_pmk - u*(center_to_pmk.dot(u)))*2.0; // mirror point of the center seed across the (straight) INTERFACE partition segment


      partition[k] = pmk;
      phi_values[k] = 0.0;

      if(k == l)
      {
        partition.insert(partition.begin() + r, plr);
        phi_values.insert(phi_values.begin() + r, 0);
        nb_seeds.insert(nb_seeds.begin() + r, tmp);
        if(r <= m0) m0++;

        /* move on to next point */
        m = mod(r + 1, partition.size());
      }
      else
      {
        partition[l] = plr;
        phi_values[l] = 0.0;
        nb_seeds[l] = tmp;

        /* move on to next point */
        m = mod(l + 1, partition.size());
      }
    }
    else
      m = mod(m + 1, partition.size());

    k = mod(m + 1, partition.size());
  } while(k != m0 && m != m0);

#ifdef CASL_THROWS
  if(partition.size() != nb_seeds.size() || phi_values.size() != nb_seeds.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->clip_Interface: error while clipping the interface.");
#endif

  compute_volume();
}


bool Voronoi2D::is_interface() const
{
  for(size_t n = 0; n < nb_seeds.size(); ++n)
    if(nb_seeds[n].n == INTERFACE)
      return true;
  return false;
}


void Voronoi2D::compute_volume()
{
  volume = 0;
  for(unsigned int m=0; m<partition.size(); ++m)
  {
    unsigned int k = mod(m+partition.size()-1, partition.size());
    Point2 u = partition[k]-center_seed;
    Point2 v = partition[m]-center_seed;
    volume += u.cross(v)/2.;
  }
}


bool Voronoi2D::is_wall() const
{
  for(unsigned int m=0; m<nb_seeds.size(); ++m)
    if(nb_seeds[m].n==WALL_m00 || nb_seeds[m].n==WALL_p00 || nb_seeds[m].n==WALL_0m0 || nb_seeds[m].n==WALL_0p0)
      return true;
  return false;
}


double Voronoi2D::area_in_negative_domain( const CF_2& ls ) const
{
  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    unsigned int k = mod(m+partition.size()-1, partition.size());

    s.x0 = center_seed.x; s.y0 = center_seed.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integral(1, 1, 1, ls(s.x0,s.y0), ls(s.x1,s.y1), ls(s.x2,s.y2));
  }

  return sum;
}


double Voronoi2D::area_in_negative_domain() const
{
  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    int k = mod(m+1, partition.size());

    s.x0 = center_seed.x; s.y0 = center_seed.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integral(1, 1, 1, phi_c, phi_values[m], phi_values[k]);
  }

  return sum;
}


double Voronoi2D::integral( const CF_2& ls, double fc, vector<double> &f ) const
{
#ifdef CASL_THROWS
  if(f.size() != partition.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->integrate_Over_Interface: wrong input dimension.");
#endif

  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    int k = mod(m+1, partition.size());

    s.x0 = center_seed.x; s.y0 = center_seed.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integral(fc, f[m], f[k], ls(s.x0,s.y0), ls(s.x1,s.y1), ls(s.x2,s.y2));
  }

  return sum;
}

double Voronoi2D::integral( double fc, vector<double> &f ) const
{
#ifdef CASL_THROWS
  if(phi_values.size() != partition.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->integrate_Over_Interface: you must call set_Level_Set_Values priori to integrating.");
  if(f.size() != partition.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->integrate_Over_Interface: wrong input dimension.");
#endif

  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    int k = mod(m+1, partition.size());

    s.x0 = center_seed.x; s.y0 = center_seed.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integral(fc, f[m], f[k], phi_c, phi_values[m], phi_values[k]);
  }

  return sum;
}

double Voronoi2D::integrate_over_interface( double fc, vector<double> &f ) const
{
#ifdef CASL_THROWS
  if(phi_values.size() != partition.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->integrate_Over_Interface: you must call set_Level_Set_Values priori to integrating over the interface.");
  if(f.size() != partition.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->integrate_Over_Interface: wrong input dimension.");
#endif

  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    int k = mod(m+1, partition.size());

    s.x0 = center_seed.x; s.y0 = center_seed.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integrate_Over_Interface(fc, f[m], f[k], phi_c, phi_values[m], phi_values[k]);
  }

  return sum;
}

double Voronoi2D::integrate_over_interface( const CF_2& f ) const
{
#ifdef CASL_THROWS
  if(phi_values.size() != partition.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->integrate_Over_Interface: you must call set_Level_Set_Values priori to integrating over the interface.");
#endif

  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    int k = mod(m+1, partition.size());

    s.x0 = center_seed.x; s.y0 = center_seed.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integrate_Over_Interface(f, phi_c, phi_values[m], phi_values[k]);
  }

  return sum;
}


void Voronoi2D::print_VTK_format( const vector<Voronoi2D>& voro, std::string file_name )
{
  FILE *fp;
  fp = fopen(file_name.c_str(), "w");
#ifdef CASL_THROWS
  if(fp==NULL) throw std::invalid_argument("[CASL_ERROR]: Voronoi2D: cannot open file.");
#endif

  std::cout << "Saving Voronoi partition in ... " << file_name << std::endl;

  int nb_vertices=0;
  int nb_polygons=0;
  for(unsigned int n=0; n<voro.size(); ++n)
  {
    nb_vertices += voro[n].partition.size();
    if(voro[n].partition.size()!=0) nb_polygons++;
  }

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "Voronoi partition\n");
  fprintf(fp, "ASCII\nDATASET POLYDATA\nPOINTS %d double\n", nb_vertices);

  for(unsigned int n=0; n<voro.size(); ++n)
    for(unsigned int m=0; m<voro[n].partition.size(); ++m)
      fprintf(fp, "%1.16e\t %1.16e\t 0.0\n", voro[n].partition[m].x, voro[n].partition[m].y);

  fprintf(fp, "POLYGONS %d %d\n", nb_polygons, nb_vertices+nb_polygons);
  int compt = 0;
  for(unsigned int n=0; n<voro.size(); ++n)
  {
    if(voro[n].partition.size()!=0)
    {
      fprintf(fp, "%lu\t ", voro[n].partition.size());
      for(unsigned int m=0; m<voro[n].partition.size(); ++m)
        fprintf(fp, "\t %d", compt++);
      fprintf(fp, "\n");
    }
  }

  fprintf(fp, "CELL_DATA %d\n", nb_polygons);

  fclose(fp);
}

void Voronoi2D::print_VTK_format( const vector<Voronoi2D> &voro, const vector<double> &f, std::string data_name, std::string file_name )
{
  FILE *fp;
  fp = fopen(file_name.c_str(), "a");
#ifdef CASL_THROWS
  if(fp==NULL) throw std::invalid_argument("[CASL_ERROR]: Voronoi2D: cannot open file.");
#endif

  fprintf(fp, "SCALARS %s double 1\n", data_name.c_str());
  fprintf(fp, "LOOKUP_TABLE default\n");

  for(unsigned int n=0; n<voro.size(); ++n)
    if(voro[n].partition.size()!=0)
      fprintf(fp, "%1.16e\n", f[n]);

  fclose(fp);
}


void Voronoi2D::print_VTK_format( const vector<Voronoi2D> &voro, const vector<double> &u, const vector<double> &v, std::string data_name, std::string file_name )
{
  FILE *fp;
  fp = fopen(file_name.c_str(), "a");
#ifdef CASL_THROWS
  if(fp==NULL) throw std::invalid_argument("[CASL_ERROR]: Voronoi2D: cannot open file.");
#endif

  fprintf(fp, "SCALARS %s double 2\n", data_name.c_str());
  fprintf(fp, "LOOKUP_TABLE default\n");

  for(unsigned int n=0; n<voro.size(); ++n)
    if(voro[n].partition.size()!=0)
      fprintf(fp, "%1.16e\t %1.16e\n", u[n], v[n]);

  fclose(fp);
}


std::ostream& operator<<(std::ostream& os, const Voronoi2D& v)
{
  os << "Center point : " << v.center_seed.x << "," << v.center_seed.y << std::endl;

  for (unsigned int n=0; n<v.nb_seeds.size(); n++)
    os << v.nb_seeds[n].n << " : (" << v.nb_seeds[n].p.x << "," << v.nb_seeds[n].p.y << "," << v.nb_seeds[n].theta << ")" << std::endl;
  os << std::endl;

  return os;
}
