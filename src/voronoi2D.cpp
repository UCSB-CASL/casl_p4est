#include <src/voronoi2D.h>
#include <src/simplex2.h>

void Voronoi2D::clear()
{
  points.resize(0);
  partition.resize(0);
  phi_values.resize(0);
}

void Voronoi2D::operator=( const Voronoi2D& voro )
{
  pc = voro.pc;
  points = voro.points;
  partition = voro.partition;
  phi_values = voro.phi_values;
  phi_c = voro.phi_c;
}

void Voronoi2D::get_Points( const vector<Voronoi2DPoint>*& points) const
{
  points = &this->points;
}

void Voronoi2D::get_Partition( const vector<Point2> *&partition ) const
{
  partition = &(this->partition);
}

void Voronoi2D::set_Partition( vector<Point2>& partition )
{
  this->partition = partition;
}

void Voronoi2D::set_Level_Set_Values( const CF_2 &ls )
{
  phi_c = ls(pc.x, pc.y);
  phi_values.resize(partition.size());
  for(unsigned  m=0; m<partition.size(); ++m)
    phi_values[m] = ls(partition[m].x, partition[m].y);
}

void Voronoi2D::set_Level_Set_Values( vector<double>& phi_values, double phi_c )
{
  this->phi_c = phi_c;
  this->phi_values = phi_values;
}

void Voronoi2D::push( int n, double x, double y )
{
  for(unsigned int m=0; m<points.size(); m++)
  {
    if(points[m].n == n)
    {
      return;
    }
  }

  Voronoi2DPoint p;
  p.n = n;
  p.p.x = x;
  p.p.y = y;
  points.push_back(p);
}

void Voronoi2D::set_Center_Point( Point2 pc )
{
  this->pc = pc;
}

void Voronoi2D::set_Center_Point( double x, double y )
{
  pc.x = x;
  pc.y = y;
}

void Voronoi2D::enforce_Periodicity( bool p_x, bool p_y, double xmin, double xmax, double ymin, double ymax )
{
  double dx = (xmax-xmin);
  double dy = (ymax-ymin);
  double xc = (xmax+xmin)/2.;
  double yc = (ymax+ymin)/2.;

  for(unsigned  m=0; m<points.size(); m++)
  {
    if(p_x && ABS(pc.x-points[m].p.x) > dx/2.)
      points[m].p.x += pc.x<xc ? -dx : dx;

    if(p_y && ABS(pc.y-points[m].p.y) > dy/2.)
      points[m].p.y += pc.y<yc ? -dy : dy;
  }
}

void Voronoi2D::construct_Partition()
{
#ifdef CASL_THROWS
  if(pc.x==DBL_MAX || pc.y==DBL_MAX) throw std::invalid_argument("[CASL_ERROR]: Voronoi2D: invalid center point to build the voronoi partition.");
  if(points.size()<3) throw std::runtime_error("[CASL_ERROR]: Voronoi2D: not enough points to build the voronoi partition.");
#endif

  // first find the closest point to (ic,jc)
  int m_min = 0;
  double d0 = (points[0].p-pc).norm_L2();
  for(unsigned  m=1; m<points.size(); ++m)
  {
    double d = (points[m].p-pc).norm_L2();
    if(d<d0)
    {
      d0    = d;
      m_min = m;
    }
  }

  // put the closest point as the head of the list, with reference theta angle 0
  Voronoi2DPoint tmp = points[0];
  points[0] = points[m_min];
  points[m_min] = tmp;
  points[0].theta = 0.;

  // compute the angle with the reference point for all points in the list
  Point2 v0(points[0].p-pc);
  for(unsigned int m=1; m<points.size(); ++m)
  {
    Point2 vm(points[m].p-pc);
    double dm = vm.norm_L2();

    double angle = MAX(-1., MIN(1., v0.dot(vm)/(dm*d0)) );
    double a = acos(angle);

    if(v0.cross(vm) < 0)
      a = 2.*PI-a;

    points[m].theta = a;
  }

  // sort the list with increasing theta angle and find bissectrix information
  vector<Point2> middle(points.size());
  vector<Point2> dir(points.size());

  middle[0] = (points[0].p+pc) / 2.;
  dir[0].x = -(points[0].p.y - pc.y);
  dir[0].y =  (points[0].p.x - pc.x);
  dir[0] /= dir[0].norm_L2();

  for(unsigned int m=1; m<points.size(); ++m)
  {
    unsigned int swp = m;
    for(unsigned int k=m+1; k<points.size(); ++k)
      if(points[k].theta < points[swp].theta)
        swp = k;
    if(swp!=m)
    {
      tmp = points[m];
      points[m] = points[swp];
      points[swp] = tmp;
    }

    middle[m] = (points[m].p + pc) / 2.;
    dir[m].x = -(points[m].p.y - pc.y);
    dir[m].y =  (points[m].p.x - pc.x);
    dir[m]  /= dir[m].norm_L2();
  }

  // construct the vertices of the voronoi partition
  vector<double> theta_vertices(points.size());
  partition.resize(points.size());
  for(unsigned int m=0; m<points.size(); ++m)
  {
    unsigned int k = (m+1) % points.size();
    double denom = dir[m].cross(dir[k]);

    // if the points are aligned, keep the point that is the closest to pc
    if(denom < EPS)
    {
      if( (points[m].p-pc).norm_L2() > (points[k].p-pc).norm_L2() )
        k = m;
      points.erase(points.begin() + k);
      partition.erase(partition.begin() + k);
      middle.erase(middle.begin() + k);
      dir.erase(dir.begin() + k);
      theta_vertices.erase(theta_vertices.begin() + k);
      m -= m==k? 2:1;
      continue;
    }

    double lambda = ( dir[k].y*(middle[k].x-middle[m].x) - dir[k].x*(middle[k].y-middle[m].y) ) / denom;

    partition[m] = middle[m] + dir[m]*lambda;

    // compute the angle between the new vertex point and the reference point
    Point2 vm(partition[m]-pc);
    double dm = vm.norm_L2();
    double a = acos(v0.dot(vm)/(dm*d0));

    if(v0.cross(vm) < 0)
      a = 2.*PI-a;
    theta_vertices[m] = a;

    // check if the new vertex point is indeed a vertex of the voronoi partition
    if(m!=0)
    {
      k = (m-1) % points.size();

      // if the new vertex is before the previous one, in trigonometric order
      // or check for a double vertex, which means the new point [m] leads to an edge of length zero
      if( (partition[k]-pc).cross((partition[m]-pc)) < 0 || ABS(fmod(theta_vertices[m],2.*PI)-fmod(theta_vertices[k],2.*PI)) < EPS )
      {
        points.erase(points.begin() + m);
        partition.erase(partition.begin() + m);
        middle.erase(middle.begin() + m);
        dir.erase(dir.begin() + m);
        theta_vertices.erase(theta_vertices.begin() + m);
        m-=2;
      }
    }
  }
}


void Voronoi2D::clip_Interface( const CF_2& ls )
{
  set_Level_Set_Values(ls);
  clip_Interface();
}


void Voronoi2D::clip_Interface()
{
#ifdef CASL_THROWS
  if(phi_values.size()!=points.size() || phi_values.size()!= partition.size())
    throw std::invalid_argument("[CASL_THROWS]: Voronoi2D: the lists of points, vertices and/or level-set values do not have the same length.");
#endif

  /* the threshold used to tell if a point is in the negative domain
   * a point is in the negative domain if phi(n)<thresh.
   * note that using epsilon instead of zero eliminate the problem of cells
   * with very small areas
   */
  double thresh = -EPS;

  /* find a vertex that is in the negative domain */
  unsigned int m0 = 0;
  for(; m0<partition.size(); ++m0)
    if(phi_values[m0]<=thresh)
      break;

  /* the partition is entirely in the positive domain */
  if(m0>=partition.size())
  {
    points.resize(0);
    partition.resize(0);
    phi_values.resize(0);
    return;
  }

  /* otherwise clip the partition */
  unsigned int m = m0;
  unsigned int k = (m+1) % partition.size();
  do
  {
    /* the next vertex needs to be clipped */
    if(phi_values[k]>thresh)
    {
      Point2 dir;
      double theta;

      /* find the vertex between m and k */
      dir = partition[k] - partition[m];
      theta = fraction_Interval_Covered_By_Irregular_Domain(phi_values[m], phi_values[k], dir.norm_L2(), dir.norm_L2());
      Point2 pmk = partition[m] + dir*theta;

      /* find the following vertex */
      unsigned int l = k;
      unsigned int r = (l+1) % partition.size();
      while(phi_values[r]>thresh)
      {
        l = r;
        r = (l+1) % partition.size();
      }

      /* remove intermediate vertices if necessary */
      if(k!=l)
      {
        unsigned int h = (k+1) % partition.size();
        while(h!=l)
        {
          points.erase(points.begin() + h);
          partition.erase(partition.begin() + h);
          phi_values.erase(phi_values.begin() + h);

          if(h<m0) m0--;
          if(h<m)  m--;
          if(h<k)  k--;
          if(h<l)  l--;
          if(h<r)  r--;

          h = (k+1) % partition.size();
        }
      }

      /* find the vertex between l and r */
      dir = partition[r] - partition[l];
      theta = fraction_Interval_Covered_By_Irregular_Domain(phi_values[r], phi_values[l], dir.norm_L2(), dir.norm_L2());
      Point2 plr = partition[r] - dir*theta;

      /* add the vertices to the partition and create the ghost points */
      Point2 u = plr - pmk;

      /* the new points are on top of each other ... which can only mean that the value at point k=l is exactly 0 */
      if(u.norm_L2()<EPS)
      {
        m = r;
        continue;
      }

      u /= u.norm_L2();
      Voronoi2DPoint tmp;
      tmp.n = INTERFACE;
      Point2 n; n.x = u.y; n.y = -u.x;
      double d = (u.x*(pc.y-pmk.y) - u.y*(pc.x-pmk.x)) / n.cross(u);
      tmp.p = pc + n*2*d;

      partition[k] = pmk;
      phi_values[k] = 0;

      if(k==l)
      {
        partition.insert(partition.begin()+r, plr);
        phi_values.insert(phi_values.begin()+r, 0);
        points.insert(points.begin()+r, tmp);
        if(r<=m0) m0++;

        /* move on to next point */
        m = (r+1) % partition.size();
      }
      else
      {
        partition[l] = plr;
        phi_values[l] = 0;
        points[l] = tmp;

        /* move on to next point */
        m = (l+1) % partition.size();
      }
    }
    else
    {
      m = (m+1) % partition.size();
    }

    k = (m+1) % partition.size();
  } while(k!=m0 && m!=m0);

#ifdef CASL_THROWS
  if(partition.size()!=points.size() || phi_values.size()!=points.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->clip_Interface: error while clipping the interface.");
#endif
}

double Voronoi2D::volume() const
{
  double sum = 0.;
  for(unsigned int m=0; m<partition.size(); ++m)
  {
    unsigned int k = (m+partition.size()-1) % partition.size();
    Point2 u = partition[k]-pc;
    Point2 v = partition[m]-pc;
    sum += u.cross(v)/2.;
  }
  return sum;
}


bool Voronoi2D::is_Wall() const
{
  for(unsigned int m=0; m<points.size(); ++m)
    if(points[m].n==WALL_m00 || points[m].n==WALL_p00 || points[m].n==WALL_0m0 || points[m].n==WALL_0p0)
      return true;
  return false;
}


double Voronoi2D::area_In_Negative_Domain( const CF_2& ls ) const
{
  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    unsigned int k = (m+partition.size()-1) % partition.size();

    s.x0 = pc.x; s.y0 = pc.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integral(1, 1, 1, ls(s.x0,s.y0), ls(s.x1,s.y1), ls(s.x2,s.y2));
  }

  return sum;
}


double Voronoi2D::area_In_Negative_Domain() const
{
  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    int k = (m+1) % partition.size();

    s.x0 = pc.x; s.y0 = pc.y;
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
    int k = (m+1) % partition.size();

    s.x0 = pc.x; s.y0 = pc.y;
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
    int k = (m+1) % partition.size();

    s.x0 = pc.x; s.y0 = pc.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integral(fc, f[m], f[k], phi_c, phi_values[m], phi_values[k]);
  }

  return sum;
}

double Voronoi2D::integrate_Over_Interface( double fc, vector<double> &f ) const
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
    int k = (m+1) % partition.size();

    s.x0 = pc.x; s.y0 = pc.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integrate_Over_Interface(fc, f[m], f[k], phi_c, phi_values[m], phi_values[k]);
  }

  return sum;
}

double Voronoi2D::integrate_Over_Interface( const CF_2& f ) const
{
#ifdef CASL_THROWS
  if(phi_values.size() != partition.size())
    throw std::invalid_argument("[CASL_ERROR]: Voronoi2D->integrate_Over_Interface: you must call set_Level_Set_Values priori to integrating over the interface.");
#endif

  double sum = 0;
  Simplex2 s;

  for(unsigned int m=0; m<partition.size(); ++m)
  {
    int k = (m+1) % partition.size();

    s.x0 = pc.x; s.y0 = pc.y;
    s.x1 = partition[m].x; s.y1 = partition[m].y;
    s.x2 = partition[k].x; s.y2 = partition[k].y;

    sum += s.integrate_Over_Interface(f, phi_c, phi_values[m], phi_values[k]);
  }

  return sum;
}


void Voronoi2D::print_VTK_Format( const vector<Voronoi2D>& voro, std::string file_name )
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
      fprintf(fp, "%e\t %e\t 0.0\n", voro[n].partition[m].x, voro[n].partition[m].y);

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

void Voronoi2D::print_VTK_Format( const vector<Voronoi2D> &voro, const vector<double> &f, std::string data_name, std::string file_name )
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
      fprintf(fp, "%e\n", f[n]);

  fclose(fp);
}


void Voronoi2D::print_VTK_Format( const vector<Voronoi2D> &voro, const vector<double> &u, const vector<double> &v, std::string data_name, std::string file_name )
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
      fprintf(fp, "%e\t %e\n", u[n], v[n]);

  fclose(fp);
}


std::ostream& operator<<(std::ostream& os, const Voronoi2D& v)
{
  os << "Center point : " << v.pc.x << "," << v.pc.y << std::endl;

  for (unsigned int n=0; n<v.points.size(); n++)
    os << v.points[n].n << " : (" << v.points[n].p.x << "," << v.points[n].p.y << "," << v.points[n].theta << ")" << std::endl;
  os << std::endl;

  return os;
}
