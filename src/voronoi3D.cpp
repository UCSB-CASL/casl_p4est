#include "voronoi3D.h"
#include <vector>

void Voronoi3D::clear()
{
  points.resize(0);
}

void Voronoi3D::get_Points( const vector<Voronoi3DPoint>*& points) const
{
  points = &this->points;
}

void Voronoi3D::push( int n, double x, double y,double z )
{
  for(unsigned int m=0; m<points.size(); m++)
  {
    if(points[m].n == n)
    {
      return;
    }
  }

  Voronoi3DPoint p;
  p.n = n;
  p.p.x = x;
  p.p.y = y;
  p.p.z = z;
  points.push_back(p);
}

void Voronoi3D::push( int n, Point3 pt )
{
  for(unsigned int m=0; m<points.size(); m++)
  {
    if(points[m].n == n)
    {
      return;
    }
  }

  Voronoi3DPoint p;
  p.n = n;
  p.p = pt;
  points.push_back(p);
}

void Voronoi3D::set_Center_Point( int nc, Point3 pc )
{
  this->nc = nc;
  this->pc = pc;
}

void Voronoi3D::set_Center_Point( int nc, double x, double y, double z)
{
  this->nc = nc;
  pc.x = x;
  pc.y = y;
  pc.z = z;
}

void Voronoi3D::construct_Partition(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
                                    bool periodic_x, bool periodic_y, bool periodic_z)
{
  /* create a container for the particles */
  voro::container voronoi(xmin, xmax, ymin, ymax, zmin, zmax,
                          1, 1, 1, periodic_x, periodic_y, periodic_z, 8);

  /* store the order in which the particles are added to the container */
  voro::particle_order po;

  /* add the center point */
  double x_tmp = ABS(pc.x-xmin)<EPS ? pc.x+EPS : ABS(pc.x-xmax)<EPS ? pc.x-EPS : pc.x;
  double y_tmp = ABS(pc.y-ymin)<EPS ? pc.y+EPS : ABS(pc.y-ymax)<EPS ? pc.y-EPS : pc.y;
  double z_tmp = ABS(pc.z-zmin)<EPS ? pc.z+EPS : ABS(pc.z-zmax)<EPS ? pc.z-EPS : pc.z;
  voronoi.put(po, nc, x_tmp, y_tmp, z_tmp);

  /* add the points potentially involved in the voronoi partition */
  for(unsigned int m=0; m<points.size(); ++m)
  {
    double x_tmp = ABS(points[m].p.x-xmin)<EPS ? points[m].p.x+EPS : ABS(points[m].p.x-xmax)<EPS ? points[m].p.x-EPS : points[m].p.x;
    double y_tmp = ABS(points[m].p.y-ymin)<EPS ? points[m].p.y+EPS : ABS(points[m].p.y-ymax)<EPS ? points[m].p.y-EPS : points[m].p.y;
    double z_tmp = ABS(points[m].p.z-zmin)<EPS ? points[m].p.z+EPS : ABS(points[m].p.z-zmax)<EPS ? points[m].p.z-EPS : points[m].p.z;

    voronoi.put(po, points[m].n, x_tmp, y_tmp, z_tmp);
  }

  voro::voronoicell_neighbor voro_cell;
  std::vector<int> neigh;
  std::vector<double> areas;

  voro::c_loop_order cl(voronoi, po);
  if(cl.start() && voronoi.compute_cell(voro_cell,cl))
  {
    vector<Voronoi3DPoint> final_points;
    volume_ = voro_cell.volume();

    voro_cell.neighbors(neigh);
    voro_cell.face_areas(areas);

    for(unsigned int n=0; n<neigh.size(); n++)
    {
      struct Voronoi3DPoint new_voro_nb;
      new_voro_nb.n = neigh[n];
      new_voro_nb.s = areas[n];

      if(neigh[n]<0)
      {
        switch(neigh[n])
        {
        case WALL_m00:
          new_voro_nb.p.x = xmin-(pc.x-xmin); new_voro_nb.p.y = pc.y; new_voro_nb.p.z = pc.z;
          break;
        case WALL_p00:
          new_voro_nb.p.x = xmax+(xmax-pc.x); new_voro_nb.p.y = pc.y; new_voro_nb.p.z = pc.z;
          break;
        case WALL_0m0:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = ymin-(pc.y-ymin); new_voro_nb.p.z = pc.z;
          break;
        case WALL_0p0:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = ymax+(ymax-pc.y); new_voro_nb.p.z = pc.z;
          break;
        case WALL_00m:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = pc.y; new_voro_nb.p.z = zmin-(pc.z-zmin);
          break;
        case WALL_00p:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = pc.y; new_voro_nb.p.z = zmax+(zmax-pc.z);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: Voronoi3D->construct_Partition: unknown boundary.");
        }
      }
      else
      {
        for(unsigned int m=0; m<points.size(); ++m)
        {
          if(points[m].n==neigh[n])
          {
            new_voro_nb.p = points[m].p;
            break;
          }
        }
      }

      final_points.push_back(new_voro_nb);
    }
    points.clear();
    points = final_points;
  }
}



void Voronoi3D::print_VTK_Format( const std::vector<Voronoi3D>& voro, const char* file_name )
{
//  std::ofstream f;
//  f.open(file_name);
//#ifdef CASL_THROWS
//  if(!f.is_open()) throw std::invalid_argument("[CASL_ERROR]: Voronoi3D: cannot open file.");
//#endif

//  struct Voro_Ngbd {
//    voro::container* voronoi;
//    voro::particle_order* po;
//    Voro_Ngbd() : voronoi(NULL), po(NULL) {}
//    ~Voro_Ngbd()
//    {
//        if(voronoi!=NULL) delete voronoi;
//        if(po!=NULL) delete po;
//    }
//  };

//  vector<Voro_Ngbd> voro_global(voro.size());
//  for(unsigned int n=0; n<voro.size(); ++n)
//  {
//    voro_global(n).voronoi = new voro::container(-1, 1, -1, 1, -1, 1,
//                                                 1, 1, 1, false, false, false, 8);
//    voro_global(n).po = new voro::particle_order;

//    voro_global(n).voronoi->put(*voro_global(n).po, voro[n].nc, voro[n].pc.x, voro[n].pc.y, voro[n].pc.z);
//    for(int m=0; m<voro[n].points.size(); ++m)
//      voro_global(n).voronoi->put(*voro_global(n).po, voro[n].points(m).n, voro[n].points(m).p.x, voro[n].points(m).p.y, voro[n].points(m).p.z);
//  }


//  voro::voronoicell_neighbor c;
//  vector<int> neigh, f_vert;
//  vector<double> v;
//  double x, y, z;
//  int pid; double r;
//  int j, k;
//  unsigned int i;

//  int nb_vertices = 0;
//  int nb_polygons = 0;
//  int nb_poly_vert = 0;

//  // first count the number of vertices and polygons
//  for(CaslInt n=0; n<voro_global.size(); n++)
//  {
//    if(voro_global(n).voronoi!=NULL)
//    {
//      voro::c_loop_order cl(*voro_global(n).voronoi,*voro_global(n).po);
//      if(cl.start() && cl.pid()==n && voro_global(n).voronoi->compute_cell(c,cl))
//      {
//        cl.pos(pid,x,y,z,r);
//        c.neighbors(neigh);
//        c.vertices(v);
//        c.face_vertices(f_vert);

//        nb_vertices += v.size() / 3;
//        nb_polygons += neigh.size();
//        nb_poly_vert += f_vert.size();
//      }
//    }
//  }

//  // add the vertices information to the VTK file
//  f << "# vtk DataFile Version 2.0" << endl;
//  f << "Voronoi partition" << endl;
//  f << "ASCII" << endl << "DATASET UNSTRUCTURED_GRID" << endl;
//  f << endl;

//  /* output the list of points */
//  f << "POINTS " << nb_vertices << " double" << endl;

//  for(CaslInt n=0; n<voro_global.size(); n++)
//  {
//    if(voro_global(n).voronoi!=NULL)
//    {
//      voro::c_loop_order cl(*voro_global(n).voronoi,*voro_global(n).po);
//      if(cl.start() && cl.pid()==n && voro_global(n).voronoi->compute_cell(c,cl))
//      {
//        cl.pos(pid,x,y,z,r);
//        c.vertices(x,y,z,v);

//        for(i=0; i<v.size(); i+=3)
//          f << v[i] << " " << v[i+1] << " " << v[i+2] << endl;
//      }
//    }
//  }

//  /* output the list of polygons */
//  f << endl << "CELLS " << nb_polygons << " " << nb_poly_vert << endl;
//  int offset = 0;
//  for(CaslInt n=0; n<voro_global.size(); n++)
//  {
//    if(voro_global(n).voronoi!=NULL)
//    {
//      voro::c_loop_order cl(*voro_global(n).voronoi,*voro_global(n).po);
//      if(cl.start() && cl.pid()==n && voro_global(n).voronoi->compute_cell(c,cl))
//      {
//        cl.pos(pid,x,y,z,r);
//        c.neighbors(neigh);
//        c.face_vertices(f_vert);
//        c.vertices(x,y,z,v);

//        for(j=0, i=0; i<neigh.size(); i++)
//        {
//          f << f_vert[j];
//          for(k=0; k<f_vert[j]; k++)
//            f << " " << f_vert[j+k+1] + offset;
//          f << endl;
//          j += f_vert[j]+1;
//        }
//        offset += v.size()/3;
//      }
//    }
//  }

//  /* now specify the type of each polygon, here a VTK_POLYGON (=7) */
//  f << endl << "CELL_TYPES " << nb_polygons << endl;
//  for(CaslInt n=0; n<voro_global.size(); n++)
//  {
//    if(voro_global(n).voronoi!=NULL)
//    {
//      voro::c_loop_order cl(*voro_global(n).voronoi,*voro_global(n).po);
//      if(cl.start() && cl.pid()==n && voro_global(n).voronoi->compute_cell(c,cl))
//      {
//        cl.pos(pid,x,y,z,r);
//        c.neighbors(neigh);

//        for(i=0; i<neigh.size(); i++)
//          f << "7" << endl;
//      }
//    }
//  }

//  f.close();

//  cout << "Saved voronoi partition in " << file_name << endl;
}
