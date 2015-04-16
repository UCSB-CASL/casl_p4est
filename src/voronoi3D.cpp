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

void Voronoi3D::push( int n, Point3 &pt )
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

void Voronoi3D::set_Center_Point( int nc, Point3 &pc, double scaling )
{
  this->nc = nc;
  this->pc = pc;
  this->scaling = scaling;
}

void Voronoi3D::set_Center_Point( int nc, double x, double y, double z, double scaling)
{
  this->nc = nc;
  pc.x = x;
  pc.y = y;
  pc.z = z;
  this->scaling = scaling;
}

void Voronoi3D::construct_Partition(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
                                    bool periodic_x, bool periodic_y, bool periodic_z)
{

//  double xmin_ = MAX((xmin-pc.x)/scaling-10, (xmin-pc.x)/scaling);
//  double xmax_ = MIN((xmax-pc.x)/scaling+10, (xmax-pc.x)/scaling);
//  double ymin_ = MAX((ymin-pc.y)/scaling-10, (ymin-pc.y)/scaling);
//  double ymax_ = MIN((ymax-pc.y)/scaling+10, (ymax-pc.y)/scaling);
//  double zmin_ = MAX((zmin-pc.z)/scaling-10, (zmin-pc.z)/scaling);
//  double zmax_ = MIN((zmax-pc.z)/scaling+10, (zmax-pc.z)/scaling);

//  /* create a container for the particles */
//  voro::container voronoi(xmin_, xmax_, ymin_, ymax_, zmin_, zmax_,
//                          1, 1, 1, periodic_x, periodic_y, periodic_z, 8);

//  /* store the order in which the particles are added to the container */
//  voro::particle_order po;

//  /* add the center point */
//  double x_tmp = 0; //pc.x/scaling;
//  double y_tmp = 0; //pc.y/scaling;
//  double z_tmp = 0; //pc.z/scaling;
//  x_tmp = ABS(x_tmp-xmin_)<EPS ? x_tmp+EPS : ABS(x_tmp-xmax_)<EPS ? x_tmp-EPS : x_tmp;
//  y_tmp = ABS(y_tmp-ymin_)<EPS ? y_tmp+EPS : ABS(y_tmp-ymax_)<EPS ? y_tmp-EPS : y_tmp;
//  z_tmp = ABS(z_tmp-zmin_)<EPS ? z_tmp+EPS : ABS(z_tmp-zmax_)<EPS ? z_tmp-EPS : z_tmp;
//  voronoi.put(po, nc, x_tmp, y_tmp, z_tmp);

//  /* add the points potentially involved in the voronoi partition */
//  for(unsigned int m=0; m<points.size(); ++m)
//  {
//    double x_tmp = (points[m].p.x-pc.x)/scaling;
//    double y_tmp = (points[m].p.y-pc.y)/scaling;
//    double z_tmp = (points[m].p.z-pc.z)/scaling;
//    x_tmp = ABS(x_tmp-xmin_)<EPS ? x_tmp+EPS : ABS(x_tmp-xmax_)<EPS ? x_tmp-EPS : x_tmp;
//    y_tmp = ABS(y_tmp-ymin_)<EPS ? y_tmp+EPS : ABS(y_tmp-ymax_)<EPS ? y_tmp-EPS : y_tmp;
//    z_tmp = ABS(z_tmp-zmin_)<EPS ? z_tmp+EPS : ABS(z_tmp-zmax_)<EPS ? z_tmp-EPS : z_tmp;
//    voronoi.put(po, points[m].n, x_tmp, y_tmp, z_tmp);
//  }




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
//    volume_ = voro_cell.volume() / (scaling*scaling*scaling);
    volume_ = voro_cell.volume();

    voro_cell.neighbors(neigh);
    voro_cell.face_areas(areas);

    for(unsigned int n=0; n<neigh.size(); n++)
    {
      struct Voronoi3DPoint new_voro_nb;
      new_voro_nb.n = neigh[n];
//      new_voro_nb.s = areas[n] / (scaling*scaling);
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



void Voronoi3D::print_VTK_Format( const std::vector<Voronoi3D>& voro, const char* file_name,
                                  double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
                                  bool periodic_x, bool periodic_y, bool periodic_z)
{
  FILE* f;
  f = fopen(file_name, "w");
#ifdef CASL_THROWS
  if(f==NULL) throw std::invalid_argument("[CASL_ERROR]: Voronoi3D: cannot open file.");
#endif

  vector<VoroNgbd> voro_global(voro.size());
  for(unsigned int n=0; n<voro.size(); ++n)
  {
    voro_global[n].voronoi = new voro::container(xmin, xmax, ymin, ymax, zmin, zmax,
                                                 1, 1, 1, periodic_x, periodic_y, periodic_z, 8);
    voro_global[n].po = new voro::particle_order;

    double x_c = ABS(voro[n].pc.x-xmin)<EPS ? voro[n].pc.x+EPS : ABS(voro[n].pc.x-xmax)<EPS ? voro[n].pc.x-EPS : voro[n].pc.x;
    double y_c = ABS(voro[n].pc.y-ymin)<EPS ? voro[n].pc.y+EPS : ABS(voro[n].pc.y-ymax)<EPS ? voro[n].pc.y-EPS : voro[n].pc.y;
    double z_c = ABS(voro[n].pc.z-zmin)<EPS ? voro[n].pc.z+EPS : ABS(voro[n].pc.z-zmax)<EPS ? voro[n].pc.z-EPS : voro[n].pc.z;
    voro_global[n].voronoi->put(*voro_global[n].po, voro[n].nc, x_c, y_c, z_c);

    for(unsigned int m=0; m<voro[n].points.size(); ++m)
      if(voro[n].points[m].n>=0)
      {
        double x_m = ABS(voro[n].points[m].p.x-xmin)<EPS ? voro[n].points[m].p.x+EPS : ABS(voro[n].points[m].p.x-xmax)<EPS ? voro[n].points[m].p.x-EPS : voro[n].points[m].p.x;
        double y_m = ABS(voro[n].points[m].p.y-ymin)<EPS ? voro[n].points[m].p.y+EPS : ABS(voro[n].points[m].p.y-ymax)<EPS ? voro[n].points[m].p.y-EPS : voro[n].points[m].p.y;
        double z_m = ABS(voro[n].points[m].p.z-zmin)<EPS ? voro[n].points[m].p.z+EPS : ABS(voro[n].points[m].p.z-zmax)<EPS ? voro[n].points[m].p.z-EPS : voro[n].points[m].p.z;
        voro_global[n].voronoi->put(*voro_global[n].po, voro[n].points[m].n, x_m, y_m, z_m);
      }
  }


  voro::voronoicell_neighbor c;
  vector<int> neigh, f_vert;
  vector<double> v;
  double x, y, z;
  int pid; double r;
  int j, k;
  unsigned int i;

  int nb_vertices = 0;
  int nb_polygons = 0;
  int nb_poly_vert = 0;

  // first count the number of vertices and polygons
  for(unsigned int n=0; n<voro_global.size(); n++)
  {
    if(voro_global[n].voronoi!=NULL)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].nc && voro_global[n].voronoi->compute_cell(c,cl))
      {
        cl.pos(pid,x,y,z,r);
        c.neighbors(neigh);
        c.vertices(v);
        c.face_vertices(f_vert);

        nb_vertices += v.size() / 3;
        nb_polygons += neigh.size();
        nb_poly_vert += f_vert.size();
      }
    }
  }

  // add the vertices information to the VTK file
  fprintf(f, "# vtk DataFile Version 2.0\n");
  fprintf(f, "Voronoi partition\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET UNSTRUCTURED_GRID\n\n");

  /* output the list of points */
  fprintf(f, "POINTS %d double\n", nb_vertices);

  for(unsigned int n=0; n<voro_global.size(); n++)
  {
    if(voro_global[n].voronoi!=NULL)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].nc && voro_global[n].voronoi->compute_cell(c,cl))
      {
        cl.pos(pid,x,y,z,r);
        c.vertices(x,y,z,v);

        for(i=0; i<v.size(); i+=3)
          fprintf(f, "%e %e %e\n", v[i], v[i+1], v[i+2]);
      }
    }
  }

  /* output the list of polygons */
  fprintf(f, "\nCELLS %d %d\n", nb_polygons, nb_poly_vert);
  int offset = 0;
  for(unsigned int n=0; n<voro_global.size(); n++)
  {
    if(voro_global[n].voronoi!=NULL)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].nc && voro_global[n].voronoi->compute_cell(c,cl))
      {
        cl.pos(pid,x,y,z,r);
        c.neighbors(neigh);
        c.face_vertices(f_vert);
        c.vertices(x,y,z,v);

        for(j=0, i=0; i<neigh.size(); i++)
        {
          fprintf(f, "%d", f_vert[j]);
          for(k=0; k<f_vert[j]; k++)
            fprintf(f, " %d", f_vert[j+k+1]+offset);
          fprintf(f, "\n");
          j += f_vert[j]+1;
        }
        offset += v.size()/3;
      }
    }
  }

  /* now specify the type of each polygon, here a VTK_POLYGON (=7) */
  fprintf(f, "\nCELL_TYPES %d\n", nb_polygons);
  for(unsigned int n=0; n<voro_global.size(); n++)
  {
    if(voro_global[n].voronoi!=NULL)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].nc && voro_global[n].voronoi->compute_cell(c,cl))
      {
        cl.pos(pid,x,y,z,r);
        c.neighbors(neigh);

        for(i=0; i<neigh.size(); i++)
          fprintf(f, "7\n");
      }
    }
  }

  fclose(f);

  printf("Saved voronoi partition in %s\n", file_name);
}
