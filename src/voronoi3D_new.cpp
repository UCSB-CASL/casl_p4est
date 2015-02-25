#include "voronoi3D_new.h"
#include <vector>
#include <algorithm>

namespace CASL
{

void Voronoi3D_NEW::clear()
{
  points.resize(0);
}

void Voronoi3D_NEW::get_Points( const vector<Voronoi3D_NEWPoint>*& points) const
{
  points = &this->points;
}

void Voronoi3D_NEW::push( int n, double x, double y,double z )
{
  if(n==nc)
    return;
  for(unsigned int m=0; m<partitions.size(); m++)
  {
    if(partitions[m].n == n)
    {
      return;
    }
  }

  candidate_t c;
  c.n = n;
  c.p.x=x; c.p.y=y; c.p.z=z;
  partitions.push_back(c);
  init_polygon(partitions.size()-1);

  for(unsigned int i=0; i<partitions.size()-1; ++i)
  {
    /* if partition i was deleted, then no need to cut the new one by i and i needs to be adjusted */
    if(cut_polygon(i, partitions.size()-1))
      i--;
    /* if i was deleted, no more modifications to bring to other partitions */
    else if(cut_polygon(partitions.size()-1, i))
      break;
  }
}

void Voronoi3D_NEW::push( int n, Point3 &pt )
{
  push(n, pt.x, pt.y, pt.z);
}

void Voronoi3D_NEW::set_Center_Point( int nc, Point3 &pc )
{
  set_Center_Point(nc, pc.x, pc.y, pc.z);
}

void Voronoi3D_NEW::set_Center_Point( int nc, double x, double y, double z)
{
  this->nc = nc;
  pc.x = x;
  pc.y = y;
  pc.z = z;

  partitions.clear();
  points.clear();

  partitions.resize(6);

  partitions[0].n = WALL_m00;
  partitions[0].p = Point3(pc.x-2*(pc.x-xmin), pc.y, pc.z);
  partitions[0].u = Point3(0, 1, 0);
  partitions[0].v = Point3(0, 0, 1);
  partitions[0].norm = Point3(-1, 0, 0);
  partitions[0].polygon.push_back(Point3(xmin, ymin, zmin));
  partitions[0].polygon.push_back(Point3(xmin, ymax, zmin));
  partitions[0].polygon.push_back(Point3(xmin, ymax, zmax));
  partitions[0].polygon.push_back(Point3(xmin, ymin, zmax));

  partitions[1].n = WALL_p00;
  partitions[1].p = Point3(pc.x+2*(xmax-pc.x), pc.y, pc.z);
  partitions[1].u = Point3(0, 1, 0);
  partitions[1].v = Point3(0, 0, 1);
  partitions[1].norm = Point3(1, 0, 0);
  partitions[1].polygon.push_back(Point3(xmax, ymin, zmin));
  partitions[1].polygon.push_back(Point3(xmax, ymax, zmin));
  partitions[1].polygon.push_back(Point3(xmax, ymax, zmax));
  partitions[1].polygon.push_back(Point3(xmax, ymin, zmax));

  partitions[2].n = WALL_0m0;
  partitions[2].p = Point3(pc.x, pc.y-2*(pc.y-ymin), pc.z);
  partitions[2].u = Point3(1, 0, 0);
  partitions[2].v = Point3(0, 0, 1);
  partitions[2].norm = Point3(0,-1, 0);
  partitions[2].polygon.push_back(Point3(xmin, ymin, zmin));
  partitions[2].polygon.push_back(Point3(xmax, ymin, zmin));
  partitions[2].polygon.push_back(Point3(xmax, ymin, zmax));
  partitions[2].polygon.push_back(Point3(xmin, ymin, zmax));

  partitions[3].n = WALL_0p0;
  partitions[3].p = Point3(pc.x, pc.y+2*(ymax-pc.y), pc.z);
  partitions[3].u = Point3(1, 0, 0);
  partitions[3].v = Point3(0, 0, 1);
  partitions[3].norm = Point3(0, 1, 0);
  partitions[3].polygon.push_back(Point3(xmin, ymax, zmin));
  partitions[3].polygon.push_back(Point3(xmax, ymax, zmin));
  partitions[3].polygon.push_back(Point3(xmax, ymax, zmax));
  partitions[3].polygon.push_back(Point3(xmin, ymax, zmax));

  partitions[4].n = WALL_00m;
  partitions[4].p = Point3(pc.x, pc.y, pc.z-2*(pc.z-zmin));
  partitions[4].u = Point3(1, 0, 0);
  partitions[4].v = Point3(0, 1, 0);
  partitions[4].norm = Point3(0, 0,-1);
  partitions[4].polygon.push_back(Point3(xmin, ymin, zmin));
  partitions[4].polygon.push_back(Point3(xmax, ymin, zmin));
  partitions[4].polygon.push_back(Point3(xmax, ymax, zmin));
  partitions[4].polygon.push_back(Point3(xmin, ymax, zmin));

  partitions[5].n = WALL_00p;
  partitions[5].p = Point3(pc.x, pc.y, pc.z+2*(zmax-pc.z));
  partitions[5].u = Point3(1, 0, 0);
  partitions[5].v = Point3(0, 1, 0);
  partitions[5].norm = Point3(0, 0, 1);
  partitions[5].polygon.push_back(Point3(xmin, ymin, zmax));
  partitions[5].polygon.push_back(Point3(xmax, ymin, zmax));
  partitions[5].polygon.push_back(Point3(xmax, ymax, zmax));
  partitions[5].polygon.push_back(Point3(xmin, ymax, zmax));

//  push(WALL_m00, pc.x-2*(pc.x-xmin), pc.y, pc.z);
//  push(WALL_p00, pc.x+2*(xmax-pc.x), pc.y, pc.z);
//  push(WALL_0m0, pc.x, pc.y-2*(pc.y-ymin), pc.z);
//  push(WALL_0p0, pc.x, pc.y+2*(ymax-pc.y), pc.z);
//  push(WALL_00m, pc.x, pc.y, pc.z-2*(pc.z-zmin));
//  push(WALL_00p, pc.x, pc.y, pc.z+2*(zmax-pc.z));
}




int Voronoi3D_NEW::cpt_restart = 0;
void Voronoi3D_NEW::init_polygon(int n)
{
  Point3 &norm = partitions[n].norm;
  switch(partitions[n].n)
  {
  case WALL_m00: norm.x=-1; norm.y= 0; norm.z= 0; break;
  case WALL_p00: norm.x= 1; norm.y= 0; norm.z= 0; break;
  case WALL_0m0: norm.x= 0; norm.y=-1; norm.z= 0; break;
  case WALL_0p0: norm.x= 0; norm.y= 1; norm.z= 0; break;
  case WALL_00m: norm.x= 0; norm.y= 0; norm.z=-1; break;
  case WALL_00p: norm.x= 0; norm.y= 0; norm.z= 1; break;
  default: norm = (partitions[n].p-pc).normalize();
  }

  Point3 pcn = (partitions[n].p+pc)*.5;

  Point3& u = partitions[n].u;
  Point3& v = partitions[n].v;
  int cptv = -1;
  do
  {
    Point3 rd((double) rand()/RAND_MAX-.5, (double) rand()/RAND_MAX-.5, (double) rand()/RAND_MAX-.5);
    int cptu = 0;

    while(rd.norm_L2()<1e-4 || fabs(rd.normalize().dot(norm))>cos(PI/20))
    {
      cptu++;
      rd = Point3((double) rand()/RAND_MAX-.5, (double) rand()/RAND_MAX-.5, (double) rand()/RAND_MAX-.5);
    }
    if(cptu!=0)
    {
      cpt_restart++;
    }

    rd /= rd.norm_L2();

    u = rd - norm*(rd.dot(norm));
    u /= u.norm_L2();

    v = u.cross(norm);
    cptv++;
  } while(v.norm_L2()<.1);
  if(cptv!=0) std::cout << nc << " : cptv = " << cptv << std::endl;

  v /= v.norm_L2();

  partitions[n].polygon.push_back(pcn + ( u+v)*big);
  partitions[n].polygon.push_back(pcn + (-u+v)*big);
  partitions[n].polygon.push_back(pcn + (-u-v)*big);
  partitions[n].polygon.push_back(pcn + ( u-v)*big);
}


/* cut polygon n by plane generated by point l */
bool Voronoi3D_NEW::cut_polygon(int n, int l)
{
  Point3 pcl = (partitions[l].p + pc)*.5;

  Point3 &u = partitions[l].u;
  Point3 &v = partitions[l].v;

  /* 0 - keep
   * 1 - new
   * 2 - delete
   */
  std::vector<int> status(partitions[n].polygon.size(), 0);

  for(int i=0; (unsigned int) i<partitions[n].polygon.size(); ++i)
  {
    int k = mod(i-1, partitions[n].polygon.size());

    Point3 &pk = partitions[n].polygon[k];
    Point3 &pi = partitions[n].polygon[i];
    Point3 pik = pk - pi;

    /* compute the intersection with the plane */
    Point3 pcli = pi - pcl;

    double det = u.x*v.y*pik.z + u.y*v.z*pik.x + u.z*v.x*pik.y - u.x*v.z*pik.y - u.y*v.x*pik.z - u.z*v.y*pik.x;

    if(fabs(det)>EPS)
    {
      double alpha = ( (u.z*v.y-u.y*v.z)*pcli.x + (u.x*v.z-u.z*v.x)*pcli.y + (u.y*v.x-u.x*v.y)*pcli.z ) / det;

      //        double a = ( (v.y*pik.z-v.z*pik.y)*pcli.x + (v.z*pik.x-v.x*pik.z)*pcli.y + (v.x*pik.y-v.y*pik.x)*pcli.z ) / det;
      //        double b = ( (u.z*pik.y-u.y*pik.z)*pcli.x + (u.x*pik.z-u.z*pik.x)*pcli.y + (u.y*pik.x-u.x*pik.y)*pcli.z ) / det;

      Point3 p_new = pi + pik*alpha;

      if((p_new-partitions[n].polygon[k]).norm_L2()>EPS && (p_new-partitions[n].polygon[i]).norm_L2()>EPS
         && alpha>0 && alpha<1)
      {

        if(partitions[l].norm.dot(pik) > 0)
          status[k] = 2;
        else
          status[i] = 2;

        partitions[n].polygon.insert(partitions[n].polygon.begin()+i, p_new);
        status.insert(status.begin()+i, 1);

        i++;
      }
      else if((p_new-partitions[n].polygon[i]).norm_L2()<EPS && partitions[l].norm.dot(pik) > 0)
      {
        status[i] = 1;
        status[k] = 2;
      }
      else if((p_new-partitions[n].polygon[k]).norm_L2()<EPS && partitions[l].norm.dot(pik) < 0)
      {
        status[i] = 2;
        status[k] = 1;
      }
    }
  }

  /* if there was no intersection at all, the polygon might need to be completely deleted */
  if(std::find(status.begin(), status.end(), 2)==status.end())
  {
    bool all_same_side = true;
    for(unsigned int i=0; i<partitions[n].polygon.size(); ++i)
    {
      if(partitions[l].norm.dot(pcl-partitions[n].polygon[i]) > EPS)
      {
        all_same_side = false;
        break;
      }
    }
    if(all_same_side)
    {
      partitions.erase(partitions.begin()+n);
      return true;
    }
  }
  else
  {
    /* propagate delete information if many vertices in a row must be removed */
    bool trig = false;
    for(int j=0; (unsigned int) j<2*status.size(); ++j)
    {
      int i = j%status.size();
      int k = mod(i-1, status.size());

      if(trig && status[i]==0)
        status[i] = 2;
      else if(status[i]==2 && status[k]==1)
        trig = true;
      else if(status[i]==1 && status[k]==2)
        trig = false;
    }

    /* delete points */
    int cpt = 0;
    for(int i=0; (unsigned int) i<status.size(); ++i)
    {
      if(status[i]==2)
      {
        partitions[n].polygon.erase(partitions[n].polygon.begin()+i-cpt);
        cpt++;
      }
    }

    if(partitions[n].polygon.size()<3)
    {
      partitions.erase(partitions.begin()+n);
      return true;
    }
  }

  return false;
}


void Voronoi3D_NEW::construct_Partition()
{
  points.resize(partitions.size());

  for(unsigned int i=0; i<partitions.size(); ++i)
  {
    points[i].n = partitions[i].n;
    points[i].p = partitions[i].p;

    Point3 center(0,0,0);
    for(int k=0; k<(int) partitions[i].polygon.size(); k++)
      center += partitions[i].polygon[k];
    center /= (double) partitions[i].polygon.size();

    points[i].s = 0;
    volume_     = 0;
    for(int k=0; k<(int) partitions[i].polygon.size(); k++)
    {
      int l = mod(k-1, partitions[i].polygon.size());
      Point3 a = partitions[i].polygon[k] - center;
      Point3 b = partitions[i].polygon[l] - center;
      points[i].s += a.cross(b).norm_L2()/2;

      a = partitions[i].polygon[k] - pc;
      b = partitions[i].polygon[l] - pc;
      Point3 c = center - pc;
      volume_ += fabs(a.dot(b.cross(c)))/6;
    }
  }

  partitions.clear();
}


bool Voronoi3D_NEW::check_Partition(const vector<Voronoi3D_NEW>& voro)
{
  for(int n=0; n<(int) voro.size(); ++n)
  {
    for(unsigned int m=0; m<voro[n].points.size(); ++m)
    {
      if(voro[n].points[m].n >= 0)
      {
        unsigned int ngbd = voro[n].points[m].n;
        unsigned int k=0;
        for(k=0; k<voro[ngbd].points.size(); ++k)
        {
          if(voro[ngbd].points[k].n == n)
            break;
        }
//        if(k==voro[ngbd].points.size())
//        {
//          std::cout << "neighbor missing " << n << ", " << ngbd << std::endl;
//          std::cout << voro[n].pc << voro[ngbd].pc;
//        }
//        else if(fabs(voro[n].points[m].s - voro[ngbd].points[k].s)>EPS)
//        {
//          std::cout << "area error " << fabs(voro[n].points[m].s - voro[ngbd].points[k].s) << std::endl;
//          std::cout << n << ", " << ngbd << std::endl;
//        }
        if(k==voro[ngbd].points.size() || fabs(voro[n].points[m].s - voro[ngbd].points[k].s)>EPS)
          return false;
      }
    }
  }

  return true;
}



void Voronoi3D_NEW::print_VTK_Format( std::vector<Voronoi3D_NEW>& voro, const char* file_name )
{
  FILE* f;
  f = fopen(file_name, "w");
#ifdef CASL_THROWS
  if(f==NULL) throw std::invalid_argument("[CASL_ERROR]: Voronoi3D: cannot open file.");
#endif

  for(unsigned int n=0; n<voro.size(); ++n)
  {
    if(voro[n].partitions.size()==0)
      for(unsigned int k=0; k<voro[n].points.size(); ++k)
      {
        voro[n].push(voro[n].points[k].n, voro[n].points[k].p);
      }
  }

  int nb_vertices = 0;
  int nb_polygons = 0;

  /* first count the number of vertices and polygons */
  for(unsigned int n=0; n<voro.size(); n++)
  {
    for(unsigned int nb=0; nb<voro[n].partitions.size(); ++nb)
    {
      if(voro[n].partitions[nb].n < (int) n)
      {
        nb_polygons += 1;
        nb_vertices += voro[n].partitions[nb].polygon.size();
      }
    }
  }

  fprintf(f, "# vtk DataFile Version 2.0\n");
  fprintf(f, "Voronoi partition\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET POLYDATA\n");


  /* add the vertices information to the VTK file */
  fprintf(f, "POINTS %d double\n", nb_vertices);
  for(unsigned int n=0; n<voro.size(); n++)
  {
    for(unsigned int nb=0; nb<voro[n].partitions.size(); ++nb)
    {
      if(voro[n].partitions[nb].n < (int) n)
      {
        for(unsigned int k=0; k<voro[n].partitions[nb].polygon.size(); ++k)
          fprintf(f, "%e %e %e\n", voro[n].partitions[nb].polygon[k].x, voro[n].partitions[nb].polygon[k].y, voro[n].partitions[nb].polygon[k].z);
      }
    }
  }


  /* output the list of polygons */
  fprintf(f, "POLYGONS %d %d\n", nb_polygons, nb_vertices+nb_polygons);
  int cpt = 0;
  for(unsigned int n=0; n<voro.size(); n++)
  {
    for(unsigned int nb=0; nb<voro[n].partitions.size(); ++nb)
    {
      if(voro[n].partitions[nb].n < (int) n)
      {
        fprintf(f, "%lu ", voro[n].partitions[nb].polygon.size());
        for(unsigned int k=0; k<voro[n].partitions[nb].polygon.size(); ++k)
        {
          fprintf(f, "%d ", cpt);
          cpt++;
        }
        fprintf(f, "\n");
      }
    }
  }

  fclose(f);

  for(unsigned int n=0; n<voro.size(); ++n)
    voro[n].partitions.clear();

  printf("Saved voronoi partition in %s\n", file_name);
}

} /* namespace CASL */
