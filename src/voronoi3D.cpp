#include "voronoi3D.h"
#include <vector>

#include <algorithm>

void Voronoi3D::set_cell(vector<ngbd3Dseed> &neighbors, double volume )
{
  this->nb_seeds = neighbors;
  this->volume = volume;
}

void Voronoi3D::push( int n, Point3 &pt, const bool* periodicity, const double* xyz_min, const double* xyz_max )
{
  if(n == idx_center_seed)
    return;
  for(unsigned int m=0; m<nb_seeds.size(); m++)
  {
    if(nb_seeds[m].n == n)
    {
      return;
    }
  }

  ngbd3Dseed ngbd_seed;
  ngbd_seed.n = n;
  ngbd_seed.p = pt;
  ngbd_seed.dist = (pt-center_seed).norm_L2();
  nb_seeds.push_back(ngbd_seed);

  if(periodicity[0] || periodicity[1] || periodicity[2]) // some periodicity ?
  {
    const double domain_diag = sqrt(SQR(xyz_max[0] - xyz_min[0]) + SQR(xyz_max[1] - xyz_min[1]) + SQR(xyz_max[2] - xyz_min[2]));
    if(periodicity[0]) // x periodic
    {
      // we use 0.49 instead of 0.5 to ensure everything goes fine even for a 1/1 grid
      int x_coeff = (fabs(pt.x-center_seed.x) > 0.49*(xyz_max[0] - xyz_min[0]))? ((pt.x<center_seed.x)?+1:-1): 0;
      if(x_coeff != 0) // add the x-wrapped if needed
      {
        ngbd3Dseed x_wrapped_neighbor;
        x_wrapped_neighbor.n      = n;
        x_wrapped_neighbor.p.x    = pt.x + ((double) x_coeff)*(xyz_max[0] - xyz_min[0]);
        x_wrapped_neighbor.p.y    = pt.y;
        x_wrapped_neighbor.p.z    = pt.z;
        x_wrapped_neighbor.dist   = (x_wrapped_neighbor.p - center_seed).norm_L2();
        if(x_wrapped_neighbor.dist < 0.51*domain_diag)
          nb_seeds.push_back(x_wrapped_neighbor);
      }
      if(periodicity[1]) // x periodic AND y periodic
      {
        int y_coeff = (fabs(pt.y-center_seed.y) > 0.49*(xyz_max[1] - xyz_min[1]))? ((pt.y<center_seed.y)?+1:-1): 0;
        // first add the y-wrapped if needed
        if(y_coeff != 0)
        {
          ngbd3Dseed y_wrapped_neighbor;
          y_wrapped_neighbor.n      = n;
          y_wrapped_neighbor.p.x    = pt.x;
          y_wrapped_neighbor.p.y    = pt.y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
          y_wrapped_neighbor.p.z    = pt.z;
          y_wrapped_neighbor.dist   = (y_wrapped_neighbor.p - center_seed).norm_L2();
          if(y_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(y_wrapped_neighbor);
        }
        // then add the xy-wrapped if needed
        if(x_coeff != 0 && y_coeff != 0)
        {
          ngbd3Dseed xy_wrapped_neighbor;
          xy_wrapped_neighbor.n     = n;
          xy_wrapped_neighbor.p.x   = pt.x + ((double) x_coeff)*(xyz_max[0] - xyz_min[0]);
          xy_wrapped_neighbor.p.y   = pt.y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
          xy_wrapped_neighbor.p.z   = pt.z;
          xy_wrapped_neighbor.dist  = (xy_wrapped_neighbor.p - center_seed).norm_L2();
          if(xy_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(xy_wrapped_neighbor);
        }
        if(periodicity[2]) // x periodic AND y periodic AND z periodic
        {
          int z_coeff = (fabs(pt.z-center_seed.z) > 0.49*(xyz_max[2] - xyz_min[2]))? ((pt.z<center_seed.z)?+1:-1): 0;
          // first add the z-wrapped if needed
          if(z_coeff != 0)
          {
            ngbd3Dseed z_wrapped_neighbor;
            z_wrapped_neighbor.n      = n;
            z_wrapped_neighbor.p.x    = pt.x;
            z_wrapped_neighbor.p.y    = pt.y;
            z_wrapped_neighbor.p.z    = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
            z_wrapped_neighbor.dist   = (z_wrapped_neighbor.p - center_seed).norm_L2();
            if(z_wrapped_neighbor.dist < 0.51*domain_diag)
              nb_seeds.push_back(z_wrapped_neighbor);
          }
          // then add the xz-wrapped if needed
          if(x_coeff != 0 && z_coeff != 0)
          {
            ngbd3Dseed xz_wrapped_neighbor;
            xz_wrapped_neighbor.n     = n;
            xz_wrapped_neighbor.p.x   = pt.x + ((double) x_coeff)*(xyz_max[0] - xyz_min[0]);
            xz_wrapped_neighbor.p.y   = pt.y;
            xz_wrapped_neighbor.p.z   = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
            xz_wrapped_neighbor.dist  = (xz_wrapped_neighbor.p - center_seed).norm_L2();
            if(xz_wrapped_neighbor.dist < 0.51*domain_diag)
              nb_seeds.push_back(xz_wrapped_neighbor);
          }
          // then add the yz-wrapped if needed
          if(y_coeff != 0 && z_coeff != 0)
          {
            ngbd3Dseed yz_wrapped_neighbor;
            yz_wrapped_neighbor.n     = n;
            yz_wrapped_neighbor.p.x   = pt.x;
            yz_wrapped_neighbor.p.y   = pt.y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
            yz_wrapped_neighbor.p.z   = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
            yz_wrapped_neighbor.dist  = (yz_wrapped_neighbor.p - center_seed).norm_L2();
            if(yz_wrapped_neighbor.dist < 0.51*domain_diag)
              nb_seeds.push_back(yz_wrapped_neighbor);
          }
          // then add the xyz-wrapped if needed
          if(x_coeff != 0 && y_coeff != 0 && z_coeff != 0)
          {
            ngbd3Dseed xyz_wrapped_neighbor;
            xyz_wrapped_neighbor.n     = n;
            xyz_wrapped_neighbor.p.x   = pt.x + ((double) x_coeff)*(xyz_max[0] - xyz_min[0]);
            xyz_wrapped_neighbor.p.y   = pt.y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
            xyz_wrapped_neighbor.p.z   = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
            xyz_wrapped_neighbor.dist  = (xyz_wrapped_neighbor.p - center_seed).norm_L2();
            if(xyz_wrapped_neighbor.dist < 0.51*domain_diag)
              nb_seeds.push_back(xyz_wrapped_neighbor);
          }
        }
      }
      else if (periodicity[2]) // x periodic, NOT periodic in y, but z-periodic
      {
        int z_coeff = (fabs(pt.z-center_seed.z) > 0.49*(xyz_max[2] - xyz_min[2]))? ((pt.z<center_seed.z)?+1:-1): 0;
        // first add the z-wrapped if needed
        if(z_coeff != 0)
        {
          ngbd3Dseed z_wrapped_neighbor;
          z_wrapped_neighbor.n      = n;
          z_wrapped_neighbor.p.x    = pt.x;
          z_wrapped_neighbor.p.y    = pt.y;
          z_wrapped_neighbor.p.z    = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
          z_wrapped_neighbor.dist   = (z_wrapped_neighbor.p - center_seed).norm_L2();
          if(z_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(z_wrapped_neighbor);
        }
        // then add the xz-wrapped if needed
        if(x_coeff != 0 && z_coeff != 0)
        {
          ngbd3Dseed xz_wrapped_neighbor;
          xz_wrapped_neighbor.n     = n;
          xz_wrapped_neighbor.p.x   = pt.x + ((double) x_coeff)*(xyz_max[0] - xyz_min[0]);
          xz_wrapped_neighbor.p.y   = pt.y;
          xz_wrapped_neighbor.p.z   = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
          xz_wrapped_neighbor.dist  = (xz_wrapped_neighbor.p - center_seed).norm_L2();
          if(xz_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(xz_wrapped_neighbor);
        }
      }
    }
    else // NOT x-periodic
    {
      if(periodicity[1]) // but y-periodic
      {
        int y_coeff = (fabs(pt.y-center_seed.y) > 0.49*(xyz_max[1] - xyz_min[1]))? ((pt.y<center_seed.y)?+1:-1): 0;
        if(y_coeff != 0)
        {
          ngbd3Dseed y_wrapped_neighbor;
          y_wrapped_neighbor.n      = n;
          y_wrapped_neighbor.p.x    = pt.x;
          y_wrapped_neighbor.p.y    = pt.y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
          y_wrapped_neighbor.p.z    = pt.z;
          y_wrapped_neighbor.dist   = (y_wrapped_neighbor.p - center_seed).norm_L2();
          if(y_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(y_wrapped_neighbor);
        }
        if(periodicity[2]) // not periodic in x but periodic in y AND z
        {
          int z_coeff = (fabs(pt.z-center_seed.z) > 0.49*(xyz_max[2] - xyz_min[2]))? ((pt.z<center_seed.z)?+1:-1): 0;
          // first add the z-wrapped if needed
          if(z_coeff != 0)
          {
            ngbd3Dseed z_wrapped_neighbor;
            z_wrapped_neighbor.n      = n;
            z_wrapped_neighbor.p.x    = pt.x;
            z_wrapped_neighbor.p.y    = pt.y;
            z_wrapped_neighbor.p.z    = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
            z_wrapped_neighbor.dist   = (z_wrapped_neighbor.p - center_seed).norm_L2();
            if(z_wrapped_neighbor.dist < 0.51*domain_diag)
              nb_seeds.push_back(z_wrapped_neighbor);
          }
          // then add the yz-wrapped if needed
          if(y_coeff != 0 && z_coeff != 0)
          {
            ngbd3Dseed yz_wrapped_neighbor;
            yz_wrapped_neighbor.n     = n;
            yz_wrapped_neighbor.p.x   = pt.x;
            yz_wrapped_neighbor.p.y   = pt.y + ((double) y_coeff)*(xyz_max[1] - xyz_min[1]);
            yz_wrapped_neighbor.p.z   = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
            yz_wrapped_neighbor.dist  = (yz_wrapped_neighbor.p - center_seed).norm_L2();
            if(yz_wrapped_neighbor.dist < 0.51*domain_diag)
              nb_seeds.push_back(yz_wrapped_neighbor);
          }
        }
      }
      else // only z-periodic
      {
        // add the z-wrapped if needed
        int z_coeff = (fabs(pt.z-center_seed.z) > 0.49*(xyz_max[2] - xyz_min[2]))? ((pt.z<center_seed.z)?+1:-1): 0;
        if(z_coeff != 0)
        {
          ngbd3Dseed z_wrapped_neighbor;
          z_wrapped_neighbor.n      = n;
          z_wrapped_neighbor.p.x    = pt.x;
          z_wrapped_neighbor.p.y    = pt.y;
          z_wrapped_neighbor.p.z    = pt.z + ((double) z_coeff)*(xyz_max[2] - xyz_min[2]);
          z_wrapped_neighbor.dist   = (z_wrapped_neighbor.p - center_seed).norm_L2();
          if(z_wrapped_neighbor.dist < 0.51*domain_diag)
            nb_seeds.push_back(z_wrapped_neighbor);
        }
      }
    }
  }
}


void Voronoi3D::set_center_point( int idx_center_seed_, Point3 &center_seed_ )
{
  this->idx_center_seed = idx_center_seed_;
  this->center_seed = center_seed_;
}

void Voronoi3D::construct_partition(const double *xyz_min, const double *xyz_max, const bool *periodic)
{
  // sort the neighbor seeds by increasing distance first (behaves much better in voro++)
  sort(nb_seeds.begin(), nb_seeds.end());

  // get the scaling factor
  const double min_dist = nb_seeds[0].dist;
  P4EST_ASSERT(min_dist > EPS*sqrt(SQR(xyz_max[0] - xyz_min[0]) + SQR(xyz_max[1] - xyz_min[1]) + SQR(xyz_max[2] - xyz_min[2])));
  const double max_dist = nb_seeds.back().dist;
  // get the seed coordinates (clamp them in case of non-periodic walls)
  double x_center = (((fabs(center_seed.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*EPS) && (!periodic[0]) )? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*EPS) : (((fabs(center_seed.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*EPS) && (!periodic[0]) )? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*EPS) : center_seed.x));
  double y_center = (((fabs(center_seed.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*EPS) && (!periodic[1]) )? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*EPS) : (((fabs(center_seed.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*EPS) && (!periodic[1]) )? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*EPS) : center_seed.y));
  double z_center = (((fabs(center_seed.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*EPS) && (!periodic[2]) )? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*EPS) : (((fabs(center_seed.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*EPS) && (!periodic[2]) )? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*EPS) : center_seed.z));

  // the 4.0*max_dist/min_dist are kinda arbitrary...
  const double xyz_min_tmp[3] = {periodic[0]?-4.0*max_dist/min_dist:(xyz_min[0] - x_center)/min_dist,
                                 periodic[1]?-4.0*max_dist/min_dist:(xyz_min[1] - y_center)/min_dist,
                                 periodic[2]?-4.0*max_dist/min_dist:(xyz_min[2] - z_center)/min_dist};
  const double xyz_max_tmp[3] = {periodic[0]?+4.0*max_dist/min_dist:(xyz_max[0] - x_center)/min_dist,
                                 periodic[1]?+4.0*max_dist/min_dist:(xyz_max[1] - y_center)/min_dist,
                                 periodic[2]?+4.0*max_dist/min_dist:(xyz_max[2] - z_center)/min_dist};

  /* create a container for the particles */
  voro::container voronoi(xyz_min_tmp[0], xyz_max_tmp[0], xyz_min_tmp[1], xyz_max_tmp[1], xyz_min_tmp[2], xyz_max_tmp[2],
                          1, 1, 1, periodic[0], periodic[1], periodic[2], 8);

  /* store the order in which the particles are added to the container */
  voro::particle_order po;

  /* add the center point */
  voronoi.put(po, idx_center_seed, 0.0, 0.0, 0.0);

  std::vector<double> point_distances(nb_seeds.size());
  std::vector<unsigned int> index(nb_seeds.size(), 0);
  for(unsigned int m=0; m<nb_seeds.size(); ++m)
  {
    point_distances.at(m) = (nb_seeds[m].p - center_seed).norm_L2();
    index.at(m) = m;
  }

  sort(index.begin(), index.end(),
      [&](const int& a, const int& b) {
          return (point_distances[a] < point_distances[b]);
      }
  );

  /* add the points potentially involved in the voronoi partition */
  for(unsigned int m=0; m<nb_seeds.size(); ++m)
  {
    double x_tmp = (((fabs(nb_seeds[m].p.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*EPS) && (!periodic[0]) )? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*EPS) : (((fabs(nb_seeds[m].p.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*EPS) && (!periodic[0]) )? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*EPS) : nb_seeds[m].p.x));
    double y_tmp = (((fabs(nb_seeds[m].p.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*EPS) && (!periodic[1]) )? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*EPS) : (((fabs(nb_seeds[m].p.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*EPS) && (!periodic[1]) )? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*EPS) : nb_seeds[m].p.y));
    double z_tmp = (((fabs(nb_seeds[m].p.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*EPS) && (!periodic[2]) )? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*EPS) : (((fabs(nb_seeds[m].p.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*EPS) && (!periodic[2]) )? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*EPS) : nb_seeds[m].p.z));
//    voronoi.put(po, nb_seeds[m].n, x_tmp, y_tmp, z_tmp);
    voronoi.put(po, nb_seeds[m].n, (x_tmp - x_center)/min_dist, (y_tmp - y_center)/min_dist, (z_tmp - z_center)/min_dist);
  }

  voro::voronoicell_neighbor voro_cell;
  vector<int> neigh;
  vector<double> areas;

  voro::c_loop_order cl(voronoi, po);
  if(cl.start() && voronoi.compute_cell(voro_cell,cl))
  {
    vector<ngbd3Dseed> final_nb_seeds;

    volume = min_dist*min_dist*min_dist*voro_cell.volume();
    voro_cell.neighbors(neigh);
    voro_cell.face_areas(areas);
    double max_area = 0.0;
    for(unsigned int n=0; n<neigh.size(); n++)
      max_area = MAX(max_area, areas[n]);

    for(unsigned int n=0; n<neigh.size(); n++)
    {
      ngbd3Dseed new_voro_nb;
      new_voro_nb.n = neigh[n];
      new_voro_nb.s = min_dist*min_dist*areas[n];

      if(neigh[n]<0)
      {
        switch(neigh[n])
        {
        case WALL_m00:
          new_voro_nb.p.x = xyz_min[0]-(center_seed.x-xyz_min[0]); new_voro_nb.p.y = center_seed.y; new_voro_nb.p.z = center_seed.z; new_voro_nb.dist = fabs(2.0*(center_seed.x - xyz_min[0]));
          break;
        case WALL_p00:
          new_voro_nb.p.x = xyz_max[0]+(xyz_max[0]-center_seed.x); new_voro_nb.p.y = center_seed.y; new_voro_nb.p.z = center_seed.z; new_voro_nb.dist = fabs(2.0*(xyz_max[0] - center_seed.x));
          break;
        case WALL_0m0:
          new_voro_nb.p.x = center_seed.x; new_voro_nb.p.y = xyz_min[1]-(center_seed.y-xyz_min[1]); new_voro_nb.p.z = center_seed.z; new_voro_nb.dist = fabs(2.0*(center_seed.y - xyz_min[1]));
          break;
        case WALL_0p0:
          new_voro_nb.p.x = center_seed.x; new_voro_nb.p.y = xyz_max[1]+(xyz_max[1]-center_seed.y); new_voro_nb.p.z = center_seed.z; new_voro_nb.dist = fabs(2.0*(xyz_max[1] - center_seed.y));
          break;
        case WALL_00m:
          new_voro_nb.p.x = center_seed.x; new_voro_nb.p.y = center_seed.y; new_voro_nb.p.z = xyz_min[2]-(center_seed.z-xyz_min[2]); new_voro_nb.dist = fabs(2.0*(center_seed.z - xyz_min[2]));
          break;
        case WALL_00p:
          new_voro_nb.p.x = center_seed.x; new_voro_nb.p.y = center_seed.y; new_voro_nb.p.z = xyz_max[2]+(xyz_max[2]-center_seed.z); new_voro_nb.dist = fabs(2.0*(xyz_max[2] - center_seed.z));
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: Voronoi3D->construct_partition: unknown boundary.");
        }
      }
      else
      {
        for(unsigned int m=0; m<nb_seeds.size(); ++m)
        {
          if(nb_seeds[m].n==neigh[n])
          {
            new_voro_nb.p = nb_seeds[m].p;
            new_voro_nb.dist = nb_seeds[m].dist;
            break;
          }
        }
      }

      if(new_voro_nb.s > EPS*max_area)
        final_nb_seeds.push_back(new_voro_nb);
    }
    nb_seeds.clear();
    nb_seeds = final_nb_seeds;
  }
  else
  {
    // the cell could not be constructed...
    std::cerr << "We're in SERIOUS TROUBLE, dude..." << std::endl;
    std::cerr << "min_dist = " << min_dist << std::endl;
    std::cerr << "cl.start() = " << cl.start()  << std::endl;
    std::cerr << "voronoi.compute_cell(voro_cell,cl) = " << voronoi.compute_cell(voro_cell,cl) << std::endl;
  }
}

void Voronoi3D::print_VTK_format( const vector<Voronoi3D>& voro, const char* file_name,
                                  const double *xyz_min, const double *xyz_max, const bool *periodic)
{
  FILE* f;
  f = fopen(file_name, "w");
#ifdef CASL_THROWS
  if(f==NULL) throw std::invalid_argument("[CASL_ERROR]: Voronoi3D: cannot open file.");
#endif

  vector<VoroNgbd> voro_global(voro.size());
    for(unsigned int n=0; n<voro.size(); ++n)
    {
      voro_global[n].voronoi = new voro::container(xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2],
                                                   1, 1, 1, periodic[0], periodic[1], periodic[2], 8);
      voro_global[n].po = new voro::particle_order;

      double x_c = ((fabs(voro[n].center_seed.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*EPS) ? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*EPS) : ((fabs(voro[n].center_seed.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*EPS) ? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*EPS) : voro[n].center_seed.x));
      double y_c = ((fabs(voro[n].center_seed.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*EPS) ? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*EPS) : ((fabs(voro[n].center_seed.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*EPS) ? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*EPS) : voro[n].center_seed.y));
      double z_c = ((fabs(voro[n].center_seed.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*EPS) ? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*EPS) : ((fabs(voro[n].center_seed.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*EPS) ? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*EPS) : voro[n].center_seed.z));
      voro_global[n].voronoi->put(*voro_global[n].po, voro[n].idx_center_seed, x_c, y_c, z_c);

      for(unsigned int m=0; m<voro[n].nb_seeds.size(); ++m)
        if(voro[n].nb_seeds[m].n>=0)
        {
          double x_m = ((fabs(voro[n].nb_seeds[m].p.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*EPS) ? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*EPS) : ((fabs(voro[n].nb_seeds[m].p.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*EPS) ? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*EPS) : voro[n].nb_seeds[m].p.x));
          double y_m = ((fabs(voro[n].nb_seeds[m].p.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*EPS) ? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*EPS) : ((fabs(voro[n].nb_seeds[m].p.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*EPS) ? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*EPS) : voro[n].nb_seeds[m].p.y));
          double z_m = ((fabs(voro[n].nb_seeds[m].p.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*EPS) ? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*EPS) : ((fabs(voro[n].nb_seeds[m].p.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*EPS) ? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*EPS) : voro[n].nb_seeds[m].p.z));
          voro_global[n].voronoi->put(*voro_global[n].po, voro[n].nb_seeds[m].n, x_m, y_m, z_m);
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
    if(voro_global[n].voronoi!=NULL && voro[n].nb_seeds.size()>0)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].idx_center_seed && voro_global[n].voronoi->compute_cell(c,cl))
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
    if(voro_global[n].voronoi!=NULL && voro[n].nb_seeds.size()>0)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].idx_center_seed && voro_global[n].voronoi->compute_cell(c,cl))
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
    if(voro_global[n].voronoi!=NULL && voro[n].nb_seeds.size()>0)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].idx_center_seed && voro_global[n].voronoi->compute_cell(c,cl))
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
    if(voro_global[n].voronoi!=NULL && voro[n].nb_seeds.size()>0)
    {
      voro::c_loop_order cl(*voro_global[n].voronoi,*voro_global[n].po);
      if(cl.start() && cl.pid()==(int) voro[n].idx_center_seed && voro_global[n].voronoi->compute_cell(c,cl))
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
