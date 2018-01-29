#include "voronoi3D.h"
#include <vector>
#include <algorithm>


void Voronoi3D::clear()
{
    points.resize(0);
}

void Voronoi3D::get_Points( const vector<Voronoi3DPoint>*& points) const
{
    points = &this->points;
}

void Voronoi3D::set_Points( vector<Voronoi3DPoint> &points, double volume )
{
    this->points = points;
    this->volume = volume;
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

void Voronoi3D::set_Center_Point( int nc, Point3 &pc )
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


/*

void Voronoi3D::construct_Partition(const double *xyz_min, const double *xyz_max, const bool *periodic)
{



    double eps = EPS;

    int numvertices = (int) pow(2,3);

    qhT qh_qh;
    qhT *qh= &qh_qh;
    qh->num_points = points.size()+numvertices+1;
    coordT array[3*(1+qh->num_points)];




    std::vector<double> point_distances(qh->num_points);
    std::vector<unsigned int> index(qh->num_points, 0);
    for(unsigned int m=0; m<qh->num_points; ++m)
    {
        point_distances.at(m) = (points[m].p - pc).norm_L2();
        index.at(m) = m;
    }


    sort(index.begin(), index.end(),
         [&](const int& a, const int& b) {
        return (point_distances[a] < point_distances[b]);
    }
    );



    // add the center point
    double x_tmp = ((fabs(pc.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*eps) : ((fabs(pc.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*eps) : pc.x));
    double y_tmp = ((fabs(pc.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*eps) : ((fabs(pc.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*eps) : pc.y));
    double z_tmp = ((fabs(pc.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*eps) : ((fabs(pc.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*eps) : pc.z));
    array[0] = x_tmp;
    array[1] = y_tmp;
    array[2] = z_tmp;

    // add cube box vertices
    for (int j=1; j<numvertices+1; j++) {
        for (int k=3; k--; ) {
            if (j & ( 1 << k))
                array[3*j+k]= xyz_max[0];
            else
                array[3*j+k]= xyz_min[0];
        }
    }

    for(unsigned int m=numvertices+1; m<qh->num_points; ++m)
    {

        unsigned int idx = index.at(m);
        double x_tmp = ((fabs(points[idx].p.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*eps) : ((fabs(points[idx].p.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*eps) : points[idx].p.x));
        double y_tmp = ((fabs(points[idx].p.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*eps) : ((fabs(points[idx].p.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*eps) : points[idx].p.y));
        double z_tmp = ((fabs(points[idx].p.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*eps) : ((fabs(points[idx].p.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*eps) : points[idx].p.z));
        array[3*m + 0] = x_tmp;
        array[3*m + 1] = y_tmp;
        array[3*m + 2] = z_tmp;
    }



    vertexT *vertex;
    setT *vertices;
    facetT *facet;

    char flags[2000];
    int exitcode;
    FILE *errfile = stdout;
    boolT ismalloc = False;

    boolT islower;
    int numfacets, numsimplicial, numridges, totneighbors, numcoplanars,
            numtricoplanars, k, num, numcenters;
    qh_RIDGE innerouter;
    int curlong, totlong;

    QHULL_LIB_CHECK qh_zero (qh, errfile);
    sprintf (flags, "qhull v Fv p");// Fv FN Fa FA");
    exitcode = qh_new_qhull(qh, P4EST_DIM, qh->num_points, array, ismalloc, flags, NULL, errfile);
    if (!exitcode)
    {
        //extract num of vertices
        qh_setvoronoi_all (qh);
        qh_countfacets (qh, qh->facet_list, NULL, !qh_ALL, &numfacets,
                        &numsimplicial, &totneighbors, &numridges,
                        &numcoplanars, &numtricoplanars);

        int nvertex;
        nvertex = numfacets;

        // allocate memory for voronoi vertices
        std::vector<double> xv, yv, zv;
        xv.resize(nvertex);
        yv.resize(nvertex);
        zv.resize(nvertex);
        std::cout<<"Hello"<<std::endl;


        //Loop over voronoi vertices
        k = 0;
        FORALLfacet_ (qh->facet_list)
        {
            num = qh->hull_dim - 1;
            if (!facet->normal || !facet->upperdelaunay || !qh->ATinfinity)
            {
                if (facet->center)
                {
                    xv[k] = facet->center[0];
                    yv[k] = facet->center[1];
                    zv[k] = facet->center[2];
                    k++;
                    std::cout<< xv[k] <<std::endl;

                }
            }
            else
            {
                printf ("qhINFINITE CASE \n");
            }
        }
        innerouter = qh_RIDGEall;


        vertices = qh_markvoronoi(qh, qh->facet_list, NULL, 0, &islower, &numcenters);

        FORALLvertices vertex->seen = False;
        realT ar;
        FORALLfacet_ (qh->facet_list){
            ar = qh_facetarea(qh, facet);
        }
        qh_settempfree (qh, &vertices);

        vertices = qh_markvoronoi(qh, qh->facet_list, NULL, 0, &islower, &numcenters);	//arg 3,0->null
        int totcount = qh_printvdiagram2(qh, NULL, NULL, vertices, innerouter, False);
        int nfacets = totcount;
        int NFAC = 5;

        // here we use NFAC which is a guess of the mean number of vertices in a facet
        //std::vector<int> adj, adji;
        adj = (int*) malloc(nfacets * NFAC * sizeof (int));
        adji = (int*) malloc(nfacets * sizeof (int));
        //adj.resize(nfacets*NFAC);
        //adji.resize(nfacets);

        for (int i = 0; i < nfacets * NFAC; i++)
            adj[i] = 0;
        for (int i = 0; i < nfacets; i++)
            adji[i] = 0;

        int nadj = 0;
        int nadjpos = 0;

        // Make a loop over all facets & look for id1, id2 and vertices ids
        totcount = 0;

        FORALLvertices vertex->seen = False;

        printvridgeT printvridge = qh_printvridge;

        int vertex_i, vertex_n;
        FOREACHvertex_i_ (qh, vertices)
        {
            if (vertex)
            {
                if (qh->GOODvertex > 0 && qh_pointid(qh, vertex->point) + 1 != qh->GOODvertex)
                    continue;
                totcount += qh_eachvoronoi_mod(qh, errfile, printvridge, vertex, !qh_ALL, innerouter, True, adj, adji, nadjpos, nadj);
            }
        }


        if (nadjpos > nfacets*NFAC) printf("WARNING, you must increase NFAC value\n");
        qh_settempfree (qh, &vertices);
        qh->NOerrexit = True;
        qh_freeqhull (qh, !qh_ALL);
        qh_memfreeshort (qh, &curlong, &totlong);
        if (curlong || totlong)
            fprintf (errfile,"warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);
    }



    throw std::invalid_argument("HERE ENOUGH");



    vector<Voronoi3DPoint> final_points;
    std::vector<int> neigh;
    std::vector<double> areas;

    // fill neigh and areas


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
                new_voro_nb.p.x = xyz_min[0]-(pc.x-xyz_min[0]); new_voro_nb.p.y = pc.y; new_voro_nb.p.z = pc.z;
                break;
            case WALL_p00:
                new_voro_nb.p.x = xyz_max[0]+(xyz_max[0]-pc.x); new_voro_nb.p.y = pc.y; new_voro_nb.p.z = pc.z;
                break;
            case WALL_0m0:
                new_voro_nb.p.x = pc.x; new_voro_nb.p.y = xyz_min[1]-(pc.y-xyz_min[1]); new_voro_nb.p.z = pc.z;
                break;
            case WALL_0p0:
                new_voro_nb.p.x = pc.x; new_voro_nb.p.y = xyz_max[1]+(xyz_max[1]-pc.y); new_voro_nb.p.z = pc.z;
                break;
            case WALL_00m:
                new_voro_nb.p.x = pc.x; new_voro_nb.p.y = pc.y; new_voro_nb.p.z = xyz_min[2]-(pc.z-xyz_min[2]);
                break;
            case WALL_00p:
                new_voro_nb.p.x = pc.x; new_voro_nb.p.y = pc.y; new_voro_nb.p.z = xyz_max[2]+(xyz_max[2]-pc.z);
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




int Voronoi3D::qh_eachvoronoi_mod (qhT * qh, FILE * fp, printvridgeT printvridge,
                        vertexT * atvertex, boolT visitall, qh_RIDGE innerouter,
                        boolT inorder, int *adj, int *adji, int nadjpos, int nadj)
{
    boolT unbounded;
    int count, ind, k;
    facetT *neighbor, **neighborp, *neighborA, **neighborAp, *facet, **facetp;
    setT *centers;
    setT *tricenters = qh_settemp (qh, qh->TEMPsize);

    vertexT *vertex, **vertexp;
    boolT firstinf;
    unsigned int numfacets = (unsigned int) qh->num_facets;
    int totridges = 0;

    qh->vertex_visit++;
    atvertex->seen = True;
    if (visitall)
    {
        FORALLvertices vertex->seen = False;
    }
    FOREACHneighbor_ (atvertex)
    {
        if (neighbor->visitid < numfacets)
            neighbor->seen = True;
    }
    FOREACHneighbor_ (atvertex)
    {
        if (neighbor->seen)
        {
            FOREACHvertex_ (neighbor->vertices)
            {
                if (vertex->visitid != qh->vertex_visit && !vertex->seen)
                {
                    vertex->visitid = qh->vertex_visit;
                    count = 0;
                    firstinf = True;
                    qh_settruncate (qh, tricenters, 0);
                    FOREACHneighborA_ (vertex)
                    {
                        if (neighborA->seen)
                        {
                            if (neighborA->visitid)
                            {
                                if (!neighborA->tricoplanar
                                        || qh_setunique (qh, &tricenters,
                                                         neighborA->center))
                                    count++;
                            }
                            else if (firstinf)
                            {
                                count++;
                                firstinf = False;
                            }
                        }
                    }
                    if (count >= qh->hull_dim - 1)
                    {
                        if (firstinf)
                        {
                            if (innerouter == qh_RIDGEouter)
                                continue;
                            unbounded = False;
                        }
                        else
                        {
                            if (innerouter == qh_RIDGEinner)
                                continue;
                            unbounded = True;
                        }
                        totridges++;

                        trace4 ((qh, qh->ferr, 4017,
                                 "qh_eachvoronoi: Voronoi ridge of %d vertices between sites %d and %d\n",
                                 count, qh_pointid (qh, atvertex->point),
                                 qh_pointid (qh, vertex->point)));

                        if (printvridge)
                        {
                            if (inorder && qh->hull_dim == 3 + 1)
                                centers = qh_detvridge3 (qh, atvertex, vertex);
                            else
                                centers = qh_detvridge (qh, vertex);

                            adj[nadjpos] = qh_setsize (qh, centers) + 2;
                            adj[nadjpos + 1] = qh_pointid (qh, atvertex->point);
                            adj[nadjpos + 2] = qh_pointid (qh, vertex->point);
                            adji[nadj] = nadjpos;
                            k = 0;
                            FOREACHfacet_ (centers)
                            {
                                adj[nadjpos + 3 + k] = facet->visitid;
                                k++;
                            }
                            nadj++;
                            nadjpos += 3 + k;
                            qh_settempfree (qh, &centers);
                        }
                    }
                }
            }
        }
    }
    FOREACHneighbor_ (atvertex) neighbor->seen = False;
    qh_settempfree (qh, &tricenters);
    return totridges;
}
*/

void Voronoi3D::construct_Partition(const double *xyz_min1, const double *xyz_max1, const bool *periodic)
{

  double eps = EPS;


  double xyz_max[3] = {0.1, 0.1, 0.1};
  double xyz_min[3] = {-0.1,-0.1, -0.1};

  voro::container voronoi(xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2],
                          1, 1, 1, periodic[0], periodic[1], periodic[2], 8);

  voro::particle_order po;


  double x_tmp = fabs(pc.x-xyz_min[0])<eps ? xyz_min[0]+eps : fabs(pc.x-xyz_max[0])<eps ? xyz_max[0]-eps : pc.x;
  double y_tmp = fabs(pc.y-xyz_min[1])<eps ? xyz_min[1]+eps : fabs(pc.y-xyz_max[1])<eps ? xyz_max[1]-eps : pc.y;
  double z_tmp = fabs(pc.z-xyz_min[2])<eps ? xyz_min[2]+eps : fabs(pc.z-xyz_max[2])<eps ? xyz_max[2]-eps : pc.z;
  voronoi.put(po, nc, x_tmp, y_tmp, z_tmp);


  for(unsigned int m=0; m<points.size(); ++m)
  {
    double x_tmp = fabs(points[m].p.x-xyz_min[0])<eps ? xyz_min[0]+eps : fabs(points[m].p.x-xyz_max[0])<eps ? xyz_max[0]-eps : points[m].p.x;
    double y_tmp = fabs(points[m].p.y-xyz_min[1])<eps ? xyz_min[1]+eps : fabs(points[m].p.y-xyz_max[1])<eps ? xyz_max[1]-eps : points[m].p.y;
    double z_tmp = fabs(points[m].p.z-xyz_min[2])<eps ? xyz_min[2]+eps : fabs(points[m].p.z-xyz_max[2])<eps ? xyz_max[2]-eps : points[m].p.z;
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
    volume = voro_cell.volume();

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
          new_voro_nb.p.x = xyz_min[0]-(pc.x-xyz_min[0]); new_voro_nb.p.y = pc.y; new_voro_nb.p.z = pc.z;
          break;
        case WALL_p00:
          new_voro_nb.p.x = xyz_max[0]+(xyz_max[0]-pc.x); new_voro_nb.p.y = pc.y; new_voro_nb.p.z = pc.z;
          break;
        case WALL_0m0:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = xyz_min[1]-(pc.y-xyz_min[1]); new_voro_nb.p.z = pc.z;
          break;
        case WALL_0p0:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = xyz_max[1]+(xyz_max[1]-pc.y); new_voro_nb.p.z = pc.z;
          break;
        case WALL_00m:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = pc.y; new_voro_nb.p.z = xyz_min[2]-(pc.z-xyz_min[2]);
          break;
        case WALL_00p:
          new_voro_nb.p.x = pc.x; new_voro_nb.p.y = pc.y; new_voro_nb.p.z = xyz_max[2]+(xyz_max[2]-pc.z);
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
  } else
  {
      std::cerr << "We're in SERIOUS TROUBLE, dude..." << std::endl;
      std::cerr << "cl.start() = " << cl.start()  << std::endl;
      std::cerr << "voronoi.compute_cell(voro_cell,cl) = " << voronoi.compute_cell(voro_cell,cl) << std::endl;
  }
}




void Voronoi3D::print_VTK_Format( const std::vector<Voronoi3D>& voro, const char* file_name,
                                  const double *xyz_min, const double *xyz_max, const bool *periodic)
{
    FILE* f;
    f = fopen(file_name, "w");
#ifdef CASL_THROWS
    if(f==NULL) throw std::invalid_argument("[CASL_ERROR]: Voronoi3D: cannot open file.");
#endif

    double eps = EPS;

    vector<VoroNgbd> voro_global(voro.size());
    for(unsigned int n=0; n<voro.size(); ++n)
    {
        voro_global[n].voronoi = new voro::container(xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2],
                1, 1, 1, periodic[0], periodic[1], periodic[2], 8);
        voro_global[n].po = new voro::particle_order;

        double x_c = ((fabs(voro[n].pc.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*eps) : ((fabs(voro[n].pc.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*eps) : voro[n].pc.x));
        double y_c = ((fabs(voro[n].pc.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*eps) : ((fabs(voro[n].pc.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*eps) : voro[n].pc.y));
        double z_c = ((fabs(voro[n].pc.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*eps) : ((fabs(voro[n].pc.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*eps) : voro[n].pc.z));
        voro_global[n].voronoi->put(*voro_global[n].po, voro[n].nc, x_c, y_c, z_c);

        for(unsigned int m=0; m<voro[n].points.size(); ++m)
            if(voro[n].points[m].n>=0)
            {
                double x_m = ((fabs(voro[n].points[m].p.x-xyz_min[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_min[0]+(xyz_max[0] - xyz_min[0])*eps) : ((fabs(voro[n].points[m].p.x-xyz_max[0])<(xyz_max[0] - xyz_min[0])*eps) ? (xyz_max[0]-(xyz_max[0] - xyz_min[0])*eps) : voro[n].points[m].p.x));
                double y_m = ((fabs(voro[n].points[m].p.y-xyz_min[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_min[1]+(xyz_max[1] - xyz_min[1])*eps) : ((fabs(voro[n].points[m].p.y-xyz_max[1])<(xyz_max[1] - xyz_min[1])*eps) ? (xyz_max[1]-(xyz_max[1] - xyz_min[1])*eps) : voro[n].points[m].p.y));
                double z_m = ((fabs(voro[n].points[m].p.z-xyz_min[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_min[2]+(xyz_max[2] - xyz_min[2])*eps) : ((fabs(voro[n].points[m].p.z-xyz_max[2])<(xyz_max[2] - xyz_min[2])*eps) ? (xyz_max[2]-(xyz_max[2] - xyz_min[2])*eps) : voro[n].points[m].p.z));
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
        if(voro_global[n].voronoi!=NULL && voro[n].points.size()>0)
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
        if(voro_global[n].voronoi!=NULL && voro[n].points.size()>0)
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
        if(voro_global[n].voronoi!=NULL && voro[n].points.size()>0)
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
        if(voro_global[n].voronoi!=NULL && voro[n].points.size()>0)
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
