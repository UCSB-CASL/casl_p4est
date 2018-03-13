#ifndef MY_P4EST_EPIDEMICS_H
#define MY_P4EST_EPIDEMICS_H
#include <vector>
#include <src/types.h>
#include <src/math.h>

#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
//#include "nearpt3/nearpt3.cc"
#include </home/pouria/Documents/ann_1.1.2/include/ANN/ANN.h>
#include <fstream>

struct Tract {
    double id;
    double x;
    double y;
    double density;
    double pop;
    double area;
};


class my_p4est_epidemics_t
{
private:
    PetscErrorCode ierr;

    class circle_t : public CF_2
    {
    private:
        double xc, yc, r;
        my_p4est_epidemics_t *prnt;
    public:
        circle_t(double xc, double yc, double r, my_p4est_epidemics_t *prnt) : xc(xc), yc(yc), r(r), prnt(prnt)
        {
            lip = 1.2;
        }
        double operator()(double x, double y) const
        {
            double d = -4*prnt->L;
            for(int i=-1; i<2; ++i)
                for(int j=-1; j<2; ++j)
                {
                    d = MAX(d, r-sqrt( SQR(x+i*prnt->L-xc) + SQR(y+j*prnt->L-yc) ) );
                }
            return d;
        }
    };

    class interp_density_t : public CF_2
    {
    private:
        /* ANN stuff  */
        int k_neighs = 20;             // number of nearest neighbors to draw
        ANNpointArray dataPts;        // data points
        ANNkd_tree* kdTree;
        double R_eff = 1e-3;          // effective radius for neighborhood

        std::vector<Tract> tracts;
        std::vector<double> densities;
        double xc_, yc_;
        double Lx_max, Lx_min, Ly_max, Ly_min;


    public:
        interp_density_t(my_p4est_epidemics_t *prnt)
        {
            // only read on rank 0 and then broadcast the result to others
            if (prnt->p4est->mpirank == 0) {
                std::ifstream infile("US_census.dat");

#ifdef CASL_THROWS
                if (!infile.fail())
                    throw std::invalid_argument("could not open the census file");
#endif

                // parse line by line
                while(!infile.eof()) {
                    Tract tract;
                    infile >> tract.id;
                    infile >> tract.x;
                    infile >> tract.y;
                    infile >> tract.density;
                    infile >> tract.pop;
                    infile >> tract.area;
                    tracts.push_back(tract);
                }
                infile.close();
            }

            size_t msg_size = tracts.size()*sizeof(Tract);
            MPI_Bcast(&msg_size, 1, MPI_UNSIGNED_LONG, 0, prnt->p4est->mpicomm);
            if (prnt->p4est->mpirank != 0)
                tracts.resize(msg_size/sizeof(Tract));
            MPI_Bcast(&tracts[0], msg_size, MPI_BYTE, 0, prnt->p4est->mpicomm);

            // compute the center of mass
            xc_ = 0;
            yc_ = 0;
            for (size_t i = 0; i<tracts.size(); i++){
                xc_ += tracts[i].x;
                yc_ += tracts[i].y;
                densities.push_back(tracts[i].density);
            }
            xc_ /= tracts.size();
            yc_ /= tracts.size();


            // compute the size of the bounding box
            Lx_max= 0;
            Lx_min=0;
            Ly_max = 0;
            Ly_min = 0;
            for (size_t i = 0; i<tracts.size(); i++){
                Lx_max = MAX(Lx_max, tracts[i].x);
                Ly_max = MAX(Ly_max, tracts[i].y);

                Lx_min = MIN(Lx_min, tracts[i].x);
                Ly_min = MIN(Ly_min, tracts[i].y);
            }
            // make room from boundaries
            Lx_min *= 1.1;
            Lx_max *= 1.1;
            Ly_min *= 1.1;
            Ly_max *= 1.1;
            // scale and recenter the tracts to middle
            translate(Lx_min, Ly_min);   // shift coordinate system to the bottom left corner
            unit_scaling();
            translate(0, -0.2);

            int maxPts = tracts.size();
            dataPts = annAllocPts(maxPts, P4EST_DIM); // allocate data points


            for(int n=0; n<tracts.size(); ++n)
            {
                dataPts[n][0] = tracts[n].x;
                dataPts[n][1] = tracts[n].y;
            }

            if(prnt->p4est->mpirank==0)
            {
                std::ofstream fout;
                fout.open("locs.txt");
                for(int n=0; n<tracts.size(); ++n)
                    fout << tracts[n].x << "\t" << tracts[n].y << "\n";
                fout.close();
            }
            set_density();
        }
        void close()
        {
            /* destroy ANN and its structure */
            delete kdTree;
            annClose();
        }

        void translate(double x_shift, double y_shift) {
            // move the tracts to the new location
            for (size_t i = 0; i<tracts.size(); i++){
                tracts[i].x -= x_shift;
                tracts[i].y -= y_shift;
            }
            xc_ -= x_shift;
            yc_ -= y_shift;
        }

        void unit_scaling() {
            // scale coordinate to be unit times unit in length
            double scale = MAX((Lx_max - Lx_min),(Ly_max - Ly_min));
            for (size_t i = 0; i<tracts.size(); i++){
                tracts[i].x /= scale;
                tracts[i].y /= scale;
            }
            xc_ /= scale;
            yc_ /= scale;
        }

        void set_density()
        {
            int nPts = tracts.size();
            kdTree = new ANNkd_tree(dataPts,  nPts, 2, 1);  // build search structure
        }


        double operator()(double x, double y) const
        {
            ANNpoint queryPt;           // query point
            queryPt = annAllocPt(2);  // allocate query point

            queryPt[0] = x;
            queryPt[1] = y;

            ANNidxArray nnIdx;          // near neighbor indices
            ANNdistArray dists;         // near neighbor distances
            nnIdx = new ANNidx[k_neighs];      // allocate near neigh indices
            dists = new ANNdist[k_neighs];     // allocate near neighbor dists
            kdTree->annkSearch( queryPt, k_neighs, nnIdx, dists, 0);

            double interpolated_density = 0;
            double denom = 0;
            double max_neigh_dens = 0;
            double min_neigh_dens = DBL_MAX;
            for (int i = 0; i < k_neighs; i++)
            {
                int nid = nnIdx[i];
                double dist = sqrt(dists[i]);
                if(dists[i]<=R_eff)
                {
                    double neigh_dens = tracts[nid].density;
                    double weight = 1/(0.1*R_eff + dist);               // softening length is 10% of effective radius
                    interpolated_density += weight*neigh_dens;
                    denom += weight;
                    max_neigh_dens = MAX(max_neigh_dens, neigh_dens);
                    min_neigh_dens = MIN(min_neigh_dens, neigh_dens);
                }
            }

            if(denom>EPS)
                interpolated_density /= denom;
            else
                interpolated_density = 0;

            if(interpolated_density>max_neigh_dens)
                interpolated_density = max_neigh_dens;

            if(interpolated_density<min_neigh_dens)
                interpolated_density = min_neigh_dens;

            if(min_neigh_dens>1e20)
                interpolated_density = 0;

            delete [] nnIdx;
            delete [] dists;

            return interpolated_density;
        }
    };


    class zero_t : public CF_2
    {
    public:
        double operator()(double x, double y) const
        {

            return 1;
        }
    } zero;


    /* grid */
    my_p4est_brick_t *brick;
    p4est_t *p4est;
    p4est_connectivity_t *connectivity;
    p4est_ghost_t *ghost;
    p4est_nodes_t *nodes;
    my_p4est_hierarchy_t *hierarchy;
    my_p4est_node_neighbors_t *ngbd;


    Vec phi_g;
    std::vector<Vec> phi;
    std::vector<Vec> v[2];

    BoundaryConditionType bc_type;

    double dxyz[P4EST_DIM];
    double xyz_min[P4EST_DIM];
    double xyz_max[P4EST_DIM];
    double L;

    /* solutions */
    Vec U_n, U_np1;
    Vec V_n, V_np1;
    Vec W_n, W_np1;
    Vec land;


    /* physical parameters */
    double dt_n;
    double R_A;
    double R_B;
    double Xi_A;
    double Xi_B;
    double D_A, D_B, D_AB;


    //  /* ANN stuff  */
    //  int k_neighs = 20;             // number of nearest neighbors to draw
    //  ANNpointArray dataPts;        // data points
    //  ANNkd_tree* kdTree;
    //  double R_eff = 1e-3;          // effective radius for neighborhood

    //  std::vector<Tract> tracts;
    //  std::vector<double> densities;
    //  double xc_, yc_;
    //  double Lx_max, Lx_min, Ly_max, Ly_min;
public:

    my_p4est_epidemics_t(my_p4est_node_neighbors_t *ngbd);
    ~my_p4est_epidemics_t();

    interp_density_t interp_density;
    double get_density(double x, double y);
    //  void read(const std::string& census);
    //  void translate(double xc, double yc);
    //  void unit_scaling();
    //  void set_density();
    //  double interp_density(double x, double y);

    void compute_phi_g();

    void set_parameters(double R_A,
                        double R_B,
                        double Xi_A,
                        double Xi_B);



    void set_D(double D_A, double D_B, double D_AB);

    inline double get_dt() { return dt_n; }

    inline p4est_t* get_p4est() { return p4est; }

    inline p4est_nodes_t* get_nodes() { return nodes; }

    void compute_velocity();

    void solve(int iter);

    void compute_dt();

    void update_grid();

    void initialize_infections();

    void save_vtk(int iter);
};


#endif /* MY_P4EST_EPIDEMICS_H */
