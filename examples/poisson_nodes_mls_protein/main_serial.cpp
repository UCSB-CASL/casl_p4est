//
// Created by Frederic Gibou on 11/21/23.
//

#include <iostream>

#include <lib/tools/CASL_types.h>
#include <lib/solvers/QuadTreeSolverNavierStokes.h>
#include <lib/solvers/QuadTreeSolverNodeBasedPoisson.h>

using namespace CASL;
using namespace std;

#define XMIN 0
#define XMAX PI
#define YMIN 0
#define YMAX PI

// Tree parameters:
#define NB_SPLITS 0
#define NB_OF_RANDOM_SPLITS 300

// Number of projections:
#define NB_ITERATIONS 1000

// Boundary condition on Hodge: Type = NEUMANN and Value = 0
class WallBCHodgeType : public WallBC2D {
public:
    BoundaryConditionType operator()( double x, double y ) const
    {
        (void) y; (void) x;
        return NEUMANN;
    }
} wall_bc_hodge_type;


class WallBCHodgeValue : public CF_2 {
public:
    double operator()(double x, double y) const { return 0.; }
} wall_bc_hodge_value;

// Define functions for the initial velocity field (combination of a div-free and curl-free part)
double uExact   (double x, double y) { return sin(x)*cos(y); }
double vExact   (double x, double y) { return -cos(x)*sin(y); }
double u_initial(double x, double y) { return uExact(x,y) + x*(PI-x)*y*y*(y/3. - PI/2.); }
double v_initial(double x, double y) { return vExact(x,y) + y*(PI-y)*x*x*(x/3. - PI/2.); }

// Prototype functions:
void check_Results(QuadTree & tr, ArrayV<double> & hodge, ArrayV<double> & un, ArrayV<double> & vn, int iter );
void compute_minus_divergence_Ustar(QuadTree & tr,
                                    const ArrayV<double> & ustar,
                                    const ArrayV<double> & vstar,
                                    QuadNgbdNodesOfNode & qnnn,
                                    ArrayV<double> & minus_div_Ustar);
void project_onto_divergence_free_space(QuadTree & tr,
                                        const ArrayV<double> & ustar,
                                        const ArrayV<double> & vstar,
                                        const ArrayV<double> & hodge,
                                        QuadNgbdNodesOfNode & qnnn,
                                        ArrayV<double> & unp1,
                                        ArrayV<double> & vnp1);
int main()
{
    // Create a random adaptive grid:
    QuadTree tr;
    tr.set_Grid(XMIN,XMAX,YMIN,YMAX);
    tr.construct_Random_Quadtree(NB_OF_RANDOM_SPLITS, 1367876610);
    for(int i = 1; i <= NB_SPLITS; ++i) tr.split_Every_Cell();
    tr.initialize_Neighbors();
    cout << "Number of nodes: " << tr.number_Of_Nodes() << "\t Number of leaves: " << tr.number_Of_Leaves() << endl;

    CaslInt  nNodes = tr.number_Of_Nodes();
    ArrayV<double> minus_div_Ustar (nNodes);
    ArrayV<double> ustar           (nNodes);
    ArrayV<double> vstar           (nNodes);
    ArrayV<double> unp1            (nNodes);
    ArrayV<double> vnp1            (nNodes);
    ArrayV<double> hodge           (nNodes);

    // Set the Ustar velocity field (Frederic checked):
    for(CaslInt n=0; n<tr.number_Of_Nodes(); n++)
    {
        double x = tr.x_fr_i(tr.get_Node(n).i);
        double y = tr.y_fr_j(tr.get_Node(n).j);
        ustar(n) = u_initial(x,y);
        vstar(n) = v_initial(x,y);
    }

    // Iteration loop on the repeated projections:
    BoundaryConditions2D bc_Hodge;
    bc_Hodge.setWallTypes(wall_bc_hodge_type);
    bc_Hodge.setWallValues(wall_bc_hodge_value);

    QuadNgbdNodesOfNode qnnn = {};

    for(int iteration = 1; iteration <= NB_ITERATIONS; ++iteration){
        cout << endl << "--------------- Stable projection --------------------" << endl;
        cout << "Iteration " << iteration << endl;

        // Compute: - div(u^star)
        compute_minus_divergence_Ustar(tr, ustar, vstar, qnnn, minus_div_Ustar);

        // Solve: Laplace(hodge) = - div(u^star):
        QuadTreeSolverNodeBasedPoisson poissonNodeBasedSolver;
        poissonNodeBasedSolver.set_Quadtree(tr);
        poissonNodeBasedSolver.set_bc(bc_Hodge);
        poissonNodeBasedSolver.set_Rhs(minus_div_Ustar);
        poissonNodeBasedSolver.solve(hodge);

        // Project onto the divergence free space:
        project_onto_divergence_free_space(tr, ustar, vstar, hodge, qnnn, unp1, vnp1);

        // Compute errors and save results to file:
        check_Results(tr, hodge, unp1, vnp1, iteration);

        // Prepare for the next iteration:
        ustar = unp1;
        vstar = vnp1;
    }

    return EXIT_SUCCESS;
}

// Standard projection on the divergence free field.
// unp1(n) = ustar(n) - qnnn.dx_Central(hodge)  // Standard weighted central differencing.
// vnp1(n) = vstar(n) - qnnn.dy_Central(hodge)  // Standard weighted central differencing.
void project_onto_divergence_free_space(QuadTree & tr,
                                        const ArrayV<double> & ustar,
                                        const ArrayV<double> & vstar,
                                        const ArrayV<double> & hodge,
                                        QuadNgbdNodesOfNode & qnnn,
                                        ArrayV<double> & unp1,
                                        ArrayV<double> & vnp1){

    for(CaslInt n = 0; n < tr.number_Of_Nodes(); ++n) {
        tr.get_Ngbd_Nodes_Of_Node(n,qnnn);

        bool isXmWall = tr.get_Node(n).is_xmWall(tr.periodic_x()); // is wall left
        bool isXpWall = tr.get_Node(n).is_xpWall(tr.periodic_x()); // is wall right
        bool isYmWall = tr.get_Node(n).is_ymWall(tr.periodic_y()); // is wall bottom
        bool isYpWall = tr.get_Node(n).is_ypWall(tr.periodic_y()); // is wall top
        if (isXmWall || isXpWall || isYmWall || isYpWall ) {       // exact solution on walls
            double x = tr.x_fr_i(tr.get_Node(n).i);
            double y = tr.y_fr_j(tr.get_Node(n).j);
            unp1(n) = uExact(x,y);
            vnp1(n) = vExact(x,y);
        }
        else {  // standard central differencing for gradient(Hodge):
            unp1(n) = ustar(n) - qnnn.dx_Central(hodge);    // Frederic: this is indeed the correct formula - check with other interpolation.
            vnp1(n) = vstar(n) - qnnn.dy_Central(hodge);    // Frederic: this is indeed the correct formula - check with other interpolation.
        }
    }
}

// Computes - div(U^*) using the scheme of Maxime.
// Both the interpolations scheme and the non-weighted central differencing are key.
void compute_minus_divergence_Ustar(QuadTree & tr,
                                    const ArrayV<double> & ustar,
                                    const ArrayV<double> & vstar,
                                    QuadNgbdNodesOfNode & qnnn,
                                    ArrayV<double> & minus_div_Ustar) {

    for(CaslInt n = 0; n < tr.number_Of_Nodes(); ++n) {

        tr.get_Ngbd_Nodes_Of_Node(n,qnnn);
        double x = tr.x_fr_i(tr.get_Node(n).i);
        double y = tr.y_fr_j(tr.get_Node(n).j);

        double u_00 = quadratic_Interpolation_WENO( tr, ustar, x                 , y                  );
        double u_p0 = quadratic_Interpolation_WENO( tr, ustar, x + ABS(qnnn.d_p0), y                  );
        double u_m0 = quadratic_Interpolation_WENO( tr, ustar, x - ABS(qnnn.d_m0), y                  );
        double v_00 = quadratic_Interpolation_WENO( tr, vstar, x                 , y                  );
        double v_0p = quadratic_Interpolation_WENO( tr, vstar, x                 , y + ABS(qnnn.d_0p) );
        double v_0m = quadratic_Interpolation_WENO( tr, vstar, x                 , y - ABS(qnnn.d_0m) );

        double d_ustar_dx, d_ustar_dy;
        bool isXmWall = tr.get_Node(n).is_xmWall(tr.periodic_x()); // is wall left
        bool isXpWall = tr.get_Node(n).is_xpWall(tr.periodic_x()); // is wall right
        bool isYmWall = tr.get_Node(n).is_ymWall(tr.periodic_y()); // is wall bottom
        bool isYpWall = tr.get_Node(n).is_ypWall(tr.periodic_y()); // is wall top

        if(!isXmWall && !isXpWall ) d_ustar_dx = (u_p0 - u_m0 ) / (qnnn.d_p0 + qnnn.d_m0 );
        else                        d_ustar_dx = isXmWall ? (u_p0 - u_00 ) / qnnn.d_p0 : (u_00 - u_m0 ) / qnnn.d_m0;

        if(!isYmWall && !isYpWall ) d_ustar_dy = (v_0p - v_0m ) / (qnnn.d_0p + qnnn.d_0m );
        else                        d_ustar_dy = isYmWall ? (v_0p - v_00 ) / qnnn.d_0p : (v_00 - v_0m ) / qnnn.d_0m;

        minus_div_Ustar(n) = - (d_ustar_dx + d_ustar_dy );
    }
}

void check_Results(QuadTree & tr, ArrayV<double> & hodge, ArrayV<double> & un, ArrayV<double> & vn, int iter )
{
    double max_err_u = 0.;
    double max_err_v = 0.;
    double u_exact;
    double v_exact;
    ArrayV<double> norm_U(tr.number_Of_Nodes());
    ArrayV<double> err_u(tr.number_Of_Nodes());
    ArrayV<double> err_v(tr.number_Of_Nodes());

    for(CaslInt n=0; n<tr.number_Of_Nodes(); ++n)
    {
        double x = tr.x_fr_i(tr.get_Node(n).i);
        double y = tr.y_fr_j(tr.get_Node(n).j);
        u_exact = uExact(x,y);
        v_exact = vExact(x,y);
        max_err_u = MAX(max_err_u, ABS(un(n) - u_exact) );
        max_err_v = MAX(max_err_v, ABS(vn(n) - v_exact) );
        norm_U(n) = sqrt(SQR(un(n)) + SQR(vn(n)) );
        err_u(n)  = ABS(un(n) - u_exact);
        err_v(n)  = ABS(vn(n) - v_exact);
    }

    ArrayV<double> err_hodge(tr.number_Of_Nodes());
    double max_err_hodge = 0.;
    double hodge_exact;
    double hodge_avg = iter==0 ? -PI*PI*PI*PI*PI*PI/144. : 0.;

    for(CaslInt n=0; n<tr.number_Of_Nodes(); n++)
    {
        double x = tr.x_fr_i(tr.get_Node(n).i);
        double y = tr.y_fr_j(tr.get_Node(n).j);
        hodge_exact =iter==0 ? -(x*x*x/3. - PI*x*x/2.) * (y*y*y/3. - PI*y*y/2.) : 0;
        err_hodge(n) = ABS(hodge(n) - hodge_exact + hodge_avg);
        max_err_hodge = MAX(max_err_hodge,ABS(hodge(n) - hodge_exact + hodge_avg));
    }

    cout << "max error Hodge : " << max_err_hodge << endl;
    cout << "max error u     : " << max_err_u     << endl;
    cout << "max error v     : " << max_err_v     << endl;

    char fileName[BUFSIZ];

    snprintf(fileName, BUFSIZ, "../../output/2D/vtk/test_proj_nbsplits=%d_%d.vtk", NB_SPLITS, iter);
    tr.print_VTK_Format(fileName);
    tr.print_VTK_Format(un, vn, "velocities", fileName);
    tr.print_VTK_Format(hodge, "hodge", fileName);
    tr.print_VTK_Format(err_u, err_v, "error_velocities", fileName);

    ofstream out_stream;
    snprintf(fileName, BUFSIZ, "../../output/2D/data/max_err_u_NB_SPLITS_%d.dat", NB_SPLITS);
    out_stream.open(fileName, ios_base::app);
    out_stream << iter << "," << max_err_u << "," << err_u.avg_Abs() << endl;
    out_stream.close();

    snprintf(fileName, BUFSIZ, "../../output/2D/data/max_err_v_NB_SPLITS_%d.dat", NB_SPLITS);
    out_stream.open(fileName, ios_base::app);
    out_stream << iter << "," << max_err_v << "," << err_v.avg_Abs() << endl;
    out_stream.close();

    snprintf(fileName, BUFSIZ, "../../output/2D/data/max_err_hodge_NB_SPLITS_%d.dat", NB_SPLITS);
    out_stream.open(fileName, ios_base::app);
    out_stream << iter << "," << max_err_hodge << endl;
    out_stream.close();
}