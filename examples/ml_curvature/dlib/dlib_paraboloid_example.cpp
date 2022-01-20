/**
 * Example for finding the minimum distance from a point to a paraboloid in three dimensions usin dlib.
 */
#include <dlib/optimization.h>
#include <dlib/timing.h>
#include <iostream>

using namespace std;

typedef dlib::matrix<double,0,1> column_vector;

const double a_par = 4;				// Paraboloid parameters: Q(u,v) = a*u^2 + b*v^2, for a,b > 0.
const double b_par = 1;
const double P[] = {0.5,0.5,0.5};	// Fixed point.

/**
 * Compute the distance from the paraboloid surface to a fixed point.
 * @param [in] m (u,v) parameters to evaluate in distance function.
 * @return D(u,v) = 0.5*norm(Q(u,v) - P)^2.
 */
double distance_paraboloid_point( const column_vector& m )
{
	const double u = m( 0 );
	const double v = m( 1 );

	return 0.5 * (pow(u-P[0], 2) + pow(v-P[1], 2) + pow(a_par*u*u + b_par*v*v - P[2], 2));
}

/**
 * Gradient of the distance function.
 * @param [in] m (u,v) parameters to evaluate in gradient of distance function.
 * @return grad(D)(u,v).
 */
column_vector distance_paraboloid_point_gradient( const column_vector& m )
{
	const double u = m( 0 );
	const double v = m( 1 );

	// Make us a column vector of length 2 to hold the gradient.
	column_vector res( 2 );

	// Now, compute the gradient vector.
	res( 0 ) = (u - P[0]) + (a_par*u*u + b_par*v*v -P[2]) * 2*u*a_par; 	// dD/du.
	res( 1 ) = (v - P[1]) + (a_par*u*u + b_par*v*v -P[2]) * 2*v*b_par; 	// dD/dv.
	return res;
}

/**
 * The Hessian matrix for the distance function to the paraboloid.
 * @param [in] m (u,v) parameters to evaluate in Hessian of distance function.
 * @return grad(grad(D))(u,v)
 */
dlib::matrix<double> distance_paraboloid_point_hessian ( const column_vector& m )
{
	const double u = m(0);
	const double v = m(1);

	// Make us a 2x2 matrix.
	dlib::matrix<double> res( 2, 2 );

	// Now, compute the second derivatives.
	res( 0, 0 ) = 1 + 6*a_par*a_par*u*u + 2*a_par*b_par*v*v - 2*a_par*P[2];	// d/du(dD/du).
	res( 1, 0 ) = res( 0, 1 ) = 4*a_par*b_par*u*v;							// d/du(dD/dv) and d/dv(dD/du).
	res( 1, 1 ) = 1 + 2*a_par*b_par*u*u + 6*b_par*b_par*v*v - 2*b_par*P[2];	// d/dv(dD/dv).
	return res;
}

/**
 * A function model for the distance function to the paraboloid.  This can be used with the find_min_trust_region()
 * routine.
 */
class distance_paraboloid_point_model
{
public:
	typedef ::column_vector column_vector;
	typedef dlib::matrix<double> general_matrix;

	double operator()( const column_vector& x ) const
	{
		return distance_paraboloid_point( x );
	}

	static void get_derivative_and_hessian( const column_vector& x, column_vector& der, general_matrix& hess )
	{
		der = distance_paraboloid_point_gradient( x );
		hess = distance_paraboloid_point_hessian( x );
	}
};

int main() try
{
	// Initial point: should be close to true minimum.
	column_vector starting_point = {1./3, 0.365079};
	const double delta_stop = 1e-7;

	// First, we check that the numerical gradient is close to the analytical one we provided.
	cout << "Difference between analytic derivative and numerical approximation of derivative: "
		 << dlib::length( dlib::derivative( distance_paraboloid_point )( starting_point ) - distance_paraboloid_point_gradient( starting_point ) )
		 << endl;

	cout << "\nFind the minimum of the distance function" << endl << endl;
	cout.precision( 9 );

	// Now we use the find_min() function to find the minimum point.
	dlib::timing::start( 0, "BFGS #1" );
	dlib::find_min( dlib::bfgs_search_strategy(),									// Use BFGS search algorithm.
					dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),	// Stop when the change in distance function is less than some threshold.
					distance_paraboloid_point, distance_paraboloid_point_gradient,
					starting_point,									// We send the starting point and get back the critical point back in the same variable.
					-1 );											// If the algorithms finds an output <= -1, stop (causes the last arugment to be disregarded).
	dlib::timing::stop( 0 );
	cout << "Critical point (BFGS):\n" << starting_point << endl;

	// Now let's try doing it again with a different starting point.
	starting_point = {0.269841, 0.492063};
	dlib::timing::start( 1, "BFGS #2" );
	dlib::find_min( dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),
					distance_paraboloid_point, distance_paraboloid_point_gradient, starting_point, -1 );
	dlib::timing::stop( 1 );
	cout << "Critical point (BFGS 2nd take):\n" << starting_point << endl;

	// Here we repeat the same thing as above but this time using the L-BFGS algorithm.  L-BFGS is very similar to the
	// BFGS algorithm, however, BFGS uses O(N^2) memory where N is the size of the starting_point vector.  The L-BFGS
	// algorithm however uses only O(N) memory.  So if you have a function of a huge number of variables the L-BFGS
	// algorithm is probably a better choice.
	starting_point = {1./3, 0.365079};
	dlib::timing::start( 2, "L-BFGS #1" );
	dlib::find_min( dlib::lbfgs_search_strategy( 10 ),  						// The 10 here is basically a measure of how much memory L-BFGS will use.
					dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),	// Adding be_verbose() causes a message to be
					distance_paraboloid_point, 									// printed for each iteration of optimization.
					distance_paraboloid_point_gradient, starting_point, -1 );
	dlib::timing::stop( 2 );
	cout << "Critical point (L-BFGS):\n" << starting_point << endl;

	starting_point = {0.269841, 0.492063};
	dlib::timing::start( 3, "L-BFGS #2" );
	dlib::find_min( dlib::lbfgs_search_strategy( 10 ), dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),
					distance_paraboloid_point, distance_paraboloid_point_gradient, starting_point, -1 );
	dlib::timing::stop( 3 );
	cout << "Critical point (L-BFGS 2nd take):\n" << starting_point << endl;

	// In many cases, it is useful if we also provide second derivative information to the optimizers.  Two examples of
	// how we can do that are shown below.
	starting_point = {1./3, 0.365079};
	dlib::timing::start( 4, "Newton #1" );
	dlib::find_min( dlib::newton_search_strategy( distance_paraboloid_point_hessian ),
					dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),
					distance_paraboloid_point, distance_paraboloid_point_gradient, starting_point, -1 );
	dlib::timing::stop( 4 );
	cout << "Critical point (Newton):\n" << starting_point << endl;

	starting_point = {0.269841, 0.492063};
	dlib::timing::start( 5, "Newton #2" );
	dlib::find_min( dlib::newton_search_strategy( distance_paraboloid_point_hessian ),
					dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),
					distance_paraboloid_point, distance_paraboloid_point_gradient, starting_point, -1 );
	dlib::timing::stop( 5 );
	cout << "Critical point (Newton 2nd take):\n" << starting_point << endl;

	// We can also use find_min_trust_region(), which is also a method which uses second derivatives.  For some kinds of
	// non-convex function it may be more reliable than using a newton_search_strategy with find_min().
	starting_point = {1./3, 0.365079};
	dlib::timing::start( 6, "Trust region #1" );
	dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),
								 distance_paraboloid_point_model(), starting_point, 0.037 );	// Initial trust region radius
	dlib::timing::stop( 6 );
	cout << "Critical point (trust region):\n" << starting_point << endl;

	starting_point = {0.269841, 0.492063};
	dlib::timing::start( 7, "Trust region #2" );
	dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( delta_stop ).be_verbose(),
								 distance_paraboloid_point_model(), starting_point, 0.037 );	// Initial trust region radius
	dlib::timing::stop( 7 );
	cout << "Critical point (trust region 2nd take):\n" << starting_point << endl;

	dlib::timing::print();
	cout << endl;
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}



