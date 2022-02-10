/**
 * A collection of local utility functions for the curvature problem.
 * Developer: Luis Ángel.
 * Created: May 2, 2021.
 * Updated: February 9, 2022.
 */

#ifndef ML_CURVATURE_LOCAL_UTILS_H
#define ML_CURVATURE_LOCAL_UTILS_H

#include <sys/stat.h>
#include <vector>
#include <string>
#include <src/my_p4est_utils.h>
#include <unordered_map>
#include <boost/math/tools/roots.hpp>
#include "data_sets/star_theta_root_finding.h"
#include "data_sets/gaussian_3d.h"

namespace kutils
{
	/**
	 * Generate the sample row of level-set function values and target hk for a node next to the star interface.
	 * @param [in] nodeIdx Query node adjancent or on the interface.
	 * @param [in] NUM_COLUMNS Number of columns in output file.
	 * @param [in] H Spacing (smallest quad/oct side-length).
	 * @param [in] stencil The full uniform stencil of indices centered at the query node.
	 * @param [in] p4est Pointer to p4est data structure.
	 * @param [in] nodes Pointer to nodes data structure.
	 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
	 * @param [in] star The level-set function with a star-shaped interface.
	 * @param [in] gen Random-number generator device.
	 * @param [in] normalDistribution A normal random variable distribution.
	 * @param [in/out] pointsFile Pointer to optional file object to write coordinates of nodes adjacent to Gamma.
	 * @param [in/out] anglesFile Pointer to optional file object to write angles of normal projected points on Gamma.
	 * @param [out] distances A vector of "true" distances from all of 9 stencil points to the star-shaped level-set.
	 * @param [out] pOnGamma Normal projection onto interface.
	 * @param [in,out] visitedNodes Hash map functioning as a memoization mechanism to speed up access to visited nodes.
	 * @param [in] normalReadPtr Pointer to normal vector components backed by parallel PETSc vectors.
	 * @param [out] tgtHK Target hk.
	 * @param [in] verbose Whether to print debugging messages or not.
	 * @return Vector of sampled, reinitialized level-set function values for the stencil centered at the nodeIdx node.
	 */
	std::vector<double> sampleNodeNextToStarInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
													   const double H, const std::vector<p4est_locidx_t>& stencil,
													   const p4est_t *p4est, const p4est_nodes_t *nodes,
													   const double *phiReadPtr, const geom::Star& star,
													   std::mt19937& gen, std::normal_distribution<double>& normalDistribution,
													   std::ofstream *pointsFile, std::ofstream *anglesFile,
													   std::vector<double>& distances, double pOnGamma[P4EST_DIM],
													   std::unordered_map<p4est_locidx_t, Point2>& visitedNodes,
													   const double *normalReadPtr[P4EST_DIM], double& tgtHK,
													   const bool& verbose=true )
	{
		std::vector<double> sample;			// Here, we write h-normalized level-set values.
		sample.reserve( NUM_COLUMNS );
		distances.clear();
		distances.reserve( NUM_COLUMNS );	// True distances and target hk.

		int s;												// Index to fill in the sample vector.
		double xyz[P4EST_DIM];
		double pOnInterface[P4EST_DIM];
		double theta, r, valOfDerivative, centerTheta;
		double dx, dy, newDistance;
		for( s = 0; s < num_neighbors_cube; s++ )			// Collect phi(x) for each of the 9 grid points.
		{
			sample.push_back( phiReadPtr[stencil[s]] );		// This is the distance obtained after reinitialization.

			// Approximate position of point projected on interface.
			const double grad[P4EST_DIM] = {DIM( normalReadPtr[0][stencil[s]], normalReadPtr[1][stencil[s]], normalReadPtr[2][stencil[s]] )};
			node_xyz_fr_n( stencil[s], p4est, nodes, xyz );
			for( int dim = 0; dim < P4EST_DIM; dim++ )
				pOnInterface[dim] = xyz[dim] - grad[dim] * sample[s];

			if( s == 4 )	// Rough estimation of point on interface, where curvature will be interpolated.
			{
				for( int dim = 0; dim < P4EST_DIM; dim++ )
					pOnGamma[dim] = pOnInterface[dim];
			}

			// Get initial angle for polar approximation to point on star interface.
			theta = atan2( pOnInterface[1], pOnInterface[0] );
			theta = ( theta < 0 )? theta + 2 * M_PI : theta;
			r = star.r( theta );
			pOnInterface[0] = r * cos( theta );
			pOnInterface[1] = r * sin( theta );				// Better approximation of projection of stencil point onto star.

//		if( s == 4 )
//		{
//			std::cout << std::setprecision( 15 )
//					  << "plot(" << xyz[0] << ", " << xyz[1] << ", 'b.', " << pOnInterfaceX << ", " << pOnInterfaceY
//					  << ", 'mo');" << std::endl;
//		}

			// Compute current distance to Gamma using the improved point on interface.
			dx = xyz[0] - pOnInterface[0];
			dy = xyz[1] - pOnInterface[1];
			distances.push_back( sqrt( SQR( dx ) + SQR( dy ) ) );

			// Find theta that yields "a" minimum distance between stencil point and star using Newton-Raphson's method.
			if( distances.back() > EPS )
			{
				if( visitedNodes.find( stencil[s] ) != visitedNodes.end() )		// Speed up queries.
				{
					theta = visitedNodes[stencil[s]].x;			// First component is the angular parameter.
					newDistance = visitedNodes[stencil[s]].y;	// Second component is the distance to Gamma.
				}
				else
				{
					valOfDerivative = 1;
					theta = distThetaDerivative_Star( stencil[s], xyz[0], xyz[1], star, theta, H, gen,
													  normalDistribution, valOfDerivative, newDistance, verbose );

//				if( s == 4 )
//				{
//					r = star.r( theta );					// Recalculating closest point on interface.
//					xOnGamma = r * cos( theta );
//					yOnGamma = r * sin( theta );
//					std::cout << std::setprecision( 15 )
//							  << "plot(" << xOnGamma << ", " << yOnGamma << ", 'ko');" << std::endl;
//				}

					double relDist = (newDistance - distances[s]) / H;
					if( relDist > 1e-4  )					// Verify that new point is closer than previous approximation.
					{
						std::ostringstream stream;
						stream << "Failure with node " << stencil[s] << ".  Val. of Der: " << std::scientific << valOfDerivative
							   << std::scientific << std::setprecision( 15 ) << ".  New dist: " << newDistance
							   << ".  Old dist: " << distances[s]
							   << ".  Rel dist: " << relDist;
						throw std::runtime_error( stream.str() );
					}

					visitedNodes[stencil[s]] = Point2( theta, newDistance );		// Memorize information for visited node.
				}

				distances[s] = newDistance;					// Root finding was successful: keep minimum distance.
			}

			if( star( xyz[0], xyz[1] ) < 0 )				// Fix sign.
				distances[s] *= -1;

			if( s == 4 )									// For center node we need theta to yield curvature.
				centerTheta = theta;

			// Normalize by H.
			sample[s] /= H;
			distances[s] /= H;
		}

		tgtHK = H * star.curvature( centerTheta );			// Write output expected hk.

		// Write center sample node index and coordinates.
		if( pointsFile )
		{
			node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
			*pointsFile << nodeIdx << "," << xyz[0] << "," << xyz[1] << std::endl;
		}

		// Write angle parameter for projected point on interface.
		if( anglesFile )
		{
			*anglesFile << ( centerTheta < 0 ? 2 * M_PI + centerTheta : centerTheta ) << std::endl;
		}

		return sample;
	}

	/**
	 * Find the axis value in the Gaussian's canonical coordinate system where curvature becomes 0 using Newton-Raphson.
	 * @param [in] Q The Gaussian Monge patch Q(u,v).
	 * @param [in] h Mesh size.
	 * @param [in] dir Either 0 for u or 1 for v.
	 * @return The p value where kappa(p,0) or kappa(0,p) is zero (depending on the chosen direction).
	 * @throws runtime exception if dir is not 0 or 1, or if bracketing or Newton-Raphson's method fails to find the root.
	 */
	double findKappaZero( const Gaussian& Q, const double& h, const unsigned char& dir )
	{
		using namespace boost::math::tools;

		if( dir != 0 && dir != 1 )
			throw std::runtime_error( "[CASL_ERROR] findKappaZero: Wrong direction!  Choose either 0 for u or 1 for v." );

		// Define parameters depending on direction: 0 for u, 1 for v.
		double s2 = (dir == 0? Q.su2() : Q.sv2());
		double t2 = (dir == 0? Q.sv2() : Q.su2());

		// Curvature with one of the directions set to zero.
		auto kappa = [&Q, &dir, &s2, &t2]( const double& p ){
			double q = (dir == 0? Q(p, 0) : Q(0, p));
			return SQR( q * p ) + SQR( s2 ) + s2 * t2 - SQR( p ) * t2;
		};

		// And this is the simplified expression to compute both kappa and its derivative kappa' with one of the dirs set to zero.
		auto kappaAndDKappa = [&Q, &kappa, &dir, &s2, &t2]( const double& p ){
			double k =  kappa( p );
			double q = (dir == 0? Q(p, 0) : Q(0, p));
			double dk = 2 * p * (SQR( q ) * (1 - SQR( p ) / s2) - t2);
			return std::make_pair( k, dk );
		};

		const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
		int getDigits = static_cast<int>( digits * 0.75 );		// Accuracy doubles with each step in Newton-Raphson's, so
																// stop when we have just over half the digits correct.
		boost::uintmax_t it = 0;
		const boost::uintmax_t MAX_IT = 10;						// Maximum number of iterations for bracketing and Newton-Raphson's.

		double s = (dir == 0? Q.su() : Q.sv());
		double start = h;										// Determine the initial bracket with a sliding a window.
		double end = 2 * s;										// We need to find an interval with different kappa signs in its endpoints.
		while( it < MAX_IT && kappa( start ) * kappa( end ) > 0 )
		{
			end += 0.5 * s;
			start += 0.5 * s;
			it++;
		}

		if( kappa( start ) * kappa( end ) > 0 )
			throw std::runtime_error( "[CASL_ERROR] findKappaZero: Failed to find a reliable bracket for " + std::to_string( dir ) + " direction!" );

		double root = (start + end) / 2;	// Initial guess.
		it = MAX_IT;						// Find zero with Newton-Raphson's.
		root = newton_raphson_iterate( kappaAndDKappa, root, start, end, getDigits, it );

		if( it >= MAX_IT )
			throw std::runtime_error( "[CASL_ERROR] findKappaZero: Couldn't find zero with Newton-Raphson's method for " + std::to_string( dir ) + " direction!" );

		return root;
	}
}

#endif // ML_CURVATURE_LOCAL_UTILS_H
