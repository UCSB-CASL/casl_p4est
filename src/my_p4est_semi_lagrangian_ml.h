#include <__bit_reference>

#ifndef ML_MASS_CONSERVATION_MY_P4EST_SEMI_LAGRANGIAN_ML_H
#define ML_MASS_CONSERVATION_MY_P4EST_SEMI_LAGRANGIAN_ML_H

#ifdef P4_TO_P8
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_nodes_along_interface.h>
#else
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_nodes_along_interface.h>
#endif

#include <vector>
#include <unordered_set>

/**
 * Machine-learning-based Semi-Lagrangian namespace.
 * @note This functionality has not yet been tested in 3D.
 *
 * Author: Luis Ángel.
 * Date: February 18, 2021.
 */
namespace slml
{
	/**
	 * The data packet containing information for machine learning processing.
	 * In regards to spatial information in P4EST_DIM dimensions, we follow the standard of x been the slowest coord,
	 * then y, then z (changing the fastest).  In 2D, for example, information is returned in the following order:
	 *   Index  Acronym
 	 *	  0	     "mm"		For the quad:    mp +--------+ pp
 	 *	  1	     "mp"                           |---*    |			*: Departure point with normalized coords w.r.t. mm.
 	 *    2		 "pm"                           |   |    |
 	 *	  3	     "pp"                        mm +--------+ pm
	 */
	struct DataPacket
	{
		p4est_locidx_t nodeIdx;			// Node index in the PETSc parallel vector.
		double phi_a;					// Level-set function value at arrival point.
		double vel_a[P4EST_DIM];		// Velocity at arrival point (u, v, and w components).
		double distance;				// Distance between arrival and departure point normalized to (min) cell width.
		double xyz_d[P4EST_DIM];		// Position of departure point in normalized coords (i.e., in [0,1]^{P4EST_DIM}).
		double phi_d[P4EST_CHILDREN];	// Level-set function values at corners of cell containing the (backtracked) departure point.
		double vel_d[P4EST_DIM * P4EST_CHILDREN];	// Serialized velocity at corners of cell containing the backtracked
													// departure point.  Order is vel_u, vel_v [, vel_w].
													// For each vel component, there are P4EST_CHILDREN values.
		double targetPhi_d;				// Reserved for expected/target phi value at departure point.
		double numBacktrackedPhi_d;		// Phi value at numerically backtracked departure point (using linear interp).
		double hk_a;					// Dimensionless curvature at the point on Gamma closest to node nodeIdx.

		/**
		 * Serialize data packet into a vector.
		 * The order is: phi_a, vel_a (by component), distance, xyz_d (by coordinate), phi_d (by child), vel_d
		 * (by component and child), targetPhi_d, numBacktrackedPhi_d.
		 * @param [out] data Output vector container.
		 * @param [in] includeNodeIdx Whether to serialize node index or not.
		 * @param [in] includeHK Whether to serialize numerical curvature or not.
		 */
		void serialize( std::vector<double>& data, bool includeNodeIdx=false, bool includeHK=false ) const
		{
			if( includeNodeIdx )					// Node index.
				data.push_back( nodeIdx );

			data.push_back( phi_a );				// Level-set value at arrival point.

			for( const auto& component : vel_a )	// Components of velocity at arrival point.
				data.push_back( component );

			data.push_back( distance );				// Scaled distance to departure point.

			for( const auto& coord : xyz_d )		// Scaled departure point w.r.t. child at lowest quad/oct coords.
				data.push_back( coord );

			for( const auto& phi : phi_d )			// Level-set values at quad/oct children.
				data.push_back( phi );

			for( const auto& componentAtChild : vel_d )		// Velocity components at quad/oct children.
				data.push_back( componentAtChild );

			data.push_back( targetPhi_d );			// Target level-set value at departure point.

			data.push_back( numBacktrackedPhi_d );	// Semi-Lagrangian approximation to level-set value at departure.

			if( includeHK )
				data.push_back( hk_a );				// Curvature at the interface.
		}

#ifndef P4_TO_P8

		/**
		 * Rotate a sample data packet by 90 degrees counter- or clockwise.
		 * @param [in] dir Rotation direction: +1 for counterclockwise, -1 for clockwise.
		 * @note This function is just tested for 2D.
		 */
		void rotate90( const int& dir=1 )
		{
			// Lambda function to perform rotation of a 2D vector.
			auto _rotate90 = (dir >= 1)? []( double& x, double& y ){
				std::swap( x, (y *= -1) );			// Rotate by positive 90 degrees (counterclockwise).
			} : []( double& x, double& y ){
				std::swap( (x *= -1), y );			// Rotate by negative 90 degrees (clockwise).
			};

			// Lambda function to rotate the children in a 2D quad.
			auto _rotateChildren = (dir >= 1)? []( double children[P4EST_CHILDREN] ){
				double c[P4EST_CHILDREN] = {children[1], children[3], children[0], children[2]};
				for( int i = 0; i < P4EST_CHILDREN; i++ )		// Rotate by positive 90 degrees.
					children[i] = c[i];
			} : []( double children[P4EST_CHILDREN] ){
				double c[P4EST_CHILDREN] = {children[2], children[0], children[3], children[1]};
				for( int i = 0; i < P4EST_CHILDREN; i++ )		// Rotate by negative 90 degrees.
					children[i] = c[i];
			};

			// phi_a remains unchanged.
			// Rotate velocity at departure point.
			_rotate90( vel_a[0], vel_a[1] );

			// distance remains unchanged.
			// Rotate departure coords.
			if( dir >= 1 )
				std::swap( xyz_d[0], (xyz_d[1] = 1.0 - xyz_d[1]) );	// (x, y) -> (1-y, x).
			else
				std::swap( (xyz_d[0] = 1.0 - xyz_d[0]), xyz_d[1] ); // (x, y) -> (y, 1-x).

			// Rotate phi_d.
			_rotateChildren( phi_d );

			// Rotate vel_d.  Requires two steps: first rotate children, then rotate the actual velocity vectors.
			_rotateChildren( &vel_d[0] );				// u component children.
			_rotateChildren( &vel_d[P4EST_CHILDREN] );	// v component children.
			for( int i = 0; i < P4EST_CHILDREN; i++ )	// Rotate actual vectors.
				_rotate90( vel_d[i], vel_d[i + P4EST_CHILDREN] );

			// targetPhi_d remains unchanged.
			// numBacktrackedPhi_d remains unchanged.
			// numK remains unchanged.
		}

		/**
		 * Rotate packet in such a way that the backtracking (i.e., negated) velocity at the arrival point has an angle
		 * has an with respect to the horizontal in the range of [0, pi/2].
		 * is negative.
	 	 */
		void rotateToFirstQuadrant()
		{
			double negVel_a[P4EST_DIM] = {DIM( -vel_a[0], -vel_a[1], -vel_a[2] )};	// This is the negated velocity at
			double theta = atan2( negVel_a[1], negVel_a[0] );						// arrival point used a reference.
			const double TWO_PI = 2. * M_PI;
			theta = (theta < 0)? TWO_PI + theta : theta;	// Make sure current angle lies in [0, 2pi].

			// Rotate only if theta not in [0, pi/2].
			if( theta > M_PI_2 )
			{
				if( theta <= M_PI )				// Quadrant II?
				{
					rotate90( -1 );				// Rotate by -pi/2.
				}
				else if( theta < M_PI_2 * 3 )	// Quadrant III?
				{
					rotate90();					// Rotate by pi.
					rotate90();
				}
				else							// Quadrant IV?
				{
					rotate90();					// Rotate by pi/2.
				}
			}
		}

		/**
		 * Reflect packet along line y = x.
		 * @note Useful for data augmentation assuming that we are using normalization to first quadrant of a local
		 * coordinate system whose origin is at the arrival point.
		 */
		void reflect_yEqx()
		{
			// phi_a remains unchanged.
			// Swap components of velocity at departure point.
			std::swap( vel_a[0], vel_a[1] );

			// distance remains unchanged.
			// Swap local departure coords.
			std::swap( xyz_d[0], xyz_d[1] );

			// Swap opposite corners of phi_d.
			std::swap( phi_d[1], phi_d[2] );

			// Swap vel_d components.  Requires two steps for opposite corners: first swap them, then change components.
			std::swap( vel_d[0 + 1], vel_d[0 + 2] );							// u component of opposite corners.
			std::swap( vel_d[P4EST_CHILDREN + 1], vel_d[P4EST_CHILDREN + 2] );	// v component.
			for( int i = 0; i < P4EST_CHILDREN; i++ )	// Now, swap actual velocity vector components.
				std::swap( vel_d[i], vel_d[i + P4EST_CHILDREN] );

			// targetPhi_d remains unchanged.
			// numBacktrackedPhi_d remains unchanged.
			// numK remains unchanged.
		}
	};
#endif


	//////////////////////////////////////////////////// DataFetcher ///////////////////////////////////////////////////

	/**
	 * A implementation of a data fetcher to retrieve data for backtracked points in space.  It leverages the
	 * parallel communication mechanisms existing in the library interpolation class.  Instead of interpolation, we
	 * retrieve the contents of the quad/oct containing a (backtracked) point.
	 * To allow the "interpolate" method to fetch all the quad information and place it in a results array, we set up a
	 * large number of "dummy" input fields.  These input fields include:
	 *  - the level-set function values and
	 *  - the velocity components u, v[, w]
	 * in that order.  This amounts to 3 input fields in 2D and 4 in 3D.  To make full children information retrieval
	 * possible, we create dummy fields to satisfy the expected outputs.
	 * Upon response, the results array contains information with:
	 *  - an error code: 0 if success, non-zero if failure (see _fetch function),
	 *  - the normalized coordinates in [0,1]^{P4EST_DIM} for the query points w.r.t. the landing quad/oct,
	 *  - the level-set function values of the quad/oct child nodes, and
	 *  - the velocity components u, v[, w], of the quad/oct child nodes.
	 * This amounts to 15 returned "fields" in 2D and 36 in 3D per query point.  For this reason, we must set up 12 dum-
	 * my fields in 2D and 32 in 3D.
	 *
	 * The information returned is serialized in a long results array.  As for data related to quad child nodes, the
	 * output is organized in the xyz order.  That is, x is the slowest changing variable, and z is the fastest changing
	 * variable.  This is the opposite to how data is organized internally in PETSc vectors (given in zyx order).  Visu-
	 * ally, the output information for a quad/oct children looks a follows (based on types.h, see
	 * my_p4est_nodes_along_interface.h too):
	 *
	 *			 2D                      3D                   #vertex  #loc
	 *		v01      v11			010      110               	000		0
	 *	 	*--------*				 *--------*					001		1
	 *	 	|        |				/.   111 /|					010		2
	 * 	 	|        |		   011 *--------* |					011		3
	 *	 	*--------*			   | *......|.*					100		4
	 *		v00      v10		   |· 000   |/ 100				101		5
	 *		 				   	   *--------*					110		6
	 *	   	   y|				  001      101	   y|			111		7
	 *			+---								+--- x
	 *		  	  x								   /z
	 */
	class DataFetcher : public my_p4est_interpolation_t
	{
	private:
		p4est_nodes_t const *_nodes;	// Pointer to nodes struct, needed to access a quad's child vertices.

		/**
		 * Collect information for a quad/oct where a query point has landed.
		 * @param [in] p4est Pointer to the forest struct.
		 * @param [in] treeId Tree index that owns the target quad.
		 * @param [in] quad Quadrant/octant where query point has landed.
		 * @param [in] fields Input fields set up during object configuration.
		 * @param [in] xyz Query point global Cartesian coordinates.
		 * @param [out] results Array where to serialize and store the output data.
		 * @param [in] nResults Number of expected results.
		 * @throws Runtime exception if number of written results doesn't match the expected number of outputs.
		 */
		static void _fetch( const p4est_t *p4est, p4est_topidx_t treeId, const p4est_quadrant_t& quad,
					  		const double *fields, const double xyz[P4EST_DIM], double *results,
					  		const size_t& nResults );

		/**
		 * Retrieve the normalized coordinates of a query point.  These coordinates are given in the range
		 * [0,1]^{P4EST_DIM}, w.r.t. minimum quad/oct corner.
		 * @note This function is based on my_p4est_utils::get_local_interpolation_weights.
		 * @param [in] p4est Pointer to forest struct.
		 * @param [in] treeId Tree that owns the quadrant where the query point lies.
		 * @param [in] quad Quadrant that owns the query point.
		 * @param [in] xyz Global Cartesian coordinates of query point.
		 * @param [out] normalizedXYZ Output normalized coordinates for query point w.r.t. minimum quad/oct corner.
		 */
		static void _getNormalizedCoords( const p4est_t *p4est, const p4est_topidx_t& treeId,
										  const p4est_quadrant_t& quad, const double xyz[P4EST_DIM],
										  double normalizedXYZ[P4EST_DIM] );

	public:
		using my_p4est_interpolation_t::interpolate;

		/**
		 * Input fields order or indices, as they should be given when setting up the inputs of the Data Fetcher object.
		 * The LAST type is only used for iteration purposes.
		 */
		enum InputFields : int {PHI = 0, VEL_U __unused, VEL_V __unused, ONLY3D( VEL_W __unused COMMA ) LAST};

		/**
		 * Constructor.
		 * @param [in] ngbd Pointer to node neighborhood struct.
		 */
		explicit DataFetcher( const my_p4est_node_neighbors_t *ngbd );

		/**
		 * Get ready by initializing the infrastructure containing the information to retrieve.
		 * @param [in] fields Input fields (artificially constructed with dummy PETSc vectors).
		 * @param [in] nFields Number of fields.
		 */
		void setInput( Vec fields[], const int& nFields );

		/**
		 * Fetch sample data for a given quadrant.
		 * @note The name is misleading: we use `interpolate` to meet the implementation requirement of the virtual
		 * function of the same name in the base class.  This function leverages the multiprocess communication infra-
		 * structure existing in the base interpolation class.
		 * @param [in] quad Local or ghost quadrant owning the query point.
		 * @param [in] xyz Query point global Cartesian coordinates.
		 * @param [out] results Array of results on output (as many as input fields were supplied).
		 * @param [in] comp Components to be considered (in our case, only ALL_COMPONENTS is allowed).
		 */
		void interpolate( const p4est_quadrant_t& quad, const double *xyz, double *results, const unsigned int& comp )
			const override;

		/**
		 * Interpolate on the fly.
		 * @note Not implemented yet!  Added just to meet the requirement from the base class.
		 * @param [in] xyz Query point global Cartesian coordinates.
		 * @param [out] results Pointer to an array of results.
		 */
		void operator()( const double *xyz, double *results ) const override;
	};


	/////////////////////////////////////////////////// CacheFetcher ///////////////////////////////////////////////////

	/**
	 * A implementation of a cache fetcher to retrieve improved level-set function values for nodes located next to the
	 * interface at time tn.  It leverages the parallel communication mechanisms existing in the library interpolation
	 * class.  Instead of interpolation, we retrieve the level-set value of the grid point and whether it was computed
	 * with the neural network.
	 * To allow the "interpolate" method to fetch the cache state and place it in a results array, we set up the input
	 * fields:
	 *  - the level-set function values at time Gn from the cache, and
	 *  - the neural flag (1 if computation was improved with neural network, 0 otherwhise)
	 * in that order.  This amounts to 2 input fields regardless of spatial dimensions.
	 * Upon response, the results array contains also 2 fields:
	 *  - the level-set value computed with the neural network or a meaningless value if query point doesn't match a
	 *    grid point at Gn for which we used the neural network, and
	 *  - a flag value: 1 if level-set value is valid, 0 if not.
	 */
	class Cache : public my_p4est_interpolation_t
	{
	private:
		p4est_nodes_t const *_nodes;	// Pointer to nodes struct, needed to access a quad's child vertices.

	public:
		using my_p4est_interpolation_t::interpolate;

		/**
		 * Input fields order or indices, as they should be given when setting up the inputs of the CacheFetcher object.
		 * The LAST type is only used for iteration purposes.
		 */
		enum Fields : int {PHI = 0, FLAG};
		static const int N_FIELDS;

		/**
		 * Constructor.
		 * @param [in] ngbd Pointer to node neighborhood struct at time tn (i.e., Gn).
		 */
		explicit Cache( const my_p4est_node_neighbors_t *ngbd );

		/**
		 * Get ready by initializing the infrastructure containing the information to retrieve.
		 * @param [in] fields Input fields.
		 * @param [in] nFields Number of fields.
		 */
		void setInput( Vec fields[], const int& nFields );

		/**
		 * Fetch cached data for a query grid point (at Gnp1 -- the grid a time tnp1).
		 * @note The name is misleading: we use `interpolate` to meet the implementation requirement of the virtual
		 * function of the same name in the base class.  This function leverages the multiprocess communication infra-
		 * structure existing in the base interpolation class.
		 * @param [in] quad Local or ghost quadrant owning the query point.
		 * @param [in] xyz Query point global Cartesian coordinates.
		 * @param [out] results Array of results on output (as many as input fields were supplied).
		 * @param [in] comp Components to be considered (in this case, only ALL_COMPONENTS is allowed).
		 */
		void interpolate( const p4est_quadrant_t& quad, const double *xyz, double *results, const unsigned int& comp )
		const override;

		/**
		 * Interpolate on the fly.
		 * @note Not implemented yet!  Added just to meet the requirement from the base class.
		 * @param [in] xyz Query point global Cartesian coordinates.
		 * @param [out] results Pointer to an array of results.
		 */
		void operator()( const double *xyz, double *results ) const override;
	};



	////////////////////////////////////////////////// SemiLangrangian /////////////////////////////////////////////////

	/**
 	 * A semi-Lagrangian implementation using Machine Learning and Neural Networks.
 	 */
	class SemiLagrangian : public my_p4est_semi_lagrangian_t
	{
	private:
		const double BAND;			// Band width around the interface: must match what was used for training (>=2).
		Vec _mlFlag = nullptr;		// A flag vector that stores 1s in nodes adjacent to Gamma for which we have
									// computed phi with the nnet, 0 otherwise.  It also distinguishes grid points with
									// valid phi stencils (i.e., uniform in each direction) to compute curvature with
									// the corresponding nnet.
		Vec _mlPhi = nullptr;		// Parallel vector to store advected phi for time tnp1 computed with nnet.

		enum HowUpdated : int {NUM = 0, NUM_BAND, NNET};	// States for determining how a grid point was updated.

		/**
		 * Compute semi-Lagrangian advection for all points and correct level-set values at time tnp1 for grid points
		 * lying next to interface at time tn by using the neural network.
		 * @note Function resets and populates _mlFlag and _mlPhi for points next to the interface.
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction.
		 * @param [in] dt Time step.
		 * @param [in] phi Parallel vector with level-set values for grid at time tn.
		 * @param [in] hk Dimensionless curvature parallel vector.  For points next to Gamma, it is hk at the closest location on the interface.
		 */
		void _computeMLSolution( Vec vel[], const double& dt, Vec phi, Vec hk );

		/**
		 * Advect level-set function using a semi-Lagrangian scheme with a single velocity step (no midpoint) with Euler
		 * along the characteristics.
		 * @param [in] dt Time step.
		 * @param [in] h Minimum cell width (assuming 1:1:1 ratios).
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction.
		 * @param [in] vel_xx Array of second derivatives for each velocity component w.r.t. each Cartesian direction.
		 * @param [in] phi Level-set function values at time n.
		 * @param [in,out] phi_np1Ptr Advected level-set function values.
		 * @param [in,out] howUpdated_np1Ptr Debugging values to indicate how the level-set was updated.
		 */
		void _advectFromNToNp1( const double& dt, const double& h, Vec vel[], Vec *vel_xx[], Vec phi,
						  		double *phi_np1Ptr, double *howUpdated_np1Ptr );

	public:
		/**
		 * Constructor.
		 * @param [in,out] p4estNp1 Pointer to a p4est object pointer.
		 * @param [in,out] nodesNp1 Pointer to a nodes' object pointer.
		 * @param [in,out] ghostNp1 Pointer to a ghost struct pointer.
		 * @param [in,out] ngbdN  Pointer to a neighborhood struct.
		 * @param [in] band Bandwidth to be used around the interface to enforce valid samples.
		 */
		SemiLagrangian( p4est_t **p4estNp1, p4est_nodes_t **nodesNp1, p4est_ghost_t **ghostNp1,
				  		my_p4est_node_neighbors_t *ngbdN, const double& band=2 );

		/**
		 * Collect samples for neural network training.  Use a semi-Lagrangian scheme with a single velocity step along
		 * the characteristics to define the departure points.  Collect samples only for grid points next to the inter-
		 * face whose velocity is essentially nonzero.
		 * @note Here, we create data packets dynamically, and you must not forget free those objects by calling the
		 * utility function freeDataPacketArray.
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction at time n.
		 * @param [in] dt Time step.
		 * @param [in] phi Level-set function values at time n.
		 * @param [out] dataPackets Vector of pointers to data packet objects.
		 * @return true if all backtracked queried points from nodes along interface lie inside within domain, false
		 * otherwise.  This serves as a warning flag.
		 */
		bool collectSamples( Vec vel[P4EST_DIM], const double& dt, Vec phi,
					   		 std::vector<DataPacket *>& dataPackets ) const;

		/**
		 * Utility function to deallocate the dynamically created data packets in collectSamples.
		 * @param [in,out] dataPackets Vector of pointers to dynamically allocated data packet objects.
		 * @return Number of freed objects.
		 */
		static size_t freeDataPacketArray( std::vector<DataPacket *>& dataPackets );

		/**
		 * Update a p4est from tn to tnp1, using a combined semi-Lagrangian scheme: numberical and machine-learning
		 * based, with a single velocity step (no midpoint) with Euler along the characteristics.
		 * The forest at time tn is copied and then refined/coarsened and balance iteratively until convergence.
		 * The method is adapted from:
		 * [*] M. Mirzadeh, A. Guittet, C. Burstedde, and F. Gibou, Parallel Level-Set Method on Adaptive Tree-Based Grids.
		 * @note You need to update the node neighborhood and hierarchy objects yourself upon exit!
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction.
		 * @param [in] dt Time step.
		 * @param [in,out] phi Level-set function values at time n, and then updated at time n + 1.
		 * @param [in] hk Dimensionless curvature.  For points next to Gamma, it is hk at the closest point on Gamma.
		 * @param [in,out] howUpdated Optional parallel vector for debugging how the level-set values were updated: 0 if
		 * 		  numerically, 1 if numerically but within a band around new interface location, 2 if using neural net.
		 */
		void updateP4EST( Vec vel[], const double& dt, Vec *phi, Vec hk, Vec *howUpdated=nullptr );
	};
}


#endif //ML_MASS_CONSERVATION_MY_P4EST_SEMI_LAGRANGIAN_ML_H
