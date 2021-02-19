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
		double phi_d[P4EST_CHILDREN];	// Level-set function values at corners of cell containing the (backtraced) departure point.
		double vel_d[P4EST_DIM * P4EST_CHILDREN];	// Serialized velocity at corners of cell containing the
													// (backtraced) departure point.  Order is vel_u, vel_v [, vel_w].
													// For each vel component, there are P4EST_CHILDREN values.
	};


	//////////////////////////////////////////////////// DataFetcher ///////////////////////////////////////////////////

	/**
	 * A implementation of a data fetcher to retrieve data for backtraced points in space.  It leverages the
	 * parallel communication mechanisms existing in the library interpolation class.  Instead of interpolation, we
	 * retrieve the contents of the quad/oct containing a (backtraced) point.
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
	class DataFetcher: public my_p4est_interpolation_t
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
		enum InputFields : int {PHI = 0, VEL_U __unused, VEL_V __unused, ONLY3D( VEL_W COMMA ) LAST};

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

	////////////////////////////////////////////////// SemiLangrangian /////////////////////////////////////////////////

	/**
 	 * A semi-Lagrangian implementation using Machine Learning and Neural Networks.
 	 */
	class SemiLagrangian: public my_p4est_semi_lagrangian_t
	{
	public:
		/**
		 * Constructor.
		 * @param [in,out] p4estNp1 Pointer to a p4est object pointer.
		 * @param [in,out] nodesNp1 Pointer to a nodes' object pointer.
		 * @param [in,out] ghostNp1 Pointer to a ghost struct pointer.
		 * @param [in,out] ngbdN  Pointer to a neighborhood struct.
		 */
		SemiLagrangian( p4est_t **p4estNp1, p4est_nodes_t **nodesNp1, p4est_ghost_t **ghostNp1,
				  		my_p4est_node_neighbors_t *ngbdN );

		/**
		 * Collect samples for neural network training.  Use a semi-Lagrangian scheme with a single velocity step along
		 * the characteristics to define the departure points.
		 * @note Here, we create data packets dynamically, and you must not forget free those objects by calling the
		 * utility function freeDataPacketArray.
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction at time n.
		 * @param [in] dt Time step.
		 * @param [in] phi Level-set function values at time n.
		 * @param [out] dataPackets Vector of pointers to data packet objects.
		 */
		void collectSamples( Vec vel[P4EST_DIM], double dt, Vec phi, std::vector<DataPacket *>& dataPackets ) const;

		/**
		 * Utility function to deallocate the dynamically created data packets in collectSamples.
		 * @param [in,out] dataPackets Vector of pointers to dynamically allocated data packet objects.
		 * @return Number of freed objects.
		 */
		static size_t freeDataPacketArray( std::vector<DataPacket *>& dataPackets );
	};
}


#endif //ML_MASS_CONSERVATION_MY_P4EST_SEMI_LAGRANGIAN_ML_H
