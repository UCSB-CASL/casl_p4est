#ifndef MY_P4EST_SEMI_LAGRANGIAN_ML_H
#define MY_P4EST_SEMI_LAGRANGIAN_ML_H

#ifdef P4_TO_P8
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_nodes_along_interface.h>
#else
#define MASS_INPUT_SIZE			22
#define MASS_INPUT_PHI_SIZE 	 6
#define MASS_INPUT_VEL_SIZE		10
#define MASS_INPUT_DIST_SIZE	 1
#define MASS_INPUT_COORDS_SIZE	 2
#define MASS_INPUT_HK_SIZE		 1
#define MASS_INPUT_PHI_XX_SIZE	 2

#define MASS_N_COMPONENTS		18	// Number of components for PCA dimensionality reduction; same as nnet input size.

#define MASS_BAND_HALF_WIDTH	 2	// Distance in min diags to enforce a uniform band around Gamma^n.

#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_nodes_along_interface.h>
#endif

#include <fdeep/fdeep.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <unordered_set>
#include <cblas.h>

/**
 * Machine-learning-based Semi-Lagrangian namespace.
 * @note This functionality has not yet been tested in 3D.  It works for MPI application and OpenMP.
 *
 * Libraries:
 * @cite JSON https://github.com/nlohmann/json.
 * @cite Frugally deep https://github.com/Dobiasd/frugally-deep, v0.15.2-p0 (02/23/2021) for loading tensorflow+keras
 * models into C++.  Main dependencies to convert models: python 3.7, tensorflow 2.4.1, and C++14.  JSON library should
 * be installed as described in frugally deep documentation.
 * @cite OpenBlas https://github.com/xianyi/OpenBLAS to speed up inference performance.  Used as a replacement for the
 * frugally deep library inference.  OpenBlas allows for batch processing rather than one sample at a time.  But I still
 * use some functionality from frugally deep to process the nnet JSON config file.
 *
 * Author: Luis Ángel.
 * Created: February 18, 2021.
 * Updated: October 11, 2021.
 */
namespace slml
{
	////////////////////////////////////////////////////// Scaler //////////////////////////////////////////////////////

	/**
	 * Abstract class to transform data into an input form that the neural network understands.
	 * Input 2D data comes in the following order (with MASS_NNET_INPUT_SIZE entries):
	 * 		phi_a,										Level-set value at arrival point.
	 *		u_a, v_a, 									Velocity components at arrival point.
	 *		d,											H-normalized distance.
	 *		x_d, y_d,									Scaled departure coords with respect to quad's lower corner.
	 *		phi_d_mm, phi_d_mp, phi_d_pm, phi_d_pp,		Level-set values at departure quad's children.
	 *		u_d_mm, u_d_mp, u_d_pm, u_d_pp,				Velocity component u at departure quad's children.
	 *		v_d_mm, v_d_mp, v_d_pm, v_d_pp,				Velocity component v at departure quad's children.
	 *		h2_phi_xx, h2_phi_yy,						h^2 * abs. value of 2nd spatial derivatives of level-set function at x_d.
	 *		hk,											Dimensionless curvature at the interface for arrival point.
	 *		numerical_phi_d								Numerically computed phi value at departure point.
	 */
	class Scaler
	{
		using json = nlohmann::json;

	protected:
		static void _printParams( const json& params, const std::string& paramsFileName );

	public:
		const int       PHI_COLS[MASS_INPUT_PHI_SIZE   ] = {0,6, 7, 8, 9,21};				// Phi column indices.
		const int       VEL_COLS[MASS_INPUT_VEL_SIZE   ] = {1,2,10,11,12,13,14,15,16,17};	// Vel column indices.
		const int      DIST_COLS[MASS_INPUT_DIST_SIZE  ] = {3};								// Dist column indices.
		const int    COORDS_COLS[MASS_INPUT_COORDS_SIZE] = {4,5};							// Coords column indices.
		const int        HK_COLS[MASS_INPUT_HK_SIZE    ] = {20};							// hk column indices.
		const int H2_PHI_XX_COLS[MASS_INPUT_PHI_XX_SIZE] = {18,19};							// h^2 * phi_xx column indices.

		/**
		 * Transform input data in place.
		 * @param [in,out] samples Data to transform.
		 * @param [in] nSamples Number of samples.
		 */
		virtual void transform( double samples[][MASS_INPUT_SIZE], const int& nSamples ) const = 0;
	};

	//////////////////////////////////////////////////// PCAScaler /////////////////////////////////////////////////////

	/**
	 * Transform data into an input form that the neural network understands.
	 * This is a transformer that applies principal component analysis [and optionally whitening] to the input.
	 * Depending on the numer of components, it also performs dimensionality reduction.
	 * See Python project ML_Mass_Conservation's Training module to understand how we applied the transformation during
	 * the learning stage.
	 */
	class PCAScaler : public Scaler
	{
		using json = nlohmann::json;

	private:
		std::vector<std::vector<double>> _components;	// Actual components.
		std::vector<double> _means;						// Mean values.
		std::vector<double> _stds;						// Standard deviation values.
		bool _whiten;									// Using whitening?

	public:
		/**
		 * Constructor.
		 * @param [in] paramsFileName JSON file name with standard scaler parameters.
		 * @param [in] printLoadedParams Whether to print loaded parameters in or not.
		 */
		explicit PCAScaler( const std::string& paramsFileName, const bool& printLoadedParams=true );

		/**
		 * Transform input data in place.
		 * In python, transform goes as ((Y-pcaw.mean_)@pcaw.components_.T)/np.sqrt(pcaw.explained_variance_)
		 * @param [in,out] samples Data to transform. Upon transformation, only the first MASS_N_COMPONENTS elements are valid.
		 * @param [in] nSamples Number of samples.
		 */
		void transform( double samples[][MASS_INPUT_SIZE], const int& nSamples ) const override;
	};

	////////////////////////////////////////////////// StandardScaler //////////////////////////////////////////////////

	/**
	 * Transform data into an input form that the neural network understands.
	 * See Python project ML_Mass_Conservation's Preprocessing module for the equivalent class that generates the params
	 * file for this implementation.
	 */
	class StandardScaler : public Scaler
	{
		using json = nlohmann::json;

	private:
		double _meanPhi = 0.;		// Mean of level-set values.
		double _stdPhi = 1.;		// Standard deviation of level-set values.

		double _meanVel = 0.;		// Mean of velocity components.
		double _stdVel = 1.;		// Standard deviation of level-set values.

		double _meanDist = 0.;		// Mean of distance between arrival and departure point.
		double _stdDist = 1.;		// Standard deviation of distance.

		double _meanCoord = 0.;		// Mean of scaled coordinates of departure point w.r.t. quad's lower corner.
		double _stdCoord = 1.;		// Standard deviation of scaled coordinates.

		double _meanHK = 0.;		// Mean of dimensionless curvature at the interface: it can be numerical or neural.
		double _stdHK = 1.;			// Standard deviation of dimensionless curvature.

		double _meanH2Phi_xx = 0.;	// Mean of second spatial derivatives of phi at departure point.
		double _stdH2Phi_xx = 1.;	// Standard deviation of second spatial derivatives of phi at departure point.

		/**
		 * Utility function to load group of parameters.
		 * @param [in] inName Parameter key name as given in JSON file.
		 * @param [in] params JSON object.
		 * @param [out] outMean Where to store the mean.
		 * @param [out] outStd Where to store the standard deviation.
		 * @throws runtime error if param key is not found in JSON object.
		 */
		static void _loadParams( const std::string& inName, const json& params, double& outMean, double& outStd );

	public:
		/**
		 * Constructor.
		 * @param [in] paramsFileName JSON file name with standard scaler parameters.
		 * @param [in] printLoadedParams Whether to print loaded parameters in or not.
		 */
		explicit StandardScaler( const std::string& paramsFileName, const bool& printLoadedParams=true );

		/**
		 * Transform input data in place.
		 * @param [in,out] samples Data to transform.
		 * @param [in] nSamples Number of samples.
		 */
		void transform( double samples[][MASS_INPUT_SIZE], const int& nSamples ) const override;
	};


	////////////////////////////////////////////////// NeuralNetwork ///////////////////////////////////////////////////

	/**
	 * Semi-Lagrangian error-correcting neural network.
	 * Internally, processes batches of samples in the following format:
	 * 										Samples (n)
	 * 						s_0  s_1  ...  s_i  ... s_{n-2}  s_{n-1}
	 * 				f_0	  |  #	  #	  ...	#   ...	   #		#	 |
	 * 				f_1	  |  #	  #	  ...	#   ...	   #		#	 |
	 * 				 :	  |	 :	  :	   ·	:	 ·	   :		:	 |
	 * Features (k)	f_j	  |	 #	  #	  ...	#   ...	   #		#	 |
	 * 				 :	  |	 :	  :	   ·	:	 ·	   :		:	 |
	 * 			  f_{k-2} |  #	  #	  ...	#   ...	   #		#	 |
	 * 			  f_{k-1} |  1    1	  ...	1   ...	   1		1	 |
	 *
	 * Features includes one row of ones to account for the bias.  This means that the very input batch to the nnet has
	 * one additional row to produce the outputs of the first hidden layer.
	 */
	class NeuralNetwork
	{
		using json = nlohmann::json;

	private:
		const PCAScaler _pcaScaler;				// Preprocessing module including type-based standard scaler followed by
		const StandardScaler _stdScaler;		// PCA dimensionality reduction and whitening.

		unsigned long N_LAYERS;					// Number of layers (hidden + output).
		std::vector<std::vector<int>> _sizes;	// Matrix size tuples (m, k).  m = layer size, k = input size, (n = number of samples).
		std::vector<std::vector<FDEEP_FLOAT_TYPE>> W;	// Weight matrices flattened.  Matrices are given in row-major order.

		const double H;							// Mesh size.

		/**
		 * ReLU activation function.
		 * @param [in] x Input.
		 * @return ReLu(x) = max(0, x).
		 */
		static FDEEP_FLOAT_TYPE _reLU( const FDEEP_FLOAT_TYPE& x );

		/**
		 * Softplus activation function.
		 * @param [in] x Input.
		 * @return softplus(x) = log(exp(x) + 1).
		 */
		static FDEEP_FLOAT_TYPE _softPlus( const FDEEP_FLOAT_TYPE& x );

	public:
		/**
		 * Constructor.
		 * @param [in] folder Full path to folder that holds the neural network (mass_nnet.json),
		 * 			   pca (mass_pca_scaler.json), and standard scaler (mass_std_scaler.json) JSON files.
		 * @param [in] h Mesh size.
		 * @param [in] verbose Whether to print debugging information or not.
		 */
		explicit NeuralNetwork( const std::string& folder, const double& h, const bool& verbose=true );

		/**
		 * Predict corrected level-set function values at departure points for a batch of samples.
		 * @note This function assumes that the inputs have been already negated when curvature is positive.  User must
		 * take care of fixing the predictions signs accordingly.
		 * @param [in,out] inputs Array of sample inputs with raw data (they'll be transformed).
		 * @param [out] outputs Array of predicted level-set function values
		 * @param [in] nSamples Batch size.
		 */
		void predict( double inputs[][MASS_INPUT_SIZE], double outputs[], const int& nSamples ) const;
	};


	//////////////////////////////////////////////////// DataPacket ////////////////////////////////////////////////////

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
		double vel_a[P4EST_DIM];		// Midpoint velocity at arrival point (u, v, and w components).
		double distance;				// Distance between arrival and departure point normalized to (min) cell width.
		double xyz_d[P4EST_DIM];		// Position of departure point in normalized coords (i.e., in [0,1]^{P4EST_DIM}).
		double phi_d[P4EST_CHILDREN];	// Level-set function values at corners of cell containing the (backtracked) departure point.
		double vel_d[P4EST_DIM * P4EST_CHILDREN];	// Serialized velocity at corners of cell containing the backtracked
													// departure point.  Order is vel_u, vel_v[, vel_w].
													// For each vel component, there are P4EST_CHILDREN values.
		double h2_phi_xx_d[P4EST_DIM];	// Serialized, h^2 scaled absolute value of 2nd spatial phi derivative at x_d.
										// To compute it, we use bilinear interpolation (invariant to 90 deg rotations).
		double targetPhi_d;				// Reserved for expected/target phi value at departure point.
		double numBacktrackedPhi_d;		// Phi value at numerically backtracked departure point (using linear interp).
		double hk_a;					// Dimensionless curvature at node nodeIdx.

		// Debugging fields.
		double theta_a;					// Angle between vel_a and normal vector at arrival point.
		double absHRelError_d;			// Absolute h-normalized phi_d error at departure point.

		/**
		 * Serialize data packet into a vector.
		 * The order is: phi_a, vel_a (by component), distance, xyz_d (by coordinate), phi_d (by child), vel_d
		 * (by component and child), phi_xx_d (by component), targetPhi_d, numBacktrackedPhi_d, hk_a.
		 * @param [out] data Output vector container.
		 * @param [in] includeNodeIdx Whether to serialize node index or not.
		 * @param [in] includeH2Phi_xx Whether to serialize h^2 * 2nd spatial derivatives of level-set function or not.
		 * @param [in] includeHK Whether to serialize numerical curvature or not.
		 * @param [in] includeTargetPhi_d Whether to serialize target phi value or not.
		 */
		void serialize( std::vector<double>& data, bool includeNodeIdx=false, bool includeH2Phi_xx=true,
						bool includeHK=false, bool includeTargetPhi_d=true ) const
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

			if( includeH2Phi_xx )					// h^2 scaled second spatial derivatives of level-set function.
			{
				for( const auto& dim : h2_phi_xx_d )
					data.push_back( dim );
			}

			if( includeTargetPhi_d )
				data.push_back( targetPhi_d );		// Target level-set value at departure point.

			data.push_back( numBacktrackedPhi_d );	// Semi-Lagrangian approximation to level-set value at departure.

			if( includeHK )
				data.push_back( hk_a );				// Dimensionless curvature at arrival point.
		}

#ifndef P4_TO_P8

		/**
		 * Rotate a sample data packet by 90 degrees counter- or clock-wise.
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
			// Rotate midpoint velocity at departure point.
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

			// h2_phi_xx_d remains unchanged.
			// targetPhi_d remains unchanged.
			// numBacktrackedPhi_d remains unchanged.
			// hk_a remains unchanged.
		}

		/**
		 * Rotate packet in such a way that the backtracking (i.e., negated midpoint) velocity at the arrival point has
		 * an angle respect to the horizontal in the range of [0, pi/2].
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

			// h2_phi_xx_d remains unchanged.
			// targetPhi_d remains unchanged.
			// numBacktrackedPhi_d remains unchanged.
			// hk_a remains unchanged.
		}
	};
#endif


	//////////////////////////////////////////////////// DataFetcher ///////////////////////////////////////////////////

	/**
	 * A implementation of a data fetcher to retrieve data for backtracked points in space.  It leverages the
	 * parallel communication mechanisms existing in the library interpolation class.  Instead of interpolation, we
	 * retrieve the contents of the quad/oct containing a (backtracked) point.
	 * To allow the "interpolate" method to fetch all the quad information and place it in a results array, we set up a
	 * large number of "dummy" input fields.  The input fields include:
	 *  - the level-set function values,
	 *  - the velocity components u, v[, w], and
	 *  - the level-set function second spatial derivatives phi_xx, phi_yy[, phi_zz]
	 * in that order.  This amounts to 5 input fields in 2D and 7 in 3D.  To make full children information retrieval
	 * possible, we create dummy fields to satisfy the expected outputs.
	 * Upon response, the results array contains information with:
	 *  - an error code: 0 if success, non-zero if failure (see _fetch function),
	 *  - the normalized coordinates in [0,1]^{P4EST_DIM} for the query points w.r.t. the landing quad/oct,
	 *  - the level-set function values of the quad/oct child nodes,
	 *  - the velocity components u, v[, w], of the quad/oct child nodes, and
	 *  - the level-set function second spatial derivatives phi_xx, phi_yy[, phi_zz] bilinearly interpolated at x_d.
	 * This amounts to 17 returned "fields" in 2D and 39 in 3D per query point.  For this reason, we must set up 12 dum-
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
		enum InputFields : int {PHI = 0, VEL_U __unused, VEL_V __unused, ONLY3D( VEL_W __unused COMMA )
								PHI_XX __unused, PHI_YY __unused, ONLY3D( PHI_ZZ __unused )};

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


	/////////////////////////////////////////////////// Cache Fetcher //////////////////////////////////////////////////

	/**
	 * A implementation of a cache fetcher to retrieve improved level-set function values for nodes next to Gamma^n. It
	 * leverages the parallel communication mechanisms existing in the library interpolation class.  However, instead of
	 * interpolation, we retrieve the grid point level-set value and whether it was computed with the neural network.
	 * To allow the "interpolate" method to fetch the cache state and place it in a results array, we set up the input
	 * fields:
	 *  - the level-set function values at G^n from the cache, and
	 *  - the neural flag (1 if computation was improved with neural network, 0 otherwhise)
	 * in that order.  This amounts to 2 input fields regardless of spatial dimensions.
	 * Upon response, the results array contains also 2 fields:
	 *  - the level-set value computed with the neural network or a meaningless value if query point doesn't match a
	 *    grid point at G^n for which we used the neural network, and
	 *  - a flag value: >= 1 if level-set value is valid (rank + 1), 0 if not.
	 */
	class Cache : public my_p4est_interpolation_t
	{
	private:
		p4est_nodes_t const *_nodes;	// Pointer to nodes struct, needed to access a quad's child vertices.

	public:
		using my_p4est_interpolation_t::interpolate;

		/**
		 * Input fields order or indices, as they should be given when setting up the inputs of the CacheFetcher object.
		 */
		enum Fields : int {PHI = 0, FLAG};
		static const int N_FIELDS;

		/**
		 * Constructor.
		 * @param [in] ngbd Pointer to node neighborhood struct at time t^n (i.e., G^n).
		 */
		explicit Cache( const my_p4est_node_neighbors_t *ngbd );

		/**
		 * Get ready by initializing the infrastructure containing the information to retrieve.
		 * @param [in] fields Input fields.
		 * @param [in] nFields Number of fields.
		 */
		void setInput( Vec fields[], const int& nFields );

		/**
		 * Fetch cached data for a query grid point (at G^np1 -- the grid a time t^np1).
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



	////////////////////////////////////////////////// SemiLagrangian //////////////////////////////////////////////////

	/**
 	 * A semi-Lagrangian implementation using Machine Learning and Neural Networks.
 	 */
	class SemiLagrangian : public my_p4est_semi_lagrangian_t
	{
	private:
		double H = 0;				// Smallest cell width.
		const bool USE_ANGLE_CONSTRAINT;	// Whether or not to collect samples based on their angle between midpoint vel_a and phi-signed normal.
		Vec _mlFlag = nullptr;		// A flag vector that stores rank+1 for nodes next to Gamma^n for which
									// we have computed phi with the nnet, 0 otherwise.  It also distinguishes vertices
									// with valid phi stencils (i.e., uniform in each direction).
		Vec _mlPhi = nullptr;		// Parallel vector to store advected phi for time t^np1 computed with nnet.

		std::unordered_set<p4est_locidx_t> _localUniformIndices;	// Set of indices of local nodes immediately next to
																	// Gamma^n with uniform stencils.

		const interpolation_method VEL_INTERP_MTHD;			// Default interpolation methods
		const interpolation_method PHI_INTERP_MTHD;			// for vel and level-set values.

		enum HowUpdated : int {NUM = 0, NNET};				// States for determining how a grid point was updated.

		const unsigned long ITERATION;						// Iteration ID for current advection step.
		bool _used = false;									// Prevents reusing a semi-Lagrangian instance of this obj.

		const NeuralNetwork * const _nnet;					// Error-correcting neural network.

		/**
		 * Create the set of indices of nodes next to Gamma^n with uniform stencils.
		 * @param [in] ngbdN Pointer to neighborhood struct at time t^n.
		 * @param [in] phi Parallel vector of level-set values at time t^n.
		 * @return Address of newly created set of node indices.
		 */
		void _computeLocalUniformIndices( const my_p4est_node_neighbors_t *ngbdN, Vec phi );

		/**
		 * Compute semi-Lagrangian advection for all points and correct level-set values at time t^np1 for grid points
		 * next to the interface at time t^n by using the neural network.
		 * @note Function resets and populates _mlFlag and _mlPhi for points where we used the nnet.
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction at time t^n.
		 * @param [in] normal Array of normal vectors in each Cartesian direction at time t^n.
		 * @param [in] vel_xx Array of spatial second derivatives for velocity components at time t^n.
		 * @param [in] dt Time step.
		 * @param [in] phi Parallel vector with level-set values for grid at time t^n.
		 * @param [in] phi_xx Level-set function spatial second derivatives at time t^n.
		 * @param [in] hk Dimensionless curvature parallel vector as computed for ALL points at time t^n.
		 */
		void _computeMLSolution( Vec vel[P4EST_DIM], Vec normal[P4EST_DIM], Vec *vel_xx[P4EST_DIM], const double& dt,
								 Vec phi, Vec phi_xx[P4EST_DIM], Vec hk );

		/**
		 * Advect level-set function using a semi-Lagrangian scheme with Euler steps along the characteristics.
		 * @param [in] dt Time step.
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction at time t^n.
		 * @param [in] vel_xx Array of second derivatives for each velocity component w.r.t. each Cartesian direction at time t^n.
		 * @param [in] phi Level-set function values at time t^n.
		 * @param [in] phi_xx Level-set function spatial second derivatives at time t^n.
		 * @param [in,out] phi_np1Ptr Advected level-set function values.
		 * @param [in,out] phiNum_np1Ptr Numerically advected level-set values (i.e., before loading nnet-corrected values).
		 * @param [in,out] howUpdated_np1Ptr Debugging values to indicate how the level-set was updated.
		 */
		void _advectFromNToNp1( const double& dt, Vec vel[P4EST_DIM], Vec *vel_xx[P4EST_DIM], Vec phi,
						  		Vec phi_xx[P4EST_DIM], double *phi_np1Ptr, double *phiNum_np1Ptr, double *howUpdated_np1Ptr );

	public:
		constexpr static double FLOW_ANGLE_THRESHOLD = 19 * M_PI / 36;	// Maximum angle between some velocity and the
																		// phi-signed normal for a point next to Gamma.

		/**
		 * Constructor.
		 * @param [in,out] p4estNp1 Pointer to a p4est object pointer.
		 * @param [in,out] nodesNp1 Pointer to a nodes' object pointer.
		 * @param [in,out] ghostNp1 Pointer to a ghost struct pointer.
		 * @param [in,out] ngbdN  Pointer to a neighborhood struct.
		 * @param [in] phi Level-set values to construct the shell of h-uniform-stencil grid points around Gamma_c^n.
		 * @param [in] useAngleConstraint Whether or not use angle constraint between midpoint vel_a and phi-signed normal.
		 * @param [in] nnet Pointer to neural network, which should be created externally to avoid recurrent spawning.
		 * @param [in] iteration Current ID for advection step.
		 */
		SemiLagrangian( p4est_t **p4estNp1, p4est_nodes_t **nodesNp1, p4est_ghost_t **ghostNp1,
						my_p4est_node_neighbors_t *ngbdN, Vec phi, const bool& useAngleConstraint=false,
						const NeuralNetwork *nnet=nullptr,
						const unsigned long& iteration=0 );

		/**
		 * Collect samples for neural network training/inference.  Use a semi-Lagrangian scheme with 2nd-order accuracy
		 * along the characteristics to define the departure points.  Collect samples for grid points next to Gamma^n
		 * with nonzero mid-point velocity, uniform h-stencils, and (optionally) an angle between phi-signed normal and
		 * midpoint vel_a in the range of [0, FLOW_ANGLE_THRESHOLD].
		 * @note Here, we create data packets dynamically, and you must not forget free those objects by calling the
		 * utility function freeDataPacketArray.
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction at time t^n.
		 * @param [in] vel_xx Array of spatial second derivatives for velocity components at time t^n.
		 * @param [in] dt Time step.
		 * @param [in] phi Level-set function values at time t^n.
		 * @param [in] normal Normal vectors at time t^n.
		 * @param [in] phi_xx Level-set function spatial second derivatives at time t^n.
		 * @param [out] dataPackets Vector of pointers to data packet objects.
		 * @return true if all backtracked queried points from nodes along interface lie inside within domain, false
		 * otherwise.  This serves as a warning flag.
		 */
		bool collectSamples( Vec vel[P4EST_DIM], Vec *vel_xx[P4EST_DIM], const double& dt, Vec phi, Vec normal[P4EST_DIM],
							 Vec phi_xx[P4EST_DIM], std::vector<DataPacket *>& dataPackets ) const;

		/**
		 * Utility function to deallocate the dynamically created data packets in collectSamples.
		 * @param [in,out] dataPackets Vector of pointers to dynamically allocated data packet objects.
		 * @return Number of freed objects.
		 */
		static size_t freeDataPacketArray( std::vector<DataPacket *>& dataPackets );

		/**
		 * Update a p4est from t^n to t^np1, using a combined semi-Lagrangian scheme: numerical and machine-learning
		 * based with Euler steps along the characteristics.
		 * The forest at time t^n is copied and then refined/coarsened and balanced iteratively until convergence.
		 * The method is adapted from:
		 * [*] M. Mirzadeh, A. Guittet, C. Burstedde, and F. Gibou, Parallel Level-Set Method on Adaptive Tree-Based Grids.
		 * @note You need to update the node neighborhood and hierarchy objects yourself upon exit!
		 * @param [in] vel Array of velocity parallel vectors in each Cartesian direction.
		 * @param [in] dt Time step.
		 * @param [in,out] phi Level-set function values at time t^n, and then updated at time t^np1.
		 * @param [in] hk Dimensionless curvature for ALL points at time t^n.
		 * @param [in] normal Normal vectors at time t^n.
		 * @param [out] howUpdated Optional parallel vector to know how the level-set values were updated: 0 if
		 * 		  numerically, 1 if using neural net.  Useful for selective reinitialization.
		 * @param [out] withTheFlow Optional parallel vector signaling (with 1s) those nodes in a narrow band (one min
		 * 		  diag) around Gamma_c^np1 whose angle between the phi^np1-signed normal^n and velocity^n is in the
		 * 		  range of [0, threshold].
		 * @param [in] flipFlow Optional constant to change the sign of the flow: 1 in its direction, -1 in the opposite direction.
		 * @param [out] phiNum Optional parallel debugging vector containing ONLY numerically advected level-set
		 * 		  values, i.e., before loading the ml-corrected trajectory.
		 */
		void updateP4EST( Vec vel[P4EST_DIM], const double& dt, Vec *phi, Vec hk, Vec normal[P4EST_DIM],
						  Vec *howUpdated=nullptr, Vec *withTheFlow=nullptr, const short int& flipFlow=1, Vec *phiNum=nullptr );

		/**
		 * Set the bit for nodes that go in the direction of the flow.  These nodes are selected if their angle between
		 * the phi^np1-signed normal^n vector and the velocity^n is in the range of [0, threshold].  To compute this
		 * angle, we interpolate data from t^n into nodes within a narrow band (one min-diag) around Gamma^np1.
		 * @param [in] vel_n Velocity at time t^n.
		 * @param [in] normal_n Normal vectors at time t^n.
		 * @param [in] neighbors_n Neighborhood struct at time t^n (for which p4est_n and nodes_n are well defined).
		 * @param [in] h Minimum cell width.
		 * @param [in] phi_np1 Updated nodal level-set values after advection.
		 * @param [in] p4est_np1 Updated p4est structure.
		 * @param [in] nodes_np1 Updated nodal structure.
		 * @param [out] withTheFlow Parallel vector with 1s for nodes to going with the flow direction at time t^np1,
		 * 		  and 0s otherwise.
		 * @param [in] flipFlow Optional constant to change the sign of the flow: 1 in its direction, -1 in the opposite direction.
		 */
		static void getNodesWithTheFlow( Vec vel_n[P4EST_DIM], Vec normal_n[P4EST_DIM],
										 const my_p4est_node_neighbors_t *neighbors_n, const double& h, Vec phi_np1,
										 const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1,
										 Vec *withTheFlow, const short int& flipFlow=1 );
	};
}


#endif //MY_P4EST_SEMI_LAGRANGIAN_ML_H
