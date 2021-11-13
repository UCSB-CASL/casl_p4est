#ifndef MY_P4EST_CURVATURE_ML_H
#define MY_P4EST_CURVATURE_ML_H

#ifdef P4_TO_P8
#include <src/my_p8est_nodes_along_interface.h>
#else
#define K_INPUT_SIZE			28	// Includes h-normalized phi values, unit normal vector components, and numerical hk.
#define K_INPUT_PHI_SIZE		 9
#define K_INPUT_NORMAL_SIZE		18
#define K_INPUT_HK_SIZE			 1

#include <src/my_p4est_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_nodes_along_interface.h>
#endif

#include <fdeep/fdeep.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <cblas.h>

/**
 * Machine-learning-based curvature namespace.
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
 * Created: November 11, 2021.
 * Updated: November 12, 2021.
 */
namespace kml
{
	////////////////////////////////////////////////////// Scaler //////////////////////////////////////////////////////

	/**
	 * Abstract class to transform data into an input form that the neural network understands.
	 * Input data in 2D comes in the following order (with K_INPUT_SIZE entries) (x is slowest changing and then y):
	 *		"mm", "m0", "mp"           =>  phi(i-1, j-1), phi(i-1, j), phi(i-1, j+1) |
	 *		"0m", "00", "0p"           =>  phi(  i, j-1), phi(  i, j), phi(  i, j+1) |  First 9 entries are level-set values.
	 *		"pm", "p0", "pp"           =>  phi(i+1, j-1), phi(i+1, j), phi(i+1, j+1) |
	 *		"nx_mm", "nx_m0", "nx_mp"  =>   nx(i-1, j-1),  nx(i-1, j),  nx(i-1, j+1) +
	 *		"nx_0m", "nx_00", "nx_0p"  =>   nx(  i, j-1),  nx(  i, j),  nx(  i, j+1) +  Second 9 entries are x-components of normal unit vectors.
	 *		"nx_pm", "nx_p0", "nx_pp"  =>   nx(i+1, j-1),  nx(i+1, j),  nx(i+1, j+1) +
	 *		"ny_mm", "ny_m0", "ny_mp"  =>   ny(i-1, j-1),  ny(i-1, j),  ny(i-1, j+1) -
	 *		"ny_0m", "ny_00", "ny_0p"  =>   ny(  i, j-1),  ny(  i, j),  ny(  i, j+1) -  Third 9 entries are y-components of normal unit vectors.
	 *		"ny_pm", "ny_p0", "ny_pp"  =>   ny(i+1, j-1),  ny(i+1, j),  ny(i+1, j+1) -
	 *		"ihk" =>  Interpolated h * kappa
	 */
	class Scaler
	{
		using json = nlohmann::json;

	protected:
		static void _printParams( const json& params, const std::string& paramsFileName );

	public:
		const int                            PHI_COLS[K_INPUT_PHI_SIZE   ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8};	// Phi column indices.
		__attribute__((unused)) const int NORMAL_COLS[K_INPUT_NORMAL_SIZE] = { 9,10,11,12,13,14,15,16,17,	// Normal components: x first,
																			  18,19,20,21,22,23,24,25,26};	// then y.
		__attribute__((unused)) const int     HK_COLS[K_INPUT_HK_SIZE    ] = {27};							// Numerical hk column index.

		/**
		 * Transform input data in place.
		 * @param [in,out] samples Data to transform.
		 * @param [in] nSamples Number of samples.
		 */
		virtual void transform( double samples[][K_INPUT_SIZE], const int& nSamples ) const = 0;
	};

	//////////////////////////////////////////////////// PCAScaler /////////////////////////////////////////////////////

	/**
	 * Transform data into an input form that the neural network understands.
	 * This is a transformer that applies principal component analysis [and optional whitening] to the input.
	 * Depending on the numer of components, it also performs dimensionality reduction.
	 * See the Hybrid_DL_Curvature Python project to understand how we applied the transformation during the learning
	 * stage.
	 */
	class PCAScaler : public Scaler
	{
		using json = nlohmann::json;

	private:
		unsigned long _nComponents = -1;				// Number of components (-1 uninitialized).
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
		 * @param [in,out] samples Data to transform. Upon transformation, only the first _nComponents elements are valid.
		 * @param [in] nSamples Number of samples.
		 */
		void transform( double samples[][K_INPUT_SIZE], const int& nSamples ) const override;

		/**
		 * Retrieve the number of components.
		 * @return
		 */
		unsigned long getNComponents() const;
	};

	////////////////////////////////////////////////// StandardScaler //////////////////////////////////////////////////

	/**
	 * Transform data into an input form that the neural network understands.
	 * Stores information about the StandardScaler object in sklearn.preprocessing module.
	 * See the Hybrid_DL_Curvature Python project to see what's exported: a list of mean values, and a list of std values.
	 * Statistics are column-wise (not type-wide as in slml).
	 */
	class StandardScaler : public Scaler
	{
		using json = nlohmann::json;

	private:
		double _mean[K_INPUT_SIZE] = {0};		// Mean feature-wise values.
		double _std[K_INPUT_SIZE] = {0};		// Standard deviation feature-wise values.

		/**
		 * Utility function to load a group of parameters.
		 * @param [in] inName Parameter key name as given in JSON file.
		 * @param [in] params JSON object.
		 * @param [out] outFeature Where to store the values.
		 * @throws runtime error if param key is not found in JSON object.
		 */
		static void _loadParams( const std::string& inName, const json& params, double outFeature[K_INPUT_SIZE] );

	public:
		/**
		 * Constructor.
		 * @param [in] paramsFileName JSON file name with standard scaler parameters.
		 * @param [in] printLoadedParams Whether to print loaded parameters.
		 */
		explicit StandardScaler( const std::string& paramsFileName, const bool& printLoadedParams=true );

		/**
		 * Transform input data in place, provided as a matrix of size (nSamples)-by-(K_INPUT_SIZE).
		 * @param [in,out] samples Data to transform.
		 * @param [in] nSamples Number of samples.
		 */
		void transform( double samples[][K_INPUT_SIZE], const int& nSamples ) const override;
	};


	////////////////////////////////////////////////// NeuralNetwork ///////////////////////////////////////////////////

	/**
	 * Curvature error-correcting neural network.
	 * Internally, processes batches of samples in the following format:
	 * 										Samples (n)
	 * 						s_0  s_1  ...  s_i  ... s_{n-2}  s_{n-1}
	 * 				f_0	  |  #	  #	  ...	#   ...	   #		#	 |
	 * 				f_1	  |  #	  #	  ...	#   ...	   #		#	 |
	 * 				 :	  |	 :	  :	   ·	:	 ·	   :		:	 |
	 * Features (k)	f_j	  |	 #	  #	  ...	#   ...	   #		#	 |
	 * 				 :	  |	 :	  :	   ·	:	 ·	   :		:	 |
	 * 			  f_{k-2} |  #	  #	  ...	#   ...	   #		#	 |
	 * 			  f_{k-1} |  1    1	  ...	1   ...	   1		1	 |   <--- This is the (constant) bias input.
	 *
	 * Features includes one row of ones to account for the bias.  This means that the very input batch to the nnet has
	 * one additional row to produce the outputs of the first hidden layer.
	 */
	class NeuralNetwork
	{
		using json = nlohmann::json;

	private:
		const PCAScaler _pcaScaler;				// Preprocessing module including type-based standard scaler followed by
		const StandardScaler _stdScaler;		// PCA dimensionality reduction (and whitening).

		unsigned long N_LAYERS;					// Number of layers (hidden + output).
		std::vector<std::vector<int>> _sizes;	// Matrix size tuples (m, k).  m = layer size, k = input size, (n = number of samples).
		std::vector<std::vector<FDEEP_FLOAT_TYPE>> W;	// Weight matrices flattened.  Matrices are given in row-major order.

		const double H;							// Mesh size.
		unsigned long _inputSize;				// Expected input size (excludes bias).

		/**
		 * ReLU activation function.
		 * @param [in] x Input.
		 * @return ReLu(x) = max(0, x).
		 */
		static FDEEP_FLOAT_TYPE _reLU( const FDEEP_FLOAT_TYPE& x );

	public:
		/**
		 * Constructor.
		 * @param [in] folder Full path to folder that holds the neural network (k_nnet.json), pca (k_pca_scaler.json),
		 * 			   and standard scaler (k_std_scaler.json) JSON files.
		 * @param [in] h Mesh size.
		 * @param [in] verbose Whether to print debugging information or not.
		 */
		explicit NeuralNetwork( const std::string& folder, const double& h, const bool& verbose=true );

		/**
		 * Predict corrected dimensionless curvature at the closest point on the interface.
		 * @note This function assumes that the inputs have been already negated when curvature is positive.  User must
		 * take care of fixing the predictions' sign accordingly.
		 * @param [in,out] inputs Array of sample inputs with raw data (they'll be transformed).
		 * @param [out] outputs Array of predicted dimensionless curvature values
		 * @param [in] nSamples Batch size.
		 * @param [in] hNormalize Whether to normalize phi values by mesh size h.
		 */
		void predict( double inputs[][K_INPUT_SIZE], double outputs[], const int& nSamples, const bool& hNormalize=true ) const;

		/**
		 * Retrieve mesh size.
		 * @return Mesh size.
		 */
		double getH() const;
	};


	//////////////////////////////////////////////////// Utilities /////////////////////////////////////////////////////

	namespace utils
	{
		/**
		 * Generate the column headers following the truth-table order with x changing slowly, then y changing faster than x,
		 * and finally z changing faster than y.  Each dimension has three states: m, 0, and p (minus, center, plus).  For
		 * example, in 2D, the columns that are generated are:
		 *		"mm", "m0", "mp"           =>  phi(i-1, j-1), phi(i-1, j), phi(i-1, j+1) |
		 *		"0m", "00", "0p"           =>  phi(  i, j-1), phi(  i, j), phi(  i, j+1) |  First 9 entries are level-set values.
		 *		"pm", "p0", "pp"           =>  phi(i+1, j-1), phi(i+1, j), phi(i+1, j+1) |
		 *		"nx_mm", "nx_m0", "nx_mp"  =>   nx(i-1, j-1),  nx(i-1, j),  nx(i-1, j+1) +
		 *		"nx_0m", "nx_00", "nx_0p"  =>   nx(  i, j-1),  nx(  i, j),  nx(  i, j+1) +  Second 9 entries are x-components of normal unit vectors.
		 *		"nx_pm", "nx_p0", "nx_pp"  =>   nx(i+1, j-1),  nx(i+1, j),  nx(i+1, j+1) +
		 *		"ny_mm", "ny_m0", "ny_mp"  =>   ny(i-1, j-1),  ny(i-1, j),  ny(i-1, j+1) -
		 *		"ny_0m", "ny_00", "ny_0p"  =>   ny(  i, j-1),  ny(  i, j),  ny(  i, j+1) -  Third 9 entries are y-components of normal unit vectors.
		 *		"ny_pm", "ny_p0", "ny_pp"  =>   ny(i+1, j-1),  ny(i+1, j),  ny(i+1, j+1) -
		 *		"hk"  =>  Exact target h * kappa (optional)
		 *		"ihk" =>  Interpolated h * kappa
 		 * @param [out] header Array of column headers to be filled up.  Must be backed by a correctly allocated array.
		 * @param [in] includeTargetHK Whether to include or not the "hk" column.
		 */
		void generateColumnHeaders( std::string header[], const bool& includeTargetHK=true );

		/**
		 * Rotate stencil of level-set function values in a sample vector by 90 degrees counter or clockwise.
		 * @param [in,out] stencil Array of level-set function values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
		 * @param [in] dir Rotation direction: > 0 for counterclockwise, <= 0 for clockwise.
		 */
		void rotateStencil90( double stencil[], const int& dir=1 );

		/**
		 * Rotate stencil of level-set function values in a sample vector by 90 degrees counter or clockwise.
		 * @param [in,out] stencil Vector of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
		 * @param [in] dir Rotation direction: > 0 for counterclockwise, <= 0 for clockwise.
		 */
		inline void rotateStencil90( std::vector<double>& stencil, const int& dir=1 )
		{
			rotateStencil90( stencil.data(), dir );
		}

		/**
		 * Reflect stencil of level-set values along line y = x.
		 * @note Useful for data augmentation assuming that we are using normalization to first quadrant of a local
		 * coordinate system whose origin is at the center of the stencil.  Exploits fact that curvature is invariant to
		 * reflections and rotations.
		 * @param [in,out] stencil Array of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
		 */
		void reflectStencil_yEqx( double stencil[] );

		/**
		 * Reflect stencil of level-set values along line y = x.
		 * @note Useful for data augmentation assuming that we are using normalization to first quadrant of a local
		 * coordinate system whose origin is at the center of the stencil.  Exploits fact that curvature is invariant to
		 * reflections and rotations.
		 * @param [in,out] stencil Vector of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
		 */
		inline void reflectStencil_yEqx( std::vector<double>& stencil )
		{
			reflectStencil_yEqx( stencil.data() );
		}

		/**
		 * Rotate stencil in such a way that the gradient computed at center node 00 has an with respect to the horizontal
		 * in the range of [0, pi/2].
		 * @note Exploits the fact that curvature is invariant to rotation.  Prior to calling this function you must have
		 * flipped the sign of the stencil (and gradient) so that the curvature is negative.
		 * @param [in,out] stencil Array of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
		 * @param [in] gradient Gradient at the center node.
		 */
		void rotateStencilToFirstQuadrant( double stencil[], const double gradient[P4EST_DIM] );

		/**
		 * Rotate stencil in such a way that the gradient computed at center node 00 has an angle with respect to the
		 * horizontal in the range of [0, pi/2].
		 * @param [in,out] stencil Vector of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
		 * @param [in] gradient Gradient at the center node.
		 */
		inline void rotateStencilToFirstQuadrant( std::vector<double>& stencil, const double gradient[P4EST_DIM] )
		{
			rotateStencilToFirstQuadrant( stencil.data(), gradient );
		}
	}


	////////////////////////////////////////////////// Curvature //////////////////////////////////////////////////

	/**
 	 * Curvature computation using machine learning and neural networks.
 	 */
	class Curvature
	{
	private:
		const double H;						// Smallest (square-)cell width.
		const double LO_MIN_HK;				// Lower- and upper-bound for minimum |hk| where we blend numerical with
		const double UP_MIN_HK;				// neural estimation for better results.
		const NeuralNetwork * const _nnet;	// Error-correcting neural network.

		/**
		 * Collect samples for locally owned nodes with full h-uniform stencil next to Gamma.  Samples include phi
		 * values and normal unit vector components plus the linearly interpolated dimensionless curvature at the
		 * interface.  Note that no negative-curvature normalization and reorientation are performed here.  Those will
		 * be considered as a preprocessing step in the function that invokes the neural inference.
		 * @param [in] ngbd Node neighborhood struct.
		 * @param [in] phi Reinitialized level-set values.
		 * @param [in] normal Nodal normal components.
		 * @param [in] numCurvature Numerical curvature (which we use for linear interpolation at Gamma).
		 * @param [out] samples Vector of samples for valid nodes next to Gamma.
		 * @param [out] indices Center nodal indices for collected samples (a one-to-one mapping).
		 */
		void _collectSamples( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM], Vec numCurvature,
							  std::vector<std::vector<double>>& samples, std::vector<p4est_locidx_t>& indices ) const;

		/**
		 * Compute the hybrid dimensionless curvature from the samples provided by using the neural network and the
		 * numerically interpolated dimensionless curvature at the interface.
		 * @param [in] samples Vector of samples for locally owned valid nodes next to Gamma.
		 * @param [out] hybHK Output dimensionless curvature computed with hybrid approach.
		 */
		void _computeHybridHK( const std::vector<std::vector<double>>& samples, std::vector<double>& hybHK ) const;

	public:
		/**
		 * Constructor.
		 * @note The loMinHK constant must be at least the MIN_HK used for training.
		 * @param [in] nnet Pointer to neural network, which should be created externally to avoid recurrent spawning.
		 * @param [in] h Mesh size.
		 * @param [in] loMinHK Strictly positive lower-bound for dimensionless curvature (e.g. 0.004) to use the nnet.
		 * @param [in] upMinHK Strictly positive upper-bound for blending nnet-computed dimensionless curvature with
		 * 		  numerical estimation (e.g. 0.007).
		 */
		Curvature( const NeuralNetwork *nnet, const double& h, const double& loMinHK=0.004, const double& upMinHK=0.007 );

		/**
		 * Compute curvature.  There are two output modes in this function.  First, it computes the numerical curvature
		 * at the nodes as usual and place the results in the numCurvature vector.  Then, it computes the curvature for
		 * grid points next to the interface using the hybrid approach.  The resulting approximation is placed in the
		 * hybCurvature vector and corresponds to the curvature *at* the normal projection of those nodes onto Gamma.
		 * The ancillary output hybFlag vector is populated with 1s where we used the hybrid approach and 0s everywhere
		 * else.
		 * @param [in] ngbd Node neighborhood structure.
		 * @param [in] phi Nodal level-set values (assuming we have already reinitialized them).
		 * @param [in] normal Nodal normal vector components.
		 * @param [out] numCurvature Numerical curvature computed at the nodes using the conventional approach.
		 * @param [out] hybCurvature Hybrid curvature computed at the normal interface-projection of nodes next to Gamma.
		 * @param [out] hybFlag Indicator vector with 1s where we used the hybrid approach and 0s everywhere else.
		 * @param [in] dimensionless Whether to scale curvature by h.
		 * @throws Runtime exception if any vector is nullptr.
		 */
		void compute( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM], Vec numCurvature,
					  Vec hybCurvature, Vec hybFlag, const bool& dimensionless=false ) const;
	};
}


#endif //MY_P4EST_CURVATURE_ML_H
