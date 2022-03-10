#ifndef MY_P4EST_CURVATURE_ML_H
#define MY_P4EST_CURVATURE_ML_H

#ifdef P4_TO_P8
#define K_INPUT_SIZE			110	// Includes h-normalized phi values, unit normal vectors, and numerical mean ihk and Gaussian ih^2k.
#define K_INPUT_SIZE_LEARN		112	// Includes two additional slots for true curvatures.
#define K_INPUT_PHI_SIZE		 27
#define K_INPUT_NORMAL_SIZE		 81
#define K_INPUT_HK_SIZE			  1	// Dimensionless mean curvature = h * H.
#define K_INPUT_H2KG_SIZE		  1	// Dimensionless Gaussian curvature = h^2 * K.

#include <src/my_p8est_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_nodes_along_interface.h>
#else
#define K_INPUT_SIZE			28	// Includes h-normalized phi values, unit normal vector components, and numerical ihk.
#define K_INPUT_SIZE_LEARN		29	// Includes an additional slot for true ihk.
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
#include <algorithm>
#include <random>

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
 * Updated: March 9, 2022.
 */
namespace kml
{
	////////////////////////////////////////////////////// Scaler //////////////////////////////////////////////////////

	/**
	 * Abstract class to transform data into an input form that the neural network understands.
	 * Input data in 2D comes in the following order (with K_INPUT_SIZE entries) (x is slowest changing and then y):
	 *		mm, m0, mp           =>  phi(i-1, j-1), phi(i-1, j), phi(i-1, j+1) |
	 *		0m, 00, 0p           =>  phi(  i, j-1), phi(  i, j), phi(  i, j+1) |  First 9 entries are level-set values.
	 *		pm, p0, pp           =>  phi(i+1, j-1), phi(i+1, j), phi(i+1, j+1) |
	 *		nx_mm, nx_m0, nx_mp  =>   nx(i-1, j-1),  nx(i-1, j),  nx(i-1, j+1) +
	 *		nx_0m, nx_00, nx_0p  =>   nx(  i, j-1),  nx(  i, j),  nx(  i, j+1) +  Second 9 entries are x-components of normal unit vectors.
	 *		nx_pm, nx_p0, nx_pp  =>   nx(i+1, j-1),  nx(i+1, j),  nx(i+1, j+1) +
	 *		ny_mm, ny_m0, ny_mp  =>   ny(i-1, j-1),  ny(i-1, j),  ny(i-1, j+1) -
	 *		ny_0m, ny_00, ny_0p  =>   ny(  i, j-1),  ny(  i, j),  ny(  i, j+1) -  Third 9 entries are y-components of normal unit vectors.
	 *		ny_pm, ny_p0, ny_pp  =>   ny(i+1, j-1),  ny(i+1, j),  ny(i+1, j+1) -
	 *		ihk                  =>  Interpolated h * kappa
	 *
	 * Input data in 3D comes in the following order (with K_INPUT_SIZE entries (x slowest, then y, and z is the fastest changing variable):
	 *      mmm, mm0, mmp, m0m, m00, m0p, mpm, mp0, mpp  =>  Face for x = m
	 *      0mm, 0m0, 0mp, 00m, 000, 00p, 0pm, 0p0, 0pp  =>  Face for x = 0
	 *      pmm, pm0, pmp, p0m, p00, p0p, ppm, pp0, ppp  =>  Face for x = p
	 * We have three such groups: phi, nx, ny, and nz, in that order.  At the end, we have ihk, the numerical dimension-
	 * less curvature bilinearly interpolated at the interface.
	 */
	class Scaler
	{
		using json = nlohmann::json;

	protected:
		static void _printParams( const json& params, const std::string& paramsFileName );

	public:
#ifndef P4_TO_P8
		static const int                         PHI_COLS   [K_INPUT_PHI_SIZE   ];	// Phi column indices.
		__attribute__((unused)) static const int NORMAL_COLS[K_INPUT_NORMAL_SIZE];	// Normal components: x first, then y.
		__attribute__((unused)) static const int     HK_COLS[K_INPUT_HK_SIZE    ];	// Numerical hk column index.
#else
		static const int PHI_COLS[K_INPUT_PHI_SIZE];
#endif

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
	 * Mean curvature error-correcting neural network.
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

		const double _h;						// Mesh size.
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
		 * Predict corrected dimensionless mean curvature at the closest point on the interface.
		 * @note This function assumes that the inputs have been already negated when mean curvature is positive.
		 * User must take care of fixing the predictions' sign accordingly.
		 * @param [in,out] inputs Array of sample inputs with raw data (they'll be transformed).
		 * @param [out] outputs Array of predicted dimensionless mean curvature values
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
		 * In 3D, we have four groups:
		 * 		phi values 			- first 27 strings:              "mmm",    "mm0",    "mmp", ... ,   "ppm",    "pp0",    "ppp".
		 * 		normal x components	- second group of 27 strings: "nx_mmm", "nx_mm0", "nx_mmp", ..., "nx_ppm", "nx_pp0", "nx_ppp".
		 * 		normal y components - third group of 27 strings:  "ny_mmm", "ny_mm0", "ny_mmp", ..., "ny_ppm", "ny_pp0", "ny_ppp".
		 * 		normal z components - fourth group of 27 strings: "nz_mmm", "nz_mm0", "nz_mmp", ..., "nz_ppm", "nz_pp0", "nz_ppp".
		 * At the end, we append true mean "hk" (optional), numerical "ihk", true Gaussian "h2kg" (optional), and numerical "ih2kg".
 		 * @param [out] header Array of column headers to be filled up.  Must be backed by a correctly allocated array.
		 * @param [in] includeTrueCurvatures Whether to include or not the (scaled) true curvature columns.
		 */
		void generateColumnHeaders( std::string header[], const bool& includeTrueCurvatures=true );

		/**
		 * Rotate stencil in a sample vector by 90 degrees about the z axis.
		 * @param [in,out] stencil Sampled data in standard order (e.g., mm[m], m0[m], mp[m],..., [p]pm, [p]p0, [p]pp).
		 * @param [in] dir Rotation direction: > 0 for counterclockwise, <= 0 for clockwise.
		 */
		void rotateStencil90z( double stencil[], const int& dir=1 );

#ifdef P4_TO_P8
		/**
		 * Rotate stencil in a sample vector by 90 degrees about the y axis.
		 * @note In y-rotations, the angle is measured from the +z axis.
		 * @param [in,out] stencil Sampled data in standard order (e.g., mm[m], m0[m], mp[m],..., [p]pm, [p]p0, [p]pp).
		 * @param [in] dir Rotation direction: > 0 for counterclockwise, <= 0 for clockwise.
		 */
		void rotateStencil90y( double stencil[], const int& dir=1 );
#endif

		/**
		 * Reflect feature stencil about y = x (or, in 3D, the plane x - y = 0, whose normal is [1, -1, 0]).
		 * @note Useful for data augmentation assuming that we are using reorientation to first quadrant (octant) of the
		 * local coordinate system with origin at the center of its stencil.  Exploits curvature reflection invariance.
		 * @param [in,out] stencil Feature array in standard order (e.g., mm[m], m0[m], mp[m],..., [p]pm, [p]p0, [p]pp).
		 */
		void reflectStencil_yEqx( double stencil[] );
		inline void reflectStencil_yEqx( std::vector<double>& stencil )
		{
			reflectStencil_yEqx( stencil.data() );
		}

		/**
		 * Rotate stencil in such a way that the gradient computed at the stencil's center node has all its Cartesian
		 * components positive.  The function doesn't modify any of the invariant quantities, such as the curvature(s).
		 * @note Exploits the fact that curvature is invariant to rotation.  Before calling this function you must have
		 * flipped the sign of the stencil (and gradient) to the desired configuration (possibly negatively normalized).
		 * @param [in,out] stencil Feature array in standard order (e.g., mm[m], m0[m], mp[m],..., [p]pm, [p]p0, [p]pp).
		 * @throws runtime_error If one or more reoriented gradient component components is negative after the process.
		 */
#ifdef P4_TO_P8
		void rotateStencilToFirstOctant( double stencil[] );
		inline void rotateStencilToFirstOctant( std::vector<double>& stencil )
		{
			rotateStencilToFirstOctant( stencil.data() );
		}
#else
		void rotateStencilToFirstQuadrant( double stencil[] );
		inline void rotateStencilToFirstQuadrant( std::vector<double>& stencil )
		{
			rotateStencilToFirstQuadrant( stencil.data() );
		}
#endif

		/**
		 * Normalize a stencil to negative (mean) curvature spectrum.
		 * @note This function doesn't modify (true and numerically interpolated) Gaussian curvature(s).  The learning
		 * flag helps to account for the additional true columns present only for training data sets.
		 * @param [in,out] stencil Feature vector with phi, normal, and (dimensionless) curvature(s).
		 * @param [in] refHK Reference dimensionless curvature to determine normalization.
		 * @param [in] learning Whether we are treating samples for learning or on-line inference.
		 */
		void normalizeToNegativeCurvature( std::vector<double>& stencil, const double& refHK, const bool& learning=false );

		/**
		 * Prepare sampling file by opening, writing the header, and setting its precision to 32-bit floating-point
		 * numbers.
		 * @param [in] mpi MPI environment.
		 * @param [in] directory Where to place samples' file.  If it doesn't exist, it'll be created by rank 0 only.
		 * @param [in] fileName File name such that the full path is 'directory/fileName'.
		 * @param [in,out] file File object.
		 * @throws runtime_error if directory can't be accessed or file can't be opened.
		 */
		void prepareSamplesFile( const mpi_environment_t& mpi, const std::string& directory,
								 const std::string& fileName, std::ofstream& file );

		/**
		 * Save buffered samples to a file.
		 * @param [in] mpi MPI environment.
		 * @param [in,out] file File object.
		 * @param [in] buffer Samples buffer.
		 * @return Number of samples written to file.
		 */
		int saveSamplesBufferToFile( const mpi_environment_t& mpi, std::ofstream& file,
									 const std::vector<std::vector<FDEEP_FLOAT_TYPE>>& buffer );

		/**
		 * Transform samples with (optional) negative-mean-curvature and phi-by-h normalization, followed by
		 * reorientation and reflection.  Then, place these samples in a cumulative array.
		 * @note Only rank 0 accumulates processed samples, but all processes receive the total number of them.
		 * @param [in] mpi MPI environment.
		 * @param [in,out] samples List of feature vectors.
		 * @param [in,out] buffer Cumulative array of feature vectors.
		 * @param [in] h Mesh size for h-normalizing phi values.
		 * @param [in] negMeanKNormalize True if we need negative-mean-curvature normalization, false otherwise.
		 * @return Number of samples collected from all processes.
		 */
		int processSamplesAndAccumulate( const mpi_environment_t& mpi, std::vector<std::vector<double>>& samples,
										 std::vector<std::vector<FDEEP_FLOAT_TYPE>>& buffer, const double& h,
										 const bool& negMeanKNormalize );

		/**
		 * Transform samples with (optional) negative-mean-curvature and phi-by-h normalization, followed by reorienta-
		 * tion and augmentation based on reflection.  Then, write these samples to a file using single precision.
		 * @note Only rank 0 writes samples to a file, but all processes received the total number of saved samples.
		 * @param [in] mpi MPI environment.
		 * @param [in,out] samples List of feature vectors.
		 * @param [in,out] file File stream where to write data (should be opened already).
		 * @param [in] h Mesh size for h-normalizing phi values.
		 * @param [in] negMeanKNormalize True if we need negative-mean-curvature normalization, false otherwise.
		 * @param [in] preAllocateSize Estimate number of samples to preallocate intermediate buffer (only rank 0).
		 * @return Number of samples saved to the input file.
		 */
		int processSamplesAndSaveToFile( const mpi_environment_t& mpi, std::vector<std::vector<double>>& samples,
										 std::ofstream& file, const double& h, const bool& negMeanKNormalize,
										 const int& preAllocateSize=1000 );

		/**
		 * Perform histogram-based subsampling by first splitting the data set into nbins intervals based on true mean
		 * |hk*|.  Then, compute the median and subsample the intervals until the number of items in each bin is at most
		 * max(frac*median, minFold*minCount), where minFold >= 1 and minCount is the smallest number of samples in any
		 * bin.  After that, save the remaining samples into a file.
		 * @note Only rank 0 writes samples to a file, but all processes receive the total number of saved samples.
		 * @param [in] mpi MPI environment.
		 * @param [in] buffer Array of buffered (already normalized and augmented) samples.
		 * @param [in,out] file File object.
		 * @param [in] minHK Minimum mean |hk*| to consider for provided buffer; if not given, it'll be computed.
		 * @param [in] maxHK Maximum mean |hk*| to consider for provided buffer; if not given, it'll be computed.
		 * @param [in] nbins Number of bins or intervals in the histogram.
		 * @param [in] frac Fraction of the median to be used for subsampling.
		 * @param [in] minFold Number of times to consider the count of the bin with the least samples.
		 * @return Number of samples that made it to the input file after subsampling.
		 * @throws invalid_argument and runtime_error if minHK >= maxHK, or if computing the histogram fails, or if frac
		 * 		   is non-positive, or if minFold is less than 1, or if the number of bins is less than 10.
		 */
		int histSubSamplingAndSaveToFile( const mpi_environment_t& mpi,
										  const std::vector<std::vector<FDEEP_FLOAT_TYPE>>& buffer,
										  std::ofstream& file, FDEEP_FLOAT_TYPE minHK=NAN, FDEEP_FLOAT_TYPE maxHK=NAN,
										  const unsigned short& nbins=100, const FDEEP_FLOAT_TYPE& frac=1,
										  const FDEEP_FLOAT_TYPE& minFold=2 );

		/**
		 * Compute an easing-off probability value based on a sinusoidal distribution in the domain [-pi/2, +pi/2].
		 * To this end, we need the minimum probability (lowProb) corresponding to sin(-pi/2) and maximum probability
		 * (upProb) corresponding to sin(+pi/2).  Similarly, we need the lowVal and upVal values matched to -pi/2 and
		 * +pi/2, respectively.  This way, the function Pr(x) returns the probability between lowProb and upProb.
		 * @param [in] x The value for which we want to compute the easing-off probability.
		 * @param [in] lowVal Lower-bound value such that Pr(x <= lowVal) = lowProb.
		 * @param [in] lowProb Lower-bound probability.
		 * @param [in] upVal Upper-bound value such that Pr(x >= upVal) = upProb.
		 * @param [in] upProb Upper-bound probability.
		 * @return Pr(x).
		 */
		double easingOffProbability( double x, const double& lowVal=0, const double& lowProb=0, const double& upVal=1,
									 const double& upProb=1 );
	}


	////////////////////////////////////////////////// Curvature //////////////////////////////////////////////////

	/**
 	 * Mean curvature computation using machine learning and neural networks.
 	 */
	class Curvature
	{
	private:
		const double _h;					// Smallest (square-)cell width.
		const double LO_MIN_HK;				// Lower- and upper-bound for minimum mean |hk| where we blend numerical
		const double UP_MIN_HK;				// with neural estimation for better results.
		const NeuralNetwork * const _nnet;	// Error-correcting neural network.

		/**
		 * Collect samples for locally owned nodes with full h-uniform stencil next to Gamma.  Samples include phi
		 * values, normal unit vector components, plus the linearly interpolated dimensionless (mean) curvature and
		 * Gaussian curvature at the interface.
		 * @note No negative-curvature normalization and reorientation are performed here.  Those will be considered as
		 * a preprocessing step in the function that invokes the neural inference.
		 * @param [in] ngbd Node neighborhood struct.
		 * @param [in] phi Reinitialized level-set values.
		 * @param [in] normal Nodal unit normal vectors.
		 * @param [in] numMeanK Numerical mean curvature (which we use for linear interpolation at Gamma).
		 * @param [out] samples Vector of samples for valid nodes next to Gamma.
		 * @param [out] indices Center nodal indices for collected samples (a one-to-one mapping).
		 */
		void _collectSamples( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM], Vec numMeanK,
							  std::vector<std::vector<double>>& samples, std::vector<p4est_locidx_t>& indices ) const;

		/**
		 * Compute the hybrid dimensionless mean curvature from the provided samples by using the neural network and the
		 * numerically interpolated dimensionless curvature(s) at the interface.
		 * @param [in] samples Vector of samples for locally owned valid nodes next to Gamma.
		 * @param [out] hybMeanHK Output dimensionless mean curvature computed with hybrid approach.
		 */
		void _computeHybridHK( const std::vector<std::vector<double>>& samples, std::vector<double>& hybMeanHK ) const;

	public:
		/**
		 * Constructor.
		 * @note The loMinHK constant must be at least the MIN_HK used for training.
		 * @param [in] nnet Pointer to neural network, which should be created externally to avoid recurrent spawning.
		 * @param [in] h Mesh size.
		 * @param [in] loMinHK Positive lower-bound for dimensionless mean curvature (e.g., 0.004) to use the nnet.
		 * @param [in] upMinHK Positive upper-bound for blending nnet-computed dimensionless mean curvature with
		 * 		  numerical estimation (e.g., 0.007).
		 */
		Curvature( const NeuralNetwork *nnet, const double& h, const double& loMinHK=0.004, const double& upMinHK=0.007 );

		/**
		 * Compute mean curvature.  There are two output modes in this function.  First, it computes the unit normals
		 * and the numerical mean curvature and place the results in the normal and numMeanK vectors.
		 * Then, it computes the mean curvature for grid points next to the interface using the hybrid approach.  The
		 * resulting approximation is placed in the hybMeanK vector and corresponds to the mean curvature *at* the
		 * normal projection of those nodes onto Gamma.
		 * The ancillary output hybFlag vector is populated with 1s where we used the hybrid approach and 0s everywhere
		 * else.
		 * TODO: Need to adjust to using compute_mean_curvature( ngbd, normal, numCurvature ) for compatibility with 3D.
		 * @param [in] ngbd Node neighborhood structure.
		 * @param [in] phi Nodal level-set values (assuming we have already reinitialized them).
		 * @param [out] normal Nodal unit normal vectors.
		 * @param [out] numMeanK Numerical mean curvature computed at all the nodes.
		 * @param [out] hybMeanK Hybrid mean curvature computed at the normal projection of nodes next to Gamma.
		 * @param [out] hybFlag Indicator vector with 1s where we used the hybrid approach and 0s everywhere else.
		 * @param [in] dimensionless Whether to scale curvature by h.
		 * @param [in] watch Optional timer.  If given, we will time numerical and hybrid curvature computations.  Timer
		 *        must be ready (i.e., called its start() method) before calling this function.
		 * @return A pair with <numerical, hybrid> timings in seconds if watch parameter is not nullptr, otherwise, the
		 *         values are set to -1.
		 * @throws invalid_argument if any vector is nullptr.
		 */
		std::pair<double, double> compute( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM],
										   Vec numMeanK, Vec hybMeanK, Vec hybFlag, const bool& dimensionless=false,
										   parStopWatch *watch=nullptr ) const;
	};
}


#endif //MY_P4EST_CURVATURE_ML_H
