# Instructions for installing libraries on a Mac M1 architecture

#### Updated on Wednesday, December 7, 2022.

The following instructions assume you'll install our development software on a Mac system with Monterey OS and M1 arm64
architecture.  For reproducibility, peform the steps in order.

First, install the xcode development tools:

```bash
% xcode-select --install
% xcode-select -p
```

## Install Homebrew for arm64

```bash
% /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Run these steps:

```bash
% echo '# Set PATH, MANPATH, etc., for Homebrew.' >> /Users/youngmin/.zprofile
% echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/youngmin/.zprofile
% eval "$(/opt/homebrew/bin/brew shellenv)"
```

## Install GNU Make

Run these steps:
```bash
% brew install make
```

Add its path to PATH in .zsh_profile:
```bash
export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
```

Close and reopen the terminal.  Check that `echo $PATH` shows Homebrew's and GNU make's paths.
Check also GNU make's version with
```bash
make --version
```
It should say something like `GNU Make 4.4` or above.


## Install GNU CC compiler

Check out (https://trinhminhchien.com/install-gcc-g-on-macos-monterey-apple-m1/) for more info.   We'll install v12.

```bash
% brew install gcc@12
```

Check:
```bash
% gcc-12 --version
```

Create softlinks:
```bash
% cd /opt/homebrew/bin
% ln -s gcc-12 gcc
% ln -s g++-12 g++
```

Close and reopen the terminal to make changes effective.

Check priorities; it should show Homebrew first.  Also, `gcc --version` and `g++ --version` should
show Homebrew's versions just installed:
```bash
% where gcc
% where g++
```

Also, check fortran:
```bash
% gfortran --version
```

If it points to old version, add a softlink too:
```bash
% cd /opt/homebrew/bin
% ln -sf gfortran-12 gfortran
```


## Install `cmake`

```bash
% brew install cmake
% cmake --version.  # Would show something like 3.24
```

Close and reopen terminal.

## Install `anaconda3` for arm64

Download the M1 command line installer from (https://www.anaconda.com/products/distribution)

```bash
% cd ~/Downloads
% chmod a+x Anaconda.sh
% ./Anaconda.sh
```

Follow the screen instructions; `anaconda` will be installed under `/Users/youngmin/anaconda3/`.
1. Let the installer initialize `anaconda`; it will write a few lines to `.zshrc`.
2. Exit the terminal and reopen it.
3. Verify that the prefix `(base)` appears in the prompt.


## Install `mpich`

Add these environment variables to `.zshrc`:
```bash
export HOMEBREW_CC=gcc-12
export HOMEBREW_CXX=g++-12
```

and source it:
```bash
% source .zshrc
```

Then, install from source and verify the installation: 
```bash
% brew install --cc=gcc-12 mpich --build-from-source
% mpichversion
```
It should show a version >= 4.0.2.

Close and reopen terminal.


## Install `OpenBLAS`

We must have `BLAS` available before installing `PETSc`.   Let's install it in a custom `~/work/` folder under your home directory.
By using `~/work/`, we won't have to provide `sudo` privileges to install software in the  `/usr/local/` folder.

```bash
% cd
% wget https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20.tar.gz
% tar xvf OpenBLAS-0.3.20.tar.gz
% cd OpenBLAS-0.3.20
% mkdir -p build && cd build
% cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/youngmin/work -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 ..
% make && make install
```

Now, add these lines to `.zshrc` in your home directory and source it:
```bash
export OPENBLAS=/Users/youngmin/work
export CFLAGS="-falign-functions=8 ${CFLAGS}"
```

Close and reopen terminal.


## Install `PETSc`

Download the latest `PETSc` version, or if you prefer, use the stable 3.18.2 version used in these instructions.
```bash
% cd
% curl -O https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.18.2.tar.gz
% tar -xvf petsc-3.18.2.tar.gz
```

Install it in the `~/work/` folder using the `mpich` and `BLAS` options:
```bash
% cd petsc-3.18.3
% ./configure --prefix=/Users/youngmin/work/petsc-3.18.2 --with-openblas=1 
  --with-openblas-lib=$OPENBLAS/lib/libopenblas.a --with-openblas-include=$OPENBLAS/include/openblas 
  --download-hypre=1 --with-debugging=0 --with-mpi-dir=/opt/homebrew/Cellar/mpich/4.0.3 --with-shared-libraries=1 
  CFLAGS=$CFLAGS COPTFLAGS=-O2 CXXOPTFLAGS=-O2 FOPTFLAGS=-O2
```

Now, build `PETSc`.  You may follow the instructions in the prompt as you complete each step.
```bash
% make PETSC_DIR=/Users/youngmin/petsc-3.18.2 PETSC_ARCH=arch-darwin-c-opt all
```

Then, install the library:
```bash
% make PETSC_DIR=/Users/youngmin/petsc-3.18.2 PETSC_ARCH=arch-darwin-c-opt install
```

Test `PETSc` with:
```bash
% make PETSC_DIR=/Users/youngmin/work/petsc-3.16.3 PETSC_ARCH="" check
```

Close and reopen terminal.


## Install `p4est`

Download the library.  Use a newer version if you prefer.
```bash
% cd
% curl -O https://p4est.github.io/release/p4est-2.8.tar.gz
% tar -xvf p4est-2.8.tar.gz
```

Install `p4est` using `mpich`, in the `~/work/` folder:
```bash
% cd p4est-2.8
% ./configure --prefix=/Users/youngmin/work/p4est-2.8 --enable-mpi --enable-shared --enable-memalign=16 
  CFLAGS=-O2 CPPFLAGS=-O2 FCFLAGS=-O2 CC=/opt/homebrew/Cellar/mpich/4.0.3/bin/mpicc
% make 
% make install
```

Close and reopen terminal.


## Install `BOOST`

Download the `BOOST` source from (https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.bz2).  Then, 
decompress it.
```bash
% cd
% tar -xvf boost_1_78_0.tar.bz2
```

Install `BOOST` by running the following commands:
```bash
% cd boost_1_78_0
% ./bootstrap.sh --prefix=/Users/youngmin/work/boost-1.78.0
% ./b2
% ./b2 install
```

Close and reopen terminal.


## Install `Voro++`

Download it from (https://math.lbl.gov/voro++/download/) and decompress it by double clicking it.
```bash
% cd voro++-0.4.6
```

Edit the `config.mk` file by changing the prefix where we'll install `Voro++`:
```bash
# Installation directory
PREFIX=/Users/youngmin/work
```

Compile and install:
```bash
% make all
% make all install
```

The library is now under `/Users/youngmin/work/`.  The executable is under `bin/`, and the library and headers under
 `lib/` and `include/voro++/`.

Check executable:
```bash
% cd ~/work/bin
% ./voro++ --help
```

Close and reopen terminal.


## Build the library with no machine learning support (yet))

You may use the script below to build the library, assuming you already downloaded/cloned `casl_p4est` to
some location in your laptop.   Suppose we want to run the **SHS project** from the `examples/` folder.

Add the `*.cmake` profile for your machine under the `cmake/` folder.   Then, add an environment variable to the 
`~/.zshrc` file like so:
```bash
export CASL_CMAKE_PROFILE=mtxt.cmake	# Replace mtxt.cmake for your *.cmake machine profile.
```
Next:
```bash
% source .zshrc
```

This template bash script will build your SHS project.   We assume that the folders `cmake-build-debug-#d/` and
`cmake-build-release-#d` exist for `#=2` or `#=3`:
```bash
#!/bin/bash
echo "Generating, compiling, and running shs_channel_flow project"
echo "Started on $(date)"
began=$(date +%s)

MODE=Release    # Choose Debug or Release.

if [ "$MODE" == "Release" ]; then
	BUILD_DIR="cmake-build-release-3d"
else 
	if [ "$MODE" == "Debug" ]; then
		BUILD_DIR="cmake-build-debug-3d"
	else
		echo "Wrong type of build: only Release and Debug are allowed!"
		exit 1
	fi
fi

# Compile.
PROJECT="/Users/youngmin/Documents/CS/CASL/casl_p4est/examples/shs_channel_flow"

# Check if build directory does not exist
if [ ! -d "$PROJECT/$BUILD_DIR" ] 
then
    echo "Directory $PROJECT/$BUILD_DIR DOES NOT exist!" 
    exit 1
fi
cd $PROJECT/$BUILD_DIR
make clean
rm -rf *
cd ..

echo "-------------- Generating Makefile using default generator --------------"
cmake -DCMAKE_BUILD_TYPE=$MODE -DDIMENSION=3d -B./$BUILD_DIR -S.

echo "--------------------------- Compiling project ---------------------------"
cmake --build ./$BUILD_DIR --target shs_channel_flow -- -j 6

echo "Finished on $(date)!"
ended=$(date +%s)
walltime=$(expr $ended - $began)
echo "Wall time: $walltime seconds"
```

Add executable permisions to the script:
```bash
% chmod a+x build_shs.sh
```

Now, build the project:
```bash
% cd
% ./build_shs.sh
```

Next, create a temporary directory and test execution:
```bash
% cd
% mkdir tmp/
% cd casl_p4est/examples/shs_channel_flow/cmake-build-release-3d	# Assuming we built the release version.
% mpiexec -n 6 ./shs_channel_flow -duration 0.01 -Re_tau 10 -adapted_dt -lmin 3 -lmax 5 -GF 0.8125 
  -wall_layer 6 -export_folder /Users/youngmin/tmp -pitch 1.0 -length 1 -height 2 -width 1 -nx 1 -ny 2 -nz 1 
  -save_drag -save_mean_profiles -save_state_dt 5.0 -save_nstates 10 -grid_update 4294967295
```

It should run with no issues!

**Note**:  If the compilation didn't succeed, try again from scratch with homebrew.  Uninstall anything installed with
it, and skip installing any of the GNU CC compilers.  When uninstalling, make sure you also remove the soft links.
You need to reinstall Anaconda too, which requires removing its initialization from the `~/.zshrc` file.


## Installing the machine learning libraries

We'll start by creating a virtual environment with tensorflow explicitly build for Mac m1.  See 
(https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706) for more information.

To install the dependencies of `casl_p4est`, let's create an environment in python.
```bash
% conda create --name py38 python=3.8
```

This is the environment that we'll use to install `frugally-deep`.
```bash
conda activate py38
```

Next:
```bash
% conda install -c apple tensorflow-deps  # Installs the dependencies.
% pip install --upgrade numpy
% pip install tensorflow-macos			  # Installs the actual tensorflow lib for current virtual environment.
% # pip install tensorflow-metal		  # Don't install this! Plugin for metal platform fails :(
```

Install the remaining packages via `pip`:
```bash
% pip install pandas pickleshare scikit-image scikit-learn scipy
```

Now, let's install the machine learning libraries for C++ and `casl_p4est`:

```bash
% conda activate py38     # Activate the appropriate environment.
```

Install the software in sequence; modify the `-DCMAKE_INSTALL_PREFIX` according to your system.
```bash
% git clone -b 'v0.2.14-p0' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus
% cd FunctionalPlus
% mkdir -p build && cd build
% cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/youngmin/work -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
% make && make install
% cd ../..

% git clone -b '3.3.9' --single-branch --depth 1 https://gitlab.com/libeigen/eigen.git
% cd eigen
% mkdir -p build && cd build
% cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/youngmin/work -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
% make && make install
% ln -s /Users/youngmin/work/include/eigen3/Eigen /Users/youngmin/work/include/Eigen
% cd ../..

% git clone -b 'v3.9.1' --single-branch --depth 1 https://github.com/nlohmann/json
% cd json
% mkdir -p build && cd build
% cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/youngmin/work -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DBUILD_TESTING=OFF ..
% make && make install
% cd ../..

% git clone -b 'v0.15.2-p0' https://github.com/Dobiasd/frugally-deep
% cd frugally-deep
% mkdir -p build && cd build
% cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/youngmin/work -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
% make && make install
% cd ../..

% wget http://dlib.net/files/dlib-19.23.tar.bz2
% tar xvf dlib-19.23.tar.bz2
% cd dlib-19.23
% mkdir -p build && cd build
% cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/youngmin/work -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
% make && make install
```

Update the cmake profile in the `casl_p4est` project with the paths for the newly added libraries.

Now, let's test the build for the `ml_curvature` project, which uses the above libraries.  To do so, create the
`~/build_ml_curvature.sh` script with the following template bash, assuming that the folders `cmake-build-debug-#d/` 
and `cmake-build-release-#d` exist for `#=2` or `#=3` under the `casl_p4est/examples/ml_curvature/` folder:
```
#!/bin/bash
echo "Generating and compiling ml_curvature project"
echo "Started on $(date)"
began=$(date +%s)

MODE=Release    # Choose Debug or Release.

if [ "$MODE" == "Release" ]; then
	BUILD_DIR="cmake-build-release-3d"
else 
	if [ "$MODE" == "Debug" ]; then
		BUILD_DIR="cmake-build-debug-3d"
	else
		echo "Wrong type of build: only Release and Debug are allowed!"
		exit 1
	fi
fi

# Compile.
PROJECT="/Users/youngmin/Documents/CS/CASL/casl_p4est/examples/ml_curvature"

# Check if build directory does not exist
if [ ! -d "$PROJECT/$BUILD_DIR" ] 
then
    echo "Directory $PROJECT/$BUILD_DIR DOES NOT exist!" 
    exit 1
fi
cd $PROJECT/$BUILD_DIR
make clean
rm -rf *
cd ..

echo "-------------- Generating Makefile using default generator --------------"
cmake -DCMAKE_BUILD_TYPE=$MODE -DDIMENSION=3d -DENABLE_ML=1 -B./$BUILD_DIR -S.

echo "--------------------------- Compiling project ---------------------------"
cmake --build ./$BUILD_DIR --target ml_curvature -- -j 6

# Some timing info.
echo "Finished on $(date)!"
ended=$(date +%s)
walltime=$(expr $ended - $began)
echo "Wall time: $walltime seconds"
```

Let's test the curvature in 3D using machine learning.   Suppose we have the neural networks in `/Users/youngmin/k_nnets/3d/`
(or `[...]/2d/` for the two-dimensional case).

```bash
% cd /Users/youngmin/casl_p4est/examples/ml_curvature/cmake-build-release-3d	# Assuming we built the release version.
% mpiexec -n 3 ./ml_curvature -nnetsDir /Users/youngmin/k_nnets
```

It should work and show the following results (with wall times varying):
```
-------------------== CASL Options Database ==------------------- 
 List of entered options:

  -nnetsDir /Users/youngmin/k_nnets
 ----------------------------------------------------------------- 
>> Began testing hybrid curvature on a Gaussian surface online with a = 1., su^2 = 0.130208, sv^2 = 0.0144676, 
   max |hk| = 0.6, and h = 0.015625 (level 6)
* Reinitializing...  done after 0.668285 secs.
* Evaluating numerical baseline...  done with the following stats:
   - Time (in secs)            = 0.710371
   - Number of grid points     = 18000
   - Mean absolute error       = 1.292019e-03
   - Maximum absolute error    = 1.886310e-01
* Computing hybrid mean curvature... done with the following stats:
   - Time (in secs)            = 1.510003
   - Number of grid points     = 18000 (14107 saddles)
   - Mean absolute error       = 3.374795e-04
   - Maximum absolute error    = 1.591395e-02
<< Done after 2.83 secs.
```

### Installing an x86 `conda` library alongside `anaconda` for arm64

See (https://taylorreiter.github.io/2022-04-05-Managing-multiple-architecture-specific-installations-of-conda-on-apple-M1/) for
details on installing `miniconda3` and launching miniconda (x86) or anaconda (arm64) depending on the whether the terminal runs
on **Rosetta** or native support.

Let's create a virtual environment with python 3.7.10 and tensorflow 2.4.1.

Open the Rosetta terminal and verify you are in an x86 environment.
```bash
% cd k_nnets	# We assume we want to load the networks we configured with frugally-deep.
% conda create -n py37 python=3.7.10
% conda activate py37
% pip install -r requirements.txt
```

where `requirements.txt` contains:
```
# Install with `pip install -r requirements.txt` if you create the environment with
# % conda create -n py37 python=3.7.10
# python==3.7.10
h5py==2.10.0
keras-applications==1.0.8
keras-preprocessing==1.1.2
matplotlib==3.3.4
numpy==1.19.5
pandas==1.2.4
pickleshare==0.7.5
scikit-image==0.15.0
scikit-learn==0.21.3
scipy==1.3.1
tensorboard==2.5.0
tensorflow==2.4.1		# Although it gets installed, we can't use it :|
tensorflow-estimator==2.4.0
wheel==0.36.2
```

Now, start python in the Rosetta console, and verify that the following script works:
```python
import numpy as np
import pickle as pk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pcaScaler = pk.load( open( "2d/7/k_pca_scaler.pkl", "rb" ) )
stdScaler = pk.load( open( "2d/7/k_std_scaler.pkl", "rb" ) )
x = np.arange( 1, 29 ) / 10
x = pcaScaler.transform( stdScaler.transform( x.reshape( 1, -1 ) ) )
print( x )
```

The result should be:
```
[[  1.39350099   1.85553189  -7.46211191 -13.68130797   4.57589644
   29.28884798  -8.17100949  -3.08813993  51.85273562   4.96443536
   26.42014694 -10.45616538   4.91592136 -52.71729041 -87.34136368
  -90.31891946  22.20266979  59.54129308]]
```

To get tensorflow 2.4 to work, we need a virtual machine with x86 emulation.  Even Linux arm64 doesn't allow it.
