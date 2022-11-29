# Instructions to install the `casl` library and its dependencies on `Stampede2`

**By _Luis Ángel_ \
November 29, 2022**

On `Stampede2` we only need to install `p4est` and `voro++`.  The other library dependencies are already installed, but we need 
to load them dynamically.

First, `shs` to Stampede2 (I'll use my username, `tg878736`, to illustrate the process):
```bash
ssh tg878736@stampede2.tacc.utexas.edu
```
Provide your passworkd at the prompt, followed by the two-factor authentication code sent through the `DUO` phone app.

On `Stampede2`, navigate to your `$WORK` directory and load the pre-installed dependencies.
```bash
cd $WORK                  # $WORK is the envioronment variable referring to your `work` directory.
module load cmake/3.16.1
module load gcc/9.1.0
module load impi/19.0.9
module load petsc/3.15    # Alternatively, load `petsc/3.15-debug` if you want to compile on debug mode.
module load python3/3.8.2
module load boost/1.72
module load fftw3/3.3.8
```

To view the description and environment variables set with each module, e.g., `petsc/3.15`, use:
```bash
module spider petsc/3.15
```

We'll install our local library dependencies under the `$WORK/local/` directory.  Let's create it:
```bash
mkdir local
```

Double check we're still in the `$WORK`, directory.  `pwd` should show something like this for my `tg878736` username:
```
/work2/08574/tg878736/stampede2
```

### Installing `p4est`

Download and decompress the library.  After running these commands, we should have a directory called `$WORK/p4est-2.8/`:
```bash
cd
curl -O https://p4est.github.io/release/p4est-2.8.tar.gz
tar xvf p4est-2.8.tar.gz
```

Install `p4est` using the pre-installed Intel's `mpi`:
```bash
cd p4est-2.8
./configure --prefix=$WORK/p4est-2.8 --enable-mpi --enable-shared --enable-memalign=16 \ 
	CFLAGS=-O2 CPPFLAGS=-O2 FCFLAGS=-O2 CC=$TACC_IMPI_BIN/mpicc		# Use the option `--enable-debug` for debug mode.
make 
make install
```

The above commands will install `p4est` under the subdirectory `$WORK/local/p4est-2.8`.  We need to create an environment 
variable for this path.  Go to your home directory and open the `.bashrc` config file:
```bash
cd
vi .bashrc
```

Inside `.bashrc`, locate `# Section 2` and place the following line inside the `if` statement:
```bash
export MY_P4EST_DIR=$WORK/local/p4est-2.8		# New env variable, MY_P4EST_DIR, to be used in the CASL library.
```

### Installing `voro++`

Now, let's install `voro++` locally, in a similar manner to `p4est`.  First, download the `voro++` library and decompress it:
```bash
cd $WORK
curl -O http://math.lbl.gov/voro++/download/dir/voro++-0.4.6.tar.gz
tar xvf voro++-0.4.6.tar.gz
```

Navigate to the `voro++-0.4.6` source folder and edit the `config.mk` file by changing the prefix where we'll install `voro++`:
```bash
# Installation directory
PREFIX=${WORK}/local/voro++-0.4.6	# Notice the `${}` operator.
```

Compile and install:
```bash
make all
make all install
```

The library is now under `$WORK/local/voro++-0.4.6`.  The executable is under `bin/`, and the library and headers under
`lib/` and `include/voro++/`.  Now, check the executable:
```bash
% cd $WORK/local/voro++-0.4.6/bin
% ./voro++ --help
```

As we did for `p4est`, let's add an environment variable for the `voro++` library to `$HOME/.bashrc`.  First, go back to your 
home director:
```bash
cd
```

Then, add this entry to the `if` statement in `# Section 2`:
```bash
export MY_VORO_DIR=$WORK/local/voro++-0.4.6
```

Since we are now done with the `Stampede2` profile, let's also add its environment variable to `.bashrc` below the entry for
`voro++`:
```bash
export CASL_CMAKE_PROFILE=stampede2.cmake
```

Source the `.bashrc` file:
```bash
source .bashrc
```

## Build the library with no machine learning support

First, clone the library into the `$WORK` directory:
```bash
cd $WORK
git clone git@bitbucket.org:cburstedde/casl_p4est.git
```

If you run into any security issues, you might need to add your `Stampede2` account's `ssh` public key to your `bitbucket` 
account.  See [this document](https://support.atlassian.com/bitbucket-cloud/docs/configure-ssh-and-two-step-verification/) for
more information.

Suppose we want to build and run the **SHS Project** (i.e., `$WORK/casl_p4est/examples/shs_channel_flow`).  Let's switch to the 
`shs_ra` branch:
```bash
cd casl_p4est
git checkout shs_ra		# Switch
git pull origin			# and update.
```

Verify that `casl_p4est/cmake/stampede2.cmake` exists.  Note this is the profile name we added to `$HOME/.bashrc` above so that
the library knows where to locate its dependencies (using `cmake`).

Next, navigate to the `shs_channel_flow/` subdirectory:
```bash
cd examples/shs_channel_flow
```

Create the build directories for release and debug modes in 3D:
```bash
mkdir cmake-build-release-3d
mkdir cmake-build-debug-3d
```

Use the following template bash script to build your SHS project.  Navigate to `$HOME` and create the script `build_shs.sh`.  
Copy the following contents into `build_shs.sh`:
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

# Loading modules.
module load cmake/3.16.1
module load gcc/9.1.0
module load impi/19.0.9
if [ "$MODE" == "Release" ]; then
        module load petsc/3.15
else
        module load petsc/3.15-debug
fi
module load python3/3.8.2
module load boost/1.72
module load fftw3/3.3.8

# Compile.
cd $WORK/casl_p4est/examples/shs_channel_flow/
cd $BUILD_DIR
make clean
rm -r *
cd ..

echo "-------------- Generating Makefile using default generator --------------"
$TACC_CMAKE_BIN/cmake -DCMAKE_BUILD_TYPE=$MODE -DDIMENSION=3d -B./$BUILD_DIR -S.

echo "--------------------------- Compiling project ---------------------------"
$TACC_CMAKE_BIN/cmake --build ./$BUILD_DIR --target shs_channel_flow

module reset

echo "Finished on $(date)!"
ended=$(date +%s)
walltime=$(expr $ended - $began)
echo "Wall time: $walltime seconds"
```

Add executable permisions to the script:
```bash
chmod a+x build_shs.sh
```

And run it.  It's ok if you're on a login node because we're not using more than a single core:
```bash
./build_shs.sh
```

### Running an SHS example

Use the following sample script to run an SHS simulation for stat collection.  You'll see the portion for running from scratch
commented out.

```bash
#!/bin/bash

#SBATCH -J run_shs_10           # Job name
#SBATCH -o shs_10.%j.out        # Name of stdout output file
#SBATCH -p normal               # Queue (partition) name
#SBATCH -N 4                    # Total # of nodes 
#SBATCH --ntasks-per-node 68    # Total # of mpi tasks
#SBATCH -t 36:00:00             # Run time (hh:mm:ss)
#SBATCH --mail-user=lal@cs.ucsb.edu
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A TG-ASC150002         # Allocation name (req'd if you have more than 1)

echo "Preparing execution of shs_channel_flow project"
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

# Loading modules.
module load cmake/3.16.1
module load gcc/9.1.0
module load impi/19.0.9
if [ "$MODE" == "Release" ]; then
        module load petsc/3.15
else
        module load petsc/3.15-debug
fi
module load python3/3.8.2
module load boost/1.72
module load fftw3/3.3.8

OUT_DIR="$SCRATCH/shs_outputs/3d_channel_10"

# Runing program.
echo "-------------------------- Running simulation ---------------------------"
cd $WORK/casl_p4est/examples/shs_channel_flow/
cd $BUILD_DIR

# Run from scratch.
#ibrun ./shs_channel_flow -duration 10.001 -Re_tau 140 -adapted_dt -lmin 6 -lmax 8 -lmid_delta_percent 0.3 -GF 0.9375 -wall_layer 27 -lip 1.2 -cfl 3 \
#       -export_folder $OUT_DIR -pitch 0.375 -length 6 -height 2 -width 3 -nx 1 -ny 1 -nz 1 \
#       -save_drag -save_mean_profiles -save_state_dt 1.00 -save_nstates 100 -grid_update 4294967295 \
#       -thresh 1000000000.0 -white_noise_rms 0.05 -u_tol 0.001 -niter_hodge 3 -cell_solver_rtol 1e-5 -cell_solver_verbose

# Restart for stat collection (noticed the `only_sum` option to avoid computing averages.  You'll have to average stats in Python.
ibrun ./shs_channel_flow -duration 5.001 -Re_tau 300 -adapted_dt -lmin 6 -lmax 8 -lmid_delta_percent 0.3 -GF 0.9375 -wall_layer 27 -lip 1.2 -cfl 3 \
        -save_state_dt 1.0 -save_nstates 100 -save_mean_profiles -save_drag -grid_update 4294967295 \
        -restart ${OUT_DIR}/end_state_tn511_rs -running_stats -running_stats_dt 0.003333333333 -running_stats_num_steps 1500 -running_stats_only_sum \
        -thresh 1000000000.0 -export_folder ${OUT_DIR} -pitch 0.375 -u_tol 0.001 -niter_hodge 3 -cell_solver_rtol 1e-5

# Some timing info.
module reset

echo "Finished on $(date)!"
ended=$(date +%s)
walltime=$(expr $ended - $began)
echo "Wall time: $walltime seconds"
```

Usually, the above script would go into `$HOME/run_shs_10.sh`, where the `10` is the experiment ID, for example.  All outputs are
placed under the `$SCRATCH/shs_outputs/3d_channel_10/`.  Upon finishing the experiment, we must move its `tar.bz2` archive to the 
`$WORK/shs_outputs/` folder because anything on `$SCRATCH/` may be purged after some period of time.