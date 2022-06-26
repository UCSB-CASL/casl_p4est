# What is `parCASL`?  

`parCASL` (Parallel Computational Applied Science Library) is a library for
solving Partial Differential Equations (PDEs) that are commonly encountered in
physics and applied sciences.  It is written in C++ and leverages state-
of-the-art algorithms as well as well-established high performance libraries to
enable fast and accurate solution of PDEs.


# License

Please see the [LICENSE](LICENSE) file included in this directory.


# Installation

Let's assume that we are on Ubuntu 18 or 20.

## Minimal dependencies

### Installing CMake

1. Download `cmake-3.22.1-linux-x86_64.sh` or a newer version from [here](https://cmake.org/download/).  Assuming
   we are in the home directory (where we have downloaded CMake), type:
```
$> sudo ./cmake-3.22.1-linux-x86_64.sh
```

2. Follow the instructions.  It'll install CMake everything under a new directory: `cmake-3.22.1-linux-x86_64.sh`.
   Rename the folder to make it easy to use CMake.
```
mv cmake-3.22.1-linux-x86_64/ cmake-3.22.1
```

3. The CMake binary is now under the `bin` folder inside `cmake-3.22.1`:
```
$> cd cmake-3.22.1
$> cd bin/
$> ./cmake --help
```
And the last line will show the help if everything went OK.

### Installing MPICH

1. Update packages, and then run the usual installation instruction with `apt`.
```
$> sudo apt update
$> sudo apt install mpich
```

2. These instructions install MPICH under the `/etc/alternatives/` folder, but it also places links to it under
   `/usr/bin/`.  Similarly, the shared libraries are linked under `/usr/lib/mpich` and the headers under
   `/usr/include/mpich`.  Now, test the installation from anywhere you like:
```
$> cd
$> mpiexec --help
```

### Installing PETSc

1. Download the latest PETSc library and decompress it.
```
$> cd
$> curl -O https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.16.3.tar.gz
$> tar -xvf petsc-3.16.3.tar.gz
```

2. The above commands creates the `petsc-3.16.3/` directory.  Switch to that folder and configure the library.
   Notice this will configure it in **release** mode.  For **debug** mode, use the option `--with-debugging=1`.
```
$> cd petsc-3.16.3
$> sudo ./configure --download-fblaslapack --download-hypre=1 --prefix=/usr/local/petsc-3.16.3 
   --with-debugging=0 --with-mpi-dir=/usr --with-shared-libraries=1 
   COPTFLAGS=-O2 CXXOPTFLAGS=-O2 FOPTFLAGS=-O2
```
Wait for the process to finish.

3. Now, build PETSc.  It'll tell you what command to use after finishing the previous step, for example:
```
$> sudo make PETSC_DIR=/home/youngmin/petsc-3.16.3 PETSC_ARCH=arch-linux-c-opt all
```

4. Install PETSc.  Again, it'll tell you what instruction to use, for example:
```
$> sudo make PETSC_DIR=/home/youngmin/petsc-3.16.3 PETSC_ARCH=arch-linux-c-opt install
```

5.  To check the installation, type:
```
$> make PETSC_DIR=/usr/local/petsc-3.16.3 PETSC_ARCH="" check
```
The library will be under the `/usr/local/petsc-3.16.3/` directory.  It comes with the `lib/` and `include/` folders.

### Installing `p4est`

1. Download the latest `p4est` library and decompress it.
```
$> cd
$> curl -O https://p4est.github.io/release/p4est-2.8.tar.gz
$> tar -xvf tar -xvf p4est-2.8.tar.gz
```

2. The above command creates the `p4est-2.8/`.  Navigate to that folder and configure the library.
   Again, this will be in **release** mode.  For **debug** mode, use the option `--enable-debug`.
```
$> cd p4est-2.8/
$> sudo ./configure --prefix=/usr/local/p4est-2.8 --enable-mpi --enable-shared --enable-memalign=16 
   CFLAGS=-O2 CPPFLAGS=-O2 FCFLAGS=-O2
```

3. Now, build and install `p4est`.
```
$> sudo make
$> sudo make install
```
The library will be under the `/usr/local/p4est-2.8/` directory.  It comes with `lib/` and `include/` folders.

### Installing BOOST

1. Download the BOOST source from [here](https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.bz2).
   Place it in the home directory and uncompress it.
```
$> cd
$> tar -xvf boost_1_78_0.tar.bz2
```

2. Install it by running the following commands:
```
$> cd boost_1_78_0
$> sudo ./bootstrap.sh --prefix=/usr/local/boost-1.78.0
$> ./b2
$> sudo ./b2 install
```
The library will then be placed in the `/usr/local/boost-1.78.0/` folder.  It comes with `lib/` and `include/boost/`
folders.

### Installing Voro++

1. Download the Voro++ library and decompress it.
```
$> cd
$> curl -O http://math.lbl.gov/voro++/download/dir/voro++-0.4.6.tar.gz
$> tar -xvf voro++-0.4.6.tar.gz
```

2. Switch to the Voro++ source folder, compile it, and install it.
```
$> cd voro++-0.4.6
$> sudo make all
$> sudo make all install
```
The library is now placed under `/usr/local/`.  The executable is under `bin/`, and the library and headers under
`lib/` and `include/voro++/`.

3. You can check the executable with:
```
$> cd /usr/local/bin
$> ./voro++ --help
```

## Machine-learning dependencies

TBD

## Cloning the repository

Follow the instructions [here](https://support.atlassian.com/bitbucket-cloud/docs/set-up-an-ssh-key/#Set-up-SSH-on-macOS-Linux)
to set up SSH before cloning.  Add your public key to your *BitBucket* personal settings.

To clone the project:
```
$> git clone git@bitbucket.org:cburstedde/casl_p4est.git
```


# Compiling `parCASL` with CMake

See the [CMake webpage](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for more details.

To create the `CMakeLists.txt` file under an example, take a look at the `ml_curvature`'s 
[CMakeLists.txt](examples/ml_curvature/CMakeLists.txt) file.  Don't forget to add your machine library dependencies
under the `cmake/` folder at the root of `parCASL`.

1. `cd` to the example you want to compile:
```
$> cd /Users/youngmin/Documents/CS/CASL/casl_p4est/examples/ml_curvature
```

2. Generate the `Makefile`.  For example, let's generate a `Makefile` for `Release` mode in `2d`:
```
$> /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -G "CodeBlocks - Unix Makefiles" 
   -DCMAKE_BUILD_TYPE=Release -DENABLE_ML=1 -DDIMENSION=2d -B./cmake-build-release-2d -S.
```

Here:
- `-DDIMENSION=2d` sets the `DIMENSION` variable to `2d`.
- `-DCMAKE_BUILD_TYPE` sets the `CMAKE_BUILD_TYPE` variable to `Release`.
- `-DENABLE_ML` enables machine-learning dependencies and classes.
- `-B./cmake-build-release-2d` tells CMake where to put all intermediate files and `Makefile`.
- `-S.` tells CMake that the source directory is the one we are currently at.
- `-G "CodeBlocks - Unix Makefiles"` indicates which generator to use.  If not given, it checks the
  `CMAKE_GENERATOR` environment variable or falls back to a builtin default selection.

3. Build the project:
```
$> /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake --build ./cmake-build-release-2d/ 
   --target ml_curvature -- -j 9
```

Here:
- `--build ./cmake-build-release-2d/` indicates where to find the `Makefile`.
- `--target ml_curvature` is the target you defined in `CMakeLists.txt`.
- `-- -j 9` gives one build-tool option.  Anything after `--` is an option. `-j 9` tells CMake to use up to 9 concurrent jobs
  when building.

4. Run the project as a single-process app:
```
$> cd cmake-build-release-2d/
$> ./ml_curvature
```
or with `MPI`:
```
$> cmake-build-release-2d/
$> mpiexec -n 3 ./ml_curvature
```