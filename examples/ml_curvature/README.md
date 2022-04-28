# `ml_curvature`

## Installing libraries for machine learning in C++

This information was taken from `frugally-deep`, which describes the steps to install that library and its dependencies using `cmake`.
Notice that you can change the default *installation prefix* (`/usr/local`) and C/C++ compiler (`cc`) by using the `cmake` options:

```bash
-DCMAKE_INSTALL_PREFIX:PATH=/$WORK/local -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=gcc
```
where we have chosen `gcc` and a user-friendly path `$WORK/local` we can write to.  Here are the steps to install all libraries we'll need:

```bash
git clone -b 'v0.2.14-p0' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus
cd FunctionalPlus
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..

git clone -b '3.3.9' --single-branch --depth 1 https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir -p build && cd build
cmake ..
make && sudo make install
sudo ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
cd ../..

git clone -b 'v3.9.1' --single-branch --depth 1 https://github.com/nlohmann/json
cd json
mkdir -p build && cd build
cmake -DBUILD_TESTING=OFF ..
make && sudo make install
cd ../..

git clone -b 'v0.15.2-p0' https://github.com/Dobiasd/frugally-deep
cd frugally-deep
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..

# To install dlib, particularly on Stampede2, we must do it on a non-sudo location.  The above libraries would require this too.
wget http://dlib.net/files/dlib-19.23.tar.bz2
tar xvf dlib-19.23.tar.bz2
cd dlib-19.23
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/$WORK/local -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=gcc ..
make && make install
```