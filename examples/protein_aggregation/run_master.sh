./configure.sh

./build_executable.sh

source ./set_env_vars.sh

mpiexec -n 12 cmake-build-release-2d/protein_aggregation
