Only execute `./run_master.sh`

Alternatively:

To configure:
`/home/samira/packages/cmake-3.22.5/bin/cmake -G "CodeBlocks - Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DDIMENSION=2d -B./cmake-build-release-2d -S.`


To build:
`/home/samira/packages/cmake-3.22.5/bin/cmake --build ./cmake-build-release-2d/  --target protein_aggregation -- -j 9`

To run: 
`cd <build-dir>`
`mpiexec -n 4 protein_aggregation`


