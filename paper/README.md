# "Tree-based Adaptive Level-set Methods on Highly Parallel Machines"

## Literature Search

- [**p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of Octrees.**][Burstedde2011]
 	This is the p4est paper that describes the low-level algorithms for parallelizing a forest of octrees.

- [**A parallel adaptive mesh method for the numerical simulation of multiphase flows**][Rodriguez2013]
	This paper tries to couple a FEM solver with a level-set mehtod on unstructured grids using a software called "PHASTA". They do not present any scaling result and simply mention that "PHASTA" has been scaled to up to 32,000 processors.

- [**Parallelization of a level set method for simulating dendritic growth**][Wang2006]
	This paper investigates the possibility of parallel computation for level-set method with applications to the dendritic growth simulation in 2D. Authors use a simple Cartesian grid for the computation and MPI for domain decomposition. Also level-set equation is solved on a narrow band to reduce the computation and the Poisson equation is solved using Guass-Sidel iteration.

	An interesting contribution of the paper is the use of "processor visualization" to improve cache efficiency and reduce communication cost. It turns out that using this techniques could improve timings depending on the degree of virtualization (50% at best). Scaling tests have been conducted up to 54 processors yielding ~83% efficiency.

- [**A Parallel Strategy for a Level Set Simulation of Droplets Moving in a Liquid Medium**][Fortmeier2011]
	In this article authors parallelize a FEM solver, "DROPS", for use in multiphase simulation of droplets in a liquid. Both the Navier-Stokes and the level-set equations are solved on hierarchical tetrahedral meshes. The library ParMetis is utilized for partitioning the unstructured tetrahedral mesh and for load balancing. Decent speedups are reported up to 256 processors (~82% equivalent at best). One part of their algorithm that does not scale well the updating of the unstructured grid and its parallel partitioning.

-	[**High performance computing for the level-set reconstruction algorithm**][Hajihashemi2010]
	In this article, authors present a parallelization of the level-set equation using MPI. Their code, which is implemented using regular Cartesian grids in 2D, is used for electromagnetic inverse scattering problems and achieves really bad scaling (~21%-33% on 256 processors). However, this seems mostly due to the bottlenecks in computing the speed functions and matrix inversion rather than solution of the level-set equations or domain decomposition. 

	Also, the domain decomposition technique used here is 1D (i.e. only in the y direction) to reduce the communication latency (NOTE: while communication *volume* decreases in 2D decomposition, the communication *latency* increases due to more separate messages)
	
-	[**A parallelized, adaptive algorithm for multiphase flows in general geometries**][Sussman2005]	

-	[**A parallel Eulerian interface tracking/Lagrangian point particle multi-scale coupling procedure**][Herrmann2010]

-	[**Parallel 3D Image Segmentation of Large Data Sets on a GPU Cluster**][Hagan2009]

[Burstedde2011]: http://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf
[Rodriguez2013]: http://www.sciencedirect.com/science/article/pii/S004579301300131X
[Wang2006]: http://www.sciencedirect.com/science/article/pii/S0743731506000244
[Sussman2005]: http://www.sciencedirect.com/science/article/pii/S0045794904004134
[Herrmann2010]: http://www.sciencedirect.com/science/article/pii/S0021999109005543
[Fortmeier2011]: http://link.springer.com/chapter/10.1007%2F978-3-642-19328-6_20
[Hajihashemi2010]: http://www.sciencedirect.com/science/article/pii/S0743731509001841
[Hagan2009]: http://link.springer.com/chapter/10.1007%2F978-3-642-10520-3_92#page-1