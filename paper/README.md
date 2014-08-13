# "Tree-based Adaptive Level-set Methods on Highly Parallel Machines"

## Literature Search

- ### General Parallel Quadtree/Octree Algorithms 
1. [**p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of Octrees.**][Burstedde2011]
 	This is the p4est paper that describes the low-level algorithms for parallelizing a forest of octrees.

2. [**Scalable Algorithms for Distributed-Memory Adaptive Mesh Refinement**][Langer2012]

3. [**Algorithms and Data Structures for Massively Parallel Generic Adaptive Finite Element Codes**][Bangerth2011]

- ### Sequential Level-Set Methods
1. [**Fast Two-Scale Methods for Eikonal Equations**][Chacon2012]
- ### Parallel Level-Set Methods
1. [**A parallel adaptive mesh method for the numerical simulation of multiphase flows**][Rodriguez2013]
	This paper tries to couple a FEM solver with a level-set mehtod on unstructured grids using a software called "PHASTA". They do not present any scaling result and simply mention that "PHASTA" has been scaled to up to 32,000 processors.

2. [**Parallelization of a level set method for simulating dendritic growth**][Wang2006]
	This paper investigates the possibility of parallel computation for level-set method with applications to the dendritic growth simulation in 2D. Authors use a simple Cartesian grid for the computation and MPI for domain decomposition. Also level-set equation is solved on a narrow band to reduce the computation and the Poisson equation is solved using Guass-Sidel iteration. An interesting contribution of the paper is the use of "processor visualization" to improve cache efficiency and reduce communication cost. It turns out that using this techniques could improve timings depending on the degree of virtualization (50% at best). Scaling tests have been conducted up to 54 processors yielding ~83% efficiency.

3. [**A Parallel Strategy for a Level Set Simulation of Droplets Moving in a Liquid Medium**][Fortmeier2011]
	In this article authors parallelize a FEM solver, "DROPS", for use in multiphase simulation of droplets in a liquid. Both the Navier-Stokes and the level-set equations are solved on hierarchical tetrahedral meshes. The library ParMetis is utilized for partitioning the unstructured tetrahedral mesh and for load balancing. Decent speedups are reported up to 256 processors (~82% equivalent at best). One part of their algorithm that does not scale well the updating of the unstructured grid and its parallel partitioning.

4. [**High performance computing for the level-set reconstruction algorithm**][Hajihashemi2010]
	In this article, authors present a parallelization of the level-set equation using MPI. Their code, which is implemented using regular Cartesian grids in 2D, is used for electromagnetic inverse scattering problems and achieves really bad scaling (~21%-33% on 256 processors). However, this seems mostly due to the bottlenecks in computing the speed functions and matrix inversion rather than solution of the level-set equations or domain decomposition. Also, the domain decomposition technique used here is 1D (i.e. only in the y direction) to reduce the communication latency (NOTE: while communication *volume* decreases in 2D decomposition, the communication *latency* increases due to more separate messages)
		
5. [**A parallelized, adaptive algorithm for multiphase flows in general geometries**][Sussman2005]	
	In this paper authors implement a parallel multiphase solver on adaptive block-Cartesian grids. As part of this they also parallelize their level-set solver to track the location of interface. They use the [boxlib][boxlib] library from the CCSE group in LLBL which utilizes MPI for parallelization. Parallel speedup study is restricted to simple situations with at most 1-16 processors. There is also a "big" simulation on 32 processors but they do not report any *parallel speedup* whatsoever.
	
6. [**A parallel Eulerian interface tracking/Lagrangian point particle multi-scale coupling procedure**][Herrmann2010]
	In this paper authors describe a hybrid Eulerian/Lagrangian framework for the interface tracking problems. The main idea is to represent small patches of liquid or gas which cannot be resolved on the grid explicitly in a Lagrangian framework. Furthermore they make use of two separate grids: 1) the flow solver grid on which NS equations are solved and could potentially be unstructured and 2) a Cartesian grid which is used for the level-set equation. They do not present scaling of level-set advection. Instead, they present scaling for their Eulerian-to-Lagrangian conversion algorithm. It is not quite clear if this also involves level-set evolution (probably does) and/or flow solver on the separate grid (probably does not). Nonetheless they report good scalings (no number -- just a figure showing the scalability) up to 2,048 processors.
	
7. [**Parallel 3D Image Segmentation of Large Data Sets on a GPU Cluster**][Hagan2009]
	Here authors present a new way of solving level-set equations (advertised as suitable for both advection and motion under curvature but in practice I can only see motion under curvature) based on using Lattice Boltzmann Method! Since LBM is embarrassingly parallel, this is useful for GPU computations. The LBM lattice used here is D3Q7 which uses regular Cartesian grid. Moreover, they decompose the domain into smaller subdomains and send each to a separate GPU on a cluster, i.e. they use a MPI-GPU hybrid approach. They claim "good performance" but only show a single timing without any reference point to measure any kind of speed up! Unsurprisingly, their method seems to considerably suffer from GPU and MPI communication overhead (~3-4 comm./comput. ratio).

8. [**Higher-order CFD and interface tracking methods on highly-Parallel MPI and GPU systems**][Appleyard2011]
	Authors present a comparison between a purely MPI and a hybrid MPI-GPU parallelization of high-order (3rd, 5th, 7th, and 9th order) level-set methods in 3D and show that the hybrid MPI-GPU method can reach greater or equal performance on 4 GPUs compared to a 256-core, purely MPI implementation. This turns out to be mainly due to the GPU having higher memory bandwidth. Moreover, they report higher performance for higher order methods, presumably due to higher FLOPs intensity. They do not report actual number for speedup but their curves seem close to linear speedup. Finally, they do not go into the details of discretization but it is a finite difference method on Cartesian grids with 2D decomposition technique.

9. [**Adaptive multi-resolution method for compressible multi-phase flows with sharp interface model and pyramid data structure**][Han2014]
	Authors propose an adaptive compressible multiphase method which uses level-set method to track the location of interface. They also make use of what they call "pyramid data structure" which just seems to be the Quadtree! They use task-based parallelism on shared memory machines in which computations within coarse blocks are treated as separate tasks and carried on by individual threads. Their algorithm is only reported in 2D and a single scaling study is performed up to 16 cores yielding 10X speedup.
	
10. [**Fast Marching Methods -- Parallel Implementation and Analysis (Ph.D. Thesis)**][Tugurlan2008]
	This thesis describes a parallelization of the Fast Marching Method (FMM) using domain decomposition and MPI. Author presents pretty good (linear) scaling up to about 50 processors in 2D even though the number of iterations to convergence grows as the number of processors is increased.

11. [**Hybrid Distributed-/Shared-Memory Parallelization For Re-Initializing Level Set Functions**][Fortmeier2010]

12. [**Data-Parallel Octrees for Surface Reconstruction**][Zhou2011]

13. [**An adaptive domain-decomposition technique for parallelization of the fast marching method**][Breub2011]

14. [**Parallel Implementations of the Fast Sweeping Method**][Zhao2007]

15. [**A Parallel Fast Sweeping Method for the Eikonal Equation**][Detrixhe2013]

16. [**A Fast Iterative Method for the Eikonal Equation**][Jeong2008]

17. [**A Parallel Heap-Cell Method for Eikonal Equations**][Chacon2013]
 
18. [**A Patchy Dynamic Programming Scheme for a Class of Hamilton-Jacobi-Bellman Equations**][Cacace2011]

19. [**Fast Iterative Method in Solving Eikonal Equations : a Multi-Level Parallel Approach**][Dang2014]


[References]: <>
[Burstedde2011]: http://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf
[Rodriguez2013]: http://www.sciencedirect.com/science/article/pii/S004579301300131X
[Wang2006]: http://www.sciencedirect.com/science/article/pii/S0743731506000244
[Sussman2005]: http://www.sciencedirect.com/science/article/pii/S0045794904004134
[Herrmann2010]: http://www.sciencedirect.com/science/article/pii/S0021999109005543
[Fortmeier2011]: http://link.springer.com/chapter/10.1007%2F978-3-642-19328-6_20
[Hajihashemi2010]: http://www.sciencedirect.com/science/article/pii/S0743731509001841
[Hagan2009]: http://link.springer.com/chapter/10.1007%2F978-3-642-10520-3_92#page-1
[boxlib]: https://ccse.lbl.gov/BoxLib/
[Appleyard2011]: http://www.sciencedirect.com/science/article/pii/S0045793010002872
[Han2014]: http://www.sciencedirect.com/science/article/pii/S0021999114000230
[Tugurlan2008]: http://etd.lsu.edu/docs/available/etd-09152008-143521/unrestricted/Tugurlandiss.pdf
[Fortmeier2010]: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5581331
[Zhou2011]: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5473223
[Breub2011]: http://www.sciencedirect.com/science/article/pii/S0096300311007107
[Zhao2007]: http://web.b.ebscohost.com/ehost/detail/detail?sid=c78ff1d5-ddf3-459d-8287-b9c3058fe63b%40sessionmgr111&vid=0&hid=112&bdata=JnNpdGU9ZWhvc3QtbGl2ZQ%3d%3d#db=mth&AN=25853983
[Detrixhe2013]: http://www.sciencedirect.com/science/article/pii/S002199911200722X
[Langer2012]: http://charm.cs.illinois.edu/newPapers/12-35/paper.pdf
[Bangerth2011]: http://p4est.github.io/papers/BangerthBursteddeHeisterEtAl11.pdf
[Chacon2012]: http://epubs.siam.org/doi/pdf/10.1137/10080909X
[Jeong2008]: http://epubs.siam.org/doi/pdf/10.1137/060670298
[Chacon2013]: http://arxiv.org/pdf/1306.4743v1.pdf
[Cacace2011]: http://hal-ensta.archivesouvertes.fr/docs/00/62/81/08/PDF/CacaceChristianiFalconePicarelli_2011.pdf
[Dang2014]: http://www.sciencedirect.com/science/article/pii/S1877050914003470