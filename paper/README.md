# "Tree-based Adaptive Level-set Methods on Highly Parallel Machines"

## Literature Search

- ### General Parallel Quadtree/Octree Algorithms 
1. [**p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of Octrees.**][Burstedde2011]
 	This is the p4est paper that describes the low-level algorithms for parallelizing a forest of octrees.

2. [**Scalable Algorithms for Distributed-Memory Adaptive Mesh Refinement**][Langer2012]

3. [**Algorithms and Data Structures for Massively Parallel Generic Adaptive Finite Element Codes**][Bangerth2011]

4. [**Fast BVH Construction on GPUs**][Lauterbach2009]

5. [**Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees**][Karras2012]

6. [**Simpler and Faster HLBVH with Work Queues**][Garanzha2011]

7. [**Data-Parallel Octrees for Surface Reconstruction**][Zhou2011]
    Authors present a parallel Octree algorithm for surface construction applications on the GPU. Their algorithm generates the octree from a cloud of points in a bottom-up approach, entirely runs on the GPU, and Compared to similar cpu algorithms they achieve about 100X-200X speed up in both the tree construction and surface generation. One key aspect of their algorithm is the use of static lookup tables to generate neighborhood information.

- ### Sequential Level-Set Methods
1. [**Fast Two-Scale Methods for Eikonal Equations**][Chacon2012]
    Authors introduce a two-level solution method for the Eikonal equation. In this method initially an FMM algorithm is used on a coarse level to find the correct updating rules for the coarse cells. Inside each coarse cell, a fine grid is initialized and the FSM is used to obtain the solution. FMM is a good method when the speed function and/or interfaces are highly irregular but is $\mathcal{O}(N\log N)$. FSM, on the other hand is an $\mathcal{O}(N)$ method but may require many iterations for complicated problems. By combining the two methods, authors come arrive at an algorithm that is both fast and robust. They suggest this approach might also be a good way to tackle the parallelization problem.

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
	
6. [**A Parallel Eulerian Interface Tracking/Lagrangian Point Particle Multi-Scale Coupling Procedure**][Herrmann2010]
	In this paper authors describe a hybrid Eulerian/Lagrangian framework for the interface tracking problems. The main idea is to represent small patches of liquid or gas which cannot be resolved on the grid explicitly in a Lagrangian framework. Furthermore they make use of two separate grids: 1) the flow solver grid on which NS equations are solved and could potentially be unstructured and 2) a Cartesian grid which is used for the level-set equation. They do not present scaling of level-set advection. Instead, they present scaling for their Eulerian-to-Lagrangian conversion algorithm. It is not quite clear if this also involves level-set evolution (probably does) and/or flow solver on the separate grid (probably does not). Nonetheless they report good scalings (no number -- just a figure showing the scalability) up to 2,048 processors.
	
7. [**Parallel 3D Image Segmentation of Large Data Sets on a GPU Cluster**][Hagan2009]
	Here authors present a new way of solving level-set equations (advertised as suitable for both advection and motion under curvature but in practice I can only see motion under curvature) based on using Lattice Boltzmann Method! Since LBM is embarrassingly parallel, this is useful for GPU computations. The LBM lattice used here is D3Q7 which uses regular Cartesian grid. Moreover, they decompose the domain into smaller subdomains and send each to a separate GPU on a cluster, i.e. they use a MPI-GPU hybrid approach. They claim "good performance" but only show a single timing without any reference point to measure any kind of speed up! Unsurprisingly, their method seems to considerably suffer from GPU and MPI communication overhead (~3-4 comm./comput. ratio).

8. [**Higher-order CFD and interface tracking methods on highly-Parallel MPI and GPU systems**][Appleyard2011]
	Authors present a comparison between a purely MPI and a hybrid MPI-GPU parallelization of high-order (3rd, 5th, 7th, and 9th order) level-set methods in 3D and show that the hybrid MPI-GPU method can reach greater or equal performance on 4 GPUs compared to a 256-core, purely MPI implementation. This turns out to be mainly due to the GPU having higher memory bandwidth. Moreover, they report higher performance for higher order methods, presumably due to higher FLOPs intensity. They do not report actual number for speedup but their curves seem close to linear speedup. Finally, they do not go into the details of discretization but it is a finite difference method on Cartesian grids with 2D decomposition technique.

9. [**Adaptive multi-resolution method for compressible multi-phase flows with sharp interface model and pyramid data structure**][Han2014]
	Authors propose an adaptive compressible multiphase method which uses level-set method to track the location of interface. They also make use of what they call "pyramid data structure" which just seems to be the Quadtree! They use task-based parallelism on shared memory machines in which computations within coarse blocks are treated as separate tasks and carried on by individual threads. Their algorithm is only reported in 2D and a single scaling study is performed up to 16 cores yielding 10X speedup.
	
10. [**Fast Marching Methods -- Parallel Implementation and Analysis (Ph.D. Thesis)**][Tugurlan2008]
	This thesis describes a parallelization of the Fast Marching Method (FMM) using domain decomposition and MPI. Author presents pretty good (linear) scaling up to about 50 processors in 2D even though the number of iterations to convergence grows as the number of processors is increased.

11. [**Parallel re-initialization of level set functions on distributed unstructured tetrahedral grids**][Fortmeier2011]
	Authors use a new geometrical approach to the reinitialization problem in which distance function is calculated by directly computing the Euclidean distance to the interface. This is done in a several pass approach. Initially, they compute the distance to just the vertices of elements that are cut by the interface. Next, they compute the projection points of these vertices onto the interface. Finally for far away vertices, they compute the minimum distance of vertices to the set of projected and close vertices. This is achieved by using a k-d tree. This k-d tree, however, is built in serial and uses ALL the points from all processors which requires a global all-to-all communication. They show scaling for their algorithm up to 128 processors. For larger problems (~16M) they achieve good scalability but only because the set of projected and close by points are quite small for their problem. They show test cases where this is not the case which, unsurprisingly, results in really bad speed up (~14 on 64 processors).

12. [**Hybrid Distributed-/Shared-Memory Parallelization For Re-Initializing Level Set Functions**][Fortmeier2010]
    Authors in this paper take their [MPI code][Fortmeier2011] (described above) and adapt it so that it uses OpenMP for on-node parallelism and MPI for off-node parallelism. They find that better results may be obtained by using OpenMP-MPI approach than a pure MPI which they attribute to reduction in the network traffic due to less MPI processes.

13. [**An adaptive domain-decomposition technique for parallelization of the fast marching method**][Breub2011]
    The authors present a novel technique for parallelization of the FMM using what they call "adaptive domain decomposition". The idea here is instead of statically partitioning the domain, as in traditional methods, they let all threads have access to the whole shared data and run a local FMM on the whole domain but for only starting from a segment of the interface. This way, threads prevent each other from further expanding their FMM's "near" node list by rewriting a smaller value that originates from another a closer interface segment.  

14. [**Parallel Implementations of the Fast Sweeping Method**][Zhao2007]
    This is the classic paper of Zhao in which he describes the domain decomposition strategy for FSM. The idea is very simple in that a regular domain decomposition is used and inside each block and after 4 iterations, ghost nodes data are updated and the FSM is repeated until convergence. No speedup information is provided, likely because they are poor as increase in number of domains increases the number of iterations to converge.

15. [**A Parallel Fast Sweeping Method for the Eikonal Equation**][Detrixhe2013]
    This is Mile's paper in which FSM is parallelized by sweeping in diagonal directions instead of regular Cartesian ones. This is useful since all nodes on a diagonal are completely independent of each other which allows for simultaneous update. He compares the method to that of Zhao and reports good scaling up to 32 threads.

16. [**A Fast Iterative Method for the Eikonal Equation**][Jeong2008]
    Authors combine ideas from FSM and FMM method to arrive at a parallel algorithm which they call Fast Iterative Method (FIM). Basically the idea is similar to FMM in that one still manages a list of "active" or "near" nodes which are to be computed but instead of sorting them and only update the smallest value, they iteratively update ALL nodes in the list and allow concurrent addition and removal of nodes. Thus the parallelization comes from concurrent calculation on the list. They implement the method on GPU and compare their results to FSM, FMM, and GMM (Group Marching Method). Overall they report nice speedups compared to FMM -- 100X for the simplest problem (i.e. when speed function is constant) all the way to 6X for the hardest problem (i.e. a maze-like problem in which characteristics change directions frequently).

17. [**A New Massive Parallel Fast-Marching Algorithm For Signed Distance Computations With Respect To Complex Triangulated Surfaces**][Croce2014]
    The idea presented here is to first compute the distance for all the ghost nodes of all processors and then use that as a boundary condition and run FMM inside each processor. To compute the distance for the ghost nodes, they literally compute the Euclidean distance the cost of which increases with number of processors. They do not discuss in details how they compute this distance for triangulated surfaces since they claim they also distribute the surface in which case this initialization step is non trivial. In any case they claim good speed up but their results are actually quite horrible! (5% at 2048)

18. [**A Parallel Heap-Cell Method for Eikonal Equations**][Chacon2013]
	This is the parallel version of the [heap-cell article][Chacon2012] described above. Basically the idea is the same, i.e. use FMM like method on a coarse grid to determine the dependency and inside each "macro-cell" solve a local Eikonal equation using LSM (Locking Sweeping Method). However, this is all done on a shared-memory system in which all heap cells of the macro-mesh are allowed to be updated simultaneously. They report decent speedups even for complicated problems up to 32 cores. Implementation is done using OpenMP.  

19. [**Parallel Algorithms for Semi-Lagrangian Advection**][Malevsky1997]
	The authors are not entirely clear but it seems that their method is based on computing a global B-spline interpolation using domain decomposition. In a sense it seems that they find approximation to the global inverse matrix for the B-spline interplant which is then used to interpolate the solution at any point. Its not clear if this "any point" refers to any point within this processor (logically this one makes sense to me) or just any global point (does not makes sense). Confusion is cause since they report solutions up to cfl ~ 8 and do not mention if they need to do anything special if a point enters a remote processor when backtracking (although they mention some approach others have taken …). Also no mention of speedup and/or how many processors this was run on. 

20. [**Massively parallel semi-Lagrangian advection**][Thomas1995]
	Authors describe a 2D, uniform Cartesian grid, parallel semi-lagrangian method for advection of passive scalar fields for atmospheric studies. Their method uses local cubic interpolations on each cell and parallelization is based on domain decomposition with a fixed ghost layer. They acknowledge that this limits the maximum allowable CFL number (they use $\text{CFL} \le 2$). Nonetheless they report reasonably good speedup for up to 128 processors for sufficiently large problems (1280X640 grid point).

21. [**High-performance high-resolution semi-Lagrangian tracer transport on a sphere**][White2011]
	Authors present a semi-lagrangian method for atmospheric  application. Their method is used for solving  2D transport equations of passive scalar on a sphere using the cubed-sphere cartesian grids. Parallelization is achieved through domain decomposition and MPI. Unlike previous paper, their algorithm allows for arbitrarily large CFL numbers by manage the interpolation in parallel pretty similar to what we do, i.e. sending list of points to other processors for interpolation. They report nice scaling for up to 1000 for a single marker and 10,000 for 100 markers when the CFL is fixed. Unfortunately they do not mention the CFL number for this experiment but seems to be ~10. However, since they are using square domains this means they already know which processors need to communicate which is much simpler than our case. They do report weak scaling for when CFL is increased (i.e. increase in spatial resolution with a fixed time step). Here they see degradation of their algorithm which they attribute to increase in communication volume and number of processors each processor needs to talk to. They do not mention how they figure out the communication topology if a CFL is too large and could result in interpolating from a remote processors (i.e. a processor that is not a direct neighbor)

22. [**Parallel Domain Decomposition Methods with Mixed Order Discretization for Fully Implicit Solution of Tracer Transport Problems on the Cubed-Sphere**][Yang2014]
	Authors present a fully implicit method for the solution of linear transport equation on cubed-sphere for atmospheric applications. The idea is to discretize the transport equation using 2nd-order LF method on the global mesh and 1st order LF method as an additive Schwarz preconditioner. The resulting linear (or nonlinear when they include flux limiters) is solved with GMRES using PETSc. Their method shows decent scaling up to 3072 processors. Also since the method is fully implicit they can use unrestricted CFL numbers, although very large CFL numbers in purely Eulerian methods is problematic and causes excessive numerical diffusion as seen in one of their tables. Nonetheless if CFL not too large, this seems like an interesting alternative to SL method since it has much more regular communication pattern.

23. [**Design and Performance of a Scalable Parallel Community Climate Model**][Drake1995]
	Authors present a climate model which includes development of a parallel semi-lagrangian method, among other components such as parallel FFT. They use a 2D longitude-latitude for this purpose and utilize domain-decomposition technique for parallelization. They note that the parallelization of semi-lagrangian is tricky since departure points could potentially lie far apart from the processors, especially close to the poles. To solve this issue they propose a "dynamic ghost layer" approach in which the location of departure points from previous iterations are used to obtain the depth of ghost layer. Also this depth could potentially be different for each boundary node. They report decent speedups on two separate machines (up to 128 on a IBM machine and 1024 on another) which look good although this timing is not exclusively for SL and also contains other parts of their solver such as FFT.

24. [**An Adaptive Semi-Lagrangian Advection Scheme and Its Parallelization**][Behrens1995] 
	Authors present a 2D parallel semi-lagrangian method on unstructured grids using FEM.At each step nodes are equally distributed among processors to balance computational load. Unlike usual domain-decomposition, authors use what they call a "virtual shared memory" environment which allows individual processors to have direct access to the memory of another processor thus eliminating manual and direct send/recv operations. Although this is not done via MPI, it looks like one-sided communication model of MPI-2 standard such as MPI_Put and MPI_Get. They report good scalability up to 26 processors (~90%)
	

25. [**A Patchy Dynamic Programming Scheme for a Class of Hamilton-Jacobi-Bellman Equations**][Cacace2011]


26. [**Fast Iterative Method in Solving Eikonal Equations : a Multi-Level Parallel Approach**][Dang2014]


27. [**A domain decomposition parallelization of the Fast Marching Method**][Hermann2003]


28. [**Parallel algorithms for approximation of distance maps on parametric surfaces**][Weber2008]


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
[Lauterbach2009]: http://luebke.us/publications/eg09.pdf
[Karras2012]: https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf
[Garanzha2011]: http://dl.acm.org/citation.cfm?id=2018333
[Malevsky1997]: http://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0363(19970830)25:4%3C455::AID-FLD572%3E3.0.CO;2-H/abstract
[Thomas1995]: http://www.sciencedirect.com/science/article/pii/0928486995000334
[White2011]: http://www.sciencedirect.com/science/article/pii/S0021999111003123
[Yang2014]: http://link.springer.com/article/10.1007/s10915-014-9828-y
[Drake1995]: http://www.sciencedirect.com/science/article/pii/0167819196800019
[Behrens1995]: http://journals.ametsoc.org/doi/abs/10.1175/1520-0493(1996)124%3C2386:AASLAS%3E2.0.CO%3B2
[Fortmeier2011]: http://www.sciencedirect.com/science/article/pii/S0021999111000878
[Hermann2003]: http://ctr.stanford.edu/ResBriefs03/herrmann1.pdf
[Weber2008]: http://dl.acm.org/citation.cfm?doid=1409625.1409626
[Croce2014]: http://icsweb.inf.unisi.ch/preprints/preprints/file201314.pdf