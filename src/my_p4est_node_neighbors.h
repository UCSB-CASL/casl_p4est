#ifndef MY_P4EST_NODE_NEIGHBORS_H
#define MY_P4EST_NODE_NEIGHBORS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_quad_neighbor_nodes_of_node.h>
#include <src/my_p8est_hierarchy.h>
#include <p8est_bits.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_quad_neighbor_nodes_of_node.h>
#include <src/my_p4est_hierarchy.h>
#include <p4est_bits.h>
#endif
#include <vector>
#include <sstream>
#include <iostream>



class my_p4est_node_neighbors_t
{
    friend class PoissonSolverNodeBase;
    friend class PoissonSolverCellBase;
    friend class InterpolatingFunctionNodeBase;
    friend class my_p4est_level_set;

    /**
     * Initialize the QuadNeighborNodeOfNode information
     */
public:
    void init_neighbors();

    void init_neighbors_with_periodicities();

    my_p4est_hierarchy_t *hierarchy;
    p4est_t *p4est;
    p4est_ghost_t *ghost;
    p4est_nodes_t *nodes;
    my_p4est_brick_t *myb;



    std::vector< quad_neighbor_nodes_of_node_t > neighbors;
    std::vector<p4est_locidx_t> layer_nodes;
    std::vector<p4est_locidx_t> local_nodes;



public:

    int max_distance;
    //NOTE:: this function is usefuul only for cartesian geometries where the geometry is simple
    // it is used for periodicities purposes only
    // in case of complex geometries such as mobius disks or taurus another approach should be used
    inline void compute_max_distance()
    {
        this->max_distance=0;
        double minx=0; double maxx=0;
        for(int i=0;i<this->p4est->connectivity->num_vertices;i++)
        {
            if(this->p4est->connectivity->vertices[3*i+0]<minx)
                minx=this->p4est->connectivity->vertices[3*i+0];
            if(this->p4est->connectivity->vertices[3*i+0]>maxx)
                maxx=this->p4est->connectivity->vertices[3*i+0];
        }
        this->max_distance=(int)(maxx-minx);
    }


    std::string IO_path;//="/Users/gaddielouaknin/p4estLocal/";
    inline std::string convert2FullPath(std::string file_name)
    {
        std::stringstream oss;
        std::string mystr;
        oss <<this->IO_path <<file_name;
        mystr=oss.str();
        return mystr;
    }

    bool px,py,pz;
    my_p4est_node_neighbors_t( my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_,bool periodic_xyz,bool px=PETSC_FALSE,bool py=PETSC_FALSE,bool pz=PETSC_FALSE)

    {

         this->hierarchy=hierarchy_;
         this->p4est=hierarchy_->p4est;
         this->ghost=hierarchy_->ghost;
         this->nodes=nodes_;
         this->myb=hierarchy_->myb;
         this->neighbors.resize(nodes_->num_owned_indeps);
        std::cout<<" compute max distance "<<std::endl;
        this->compute_max_distance();

        std::cout<<" inite neighbors "<<std::endl;

        if(periodic_xyz)
        {
            std::cout<<" inite neighbors with periodicities"<<std::endl;
            this->px=PETSC_TRUE; this->py=PETSC_TRUE; this->pz=PETSC_TRUE;
            this->init_neighbors_with_periodicities();
        }
        else
        {
            std::cout<<" inite neighbors without periodicities "<<std::endl;
            if(!px&& !py &&!pz)
            {
                this->init_neighbors();
            }
            else
            {
                this->px=px;
                this->py=py;
                this->pz=pz;
                this->init_neighbors_with_periodicities();
            }
        }

        /* compute the layer and local nodes.
     * layer_nodes: This is a list of indices for nodes in the local range on this
     * processor (i.e. 0<= i < nodes->num_owned_indeps) that are taged as ghost
     * on at least another processor
     * local_nodes: This is a list of indices for nodes in the local range on this
     * processor that are not included in the layer_nodes
     *
     * With this subdivision, ANY computation on the local nodes should be decomposed
     * into four stages:
     * 1) do computation on the layer nodes
     * 2) call VecGhostUpdateBegin so that each processor begins sending messages
     * 3) do computation on the local nodes
     * 4) call VecGhostUpdateEnd to finish the update process
     *
     * This will effectively hide the communication steps 2,4 with the computation
     * step 3
     */

        std::cout<<" do whatetever you want"<<std::endl;
        layer_nodes.reserve(nodes->num_owned_shared);
        local_nodes.reserve(nodes->num_owned_indeps - nodes->num_owned_shared);

        for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i){
            p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->offset_owned_indeps);
            ni->pad8 == 0 ? local_nodes.push_back(i) : layer_nodes.push_back(i);
        }

        std::cout<<" finished to initiate neighbors "<<std::endl;
    }

    ~my_p4est_node_neighbors_t()
    {

//        while(!this->neighbors.empty()) {
//               delete this->neighbors.back();
//               this->neighbors.pop_back();
//}

        this->neighbors.clear();
        if(this->neighbors.empty())
        {
            std::cout<<" the vector is empty but its size is "<<this->neighbors.size()<<std::endl;
        }
        else
        {
            std::cout<<" the vector is not empty but its size is "<<this->neighbors.size()<<std::endl;
        }

        this->layer_nodes.clear();
        this->local_nodes.clear();
//        p4est_destroy(this->p4est);
//        p4est_ghost_destroy(this->ghost);
//        p4est_nodes_destroy(this->nodes);

//        delete this->myb;
//        delete this->hierarchy;
    }

    inline const quad_neighbor_nodes_of_node_t& operator[]( p4est_locidx_t n ) const {
#ifdef CASL_THROWS
        if (n<0 || n>=nodes->num_owned_indeps){
            std::ostringstream oss;
            oss << "[ERROR]: Trying to access neighboring nodes of element " << n
                << " in the QNNN structure which is out of bound [0, " << nodes->num_owned_indeps
                << "). This probably means you are trying to acess neighboring nodes"
                   " of a ghost nod. This is not supported." << std::endl;
            throw std::invalid_argument(oss.str());
        }
#endif
        return neighbors[n];
    }

    /**
     * This function is finds the neighboring cell of a node in the given (i,j) direction. The direction must be diagonal
     * for the function to work ! (e.g. (-1,1) ... no cartesian direction!).
     * \param [in] node          a pointer to the node whose neighboring cells are looked for
     * \param [in] i             the x search direction, -1 or 1
     * \param [in] j             the y search direction, -1 or 1
     * \param [out] quad         the index of the found quadrant, in mpirank numbering. To fetch this quadrant from its corresponding tree
     *                           you need to substract the tree quadrant offset. If no quadrant was found, this is set to -1 (e.g. edge of domain)
     * \param [out] nb_tree_idx  the index of the tree in which the quadrant was found
     *
     */
#ifdef P4_TO_P8
    void find_neighbor_cell_of_node( p4est_indep_t *node, char i, char j, char k, p4est_locidx_t& quad_idx, p4est_topidx_t& nb_tree_idx ) const;
#else
    void find_neighbor_cell_of_node( p4est_indep_t *node, char i, char j, p4est_locidx_t& quad_idx, p4est_topidx_t& nb_tree_idx ) const;
#endif

    /*!
   * \brief dxx_central compute dxx_central on all nodes and update the ghosts
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fxx PETSc vector to store the results in. A check is done to ensure they have the same size
   */
    void dxx_central(const Vec f, Vec fxx) const;

    /*!
   * \brief dyy_central compute dyy_central on all nodes and update the ghosts
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fyy PETSc vector to store the results in. A check is done to ensure they have the same size
   */
    void dyy_central(const Vec f, Vec fyy) const;

#ifdef P4_TO_P8
    /*!
   * \brief dzz_central compute dzz_central on all nodes and update the ghosts
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fzz PETSc vector to store the results in. A check is done to ensure they have the same size
   */
    void dzz_central(const Vec f, Vec fzz) const;
#endif

    /*!
   * \brief second_derivatives_central computes both dxx_central and dyy_central at all
   * points. Theoretically this should have a better chance at hiding communications
   * than above calls combined.
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fdd PETSc _BLOCK_ vector to store dxx adn dyy results in.
   * A check is done to ensure it has the same size as f and block size = P4EST_DIM
   */
    void second_derivatives_central(const Vec f, Vec fdd) const;

    /*!
   * \brief second_derivatives_central computes dxx, dyy, and dzz central at all
   * points. Similar to the function above except it use two regular vector in
   * place of a single blocked vector. Easier to use but more expensive in terms
   * of MPI. Also note that fxx, fyy, and fzz cannot be obtained via VecDuplicate as
   * this would share the same VecScatter object and avoid simaltanous update.
   *
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fxx PETSc vector to store the results in. A check is done to ensure they have the same size as f
   * \param [out] fyy PETSc vector to store the results in. A check is done to ensure they have the same size as f
   * \param [out] fzz PETSc vector to store the results in. A check is done to ensure they have the same size as f (only inn 3D)
   */
#ifdef P4_TO_P8
    void second_derivatives_central(const Vec f, Vec fxx, Vec fyy, Vec fzz) const;
#else
    void second_derivatives_central(const Vec f, Vec fxx, Vec fyy) const;
#endif

private:
#ifdef P4_TO_P8
    void dxx_and_dyy_central_using_block(const Vec f, Vec fxx, Vec fyy, Vec fzz) const;
#else
    void second_derivatives_central_using_block(const Vec f, Vec fxx, Vec fyy) const;
#endif
};

#endif /* !MY_P4EST_NODE_NEIGHBORS_H */
