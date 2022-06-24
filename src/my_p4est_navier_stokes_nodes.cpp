class my_p4est_navier_stokes_nodes_t
{
  my_p4est_navier_stokes_nodes_t(const my_p4est_navier_stokes_nodes_t& other);
  my_p4est_navier_stokes_nodes_t& operator = (const my_p4est_navier_stokes_nodes_t& other);

public:
  my_p4est_navier_stokes_nodes_t(my_p4est_node_neighbors_t *node_neighbors, int num_comps);
  ~my_p4est_navier_stokes_nodes_t();

private:
  PetscErrorCode ierr;
  my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  // p4est objects
  p4est_t           *p4est_;
  p4est_nodes_t     *nodes_;
  p4est_ghost_t     *ghost_;
  my_p4est_brick_t  *myb_;

  my_p4est_interpolation_nodes_t interp_;
  my_p4est_interpolation_nodes_t interp_bc_points;

  //Geometry
  vec_and_ptr_t phi;
  // probably another phi


}
