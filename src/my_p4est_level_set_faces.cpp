#ifdef P4_TO_P8
#include "my_p8est_level_set_faces.h"
#include <src/point3.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_refine_coarsen.h>
#else
#include "my_p4est_level_set_faces.h"
#include <src/point2.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_refine_coarsen.h>
#endif

#include "petsc_compatibility.h"
#include <petsclog.h>

#undef MAX
#undef MIN

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_level_set_faces_extend_over_interface;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif



#ifdef P4_TO_P8
void my_p4est_level_set_faces_t::extend_Over_Interface( Vec phi, Vec q, BoundaryConditions3D &bc, int dir, Vec face_is_well_defined, int order, int band_to_extend ) const
#else
void my_p4est_level_set_faces_t::extend_Over_Interface( Vec phi, Vec q, BoundaryConditions2D &bc, int dir, Vec face_is_well_defined, int order, int band_to_extend ) const
#endif
{
#ifdef CASL_THROWS
  if(bc.interfaceType()==NOINTERFACE) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: no interface defined in the boundary condition ... needs to be dirichlet or neumann.");
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: invalid order. Choose 0, 1 or 2");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_faces_extend_over_interface, phi, q, 0, 0); CHKERRXX(ierr);

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  /* first compute the derivatives of phi */
  Vec phi_x;
  ierr = VecDuplicate(phi, &phi_x); CHKERRXX(ierr);
  double *phi_x_p;
  ierr = VecGetArray(phi_x, &phi_x_p); CHKERRXX(ierr);
  Vec phi_y;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_y); CHKERRXX(ierr);
  double *phi_y_p;
  ierr = VecGetArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec phi_z;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_z); CHKERRXX(ierr);
  double *phi_z_p;
  ierr = VecGetArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    phi_x_p[n] = qnnn.dx_central(phi_p);
    phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    phi_z_p[n] = qnnn.dz_central(phi_p);
#endif
  }

  ierr = VecGhostUpdateBegin(phi_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(phi_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGhostUpdateBegin(phi_z, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    phi_x_p[n] = qnnn.dx_central(phi_p);
    phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    phi_z_p[n] = qnnn.dz_central(phi_p);
#endif
  }

  ierr = VecGhostUpdateEnd(phi_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGhostUpdateEnd(phi_z, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_x, &phi_x_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif

  my_p4est_interpolation_nodes_t interp_phi  (ngbd_n); interp_phi  .set_input(phi  , linear);
  my_p4est_interpolation_nodes_t interp_phi_x(ngbd_n); interp_phi_x.set_input(phi_x, linear);
  my_p4est_interpolation_nodes_t interp_phi_y(ngbd_n); interp_phi_y.set_input(phi_y, linear);
#ifdef P4_TO_P8
  my_p4est_interpolation_nodes_t interp_phi_z(ngbd_n); interp_phi_z.set_input(phi_z, linear);
#endif

  Vec bc_interface_values;
  ierr = VecDuplicate(phi, &bc_interface_values); CHKERRXX(ierr);
  double *bc_p;
  ierr = VecGetArray(bc_interface_values, &bc_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
#ifdef P4_TO_P8
    bc_p[n] = bc.interfaceValue(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes), node_z_fr_n(n, p4est, nodes));
#else
    bc_p[n] = bc.interfaceValue(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes));
#endif
  }
  ierr = VecGhostUpdateBegin(bc_interface_values, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
#ifdef P4_TO_P8
    bc_p[n] = bc.interfaceValue(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes), node_z_fr_n(n, p4est, nodes));
#else
    bc_p[n] = bc.interfaceValue(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes));
#endif
  }
  ierr = VecGhostUpdateEnd(bc_interface_values, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(bc_interface_values, &bc_p); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp0(ngbd_n); interp0.set_input(bc_interface_values, linear);
  my_p4est_interpolation_faces_t interp1(ngbd_n, faces); interp1.set_input(q, dir, 2, face_is_well_defined);
  my_p4est_interpolation_faces_t interp2(ngbd_n, faces); interp2.set_input(q, dir, 2, face_is_well_defined);

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif

#ifdef P4_TO_P8
  double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
  double diag = sqrt(dx*dx + dy*dy);
#endif

  std::vector<double> q0;
  std::vector<double> q1;
  std::vector<double> q2;

  q0.resize(faces->num_local[dir]);
  if(order >= 1 || (order==0 && bc.interfaceType()==NEUMANN)) q1.resize(faces->num_local[dir]);
  if(order >= 2)                                              q2.resize(faces->num_local[dir]);

  const PetscScalar *face_is_well_defined_p;
  ierr = VecGetArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  /* now buffer the interpolation points */
  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
  {
    if(!face_is_well_defined_p[f_idx])
    {
      double x = faces->x_fr_f(f_idx, dir);
      double y = faces->y_fr_f(f_idx, dir);

#ifdef P4_TO_P8
      double z = faces->z_fr_f(f_idx, dir);
      double phi_f = interp_phi(x,y,z);
      Point3 grad_phi(interp_phi_x(x,y,z), interp_phi_y(x,y,z), interp_phi_z(x,y,z));
#else
      double phi_f = interp_phi(x,y);
      Point2 grad_phi(interp_phi_x(x,y), interp_phi_y(x,y));
#endif

      if(phi_f<band_to_extend*diag && grad_phi.norm_L2()>EPS)
      {
        grad_phi /= grad_phi.norm_L2();

        double xyz_i[] =
        {
          x - grad_phi.x*phi_f,
          y - grad_phi.y*phi_f
  #ifdef P4_TO_P8
          ,z - grad_phi.z*phi_f
  #endif
        };

        interp0.add_point(f_idx, xyz_i);

        if(order >= 1 || (order==0 && bc.interfaceType()==NEUMANN))
        {
          double xyz_ [] =
          {
            x - grad_phi.x * (2*diag + phi_f),
            y - grad_phi.y * (2*diag + phi_f)
  #ifdef P4_TO_P8
            , z - grad_phi.z * (2*diag + phi_f)
  #endif
          };
          interp1.add_point(f_idx, xyz_);
        }

        if(order >= 2)
        {
          double xyz_ [] =
          {
            x - grad_phi.x * (3*diag + phi_f),
            y - grad_phi.y * (3*diag + phi_f)
  #ifdef P4_TO_P8
            , z - grad_phi.z * (3*diag + phi_f)
  #endif
          };
          interp2.add_point(f_idx, xyz_);
        }
      }
    }
  }

  interp0.interpolate(q0.data());
  interp1.interpolate(q1.data());
  interp2.interpolate(q2.data());

  /* now compute the extrapolated values */
  double *q_p;
  ierr = VecGetArray(q, &q_p); CHKERRXX(ierr);
  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
  {
    if(!face_is_well_defined_p[f_idx])
    {
      double x = faces->x_fr_f(f_idx, dir);
      double y = faces->y_fr_f(f_idx, dir);

#ifdef P4_TO_P8
      double z = faces->z_fr_f(f_idx, dir);
      double phi_f = interp_phi(x,y,z);
      Point3 grad_phi(interp_phi_x(x,y,z), interp_phi_y(x,y,z), interp_phi_z(x,y,z));
#else
      double phi_f = interp_phi(x,y);
      Point2 grad_phi(interp_phi_x(x,y), interp_phi_y(x,y));
#endif

      if(phi_f<band_to_extend*diag && grad_phi.norm_L2()>EPS)
      {
        grad_phi /= grad_phi.norm_L2();

        if(order==0)
        {
          if(bc.interfaceType()==DIRICHLET)
            q_p[f_idx] = q0[f_idx];
          else /* interface neumann */
            q_p[f_idx] = q1[f_idx];
        }

        else if(order==1)
        {
          if(bc.interfaceType()==DIRICHLET)
          {
            double dif01 = (q1[f_idx] - q0[f_idx])/(2*diag - 0);
            q_p[f_idx] = q0[f_idx] + (-phi_f - 0) * dif01;
          }
          else /* interface Neumann */
          {
            double dif01 = -q0[f_idx];
            q_p[f_idx] = q1[f_idx] + (-phi_f - 2*diag) * dif01;
          }
        }

        else if(order==2)
        {
          if(bc.interfaceType()==DIRICHLET)
          {
            double dif01  = (q1[f_idx] - q0[f_idx]) / (2*diag);
            double dif12  = (q2[f_idx] - q1[f_idx]) / (diag);
            double dif012 = (dif12 - dif01) / (3*diag);
            q_p[f_idx] = q0[f_idx] + (-phi_f - 0) * dif01 + (-phi_f - 0)*(-phi_f - 2*diag) * dif012;
          }
          else if (bc.interfaceType() == NEUMANN) /* interface Neumann */
          {
            double x1 = 2*diag;
            double x2 = 3*diag;

            double b = -q0[f_idx];
            double a = (q2[f_idx] - q1[f_idx] + b*(x1 - x2)) / (x2*x2 - x1*x1);
            double c = q1[f_idx] - a*x1*x1 - b*x1;

            double x = -phi_f;
            q_p[f_idx] = a*x*x + b*x + c;
          }
        }
      }
      else
        q_p[f_idx] = 0;
    }
  }

  ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  ierr = VecDestroy(phi_x); CHKERRXX(ierr);
  ierr = VecDestroy(phi_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(phi_z); CHKERRXX(ierr);
#endif

  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  ierr = VecDestroy(bc_interface_values); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(q, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (q, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_faces_extend_over_interface, phi, q, 0, 0); CHKERRXX(ierr);
}
