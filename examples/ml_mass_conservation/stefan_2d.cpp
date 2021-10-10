/**
 * Testing rror-correcting neural networks for semi-Lagrangian advection in Stefan problem instances.  There are two
 * tests: Frank-sphere and anisotropic unstable growth due to curvature.
 *
 * Code is based on examples/stefan/main_2d.cpp
 *
 * @cite H. Chen, C. Min, and F. Gibou.  A numerical scheme for the Stefan problem on adaptive Cartesian grids with
 *       supralinear convergence rate.  J. Comput. Phys., 228:5803-5818, 2009.
 *
 * TODO: Migrate the tests to stefan_mls after merging the "develop" and "long_overdue_merge" branches.
 *
 * Author: Luis Ángel (임 영민)
 * Created: September 16, 2021.
 * Updated: October 9, 2021.
 */

#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_semi_lagrangian_ml.h>
#include <src/casl_geometry.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <random>

/* Available options in 2d
 * 0 - frank sphere
 * 1 - a single seed
 */
int test_number;

double k_s = 1;
double k_l = 1;
double L = 1;
const double S0 = 0.5;
double X0 = 0, Y0 = 0;		// Frank-sphere center.

double t_interface = 0;		// Anisotropic growth.
double epsilon_c = 5e-6;
double epsilon_anisotropy = 0.5;
double N_anisotropy = 5;
double theta_0 = 2 * M_PI / 3.0;

double tn;
double dt;

/* error function */
double E1( double x )
{
	const double EULER = 0.5772156649;
	const int MAXIT = 100;
	const double FPMIN = 1.0e-20;

	int i, ii;
	double a, b, c, d, del, fact, h, psi, ans = 0;

	int n = 1;
	int nm1 = 0;

	if( x > 1.0 )
	{        /* Lentz's algorithm */
		b = x + n;
		c = 1.0 / FPMIN;
		d = 1.0 / b;
		h = d;
		for( i = 1; i <= MAXIT; i++ )
		{
			a = -i * (nm1 + i);
			b += 2.0;
			d = 1.0 / (a * d + b);    /* Denominators cannot be zero */
			c = b + a / c;
			del = c * d;
			h *= del;
			if( fabs( del - 1.0 ) < EPS )
			{
				ans = h * exp( -x );
				return ans;
			}
		}
		printf( "Continued fraction failed in expint\n" );
	}
	else
	{
		ans = (nm1 != 0? 1.0 / nm1 : -log( x ) - EULER);    /* Set first term */
		fact = 1.0;
		for( i = 1; i <= MAXIT; i++ )
		{
			fact *= -x / i;
			if( i != nm1 ) del = -fact / (i - nm1);
			else
			{
				psi = -EULER;  /* Compute psi(n) */
				for( ii = 1; ii <= nm1; ii++ ) psi += 1.0 / ii;
				del = fact * (-log( x ) + psi);
			}
			ans += del;
			if( fabs( del ) < fabs( ans ) * EPS ) return ans;
		}
		printf( "series failed in expint\n" );
	}

	return ans;
}

double F( double s )
{
	return E1( .25 * s * s );
}

double dF( double s )
{
	return -0.5 * s * exp( -s * s / 4 ) / (s * s / 4);
}

double t_frank_sphere( double x, double y, double t )
{
	double s = sqrt( SQR( x - X0 ) + SQR( y - Y0 ) ) / sqrt( t );
	double Tinf = .5 * S0 * F( S0 ) / dF( S0 );
	if( s <= S0 ) return 0;
	else return Tinf * (1 - F( s ) / F( S0 ));
}

struct level_set_t : CF_2
{
	geom::Star *starPtr = nullptr;

	level_set_t()
	{
		lip = 1.2;
		if( test_number == 1 )
			starPtr = new geom::Star( 0.02, 0.09, 3 );    // This star has curvature varying between -25 to +25.
	}

	double operator()( double x, double y ) const override
	{
		switch( test_number )
		{
			case 0:
				return sqrt( SQR( x - X0 ) + SQR( y - Y0 )) - S0 * sqrt( tn );
			case 1:
				return starPtr->operator()( x, y );
			default:
				throw std::invalid_argument( "[ERROR]: choose a valid test." );
		}
	}

	~level_set_t() override
	{
		delete starPtr;
	}
};

struct init_temperature_l_t:CF_2
{
	double operator()( double x, double y ) const override
	{
		switch( test_number )
		{
			case 0:
				return t_frank_sphere( x, y, tn );
			case 1:
				return -0.08;
			default:
				throw std::invalid_argument( "[ERROR]: choose a valid test." );
		}
	}
} init_temperature_l;

struct init_temperature_s_t:CF_2
{
	double operator()( double x, double y ) const override
	{
		switch( test_number )
		{
			case 0:
				return t_frank_sphere( x, y, tn );
			case 1:
				return 0;
			default:
				throw std::invalid_argument( "[ERROR]: choose a valid test." );
		}
	}
} init_temperature_s;

struct bc_wall_type_t : WallBC2D
{
	BoundaryConditionType operator()( double, double ) const override
	{
		switch( test_number )
		{
			case 0:
				return DIRICHLET;
			case 1:
				return NEUMANN;
			default:
				throw std::invalid_argument( "[ERROR]: choose a valid test." );
		}
	}
} bc_wall_type;

struct bc_wall_value_t : CF_2
{
	double operator()( double x, double y ) const override
	{
		switch( test_number )
		{
			case 0:
				return t_frank_sphere( x, y, tn + dt );
			case 1:
				return 0;
			default:
				throw std::invalid_argument( "[ERROR]: choose a valid test." );
		}
	}
} bc_wall_value;

struct BCInterfaceValue : CF_2
{
private:
	my_p4est_interpolation_nodes_t interp;
	my_p4est_interpolation_nodes_t interp_phi_x;
	my_p4est_interpolation_nodes_t interp_phi_y;
public:
	BCInterfaceValue( my_p4est_node_neighbors_t *ngbd_,
					  Vec *d_phi, Vec kappa_ )
		: interp( ngbd_ ),
		  interp_phi_x( ngbd_ ),
		  interp_phi_y( ngbd_ )
	{
		interp.set_input( kappa_, linear );
		interp_phi_x.set_input( d_phi[0], linear );
		interp_phi_y.set_input( d_phi[1], linear );
	}

	double operator()( double x, double y ) const override
	{
		/* frank sphere: no surface tension */
		if( test_number == 0 ) return 0;

		double theta = atan2( interp_phi_y( x, y ), interp_phi_x( x, y ));
		return t_interface -
			   epsilon_c * (1. + epsilon_anisotropy * cos( N_anisotropy * (theta + theta_0))) * interp( x, y );
		/* T = -eps_c kappa - eps_v V */
	}
};

void save_VTK( p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_brick_t *brick, Vec phi, Vec phiExact, Vec T_l, Vec T_s,
			   Vec *v, Vec kappa, Vec howUpdated, int compt, const bool& mode, const int& maxRL )
{
	PetscErrorCode ierr;
	const char *out_dir = getenv( "OUT_DIR" );
	if( !out_dir )
		out_dir = (test_number == 0? "stefan/frank_sphere" : "stefan/anisotropic");

	std::ostringstream oss, command;
	oss << out_dir << (mode? "/nnet" : "/num") << maxRL;

	command << "mkdir -p " << oss.str();
	system( command.str().c_str());

	struct stat st{};
	if( stat( oss.str().data(), &st ) != 0 || !S_ISDIR( st.st_mode ))
	{
		ierr = PetscPrintf( p4est->mpicomm, "Trying to save files in %s\n", oss.str().data());
		CHKERRXX( ierr );
		throw std::invalid_argument( "[ERROR]: the directory specified to export vtu images does not exist." );
	}

	oss << "/stefan_"
		<< p4est->mpisize << "_"
		<< brick->nxyztrees[0] << "x"
		<< brick->nxyztrees[1] <<
		"." << compt;

	const double *phi_p, *t_l_p, *t_s_p, *kappa_p, *phiExactPtr, *howUpdatedPtr;
	const double *v_p[P4EST_DIM];

	ierr = VecGetArrayRead( phi, &phi_p );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( T_s, &t_s_p );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( T_l, &t_l_p );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( kappa, &kappa_p );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( phiExact, &phiExactPtr );
	CHKERRXX( ierr );
	if( howUpdated )
	{
		ierr = VecGetArrayRead( howUpdated, &howUpdatedPtr );
		CHKERRXX( ierr );
	}

	for( int dir = 0; dir < P4EST_DIM; ++dir )
	{
		ierr = VecGetArrayRead( v[dir], &v_p[dir] );
		CHKERRXX( ierr );
	}

	/* compute the temperature in the domain */
	Vec t;
	ierr = VecDuplicate( phi, &t );
	CHKERRXX( ierr );
	double *t_p;
	ierr = VecGetArray( t, &t_p );
	CHKERRXX( ierr );
	for( size_t n = 0; n < nodes->indep_nodes.elem_count; ++n )
		t_p[n] = phi_p[n] < 0? t_s_p[n] : t_l_p[n];

	if( test_number == 0 )
	{
		my_p4est_vtk_write_all( p4est, nodes, nullptr, P4EST_TRUE, P4EST_TRUE, 7, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phi_p,
								VTK_POINT_DATA, "phiExact", phiExactPtr,
								VTK_POINT_DATA, "temperature", t_p,
								VTK_POINT_DATA, "vx", v_p[0],
								VTK_POINT_DATA, "vy", v_p[1],
								VTK_POINT_DATA, "kappa", kappa_p,
								VTK_POINT_DATA, "howUpdated", howUpdated? howUpdatedPtr : phi_p );
	}
	else if( test_number == 1 )
	{
		my_p4est_vtk_write_all( p4est, nodes, nullptr, P4EST_TRUE, P4EST_TRUE, 6, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phi_p,
								VTK_POINT_DATA, "temperature", t_p,
								VTK_POINT_DATA, "vx", v_p[0],
								VTK_POINT_DATA, "vy", v_p[1],
								VTK_POINT_DATA, "kappa", kappa_p,
								VTK_POINT_DATA, "howUpdated", howUpdated? howUpdatedPtr : phi_p );
	}
	else
		throw std::invalid_argument( "[ERROR]: choose a valid test." );

	ierr = VecRestoreArray( t, &t_p );
	CHKERRXX( ierr );
	ierr = VecDestroy( t );
	CHKERRXX( ierr );

	if( howUpdated )
	{
		ierr = VecRestoreArrayRead( howUpdated, &howUpdatedPtr );
		CHKERRXX( ierr );
	}
	ierr = VecRestoreArrayRead( phi, &phi_p );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( T_l, &t_l_p );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( T_s, &t_s_p );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( kappa, &kappa_p );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( phiExact, &phiExactPtr );
	CHKERRXX( ierr );
	for( int dir = 0; dir < P4EST_DIM; ++dir )
	{
		ierr = VecRestoreArrayRead( v[dir], &v_p[dir] );
		CHKERRXX( ierr );
	}

	ierr = PetscPrintf( p4est->mpicomm, "Saved in %s, time = %f\n", oss.str().data(), tn );
	CHKERRXX( ierr );
}



void update_p4est( my_p4est_brick_t *brick, p4est_t *&p4est, p4est_ghost_t *&ghost, p4est_nodes_t *&nodes,
				   my_p4est_hierarchy_t *&hierarchy, my_p4est_node_neighbors_t *&ngbd, Vec& phi, Vec normal[P4EST_DIM],
				   Vec v[P4EST_DIM], Vec& t_l, Vec& t_s, const slml::NeuralNetwork *nnet, Vec hk, Vec& howUpdated,
				   const int& iter, const int& REINIT_NUM_ITER, const signed char& MAX_RL, bool& nnetTurn )
{
	PetscErrorCode ierr;

	p4est_t *p4est_np1 = p4est_copy( p4est, P4EST_FALSE );
	p4est_ghost_t *ghost_np1 = my_p4est_ghost_new( p4est_np1, P4EST_CONNECT_FULL );
	p4est_nodes_t *nodes_np1 = my_p4est_nodes_new( p4est_np1, ghost_np1 );

	// Create semi-Lagrangian object and update level-set values.
	slml::SemiLagrangian *mlSemiLagrangian;
	my_p4est_semi_lagrangian_t *numSemiLagrangian;
	if( nnetTurn )    // Use nnet if dt==dx, max|u| <= 1, and it's nnet turn.
	{
		mlSemiLagrangian = new slml::SemiLagrangian( &p4est_np1, &nodes_np1, &ghost_np1, ngbd, phi, false, nnet, iter );
		mlSemiLagrangian->updateP4EST( v, dt, &phi, hk, normal, &howUpdated );
	}
	else
	{
		numSemiLagrangian = new my_p4est_semi_lagrangian_t( &p4est_np1, &nodes_np1, &ghost_np1, ngbd );
		numSemiLagrangian->set_phi_interpolation( interpolation_method::quadratic );
		numSemiLagrangian->set_velo_interpolation( interpolation_method::quadratic );
		numSemiLagrangian->update_p4est( v, dt, phi, nullptr, nullptr, MASS_BAND_HALF_WIDTH );

		ierr = VecDestroy( howUpdated );		// For numerical method, howUpdated flag is zero everywhere.
		CHKERRXX( ierr );
		ierr = VecCreateGhostNodes( p4est_np1, nodes_np1, &howUpdated );
		CHKERRXX( ierr );
	}

	// Interpolate the quantities on the new mesh.
	Vec tnp1_l, tnp1_s;
	ierr = VecDuplicate( phi, &tnp1_l );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi, &tnp1_s );
	CHKERRXX( ierr );
	my_p4est_interpolation_nodes_t interp( ngbd );

	for( p4est_locidx_t n = 0; n < nodes_np1->indep_nodes.elem_count; ++n )
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( n, p4est_np1, nodes_np1, xyz );
		interp.add_point( n, xyz );
	}

	interp.set_input( t_l, interpolation_method::quadratic );
	interp.interpolate( tnp1_l );

	interp.set_input( t_s, interpolation_method::quadratic );
	interp.interpolate( tnp1_s );

	ierr = VecDestroy( t_l );
	CHKERRXX( ierr );
	t_l = tnp1_l;

	ierr = VecDestroy( t_s );
	CHKERRXX( ierr );
	t_s = tnp1_s;

	// Destroy old forest and create new structures.
	p4est_destroy( p4est );
	p4est = p4est_np1;
	p4est_ghost_destroy( ghost );
	ghost = ghost_np1;
	p4est_nodes_destroy( nodes );
	nodes = nodes_np1;

	delete hierarchy;
	hierarchy = new my_p4est_hierarchy_t( p4est, ghost, brick );
	delete ngbd;
	ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
	ngbd->init_neighbors();

	for( int dir = 0; dir < P4EST_DIM; ++dir )		// Allocate new velocity vector components (computed in main body).
	{
		ierr = VecDestroy( v[dir] );
		CHKERRXX( ierr );
		ierr = VecCreateGhostNodes( p4est, nodes, &v[dir] );
		CHKERRXX( ierr );
	}

	// Reinitialize.
	my_p4est_level_set_t ls( ngbd );
	if( nnetTurn )
	{
		const double *phiReadPtr;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Selective reinitialization of level-set function: protect nodes updated with the nnet whose level-set
		// value is negative and are immediately next to Gamma^np1.
		Vec mask;
		ierr = VecCreateGhostNodes( p4est, nodes, &mask );		// Mask vector to flag updatable nodes.
		CHKERRXX( ierr );

		double *howUpdatedPtr;
		ierr = VecGetArray( howUpdated, &howUpdatedPtr );
		CHKERRXX( ierr );

		double *maskPtr;
		ierr = VecGetArray( mask, &maskPtr );
		CHKERRXX( ierr );

		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )	// No need to check all independent nodes.
			maskPtr[n] = 1;												// Initially, all are 1 => updatable.

		NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd, MAX_RL );
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		int numMaskedNodes = 0;
		for( const auto& n : indices )			// Now, check only points next to Gamma^np1.
		{
			if( howUpdatedPtr[n] == 1 && phiReadPtr[n] <= 0 )
			{
				numMaskedNodes++;
				maskPtr[n] = 0;					// 0 => nonupdatable.
				howUpdatedPtr[n] = 2;
			}
			else
				maskPtr[n] = 1;
		}

		ierr = VecRestoreArray( mask, &maskPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( howUpdated, &howUpdatedPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		ls.reinitialize_2nd_order_with_mask( phi, mask, numMaskedNodes, REINIT_NUM_ITER );

		ierr = VecDestroy( mask );
		CHKERRXX( ierr );
	}
	else
	{
		ls.reinitialize_2nd_order( phi, REINIT_NUM_ITER );
	}

//	ls.perturb_level_set_function( phi, EPS );

	// Destroy semi-Lagrangian objects and switch turns.
	if( nnetTurn )
	{
		delete mlSemiLagrangian;
		nnetTurn = false;			// Done using nnet -- enable it back after an iteration with numerical computation.
	}
	else
	{
		delete numSemiLagrangian;
		nnetTurn = true;			// Attempt to set chance for nnet in next round.  It'll be confirmed if conditions are met.
	}
}


void compute_normal_and_curvature( my_p4est_node_neighbors_t *ngbd, Vec phi, Vec *normal, Vec kappa, Vec& hk, const double& h )
{
	PetscErrorCode ierr;

	// Shortcuts.
	const p4est_t *p4est = ngbd->get_p4est();
	const p4est_nodes_t *nodes = ngbd->get_nodes();

	// Prepare output parallel vector with dimensionless curvature values.
	ierr = hk? VecDestroy( hk ) : 0;
	CHKERRXX( ierr );
	ierr = VecCreateGhostNodes( p4est, nodes, &hk );			// By default, all values are zero.
	CHKERRXX( ierr );

	for(int dim = 0; dim < P4EST_DIM; dim++ )
	{
		ierr = normal[dim]? VecDestroy( normal[dim] ) : 0;
		CHKERRXX( ierr );

		ierr = VecCreateGhostNodes( p4est, nodes, &normal[dim] );
		CHKERRXX( ierr );
	}

	// Compute normals and temporary curvature.
	Vec kappa_tmp;
	ierr = VecDuplicate( kappa, &kappa_tmp );
	CHKERRXX( ierr );

	compute_normals( *ngbd, phi, normal );
	compute_mean_curvature( *ngbd, normal, kappa_tmp );

	double *hkPtr;
	ierr = VecGetArray( hk, &hkPtr );
	CHKERRXX( ierr );

	const double *kappa_tmpReadPtr;
	ierr = VecGetArrayRead( kappa_tmp, &kappa_tmpReadPtr );
	CHKERRXX( ierr );

	// Scaling curvature by h.
	for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
		hkPtr[n] = h * kappa_tmpReadPtr[n];

	ierr = VecRestoreArrayRead( kappa_tmp, &kappa_tmpReadPtr );
	CHKERRXX( ierr );

	ierr = VecRestoreArray( hk, &hkPtr );
	CHKERRXX( ierr );

	// Extend curvature from interface to all points.
	my_p4est_level_set_t ls( ngbd );
	ls.extend_from_interface_to_whole_domain_TVD( phi, kappa_tmp, kappa );

	ierr = VecDestroy( kappa_tmp );
	CHKERRXX( ierr );
}


void solve_temperature( my_p4est_node_neighbors_t *ngbd, Vec phi_s, Vec phi_l, Vec *d_phi, Vec kappa, Vec t_l, Vec t_s )
{
	BoundaryConditions2D bc;
	BCInterfaceValue bc_interface_value( ngbd, d_phi, kappa );

	bc.setInterfaceType( DIRICHLET );
	bc.setInterfaceValue( bc_interface_value );
	bc.setWallTypes( bc_wall_type );
	bc.setWallValues( bc_wall_value );

	/* solve for the liquid phase */
	my_p4est_poisson_nodes_t solver_l( ngbd );
	solver_l.set_phi( phi_l );
	solver_l.set_mu( k_l * dt );
	solver_l.set_diagonal( 1 );
	solver_l.set_bc( bc );
	solver_l.set_rhs( t_l );

	solver_l.solve( t_l );

	/* solve for the solid phase */
	my_p4est_poisson_nodes_t solver_s( ngbd );
	solver_s.set_phi( phi_s );
	solver_s.set_mu( k_s * dt );
	solver_s.set_diagonal( 1 );
	solver_s.set_bc( bc );
	solver_s.set_rhs( t_s );

	solver_s.solve( t_s );
}


void extend_temperatures_over_interface( my_p4est_node_neighbors_t *ngbd, Vec phi_s, Vec phi_l, Vec t_l, Vec t_s )
{
	my_p4est_level_set_t ls( ngbd );
	ls.extend_Over_Interface_TVD( phi_l, t_l );
	ls.extend_Over_Interface_TVD( phi_s, t_s );
}


void compute_velocity( my_p4est_node_neighbors_t *ngbd, Vec phi, Vec t_l, Vec t_s, Vec *v )
{
	PetscErrorCode ierr;

	double *t_l_p, *t_s_p;
	ierr = VecGetArray( t_l, &t_l_p );
	CHKERRXX( ierr );
	ierr = VecGetArray( t_s, &t_s_p );
	CHKERRXX( ierr );

	Vec jump[P4EST_DIM];
	double *jump_p[P4EST_DIM];
	for( int dir = 0; dir < P4EST_DIM; ++dir )
	{
		ierr = VecDuplicate( v[dir], &jump[dir] );
		CHKERRXX( ierr );
		ierr = VecGetArray( jump[dir], &jump_p[dir] );
		CHKERRXX( ierr );
	}

	quad_neighbor_nodes_of_node_t qnnn{};
	for( size_t i = 0; i < ngbd->get_layer_size(); ++i )
	{
		p4est_locidx_t n = ngbd->get_layer_node( i );
		ngbd->get_neighbors( n, qnnn );
		jump_p[0][n] = (k_s * qnnn.dx_central( t_s_p ) - k_l * qnnn.dx_central( t_l_p )) / L;
		jump_p[1][n] = (k_s * qnnn.dy_central( t_s_p ) - k_l * qnnn.dy_central( t_l_p )) / L;
	}
	for( auto &dir: jump )
	{
		ierr = VecGhostUpdateBegin( dir, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );
	}
	for( size_t i = 0; i < ngbd->get_local_size(); ++i )
	{
		p4est_locidx_t n = ngbd->get_local_node( i );
		ngbd->get_neighbors( n, qnnn );
		jump_p[0][n] = (k_s * qnnn.dx_central( t_s_p ) - k_l * qnnn.dx_central( t_l_p )) / L;
		jump_p[1][n] = (k_s * qnnn.dy_central( t_s_p ) - k_l * qnnn.dy_central( t_l_p )) / L;
	}
	for( auto &dir: jump )
	{
		ierr = VecGhostUpdateEnd( dir, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );
	}

	ierr = VecRestoreArray( t_l, &t_l_p );
	CHKERRXX( ierr );
	ierr = VecRestoreArray( t_s, &t_s_p );
	CHKERRXX( ierr );

	my_p4est_level_set_t ls( ngbd );
	for( int dir = 0; dir < P4EST_DIM; ++dir )
	{
		ierr = VecRestoreArray( jump[dir], &jump_p[dir] );
		CHKERRXX( ierr );
		ls.extend_from_interface_to_whole_domain_TVD( phi, jump[dir], v[dir], 20 );
		ierr = VecDestroy( jump[dir] );
		CHKERRXX( ierr );
	}
}


void check_error_frank_sphere( p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec t_l )
{
	PetscErrorCode ierr;

	// Retrieve grid size data.
	double dxyz[P4EST_DIM];
	double dxyz_min;
	double diag_min;
	get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );

	const double *t_l_p, *phi_p;
	ierr = VecGetArrayRead( phi, &phi_p );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( t_l, &t_l_p );
	CHKERRXX( ierr );

	double err[] = {0, 0};
	double r = S0 * sqrt( tn );
	int numPoints = 0;
	double cumulativeError = 0;

	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n )
	{
		double x = node_x_fr_n( n, p4est, nodes );
		double y = node_y_fr_n( n, p4est, nodes );

		if( ABS( phi_p[n] ) < diag_min )
		{
			double phi_exact = sqrt( SQR( x - X0 ) + SQR( y - Y0 ) ) - r;
			double error = ABS( phi_p[n] - phi_exact );
			err[0] = MAX( err[0], error );
			numPoints++;
			cumulativeError += error;
		}

		if( phi_p[n] > 0 )
			err[1] = MAX( err[1], fabs( t_frank_sphere( x, y, tn ) - t_l_p[n] ) );
	}
	ierr = VecRestoreArrayRead( phi, &phi_p );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( t_l, &t_l_p );
	CHKERRXX( ierr );

	int mpiret;
	mpiret = MPI_Allreduce( MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_MAX, p4est->mpicomm );
	SC_CHECK_MPI( mpiret );
	mpiret = MPI_Allreduce( MPI_IN_PLACE, &numPoints, 1, MPI_INT, MPI_SUM, p4est->mpicomm );		// Total error.
	SC_CHECK_MPI( mpiret );
	mpiret = MPI_Allreduce( MPI_IN_PLACE, &cumulativeError, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm );
	SC_CHECK_MPI( mpiret );

	double l1Error = cumulativeError / numPoints;

	double area = area_in_negative_domain( p4est, nodes, phi );
	double expectedArea = M_PI * SQR( r );
	double massLossPercentage = (1.0 - area / expectedArea) * 100.0;

	ierr = PetscPrintf( p4est->mpicomm, "\n   mean abs error %.3e\n   max abs error %.3e\n   temperature error %.3e\n   area %.3e (expected %.3e, loss %.2f)",
			 			l1Error, err[0], err[1], area, expectedArea, massLossPercentage );
	CHKERRXX( ierr );
}


int main (int argc, char* argv[])
{
	double T_FINAL;						// Simulation final time.
	const int MAX_RL = 6;				// Grid's maximum refinement level.
	const int REINIT_NUM_ITER = 10;		// Number of iterations for level-set renitialization.
	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.

	const int MIN_D = -1;				// Domain minimum and maximum values for each dimension.
	const int MAX_D = -MIN_D;
	const int NUM_TREES_PER_DIM = 2;	// Number of macrocells per dimension.
	const int PERIODICITY = 0;			// Domain periodicity.
	const double BAND = 2; 				// Minimum number of cells around interface.  Must match what was used in training.

	const double H = 1. / (1 << MAX_RL);
	slml::NeuralNetwork nnet( "/Users/youngmin/nnets", H, false );

	std::mt19937 gen; 			// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
	std::uniform_real_distribution<double> uniformDistributionAroundCenter( -H/2.0, +H/2.0 );
	X0 = uniformDistributionAroundCenter( gen );
	Y0 = uniformDistributionAroundCenter( gen );

	mpi_environment_t mpi{};
	mpi.init( argc, argv );

	PetscErrorCode ierr;

	cmdParser cmd;
	cmd.add_option( "save_vtk", "1 to export vtu images, 0 otherwise" );
	cmd.add_option( "save_every_n", "export images every n iterations" );
	cmd.add_option( "max_iter", "maximum number of iterations" );
	cmd.add_option( "mode", "solver mode: 1 for nnet, 0 for numerical" );
	cmd.add_option( "test", "the test to run. Available options are"
							"\t 0 - frank sphere\n"
							"\t 1 - a 3-petal seed\n");
	cmd.parse( argc, argv );

	// TODO: modify simulation options.
	bool save_vtk = cmd.get( "save_vtk", (int)0 );
	int save_every_n = cmd.get( "save_every_n", 2 );
	int max_iter = cmd.get( "max_iter", INT_MAX );
	bool mode = cmd.get( "mode", (int)1 );
	test_number = cmd.get( "test", (int)0 );

	// TODO: modify simulation times per test.
	switch( test_number )
	{
		case 0: k_s=1; k_l=1; tn=0.25; T_FINAL=0.875; break;
		case 1: tn=0; T_FINAL=1.5; break;
		default: throw std::invalid_argument( "[ERROR]: choose a valid test." );
	}

	// Begin simulation.
	parStopWatch watch;
	watch.start();

	ierr = PetscPrintf( mpi.comm(), ">> Began 2D %s test with MAX_RL = %d in %s mode\n",
						!test_number? "FRANK-SPHERE" : "ANISOTROPY", MAX_RL, mode? "NNET" : "NUMERICAL" );
	CHKERRXX( ierr );

	// Create the connectivity object
	p4est_connectivity_t *connectivity;
	my_p4est_brick_t brick;

	int n_xyz[] = {NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM};
	double xyz_min[] = {MIN_D, MIN_D, MIN_D};
	double xyz_max[] = {MAX_D, MAX_D, MAX_D};
	int periodic[P4EST_DIM] = {PERIODICITY};
	connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	p4est_t *p4est;
	p4est_ghost_t *ghost;
	p4est_nodes_t *nodes;

	// Create forest using a level-set as refinement criterion.
	level_set_t level_set;
	p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
	splitting_criteria_cf_and_uniform_band_t lsSplittingCriterion( 1, MAX_RL, &level_set, BAND );
	p4est->user_pointer = &lsSplittingCriterion;

	// Refine and partition forest.
	for( int i = 0; i < MAX_RL; i++ )
	{
		my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
		my_p4est_partition( p4est, P4EST_FALSE, nullptr );
	}

	// Create the ghost (cell) and node structures.
	ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
	nodes = my_p4est_nodes_new( p4est, ghost );

	// Initialize neighbor node structure and hierarchy.
	auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
	auto *ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
	ngbd->init_neighbors();

	// Retrieve grid size data.
	double dxyz[P4EST_DIM];
	double dxyz_min;
	double diag_min;
	get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );

	// Initialize the level-set function.
	Vec phi_s;
	Vec phi_l;
	Vec phiExact;
	Vec kappa, hk;
	Vec d_phi[P4EST_DIM];
	Vec t_l, t_s;
	Vec howUpdated = nullptr;
	ierr = VecCreateGhostNodes( p4est, nodes, &phi_s );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi_s, &phi_l );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi_s, &phiExact );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi_s, &t_l );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi_s, &t_s );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi_s, &kappa );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi_s, &hk );
	CHKERRXX( ierr );

	if( mode )
	{
		ierr = VecCreateGhostNodes( p4est, nodes, &howUpdated );
		CHKERRXX( ierr );
	}

	std::unordered_set<p4est_locidx_t> localUniformIndices;

	sample_cf_on_nodes( p4est, nodes, level_set, phi_s );
	sample_cf_on_nodes( p4est, nodes, init_temperature_l, t_l );
	sample_cf_on_nodes( p4est, nodes, init_temperature_s, t_s );

	if( test_number == 1 )		// Star-shaped level-set needs reinitialization.
	{
		my_p4est_level_set_t ls( ngbd );
		ls.reinitialize_2nd_order( phi_s, REINIT_NUM_ITER );
	}

	double *phi_l_p;
	const double *phi_s_p;
	ierr = VecGetArray( phi_l, &phi_l_p );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( phi_s, &phi_s_p );
	CHKERRXX( ierr );
	for( size_t n = 0; n < nodes->indep_nodes.elem_count; ++n )
		phi_l_p[n] = -phi_s_p[n];
	ierr = VecRestoreArray( phi_l, &phi_l_p );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( phi_s, &phi_s_p );
	CHKERRXX( ierr );

	// Initialize the velocity field.
	Vec v[P4EST_DIM];
	double *v_p[P4EST_DIM];
	for( int dir = 0; dir < P4EST_DIM; ++dir )
	{
		ierr = VecCreateGhostNodes( p4est, nodes, &d_phi[dir] );
		CHKERRXX( ierr );
		ierr = VecDuplicate( d_phi[dir], &v[dir] );
		CHKERRXX( ierr );
	}

	compute_normal_and_curvature( ngbd, phi_s, d_phi, kappa, hk, dxyz_min );
	extend_temperatures_over_interface( ngbd, phi_s, phi_l, t_l, t_s );
	compute_velocity( ngbd, phi_s, t_l, t_s, v );

	// Save the initial state.
	save_VTK( p4est, nodes, &brick, phi_s, phiExact, t_l, t_s, v, kappa, howUpdated, 0, mode, MAX_RL );

	// Loop over time.
	int iter = 0;
	int vtkIdx = 1;
	dt = 0;
	bool nnetTurn = true;	// Indicates if it's nnet's turn (if all conditions hold and comes after a numerical step).
	while( tn < T_FINAL && iter <= max_iter )
	{
		// Compute the time step dt.
		for( int dir = 0; dir < P4EST_DIM; dir++ )
		{
			ierr = VecGetArray( v[dir], &v_p[dir] );
			CHKERRXX( ierr );
		}

		double max_norm_u = 0;
		ierr = VecGetArrayRead( phi_s, &phi_s_p );
		CHKERRXX( ierr );
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n )
		{
			if( ABS( phi_s_p[n] ) < 3 * dxyz_min )
			{
				double unorm = sqrt( SQR( v_p[0][n] ) + SQR( v_p[1][n] ) );
				max_norm_u = MAX( max_norm_u, unorm );
			}
		}
		ierr = VecRestoreArrayRead( phi_s, &phi_s_p );
		CHKERRXX( ierr );

		for( int dir = 0; dir < P4EST_DIM; ++dir )
		{
			ierr = VecRestoreArray( v[dir], &v_p[dir] );
			CHKERRXX( ierr );
		}
		MPI_Allreduce( MPI_IN_PLACE, &max_norm_u, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm );

		// To enable the nnet, dt should equal dx.
		dt = MIN( 1., 1 / max_norm_u ) * CFL * dxyz_min;

		if( tn + dt > T_FINAL )
			dt = T_FINAL - tn;

		if( mode && nnetTurn )
		{
			if( ABS( dt - dxyz_min ) > PETSC_MACHINE_EPSILON || max_norm_u > 1 || max_norm_u < PETSC_MACHINE_EPSILON )	// It's nnet's turn, but can't use it.
				nnetTurn = false;
		}
		else
			nnetTurn = false;

		std::cout << (nnetTurn? "NNet" : "Numerical") << ", Max norm u = " << max_norm_u << ", dt = " << dt << std::endl;

		// Contruct the mesh at time t^np1
		update_p4est( &brick, p4est, ghost, nodes, hierarchy, ngbd, phi_s, d_phi, v, t_l, t_s, &nnet, hk, howUpdated, iter, REINIT_NUM_ITER, MAX_RL, nnetTurn );

		ierr = VecDestroy( phi_l );
		ierr = VecDuplicate( phi_s, &phi_l );
		CHKERRXX( ierr );
		ierr = VecGetArray( phi_l, &phi_l_p );
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( phi_s, &phi_s_p );
		CHKERRXX( ierr );
		for( size_t n = 0; n < nodes->indep_nodes.elem_count; ++n )
			phi_l_p[n] = -phi_s_p[n];
		ierr = VecRestoreArray( phi_l, &phi_l_p );
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( phi_s, &phi_s_p );
		CHKERRXX( ierr );

		// Compute the curvature for boundary conditions.
		ierr = VecDestroy( kappa );
		CHKERRXX( ierr );
		ierr = VecDuplicate( phi_s, &kappa );
		CHKERRXX( ierr );
		compute_normal_and_curvature( ngbd, phi_s, d_phi, kappa, hk, dxyz_min );

		// Solve for the temperatures.
		solve_temperature( ngbd, phi_s, phi_l, d_phi, kappa, t_l, t_s );

		// Extend the temperature over the interface.
		extend_temperatures_over_interface( ngbd, phi_s, phi_l, t_l, t_s );

		// Compute the velocity of the interface.
		compute_velocity( ngbd, phi_s, t_l, t_s, v );

		tn += dt;

		// Exact level-set function for Frank-sphere problem.
		if( test_number == 0 )
		{
			ierr = VecDestroy( phiExact );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( p4est, nodes, &phiExact );
			CHKERRXX( ierr );
			double *phiExactPtr;
			ierr = VecGetArray( phiExact, &phiExactPtr );
			CHKERRXX( ierr );
			double r = S0 * sqrt( tn );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n )
			{
				double x = node_x_fr_n( n, p4est, nodes );
				double y = node_y_fr_n( n, p4est, nodes );
				phiExactPtr[n] = sqrt( SQR( x - X0 ) + SQR( y - Y0 ) ) - r;
			}
			ierr = VecRestoreArray( phiExact, &phiExactPtr );
			CHKERRXX( ierr );
		}

		iter++;

		// Display iteration message.
		ierr = PetscPrintf( mpi.comm(), "\tIteration %04d: t = %1.4f \n", iter, tn );
		CHKERRXX( ierr );

		if( (save_vtk && iter % save_every_n == 0) || tn >= T_FINAL || iter == max_iter )
		{
			save_VTK( p4est, nodes, &brick, phi_s, phiExact, t_l, t_s, v, kappa, howUpdated, vtkIdx, mode, MAX_RL );
			vtkIdx++;
		}
	}

	ierr = PetscPrintf( mpi.comm(), "Final time of the simulation: tf = %g\n", T_FINAL );
	CHKERRXX( ierr );

	ierr = PetscPrintf( mpi.comm(), "<< Finished after %.3f secs.", watch.get_duration_current() );
	CHKERRXX( ierr );

	if( test_number == 0 )
		check_error_frank_sphere( p4est, nodes, phi_s, t_l );

	for( int dir = 0; dir < P4EST_DIM; ++dir )
	{
		ierr = VecDestroy( d_phi[dir] );
		CHKERRXX( ierr );
		ierr = VecDestroy( v[dir] );
		CHKERRXX( ierr );
	}
	ierr = VecDestroy( hk );
	CHKERRXX( ierr );
	ierr = VecDestroy( kappa );
	CHKERRXX( ierr );
	ierr = VecDestroy( phi_s );
	CHKERRXX( ierr );
	ierr = VecDestroy( phi_l );
	CHKERRXX( ierr );
	ierr = VecDestroy( phiExact );
	CHKERRXX( ierr );
	ierr = VecDestroy( t_s );
	CHKERRXX( ierr );
	ierr = VecDestroy( t_l );
	CHKERRXX( ierr );
	ierr = VecDestroy( howUpdated );
	CHKERRXX( ierr );

	// Destroy the p4est and its connectivity structure
	delete ngbd;
	delete hierarchy;
	p4est_nodes_destroy( nodes );
	p4est_ghost_destroy( ghost );
	p4est_destroy( p4est );
	my_p4est_brick_destroy( connectivity, &brick );

	return 0;
}
