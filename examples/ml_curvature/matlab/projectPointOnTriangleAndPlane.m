function [inside, u, v, P, x, y] = projectPointOnTriangleAndPlane( p, v0, v1, v2 )
	% Compute triangle's plane's normal vector.
	v0v1 = v1 - v0;
	v0v2 = v2 - v0;
	inside = false;

	% No need to normalize.
	N = cross( v0v1, v0v2 );
	denom = dot( N, N );

	if( sqrt( denom ) <= eps )		% Check for colinear points.
		error( "[projectPointOnTriangleAndPlane] Triangle's vertices are collinear!" );
	end

	% Step 1: finding P, the projection of p onto the triangle's plane.
	vp0 = p - v0;
	P = p - N * dot( vp0, N ) / denom;

	% Step 2: inside-outside test.
	% This test uses the cross product with a reference vector (i.e. a triangle's side).  When a point is outside
	% the triangle, we have at most two sides for which the inside/outside test fails:
	%         \                          \
	%          \   o Point                \
	%           \                    ------*.....
	%     -------*.....                     .  o Point
	%             .                          .
	%          (a)                       (b)
	% In case (a), there is just one side failing the inside/outside test.  In case (b), there are two.  In the
	% latter, just the first triangle's side to fail the test will be reported back.  However, in case (b) the
	% closest point on the triangle is the shared corner, here denoted as '*'.
	x = 0;
	y = 0;
	u = 0;
	v = 0;

	edge0 = v1 - v0;				% Edge 0.
	vp0 = P - v0;
	C = cross( edge0, vp0 );
	if( dot( N, C ) < 0 )			% P is on the right side of edge 0, opposed to v2.
		x = v0;
		y = v1;
		return;
	end

	edge1 = v2 - v1;				% Edge 1.
	vp1 = P - v1;
	C = cross( edge1, vp1 );		% Vector perpendicular to plane used for inside-outside test.
	u = dot( N, C );
	if( u < 0 )						% P is on the right side of edge 1, opposed to v0.
		x = v1;
		y = v2;
		return;
	end

	edge2 = v0 - v2;				% Edge 2.
	vp2 = P - v2;
	C = cross( edge2, vp2 );
	v = dot( N, C );
	if( v < 0 )						% P is on the right side of edge 2, opposed to v1.
		x = v2;
		y = v0;
		return;
	end

	u = u / denom;
	v = v/ denom;

	inside = true; 					% The projected point P falls within the triangle.
end