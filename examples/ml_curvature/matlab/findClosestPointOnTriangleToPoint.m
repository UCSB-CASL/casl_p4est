function [R, P] = findClosestPointOnTriangleToPoint( p, v0, v1, v2 )
	P = 0;				% Projected point triangle's plane.
	u0 = 0; u1 = 0;		% Vertices of line segment closest to P if the latter doesn't fall within the triangle.
	[inside, ~, ~, P, u0, u1] = projectPointOnTriangleAndPlane( p, v0, v1, v2 );
	if( inside )
		R = P;
		return;
	else				% Find closest point from projected point to nearest triangle's segment that failed in/out test.
		R = findClosestPointOnLineSegmentToPoint( P, u0, u1 );
	end
end