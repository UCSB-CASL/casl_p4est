function [P] = findClosestPointOnLineSegmentToPoint( p, v0, v1 )
	v = v1 - v0;
	denom = dot( v, v );
	if( sqrt( denom ) <= eps )		% Degenerate line segment?
		P = v0;
		return;
	end

	t = dot( p - v0, v ) / denom;	% Parameter t in Q = v0 + tv.
	if( t <= 0 )
		P = v0;
		return;
	end
	
	if( t >= 1 )
		P = v1;
		return;
	end
	
	P = v0 + v * t;
end

