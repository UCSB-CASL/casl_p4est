% Find the intersection of a ray R(t)=A+ct to a plane dot(n, P-B)=0.
% A: Ray's origin point.
% c: Ray's vector direction.
% B: Point on plane.
% n: Plane's normal vector.
% Returns: isHit=true, the hit time tHit, and the hit point PHit if ray intersects plane.  If not, 
% isHit is set to false and tHit and PHit become nan.
function [isHit, tHit, PHit] = rayPlaneIntersection( A, c, B, n )
	den = dot( n, c );		% If ray is parallel to (or lies on) the plane, den is zero.
	if abs(den) <= eps
		isHit = false;
		tHit = nan;
		PHit = nan;
		return;
	end
	
	% Ray is not parallel.  Let's find tHit (which can be negative, btw --if so, it means the plane
	% is behind the ray).
	tHit = dot( n, B - A ) / den;
	PHit = A + tHit * c;
	isHit = true;
end