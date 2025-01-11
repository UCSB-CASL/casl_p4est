% Investigating how to determine the cubic domain to minimize the number of
% triangles in the discretization of Q(u,v), but making it possible, at the
% same time, to get some minimum curvature along the surface.
clc; clear;

% Paraboloid parameters.
cellsPerUnitLengh = 64;
h = 1/cellsPerUnitLengh;
a = 1/(6*h);		% Here, we look at k_max^low = 0.5*k_max^up = 0.5*4/(3h), and a = k_max^low/4.
b = 3/(6*h);		% We want b to produce k_max^up in combination with a.  Thus, b = k_max^up/2 - a.

% Paraboloid function.
Q = @(u,v) a*u.^2 + b*v.^2;

% Curvature function.
K = @(u,v) (2*a*(1+4*b^2*v.^2) + 2*b*(1+4*a^2*u.^2)) ./ (1+4*a^2*u.^2 + 4*b^2*v.^2).^1.5;

% Let's use hk_min=0.005 in 2D. In 3D, hk_min=2*0.005=0.01.
hk_min = 0.01;
kmin = hk_min/h;

%%%%%%%%%%%%%%% 1) Find level curve on Q where curvature achieves the minimum faster %%%%%%%%%%%%%%%

% Let's use Newton's method to find what u value (with v=0) yields kmin.
fu = @(x) kmin - (2*a+2*b*(1+4*a^2*x.^2))./(1+4*a^2*x.^2).^1.5;
dfdu = @(x) (8*a^2*x.*(4*b*a^2*x.^2 + 3*a + b))./(4*a^2*x.^2 + 1).^(2.5);
ulim = abs( findZero( fu, dfdu, h, 100, 1e-8*h ) );
qulim = Q( ulim, 0 );		% We've gotten the points (-ulim, 0, qulim) and (ulim, 0, qulim).

% Same as before, but now for v (with u=0).
fv = @(y) kmin - (2*a*(1+4*b^2*y.^2)+2*b)./(1+4*b^2*y.^2).^1.5;
dfdv = @(y) (8*b^2*y.*(4*a*b^2*y.^2 + 3*b + a))./(4*b^2*y.^2 + 1).^(2.5);
vlim = abs( findZero( fv, dfdv, h, 100, 1e-8*h ) );
qvlim = Q( 0, vlim );		% We've gotten the points (0, -vlim, qvlim) and (0, vlim, qvlim).

%%%%%%% 2) Pick the minimum Q value and modify the other axis so that it produces the same Q %%%%%%%

if qulim < qvlim			% Level up the v axis?
	vlim = sqrt(qulim / b);
	qvlim = Q(0, vlim);		% Must yield qulim too --now we have (-vlim, qulim) and (vlim, qulim).
else
	ulim = sqrt(qvlim / a);
	qulim = Q(ulim, 0);		% Must yield qvlim too --now we have (-ulim, qvlim) and (ulim, qvlim).
end

qlim = qvlim;				% Level curve where we achieve the minimum curvature (=qulim works too).

%%%%%%%%%%%%%%%%%%% 3) Transform Q(u,v) and its axes to find enclosing cylinder %%%%%%%%%%%%%%%%%%%%

% Building the transformation matrix based on a translation and rotation.
beta = 11*pi/36;			% Angle of rotation (11*pi/36 puts the corner up).
u = [1,-1,0];				% Axis of rotation (for above, [1,-1,0]).
u = u ./ norm(u);
ux = u(1); uy = u(2); uz = u(3);
c = cos(beta); s = sin(beta);
R = [    c+(1-c)*ux^2, (1-c)*uy*ux-s*uz, (1-c)*uz*ux+s*uy;...	% Rotation matrix.
	 (1-c)*ux*uy+s*uz,     c+(1-c)*uy^2, (1-c)*uz*uy-s*ux;...
	 (1-c)*ux*uz-s*uy, (1-c)*uy*uz+s*ux,     c+(1-c)*uz^2];
T = [-0.125, 0.125, -0.125]';				% Translation vector.

% Finding the coords of cylinder containing Q(u,v).
% Top coords (the four points lying on the same qlim found above).
Qt = zeros(3,4);	% 3 dimensions (x,y,z) and 4 points (-ulim,ulim,-vlim,vlim), one per column.
Qt(:,1) = R * [-ulim, 0, qlim]' + T;			% (-ulim, 0, qlim).
Qt(:,2) = R * [+ulim, 0, qlim]' + T;			% (+ulim, 0, qlim).
Qt(:,3) = R * [0, -vlim, qlim]' + T;			% (0, -vlim, qlim).
Qt(:,4) = R * [0, +vlim, qlim]' + T;			% (0, +vlim, qlim).

% Base coords (the four points lying on the base of the paraboloid: q axis = 0).
Q0 = zeros(3,4);	% Same as above, but for q=0.
Q0(:,1) = R * [-ulim, 0, 0]' + T;				% (-ulim, 0, 0).
Q0(:,2) = R * [+ulim, 0, 0]' + T;				% (+ulim, 0, 0).
Q0(:,3) = R * [0, -vlim, 0]' + T;				% (0, -vlim, 0).
Q0(:,4) = R * [0, +vlim, 0]' + T;				% (0, +vlim, 0).

%%%%%%%%%%%%%%%%%%%% 4) Get the range along each axis of the enclosing cylinder %%%%%%%%%%%%%%%%%%%%

A = [Qt, Q0];
minA = min(A, [], 2);
maxA = max(A, [], 2);
rangeA = maxA - minA;

%%%%%%%%%%%%%%%% 5) Use the x,y,z ranges to find the cube width and define its faces %%%%%%%%%%%%%%%

cubeSideLen = max(rangeA);
centroid = mean(A, 2);
dp = cubeSideLen/2;			% Each axis begins at the centroid and extends to -dp and +dp.
dp = dp + 2*h;				% Padding each direction (we won't sample points beyond dp, but need info from there).

% Cube faces: six planes defined by a point and an outward-pointing normal vector.
% Assume the z axis points up and y axis towards the screen.
Bp = zeros(3,3);			% Points on positive faces (a column per dim).
np = zeros(3,3);			% Outward normals on positive faces (a column per dim).
Bm = zeros(3,3);			% Points on negative faces.
nm = zeros(3,3);			% Outward normals on negative faces.
Bp(:,1) = centroid + [dp,0,0]'; np(:,1) = [ 1, 0, 0]';		% Right.
Bm(:,1) = centroid - [dp,0,0]'; nm(:,1) = [-1, 0, 0]';		% Left.
Bp(:,2) = centroid + [0,dp,0]'; np(:,2) = [ 0, 1, 0]';		% Back.
Bm(:,2) = centroid - [0,dp,0]'; nm(:,2) = [ 0,-1, 0]';		% Front.
Bp(:,3) = centroid + [0,0,dp]'; np(:,3) = [ 0, 0, 1]';		% Top.
Bm(:,3) = centroid - [0,0,dp]'; nm(:,3) = [ 0, 0,-1]';		% Bottom.

%%%%%%%%% 6) Fill the gaps by redefining the canonical domain and shooting rays to the cube %%%%%%%%

% Let's find the min hit with any face for each of the four rays.
tHits = nan(4,1);
for ray = 1:4				% Four rays.
	c = Qt(:,ray) - Q0(:,ray);
	c = c / norm(c);		% Unit direction vector --that way, tHit indicates the target height Q.
	P = Qt(:,ray);			% Ray origin: must be the top because we don't want the base to interfere.
	for i = 1:3				% Three groups of faces: two on x, two on y, and two on z.
		
		% First, check positive face in the ith direction.
		[isHit, tHit, ~] = rayPlaneIntersection( P, c, Bp(:,i), np(:,i) );
		if isHit && tHit >= 0	% Is face in front of ray and ray aims along its face's normal?
			if isnan(tHits(ray)) || tHit < tHits(ray)
				tHits(ray) = tHit;
			end
		end
		
		% Then, check negative face in the ith direction.
		[isHit, tHit, ~] = rayPlaneIntersection( P, c, Bm(:,i), nm(:,i) );
		if isHit && tHit >= 0	% Is face in front of ray and ray aims along its face's normal?
			if isnan(tHits(ray)) || tHit < tHits(ray)
				tHits(ray) = tHit;
			end
		end
	end
end

% For each paraboloid main direction select the max tHit. That's the additional Q value we need.
tHits( isnan(tHits) ) = 0;			% If q stayed as nan, we set its height to zero for consistency.
qu = qlim + max(tHits(1:2));		% u=[-ulim,+ulim], v=0.
qv = qlim + max(tHits(3:4));		% u=0, v=[-vlim,+vlim].
q = max(qu, qv);					% Definite height.

mu = ceil(sqrt(q / a)/h)*h;	% New canonical domain boundaries in u and v directions.
mv = ceil(sqrt(q / b)/h)*h;

[U,V] = meshgrid( linspace( -mu, mu, 2*mu/h + 1 ), linspace( -mv, mv, 2*mv/h + 1 ) );
Z = Q( U, V );
figure;
hold on;
surf(U, V, Z, 'edgecolor', 'none');
xlabel( "x" );
ylabel( "y" );
zlabel( "z" );

% A cube representing the domain.
s = -pi : pi/2 : pi;                                % Define corners.
ph = pi/4;                                          % Define angular orientation ('phase').
x = dp*[cos(s+ph); cos(s+ph)]/cos(ph);
y = dp*[sin(s+ph); sin(s+ph)]/sin(ph);
z = dp*[-ones(size(s)); ones(size(s))];

corners1 = R'*[x(1,:) - T(1) + centroid(1); y(1,:) - T(2) + centroid(2); z(1,:) - T(3) + centroid(3)];
corners2 = R'*[x(2,:) - T(1) + centroid(1); y(2,:) - T(2) + centroid(2); z(2,:) - T(3) + centroid(3)];
corners = [corners1; corners2];

% Plot cube domain with coordinates with respect to paraboloid coordinate system.
surf( corners([1,4],:), corners([2,5],:), corners([3,6],:), 'FaceColor', 'm', 'FaceAlpha', 0.15 );
patch( corners([1,4],:)', corners([2,5],:)', corners([3,6],:)', 'm', 'FaceAlpha', 0.3 );

title( "Paraboloid" );
rotate3d on;
grid on;
axis equal;
zlim([min(zlim)-1, max(zlim)])
hold off;

% Let's plot the canonical domain in 2D and show points that we should keep to speed up queries to 
% the balltree.
ru2 = q/a;				% Semiaxis along u.
rv2 = q/b;				% Semiaxis along v.
t = linspace(0, 2*pi, 200);
figure;
plot( sqrt(ru2)*cos(t), sqrt(rv2)*sin(t), "k-" )
hold on;
xticks(linspace( -mu, mu, 2*mu/h + 1 ));
yticks(linspace( -mv, mv, 2*mv/h + 1 ));

for x = linspace(-mu, mu, 2*mu/h+1)		% Points lying inside the projected ellipse.
	for y = linspace(-mv, mv, 2*mv/h+1)
		if x^2/ru2 + y^2/rv2 <= 1
			% Bring in the 8 neighbors. to valid set.
			for px = [x-h, x, x+h]
				for py = [y-h, y, y+h]
					if px^2/ru2 + py^2/rv2 <= 1
% 						plot(px, py, "g.");		% Plotting causes very slow response.
					else
						plot(px, py, "b.");
					end
				end
			end
		end
	end
end

grid on;
axis equal;
set( gca, 'YTickLabel', [] );
set( gca, 'XTickLabel', [] );
xlim([-mu, mu]);
ylim([-mv, mv]);
title( "Canonical domain and its seed and boundary points" );