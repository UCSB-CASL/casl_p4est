% Investigating how to determine the cubic domain to minimize the number of
% triangles in the discretization of Q(u,v), but making it possible, at the
% same time, to get 0 curvature on the surface.
clc; clear;

% Paraboloid parameters.
h = 1/64;				% Mesh size.
start_k_max = 2/(3*h);	% Starting max desired curvature; hk_max^up = 4/3  and  hk_max^low = 2/3 (2/3 and 1/3 in 2D).
A = 200*h;				% Equivalent to the sphere of max radius: hk_min = 0.01 (0.005 in 2D), and 4*r_min = 6h.
su = sqrt(2*A / start_k_max);
k_max = 2*start_k_max;
denom = k_max/A*su^2 - 1;
assert( denom > 0 );
sv = sqrt(su^2 / denom);

% Defining Gaussian Monge patch and its curvature function.
Q = @(u,v) A*exp(-0.5*(u.^2/su^2 + v.^2/sv^2));
Qu = @(u,v) -Q(u,v).*(u/su^2);
Qv = @(u,v) -Q(u,v).*(v/sv^2);
Quu = @(u,v) -1/su^2 * (Qu(u,v).*u + Q(u,v));
Qvv = @(u,v) -1/sv^2 * (Qv(u,v).*v + Q(u,v));
Quv = @(u,v) Q(u,v).*u.*v / (su^2 * sv^2);
kappa = @(u,v) ((1+Qv(u,v).^2).*Quu(u,v) - 2*Qu(u,v).*Qv(u,v).*Quv(u,v) + (1+Qu(u,v).^2).*Qvv(u,v)) ...
			   ./ (1+Qu(u,v).^2+Qv(u,v).^2).^1.5;

%%%%%%%%%%%% 1) Find the u and v values where curvature reaches 0 using Newton's method %%%%%%%%%%%%

fu = @(u) (Q(u,0).*u).^2 + su^4 + su^2*sv^2 - u.^2*sv^2;
dfu = @(u) 2*u.*(Q(u,0).^2.*(1-u.^2/su^2) - sv^2);
uZero = abs(findZero( fu, dfu, su, 100, 1e-8*h ));	% Start one su from the origin in the u-axis.

fv = @(v) (Q(0,v).*v).^2 + sv^4 + su^2*sv^2 - v.^2*su^2;
dfv = @(v) 2*v.*(Q(0,v).^2.*(1-v.^2/sv^2) - su^2);
vZero = abs(findZero( fv, dfv, sv, 100, 1e-8*h ));	% Start one sv from the origin in the v-axis.

% Let's define limiting u and v values.  This will define an elliptical cylinder we want to have 
% inside the discretized domain.
ulim = uZero + su;		% One su and sv away from where we found the zeros.
vlim = vZero + sv;		% The cylinder extends from the uv-plane to the plane at Q(0,0)=A.
qlim = A;

%%%%%%%%%%%%%%%%%%% 2) Transform Q(u,v) and its axes to find enclosing cylinder %%%%%%%%%%%%%%%%%%%%

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

% Finding the world coords of cylinder containing Q(u,v).
% Top coords (the four points lying on the same qlim found above).
Qt = zeros(3,4);	% 3 dimensions (x,y,z) and 4 points (-ulim,ulim,-vlim,vlim), one per column.
Qt(:,1) = R * [-ulim, 0, qlim]' + T;			% (-ulim, 0, qlim).
Qt(:,2) = R * [+ulim, 0, qlim]' + T;			% (+ulim, 0, qlim).
Qt(:,3) = R * [0, -vlim, qlim]' + T;			% (0, -vlim, qlim).
Qt(:,4) = R * [0, +vlim, qlim]' + T;			% (0, +vlim, qlim).

% Base coords (the four points lying on the uv-plane).
Q0 = zeros(3,4);	% Same as above, but for q=0.
Q0(:,1) = R * [-ulim, 0, 0]' + T;				% (-ulim, 0, 0).
Q0(:,2) = R * [+ulim, 0, 0]' + T;				% (+ulim, 0, 0).
Q0(:,3) = R * [0, -vlim, 0]' + T;				% (0, -vlim, 0).
Q0(:,4) = R * [0, +vlim, 0]' + T;				% (0, +vlim, 0).

%%%%%%%%%%%%%%%%%%%% 3) Get the range along each axis of the enclosing cylinder %%%%%%%%%%%%%%%%%%%%

B = [Qt, Q0];	% TODO: Give it a different name!  It's polluting the scope.
minR = min(B, [], 2);
maxR = max(B, [], 2);
rangeB = maxR - minR;

%%%%%%%%%% 5) Use the x,y,z ranges to find the cube side length and circumscribing sphere %%%%%%%%%%

cubeSideLen = max(rangeB);
centroid = mean(B, 2);
dp = cubeSideLen/2;			% Each axis begins at the centroid and extends to -dp and +dp.
dp = dp + 2*h;				% Padding each direction (we won't sample points beyond dp, but need info from there).
r = dp * sqrt(3);			% Radius of circumscribing sphere.

%%%%%%%%%%% 6) Redefining the canonical domain by using the cube's circumscribing sphere %%%%%%%%%%%

% Transform the centroid from world to paraboloid canonical coords: only the q(=z) component != 0.
centroidQ = R' * (centroid - T);
z = centroidQ(end);

% Set v=0 and find the intersection of circle with radius r centered at (0,0,z) and Gaussian Q(u,0).
f = @(u) u.^2 + (Q(u,0)-z).^2 - r^2;
fu = @(u) 2*u + 2*(Q(u,0)-z).*Qu(u,0);
uu = findZero(f, fu, ulim, 100, 1e-8*h);
assert(uu >= ulim);			% New ulim value must be at least its previous estimation.

% Set u=0 and find the intersectino of circle with radius r centered at (0,0,z) and Gaussian Q(0,v).
f = @(v) v.^2 + (Q(0,v)-z).^2 - r^2;
fv = @(v) 2*v + 2*(Q(0,v)-z).*Qv(0,v);
vv = findZero(f, fv, vlim, 100, 1e-8*h);
assert(vv >= vlim);

% Adjust the canonical domain for new height.
mu = ceil(uu/h)*h;			% New canonical domain boundaries in u and v directions as multiples of h.
mv = ceil(vv/h)*h;

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
surf( corners([1,4],:), corners([2,5],:), corners([3,6],:), 'FaceColor', 'm', 'FaceAlpha', 0.05 );
patch( corners([1,4],:)', corners([2,5],:)', corners([3,6],:)', 'm', 'FaceAlpha', 0.10 );

% Zero and limiting ellipses.
t = linspace(0, 2*pi, 300);
plot3( ulim*cos(t), vlim*sin(t), Q(ulim*cos(t), vlim*sin(t)), "r-" );	% Limiting ellipse.
plot3( uZero*cos(t), vZero*sin(t), zeros(size(t)), "k-" );				% "Zero"-curvature ellipse.

title( "Gaussian Monge Patch" );
rotate3d on;
grid on;
axis equal;
zlim([min(zlim)-1, max(zlim)])
hold off;

% Let's plot the canonical domain in 2D and show points that we should keep to speed up queries to 
% the balltree.
figure;
plot( uu*cos(t), vv*sin(t), "k-" )
hold on;
plot( ulim*cos(t), vlim*sin(t), "r-" );	% Limiting ellipse.
plot( uZero*cos(t), vZero*sin(t), "y-" );	% "Zero"-curvature ellipse.
xticks(linspace( -mu, mu, 2*mu/h + 1 ));
yticks(linspace( -mv, mv, 2*mv/h + 1 ));

for x = linspace(-mu, mu, 2*mu/h+1)		% Points lying inside the projected ellipse.
	for y = linspace(-mv, mv, 2*mv/h+1)
		if x^2/uu^2 + y^2/vv^2 <= 1
			% Bring in the 8 neighbors. to valid set.
			for px = [x-h, x, x+h]
				for py = [y-h, y, y+h]
					if px^2/uu^2 + py^2/vv^2 > 1
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