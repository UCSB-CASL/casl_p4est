% Investigating how to determine the cubic domain to minimize the number of
% triangles in the discretization of Q(u,v), but making it possible, at the
% same time, to get some minimum curvature along the surface.
clc; clear;

% Paraboloid parameters.
cellsPerUnitLengh = 64;
h = 1/cellsPerUnitLengh;
a = 1/(6*h);		% Here, we look at k_max^low = 0.5*k_max^up = 0.5*4/(3h), and a = k_max^low/4.
b = 3/(6*h);		% We want b to produce k_max^up in combination with a.  Thus, b = k_max^up/2 - a.
Q = @(u,v) a*u.^2 + b*v.^2;		% Paraboloid function.
K = @(u,v) (2*a*(1+4*b^2*v.^2) + 2*b*(1+4*a^2*u.^2)) ./ (1+4*a^2*u.^2 + 4*b^2*v.^2).^1.5;	% Curvature function.

% In 2D, we had hk_min=0.005. In 3D, hk_min=0.01.
kmin = 1/(100*h);

% Let's use Newton's method to find what u value (with v=0) yields kmin.
fu = @(x) kmin - (2*a+2*b*(1+4*a^2*x.^2))./(1+4*a^2*x.^2).^1.5;
dfu = @(x) (8*a^2*x.*(4*b*a^2*x.^2 + 3*a + b))./(4*a^2*x.^2 + 1).^(2.5);
ulim = abs( findZero( fu, dfu, h, 100, 1e-8*h ) );
qulim = Q( ulim, 0 );
dulim = max( ulim, qulim );	% For a cubic domain, we need to include the point with the desired k,
							% curvature.  Thus, we must check the u and Q direction.

% Same as before, but now for v (with u=0).
fv = @(y) kmin - (2*a*(1+4*b^2*y.^2)+2*b)./(1+4*b^2*y.^2).^1.5;
dfv = @(y) (8*b^2*y.*(4*a*b^2*y.^2 + 3*b + a))./(4*b^2*y.^2 + 1).^(2.5);
vlim = abs( findZero( fv, dfv, h, 100, 1e-8*h ) );
qvlim = Q( 0, vlim );
dvlim = max( vlim, qvlim );

d = min( dulim, dvlim );	% Domain is [-d, d]^3.
r = (d+h/2) * sqrt(3);		% Radius of circumscribing circle (containing the domain cube).
hu = 0.5 * (-1/a + sqrt(a^(-2) + 4*r^2));		% Expected min height along u axis (for v=0).
mu = ceil(sqrt(hu / a)/h)*h;
hv = 0.5 * (-1/b + sqrt(b^(-2) + 4*r^2));		% Expected min height along v axis (for u=0).
mv = ceil(sqrt(hv / b)/h)*h;

t = linspace( 0, 2*pi, 100 );

[U,V] = meshgrid( linspace( -mu, mu, 2*mu/h + 1 ), linspace( -mv, mv, 2*mv/h + 1 ) );
Z = Q( U, V );
figure;
hold on;
surf(U, V, Z, 'edgecolor', 'none');
plot3(r*cos(t), zeros(length(t)), r*sin(t), "b-");			% Circle on xz plane.
plot3(zeros(length(t)), r*cos(t), r*sin(t), "r-");			% Circle on yz plane.
xlabel( "x" );
ylabel( "y" );
zlabel( "z" );
axis equal;
% xlim([-0.5, 0.5]);
% ylim([-0.5, 0.5]);
% zlim([-0.5, 0.5]);

plot3( 0, vlim, qvlim, "r*" );

% A cube representing the domain.
s = -pi : pi/2 : pi;                                % Define corners.
ph = pi/4;                                          % Define angular orientation ('phase').
x = d*[cos(s+ph); cos(s+ph)]/cos(ph);
y = d*[sin(s+ph); sin(s+ph)]/sin(ph);
z = d*[-ones(size(s)); ones(size(s))];

beta = 11*pi/36;			% Angle of rotation (11*pi/36 puts the corner up).
u = [1,-1,0];		% Axis of rotation.
u = u ./ norm(u);
ux = u(1); uy = u(2); uz = u(3);
c = cos(beta); s = sin(beta);
R = [    c+(1-c)*ux^2, (1-c)*uy*ux-s*uz, (1-c)*uz*ux+s*uy;...	% Rotation matrix.
	 (1-c)*ux*uy+s*uz,     c+(1-c)*uy^2, (1-c)*uz*uy-s*ux;...
	 (1-c)*ux*uz-s*uy, (1-c)*uy*uz+s*ux,     c+(1-c)*uz^2];
T = [0, 0, 0]';		% Translation vector.

corners1 = R'*[x(1,:) - T(1); y(1,:) - T(2); z(1,:) - T(3)];
corners2 = R'*[x(2,:) - T(1); y(2,:) - T(2); z(2,:) - T(3)];
corners = [corners1; corners2];

% Plot cube domain with coordinates with respect to paraboloid coordinate system.
surf( corners([1,4],:), corners([2,5],:), corners([3,6],:), 'FaceColor', 'm', 'FaceAlpha', 0.15 );
patch( corners([1,4],:)', corners([2,5],:)', corners([3,6],:)', 'm', 'FaceAlpha', 0.3 );

title( "Paraboloid" );
rotate3d on;
grid on;
hold off;