% Three-dimensional sinusoid implemented as a Monge patch.
wx = 3.6950417228136048;	% How many times you want wave to oscillate in x-dir.
wy = 6.3999999999999995;	% How many times you want wave to oscillate in y-dir.
A = 1.5625;					% Amplitude.

% The actual function.
h = @(x,y) A * sin( wx * x ) .* sin( wy * y );

% The partial derivatives.
hy = @(x,y) A * sin( wx * x ) .* cos( wy * y ) * wy;
hyy = @(x,y) -(wy^2) * h(x, y);
hx = @(x,y) A * wx * cos( wx * x ) .* sin( wy * y );
hxx = @(x,y) -(wx^2) * h(x, y);
hxy = @(x,y) A * wx * wy * cos( wx * x ) .* cos( wy * y );

% The mean curvature. Note: in reality, we are computing 2*H because of how it's used in fluid
% dynamics applications and the library.
% H = @(x,y) ((1+hy(x,y).^2).*hxx(x,y) - 2*hx(x,y).*hy(x,y).*hxy(x,y) + (1+hx(x,y).^2).*hyy(x,y)) ./ ...
% 	(1 + hx(x,y).^2 + hy(x,y).^2).^1.5;
% H = @(x,y) ((1+hy(x,y).^2).*hxx(x,y) - 2*h(x,y).*hxy(x,y).^2 + (1+hx(x,y).^2).*hyy(x,y)) ./ ...
% 	(1 + hx(x,y).^2 + hy(x,y).^2).^1.5;
H = @(x,y) -h(x,y).*((1+hy(x,y).^2)*wx^2 + 2*hxy(x,y).^2 + (1+hx(x,y).^2)*wy^2) ./ ...
	(1 + hx(x,y).^2 + hy(x,y).^2).^1.5;

P = [-1.3229332857166214, -1.5416832857166214, 0.63180947678871369];	% Problematic point.
T = [-1.34375, -1.5400491201926108, 0.63199483602864381];	% Closest point on triangle.
Q = [-1.3230674438806957, -1.5387913092002445, 0.63213169724749074];	% Closest point trust region.
xrange = linspace(P(1) - 0.04, P(1) + 0.04, 20);
yrange = linspace(P(2) - 0.03, P(2) + 0.03, 20);
% v = T - P;		% Direction vector.

[X,Y] = meshgrid( xrange, yrange );
Z = h( X, Y );
figure;
surf(X, Y, Z);
hold on;
plot3(P(1),P(2),P(3),'r*');
plot3(T(1),T(2),T(3),'m*');
plot3(Q(1),Q(2),Q(3),'y*');
% v1 = P + v*1.375; plot3(v1(1),v1(2),v1(3),"go");
% plot3(v1(1),v1(2),h(v1(1),v1(2)),"k*")
xlabel( "x" );
ylabel( "y" );
zlabel( "z" );
axis equal;
% zlim([-1, 1]);
title( "Sinusoidal wave in 3D" );
shading interp;

%% Let's compute curvature to determine the sign: concave and convex regions
% on the interface.
kappa = H( X, Y );
figure;
surf(X, Y, kappa);
xlabel( "x" );
ylabel( "y" );
zlabel( "\kappa" );
axis equal;
title( "Mean curvature for sinusoidal wave in 3D" );
colormap turbo;