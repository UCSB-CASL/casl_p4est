% Three-dimensional sinusoid implemented as a Monge patch.
wx = 1;				% How many times you want wave to oscillate in x-dir.
wy = 2;				% How many times you want wave to oscillate in y-dir.
A = 0.5;			% Amplitude.

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
H = @(x,y) ((1+hy(x,y).^2).*hxx(x,y) - 2*hx(x,y).*hy(x,y).*hxy(x,y) + (1+hx(x,y).^2).*hyy(x,y)) ./ ...
	(1 + hx(x,y).^2 + hy(x,y).^2).^1.5;

[X,Y] = meshgrid( linspace( -pi, pi ) );
Z = h( X, Y );
figure;
surf(X, Y, Z);
xlabel( "x" );
ylabel( "y" );
zlabel( "z" );
axis equal;
zlim([-1, 1]);
title( "Sinusoidal wave in 3D" );

% Let's compute curvature to determine the sign: concave and convex regions
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