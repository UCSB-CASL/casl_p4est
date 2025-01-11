%% Modeling a hyperbolic paraboloid surface.
clc; clear; close all;
b = 4; a = 5*b;		% Paraboloid coefficients.

% The hyperbolic paraboloid.
Q = @(u,v) a*u.^2 - b*v.^2;

% The mean curvature function.
H = @(u,v) (a*(1+4*b^2*v.^2) - b*(1+4*a^2*u.^2)) ./ (1+4*a^2*u.^2 + 4*b^2*v.^2).^1.5;

% The Gaussian curvature function.
K = @(u,v) -4*a*b ./ (1+4*a^2*u.^2 + 4*b^2*v.^2).^2;

cellsPerUnitLengh = 128;
h = 1/cellsPerUnitLengh;

%% Plotting the surface.

[U,V] = meshgrid(linspace( -0.5, 0.5, cellsPerUnitLengh+1 ), linspace( -0.5, 0.5, cellsPerUnitLengh+1 ));
Z = Q( U, V );
figure;
surf(U, V, Z);
hold on;
xlabel( "u" );
ylabel( "v" );
zlabel( "Q(u,v)" );
% axis equal;
title( "Hyperbolic paraboloid" );
% xlim([-0.5, 0.5]);
% ylim([-0.5, 0.5]);
% zlim([-0.5, 0.5]);
rotate3d on;
grid on;
shading interp;
hold off;

%% Plotting its mean curvature.

Z = H( U, V );
figure;
hold on;
surf(U, V, Z);
shading interp;
xlabel( "u" );
ylabel( "v" );
zlabel( "H(u,v)" );
% axis equal;
title( "Mean curvature" );
rotate3d on;
grid on;
hold off;

%% Plotting its Gaussian curvature.

Z = K( U, V );
figure;
hold on;
surf(U, V, Z);
shading interp;
xlabel( "u" );
ylabel( "v" );
zlabel( "K(u,v)" );
axis equal;
title( "Gaussian curvature" );
rotate3d on;
grid on;
hold off;