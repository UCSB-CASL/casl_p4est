% Modeling the distance function from a 3D query point to a paraboloid.
a = 4; b = 1;		% Paraboloid coefficients.
p = [0.5,0.5,0.5];		% Query point.

% The paraboloid.
f = @(u,v) a*u.^2 + b*v.^2;

% The distance function.
D = @(u,v) 0.5 * ((u-p(1)).^2 + (v-p(2)).^2 + (a*u.^2 + b*v.^2 - p(3)).^2);

[U,V] = meshgrid( linspace( -0.5, 0.5, 64 ) );
Z = f( U, V );
figure;
hold on;
surf(U, V, Z);
plot3(p(1), p(2), p(3), "o");
pStar = [0.311190426, 0.434147127, f(0.311190426, 0.434147127)];
plot3(pStar(1), pStar(2), pStar(3), "o");
plot3([p(1), pStar(1)], [p(2), pStar(2)], [p(3), pStar(3)], "-");
xlabel( "x" );
ylabel( "y" );
zlabel( "z" );
axis equal;
xlim([-0.5, 0.5]);
ylim([-0.5, 0.5]);
zlim([-0.5, 0.5]);
title( "Paraboloid" );
rotate3d on;
grid on;
hold off;

Z = D( U, V );
figure;
hold on;
surf(U, V, Z);
xlabel( "x" );
ylabel( "y" );
zlabel( "z" );
axis equal;
%zlim([-0.5, 1.5]);
title( "Distance function" );
rotate3d on;
grid on;
hold off;