% Test/validating distance computation from points to a triangle in three-dimensional space.

rng( 1 );
% Rot = orth( rand(3,3) );	% Random orthogonal transformation.
Rot = eye( 3 );

% Triangle vertices and query points.
vertices = Rot * [0,0,0; 2,0,0; 0,3,0]';			% Columns are actual vertices.
perm = [1,2,3; 1,3,2; 2,3,1; 2,1,3; 3,2,1; 3,1,2];	% Let's try different orders in triangle vertices.
vertices = vertices(:,perm(6,:));
Q = [1,0,1]';			% Inside.
% Q = [1,1,1]';			% Inside.
% Q = [0.5,2,-1]';		% Inside.
% Q = [-1,2,2]';		% Outside.
% Q = [3,0,-1]';		% Outside.
% Q = [1.5,1.5,-2]';	% Outside.
% Q = [1.5,-1,-1]';		% Outside.
% Q = [-0.5,4,1]';		% Outside.

Q = Rot * Q;			% Let's apply the random transform.

% Get projections.
% [inside, u, v, P, x, y] = projectPointOnTriangleAndPlane( Q, vertices(:,1), vertices(:,2), vertices(:,3) );
% if inside
% 	PP = u * vertices(:,1) + v * vertices(:,2) + ( 1 - u - v ) * vertices(:,3);
% 	fprintf( "In triangle.  Diff: %f\n", norm( P - PP ) );
% end
[R, P] = findClosestPointOnTriangleToPoint( Q, vertices(:,1), vertices(:,2), vertices(:,3) );

% Let's plot.
figure; hold on;
plot3( [vertices(1,:), vertices(1,1)], [vertices(2,:), vertices(2,1)], [vertices(3,:), vertices(3,1)], "-" );
plot3( Q(1), Q(2), Q(3), "o" );		% Query point.
plot3( P(1), P(2), P(3), "*" );		% Projection onto triangle's plane.
plot3( R(1), R(2), R(3), "^" );		% Closest point on triangle.
plot3( [Q(1), P(1)], [Q(2), P(2)], [Q(3), P(3)], ":", "linewidth", 2 );	% Line between query and projection.
plot3( [Q(1), R(1)], [Q(2), R(2)], [Q(3), R(3)], "-" );					% Line between query and closest point on triangle.
% if( ~inside )
% 	plot3( [x(1), y(1)], [x(2), y(2)], [x(3), y(3)], "-", "linewidth", 2 );	% First edge that failed inside test.
% end
axis equal; grid on;
xlim([-5, 5]); ylim([-5, 5]); zlim([-5, 5]);
axis square; grid on;
xlabel("x"); ylabel("y"); zlabel("z");
hold off;
rotate3d on;

fprintf( "Distance: %.15f\n", norm(Q - R) );
disp( R );