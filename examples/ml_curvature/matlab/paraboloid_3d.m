% Modeling the distance function from a 3D query point to a paraboloid.
a = 4; b = 1;		% Paraboloid coefficients.
p = [0.5,0.5,0.5];	% Query point.

% The paraboloid.
f = @(u,v) a*u.^2 + b*v.^2;

% The distance function.
D = @(u,v) 0.5 * ((u-p(1)).^2 + (v-p(2)).^2 + (a*u.^2 + b*v.^2 - p(3)).^2);

cellsPerUnitLengh = 64;
h = 1/cellsPerUnitLengh;

% Finding how far to go in the half-axes to get a lower bound on the
% maximum height in Q(u,v) = a*u^2 + b*v^2.
d = 0.5;			% Domain is [-d, d]^3.
r = d * sqrt(3);	% Radius of circumscribing circle (containing the domain cube).
hu = 0.5 * (-1/a + sqrt(a^(-2) + 4*r^2));		% Expected min height along u axis (for v=0).
mu = ceil(sqrt(hu / a)/h)*h;
hv = 0.5 * (-1/b + sqrt(b^(-2) + 4*r^2));		% Expected min height along v axis (for u=0).
mv = ceil(sqrt(hv / b)/h)*h;

t = linspace( 0, 2*pi, 100 );

[U,V] = meshgrid( linspace( -mu, mu, 2*mu/h + 1 ), linspace( -mv, mv, 2*mv/h + 1 ) );
Z = f( U, V );
figure;
hold on;
surf(U, V, Z);
% plot3(p(1), p(2), p(3), "o");
% pStar = [0.311190426, 0.434147127, f(0.311190426, 0.434147127)];
% plot3(pStar(1), pStar(2), pStar(3), "o");
% plot3([p(1), pStar(1)], [p(2), pStar(2)], [p(3), pStar(3)], "-");
plot3(r*cos(t), zeros(length(t)), r*sin(t), "b-");			% Circle on xz plane.
plot3(zeros(length(t)), r*cos(t), r*sin(t), "r-");			% Circle on yz plane.
xlabel( "x" );
ylabel( "y" );
zlabel( "z" );
axis equal;
% xlim([-0.5, 0.5]);
% ylim([-0.5, 0.5]);
% zlim([-0.5, 0.5]);

% A cube representing the domain.
s = -pi : pi/2 : pi;                                % Define corners.
ph = pi/4;                                          % Define angular prientation ('phase').
x = d*[cos(s+ph); cos(s+ph)]/cos(ph);
y = d*[sin(s+ph); sin(s+ph)]/sin(ph);
z = d*[-ones(size(s)); ones(size(s))];
beta = pi/4;
% Rot = [cos(beta), -sin(beta), 0; sin(beta), cos(beta), 0; 0, 0, 1];
% Rot = Rot * [cos(beta), 0, sin(beta); 0, 1, 0; -sin(beta), 0, cos(beta)];
Rot = eye(3);
corners1 = Rot*[x(1,:);y(1,:);z(1,:)];
corners2 = Rot*[x(2,:);y(2,:);z(2,:)];
corners = [corners1; corners2];
surf( corners([1,4],:), corners([2,5],:), corners([3,6],:), 'FaceColor', 'm', 'FaceAlpha', 0.15 )	% Plot cube domain.
patch( corners([1,4],:)', corners([2,5],:)', corners([3,6],:)', 'm', 'FaceAlpha', 0.3 )

title( "Paraboloid" );
rotate3d on;
grid on;
hold off;

% Z = D( U, V );
% figure;
% hold on;
% surf(U, V, Z);
% xlabel( "x" );
% ylabel( "y" );
% zlabel( "z" );
% axis equal;
% zlim([-0.5, 1.5]);
% title( "Distance function" );
% rotate3d on;
% grid on;
% hold off;