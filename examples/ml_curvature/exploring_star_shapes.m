% Generating star-shaped interfaces and finding their curvature as a way to
% choosing lower and upper bounds where the numerical method excels (i.e.,
% well resolved regions).

d = 0;		% Shape parameters: phase.
p = 5;		% Number of arms.
a = 0.225;
b = 0.355;

r = @(theta) a * cos( p * theta - d ) + b;
rPrime = @(theta) -a * p * sin( p * theta - d );
rPrimePrime = @(theta) -a * p^2 * cos( p * theta - d );
curvature = @(theta) (r(theta).^2 + 2 * rPrime(theta).^2 - r(theta).*rPrimePrime(theta))./(r(theta).^2 + rPrime(theta).^2).^1.5;

t = linspace(0, 2*pi, 1000);
R = r(t);
figure; 
plot( R.*cos(t), R.*sin(t), "-" ); 
axis equal; 
grid on; 
limits = [-1, 1];
xlim( limits ); ylim( limits );

figure;
k = curvature(t);
plot( t, k, "-" );
grid on;
axis normal;