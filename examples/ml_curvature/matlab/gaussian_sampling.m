% Exploring bivariate random point generation within the projected ellipse for the corresponding 
% paraboloid canonical domain.
% About 95% of samples lie within -2sigma and +2sigma.  So, we use the ellipse semiaxes to define 
% the variances we need.
clear; clc;
q = 9;			% Say, this is the paraboloid height.
h = 1/64;		% Mesh size.
a = 1/(6*h);	% Q(u,v) = a*u^2 + b*v^2.
b = 3/(6*h);
ru = sqrt(q/a);	% Semiaxes along u and v.
rv = sqrt(q/b);

su = ru/2;		% Standard deviations for both axes.
sv = rv/2;
n = round(pi*ru*rv / h^2);	% Number of samples.  We want bivariate sampling with a diagonal covariance matrix.
rng(1);
u = randn(n,1) * su;		% Gaussian coords pulled independently for each direction.
v = randn(n,1) * sv;

insideIdx = (u.^2)./ru^2 + (v.^2)./rv^2 <= 1;
fprintf( "Inside %i (%f%%)\n", sum(insideIdx), sum(insideIdx)/n*100 );

figure;
t = linspace(0, 2*pi, 200);
plot( ru*cos(t), rv*sin(t), "k-" );
hold on;
plot( u(insideIdx), v(insideIdx), "b." );
plot( u(~insideIdx), v(~insideIdx), ".", "color", "#aaa" );
grid on;
xlabel("u");
ylabel("v");
axis equal;
title("Sampling from a bivariate Gaussian distribution with \sigma_i = r_i/2");

% Another way, using the multivariate builtin function--produces exactly
% the same values as above.  This means I can generate the coords
% separately.  I can't run the code below locally, unless I install the
% statistical toolbox.  Tested this online though.
% rng(1);
% UV = mvnrnd( [0,0], [su^2, 0; 0, sv^2], n);
% u = UV(:,1);
% v = UV(:,2);