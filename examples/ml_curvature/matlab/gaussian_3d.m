% Investigating Gaussian patch to generate samples.
% Questions are: how big the domain should be?  where does Q(u,v) reach a zero curvature?
clear; clc;
h = 1/128;				% Mesh size.
start_k_max = 2/(3*h);	% Starting max desired curvature; hk_max^up = 4/3  and  hk_max^low = 2/3 (2/3 and 1/3 in 2D).
A = 200*h;				% Equivalent to the sphere of max radius: hk_min = 0.01 (0.005 in 2D), and 4*r_min = 6h.
su = sqrt(2*A / start_k_max);
k_max = 2*start_k_max;
denom = k_max/A*su^2 - 1;
assert( denom > 0 );
sv = sqrt(su^2 / denom);

% Defining Gaussian Monge patch.
Q = @(u,v) A*exp(-0.5*(u.^2/su^2 + v.^2/sv^2));
Qu = @(u,v) -Q(u,v).*(u/su^2);
Qv = @(u,v) -Q(u,v).*(v/sv^2);
Quu = @(u,v) -1/su^2 * (Qu(u,v).*u + Q(u,v));
Qvv = @(u,v) -1/sv^2 * (Qv(u,v).*v + Q(u,v));
Quv = @(u,v) Q(u,v).*u.*v / (su^2 * sv^2);
kappa = @(u,v) ((1+Qv(u,v).^2).*Quu(u,v) - 2*Qu(u,v).*Qv(u,v).*Quv(u,v) + (1+Qu(u,v).^2).*Qvv(u,v)) ...
			   ./ (1+Qu(u,v).^2+Qv(u,v).^2).^1.5;

% Using Newton's method to find u (and v) where kappa is zero.
fu = @(u) (Q(u,0).*u).^2 + su^4 + su^2*sv^2 - u.^2*sv^2;
dfu = @(u) 2*u.*(Q(u,0).^2.*(1-u.^2/su^2) - sv^2);
uZero = abs(findZero( fu, dfu, su, 100, 1e-8*h ));		% Start one su from the origin in the u-axis.

fv = @(v) (Q(0,v).*v).^2 + sv^4 + su^2*sv^2 - v.^2*su^2;
dfv = @(v) 2*v.*(Q(0,v).^2.*(1-v.^2/sv^2) - su^2);
vZero = abs(findZero( fv, dfv, sv, 100, 1e-8*h ));

% Plotting Gaussian surface.
figure;
t = linspace(0,2*pi,500);
zeroEllipseX = uZero*cos(t);
zeroEllipseY = vZero*sin(t);
zeroEllipseQ = Q(zeroEllipseX, zeroEllipseY);
limitEllipseX = (uZero+su)*cos(t);
limitEllipseY = (vZero+sv)*sin(t);
limitEllipseQ = Q(limitEllipseX, limitEllipseY);
[U, V] = meshgrid( linspace(-su-uZero, uZero+su, 300), linspace(-sv-vZero, vZero+sv, 300) );
G = Q(U,V);
surf(U,V,G);
shading interp;
hold on;
plot3(zeroEllipseX, zeroEllipseY, zeroEllipseQ, "k-");		% Approximate curvature's zero level set with an ellipse.
plot3(limitEllipseX, limitEllipseY, limitEllipseQ, "y-");	% Limiting ellipse.
xlabel("u");
ylabel("v");
zlabel("Q(u,v)");
title( "Gaussian surface" );
axis equal;
		   
% Plotting curvature.
figure;
K = kappa(U,V);
surf(U,V,K);
shading interp;
hold on;
plot3(uZero, 0, kappa(uZero,0), "b*");
plot3(0, vZero, kappa(0,vZero), "m*");
plot3(zeroEllipseX, zeroEllipseY, zeros(size(t)), "k-");	% Approximate curvature's zero level set with an ellipse.
plot3(limitEllipseX, limitEllipseY, kappa(limitEllipseX, limitEllipseY), "b-");	% Limiting ellipse.
xlabel("u");
ylabel("v");
zlabel("\kappa(u,v)");
title( "Curvature" );

% Is the isoline at curvature level 0 an ellipse?
z = sum(abs(kappa(uZero*cos(t), vZero*sin(t))));	% Should be zero, but it's not.
fprintf( "Sum of Q along zero-ellipse is %g\n", z );

% Plotting curvature's isolines.
figure;
contourf(U,V,K, [-10, 0], "ShowText", "on");
xlabel("u")
ylabel("v");
title("Curvature isolines");
axis equal;