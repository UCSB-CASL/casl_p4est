% This script solves a nonlinear PDE using MATLAB's interface to fsolve
% The PDE is the nonlinear Poisson-Boltzman:
%       u''=k^2*sinh(u)
% The boundary conditions are:
%       u'(0) = q, u(1) = 0;

pb.k = 20;
pb.bc.left  = -100;
pb.bc.right = 0;

pb.x = linspace(0,1,200);
pb.fsolve.u0 = -pb.bc.left/pb.k * exp(-pb.k*pb.x); % initial guess based on linear pb solution

%% solve for the PB system
pb.fsolve.ops = optimoptions('fsolve', 'Display', 'iter');
pb.u = fsolve(@(u) pb_pde(u, pb), pb.fsolve.u0, pb.fsolve.ops);

plot(pb.x, pb.u, 'k-'); hold on;
plot(pb.x, pb.fsolve.u0, 'k--'); hold off; shg;