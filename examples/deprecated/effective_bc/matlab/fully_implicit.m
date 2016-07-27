% This script solves the effective boundary condition problem in a fully
% implicit discretization using MATLAB's fsolve capabilities. Specifically
% we solve for (c,u) where:
%   D(c*D(u,x),x) = 0
%   D(c,t) = D(c,{x,2})
% subjected to flux conditon at x = 0,
%   eps * D(q(c,du), t) = i and eps*D(w(c,du),t) = j
% where du = u - ue is the potential drop acros the EDL and q and w are
% excess charge and salt densities inside the EDL given by GC theory. i an
% d j are the normal fluxes **into** the EDL given by:
%   i = c*D(u,x), j = D(c,x) @x=0
% The other boundary condition is the bulk condition, i.e.:
%   u = 0 and c = 1 @x=1

pde.eps = 0.01;
pde.ue  = 10;
pde.ub  = 0;
pde.cb  = 1;
pde.x = linspace(0,1); %unique([linspace(0,3*pde.eps, 50) linspace(3*pde.eps,1,10)]);
pde.dx = diff(pde.x);
pde.dt = 1e-5;%1e-2*pde.eps;

pde.q = @(c,du) 2*sqrt(c)*sinh(0.50*du);
pde.w = @(c,du) 4*sqrt(c)*sinh(0.25*du)^2;
pde.qmax = pde.q(1, -pde.ue);
pde.wmax = pde.w(1, -pde.ue);

pde.un = pde.ue*(1-pde.x); % constant field as initial condition
pde.cn = pde.cb*ones(size(pde.x)); % constant concentration as initial conditon

%% solve for the transient system
pde.fsolve.ops   = optimoptions('fsolve', 'Display', 'iter', 'MaxIter', 500, 'MaxFunEvals', 1e5);
pde.fsolve.guess = reshape([pde.cn; pde.un], 2*length(pde.x), 1);

close all;
figure(1); shg;
subplot(2,2,1);
plot(0, pde.q(pde.cn(1), pde.un(1) - pde.ue)/pde.qmax, 'bo');  hold on;

subplot(2,2,2);
plot(0, pde.w(pde.cn(1), pde.un(1) - pde.ue)/pde.wmax, 'bo');  hold on;

subplot(2,2,3);
plot(0, pde.cn(1), 'bo');  hold on;

subplot(2,2,4);
plot(0, pde.un(1), 'bo');  hold on;

figure(2); shg;
subplot(2,1,1);
plot(pde.x, pde.cn); hold on;

subplot(2,1,2);
plot(pde.x, pde.un); hold on;

t = 0;
for n=1:200
    sol = fsolve(@(x) fully_implicit_system(x, pde), pde.fsolve.guess, pde.fsolve.ops);    
    pde.fsolve.guess = sol; %2*sol - pde.fsolve.guess;
    pde.cn = sol(1:2:end);
    pde.un = sol(2:2:end);

    figure(1); shg;
    t = t + pde.dt;
    pde.dt = min(1.3*pde.dt, 10*pde.eps);
    subplot(2,2,1);
    plot(t, pde.q(pde.cn(1), pde.un(1) - pde.ue)/pde.qmax, 'bo');

    subplot(2,2,2);
    plot(t, pde.w(pde.cn(1), pde.un(1) - pde.ue)/pde.wmax, 'bo');

    subplot(2,2,3);
    plot(t, pde.cn(1), 'bo');

    subplot(2,2,4);
    plot(t, pde.un(1), 'bo');    
    
    if (mod(n, 10) == 0)
        figure(2); shg;
        subplot(2,1,1);
        plot(pde.x, pde.cn, 'bo-'); hold on;

        subplot(2,1,2);
        plot(pde.x, pde.un, 'bo-'); hold on;
    end
end