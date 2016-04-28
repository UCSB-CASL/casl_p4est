% Matlab scrip to solve the 1d probelm
%% constants
x0 = [1.0; -0.5]; % initial guess
xs = [0.8196; 8.469-10]; % solution we want to converge to

%% Using Matlab's FSolve
options = optimoptions('fsolve', 'Display', 'iter');
[x, fval] = fsolve(@(x) GC(x) - GC(xs), x0, options)
fprintf('MATLAB`s fsolve err = %e\n', max(abs(x-xs)));

%% using My nonlinear solver
it = 1; itmax = 100;
tol = 1e-8;
err = 1 + tol;

x = x0 + 1e-1;
fprintf('Newton solver: \n');
close all
while(it < itmax && err > tol)
    F  = GC(x) - GC(xs);
    J  = JGC(x);
    dx = -J\F;
    err = max(abs(dx));
    errs(it) = err;
    fprintf('   it = %d err = %e\n', it, err);
    plot((it), (err), 'bo'); hold on;
    x = x + dx; it = it+1;
end
x
abs(GC(x) - GC(xs))
