function F = fully_implicit_system(sol, pde)
% This function calculates the nonlinear residual of the system of
% equations we are trying to solve, which here is the fully implicit
% discretization.
% Note that we store the solution as pairs of (c,u), meaning that all odd
% indecis in x point to c whereas even indecies point to u;

c = sol(1:2:end);
u = sol(2:2:end);

qn   = pde.q(pde.cn(1), pde.un(1) - pde.ue);
qnp1 = pde.q(c(1), u(1) - pde.ue);
wn   = pde.w(pde.cn(1), pde.un(1) - pde.ue);
wnp1 = pde.w(c(1), u(1) - pde.ue);

%  Compute the residual
n = length(pde.x);
F = zeros(size(sol));

% left boundary conditions
ug = u(2) - pde.eps*(qnp1 - qn)/pde.dt*2*pde.dx(1)/c(1); % ghost value for the potential
cg = c(2) - pde.eps*(wnp1 - wn)/pde.dt*2*pde.dx(1); % ghost value for the concentration

F(1) = -((c(2)-c(1))/pde.dx(1) - (c(1)-cg)/pde.dx(1))*2/(pde.dx(1)+pde.dx(1)) + (c(1) - pde.cn(1))/pde.dt; % left bc for concentration
F(2) = -(0.5*(c(1)+c(2))*(u(2)-u(1))/pde.dx(1) - 0.5*(cg + c(1))*(u(1)-ug)/pde.dx(1))*2/(pde.dx(1)+pde.dx(1)); % left bc for potential

for i=2:n-1
    F(2*i-1) = -((c(i+1)-c(i))/pde.dx(i) - (c(i)-c(i-1))/pde.dx(i-1))*2/(pde.dx(i)+pde.dx(i-1)) + (c(i) - pde.cn(i))/pde.dt;
    F(2*i)   = -(0.5*(c(i)+c(i+1))*(u(i+1)-u(i))/pde.dx(i) - 0.5*(c(i-1) + c(i))*(u(i)-u(i-1))/pde.dx(i-1))*2/(pde.dx(i)+pde.dx(i-1));    
end
    
F(2*n-1) = c(n) - pde.cb; % right bc for concentration
F(2*n)   = u(n) - pde.ub; % right bc for potential
end