function [c,u] = bc_solve_coupled(pde)
x0  = [pde.cn(1); pde.un(1)];
% ops = optimoptions('fsolve', 'Display', 'iter');
x   = fsolve(@(x) func(x, pde), x0);
c   = x(1); 
u   = x(2);
end

function F = func(x,pde)
w1 = (2*pde.dx(1)+pde.dx(2))/(pde.dx(1)+pde.dx(2));
w2 = -pde.dx(1)/(pde.dx(1)+pde.dx(2));
flux = [
    w1*(pde.cn(2)-x(1))/pde.dx(1) + w2*(pde.cn(3)-pde.cn(2))/pde.dx(2)
    w1*0.5*(x(1)+pde.cn(2))*(pde.un(2)-x(2))/pde.dx(1) + ...
    w2*0.5*(pde.cn(2)+pde.cn(3))*(pde.un(3)-pde.un(2))/pde.dx(2)
    ];

wn   = pde.w(pde.cn(1), pde.un(1)-pde.ue);
qn   = pde.q(pde.cn(1), pde.un(1)-pde.ue);
wnp1 = pde.w(x(1), x(2)-pde.ue);
qnp1 = pde.q(x(1), x(2)-pde.ue);

F = [
    pde.eps*(wnp1-wn)/pde.dt
    pde.eps*(qnp1-qn)/pde.dt
    ] - flux;    
end