function [c,u] = bc_solve_decoupled(pde)
% ops = optimoptions('fsolve', 'Display', 'iter');
u = fsolve(@(u) q_res(pde.cn(1), u, pde), pde.un(1));
c = fsolve(@(c) w_res(c, pde.un(1), pde), pde.cn(1));
end

function F = q_res(c, u,pde)
w1 = (2*pde.dx(1)+pde.dx(2))/(pde.dx(1)+pde.dx(2));
w2 = -pde.dx(1)/(pde.dx(1)+pde.dx(2));

flux = w1*0.5*(pde.cn(1)+pde.cn(2))*(pde.un(2)-pde.un(1))/pde.dx(1) + ...
       w2*0.5*(pde.cn(2)+pde.cn(3))*(pde.un(3)-pde.un(2))/pde.dx(2);

qn   = pde.q(pde.cn(1), pde.un(1)-pde.ue);
qnp1 = pde.q(c, u-pde.ue);

F = pde.eps*(qnp1-qn)/pde.dt - flux;    
end

function F = w_res(c,u,pde)
w1 = (2*pde.dx(1)+pde.dx(2))/(pde.dx(1)+pde.dx(2));
w2 = -pde.dx(1)/(pde.dx(1)+pde.dx(2));
flux = w1*(pde.cn(2)-pde.cn(1))/pde.dx(1) + w2*(pde.cn(3)-pde.cn(2))/pde.dx(2);

wn   = pde.w(pde.cn(1), pde.un(1)-pde.ue);
wnp1 = pde.w(c, u-pde.ue);

F = pde.eps*(wnp1-wn)/pde.dt - flux;
end