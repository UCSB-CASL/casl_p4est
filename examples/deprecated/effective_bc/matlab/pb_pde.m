function F = pb_pde(u, pb)
% Compute the discretization matvec operation corresponding to the PDE
% F = u''-k^2*sinh(u) = 0

n = length(u);
F = zeros(size(u));
dx = pb.x(2) - pb.x(1);

F(1) = 2*(u(2) - u(1))/dx^2 - pb.k^2*sinh(u(1)) - 2*pb.bc.left/dx; % left bc
for i=2:n-1
    F(i) = (u(i+1) - 2*u(i) + u(i-1))/dx^2 - pb.k^2*sinh(u(i));
end
F(n) = u(n) - pb.bc.right; % right bc

end