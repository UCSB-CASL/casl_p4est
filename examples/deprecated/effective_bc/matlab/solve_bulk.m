function [cnp1, unp1] = solve_bulk(c, u, pde)
n = length(pde.x);

% solve for the concentration field first
A = zeros(n,n);
b = zeros(n,1);
A(1,1) = 1; b(1) = c;
for i=2:n-1
    A(i,i-1) = -2.0/pde.dx(i-1)/(pde.dx(i-1)+pde.dx(i));
    A(i,i+1) = -2.0/pde.dx(i)/(pde.dx(i-1)+pde.dx(i));
    A(i,i)   = 1/pde.dt -(A(i,i-1)+A(i,i+1));
    b(i)     = pde.cn(i)/pde.dt;
end
A(n,n) = 1; b(n) = pde.cb;
A = sparse(A);
cnp1 = A\b;

% solve for the potential field
A = zeros(n,n);
b = zeros(n,1);
A(1,1) = 1; b(1) = u;
for i=2:n-1
    A(i,i-1) = -2.0/pde.dx(i-1)/(pde.dx(i-1)+pde.dx(i))*0.5*(cnp1(i-1)+cnp1(i));
    A(i,i+1) = -2.0/pde.dx(i)/(pde.dx(i-1)+pde.dx(i))*0.5*(cnp1(i+1)+cnp1(i));
    A(i,i)   = -(A(i,i-1)+A(i,i+1));
end
A(n,n) = 1; b(n) = pde.ub;
A = sparse(A);
unp1 = A\b;

end