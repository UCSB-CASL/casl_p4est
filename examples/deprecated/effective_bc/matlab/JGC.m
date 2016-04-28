function J = JGC(x)
J = [1.0./sqrt(x(1)).*sinh(0.50*x(2))    sqrt(x(1)).*cosh(0.50*x(2));
     2.0./sqrt(x(1)).*sinh(0.25*x(2)).^2 sqrt(x(1)).*sinh(0.50*x(2))];
end