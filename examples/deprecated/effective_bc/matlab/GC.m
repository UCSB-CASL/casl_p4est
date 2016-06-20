function F = GC(x)
F = [2*sqrt(x(1)).*sinh(0.50*x(2));
     4*sqrt(x(1)).*sinh(0.25*x(2)).^2];
end