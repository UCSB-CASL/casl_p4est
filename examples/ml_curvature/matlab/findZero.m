% Find a zero of a single-variable function by using Newton's method.
function [x] = findZero( f, df, x0, maxIter, tol )
	iter = 0;
	xk = x0;
	fprintf( "Iter %d, x = %.15f\n", iter, xk );
	while iter < maxIter
		xkp1 = xk - f(xk) / df(xk);
		fprintf( "Iter %d, x = %.15f\n", iter, xkp1 );
		d = abs(xk - xkp1); 
		xk = xkp1;
		if d <= tol
			break
		end
		iter = iter + 1;
	end
	x = xk;
end