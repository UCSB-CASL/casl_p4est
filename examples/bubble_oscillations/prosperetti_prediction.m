function sigma = prosperetti_prediction(rho_in, rho_out, mu_in, mu_out, gamma, R0)
    
    initial_guess = lamb_prediction(rho_in, rho_out, mu_in, mu_out, gamma, R0);
    
    x0 = [real(initial_guess), imag(initial_guess)];
    
    eqn = @(x) characteristic_equation(x, 2, rho_in, rho_out, mu_in, mu_out, gamma, R0);
    
    sol = fsolve(eqn, x0);
    
    sigma = sol(1) + 1i*sol(2);
    
    return
end

function h_tilde = H_tilde(x, n)
    h_tilde = x*besselh(n + 1, x)/besselh(n, x);
    return;
end

function jj = J(x, n)
    jj = x*besselj(n - 1, x)/besselj(n, x);
    return;
end

function F = characteristic_equation(x, n, rho_in, rho_out, mu_in, mu_out, gamma, R0)
    z = x(1) + 1i*x(2);
    vv = R0*R0*(n*rho_out + (n + 1)*rho_in)*(mu_in*J(R0*sqrt(z*rho_in/mu_in), n + 1.5) + mu_out*H_tilde(R0*sqrt(z*rho_out/mu_out), n - 0.5) + 2*(mu_out - mu_in))*(z*z + ((n - 1)*n*(n + 1)*(n + 2)*gamma/((n*rho_out + (n + 1)*rho_in)*R0*R0*R0))) ...
        - z*(((2*n + 1)*mu_in*J(R0*sqrt(z*rho_in/mu_in), n + 1.5) + 2*n*(n + 2)*(mu_out - mu_in))*((2*n + 1)*mu_out*H_tilde(R0*sqrt(z*rho_out/mu_out), n - 0.5) - 2*(n - 1)*(n + 1)*(mu_out - mu_in)));
    
    F(1) = real(vv);
    F(2) = imag(vv);
    return
end
