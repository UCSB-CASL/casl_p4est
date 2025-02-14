function sigma = lamb_prediction(rho_in, rho_out, mu_in, mu_out, gamma, R0)
    n = 2;
    
    if rho_in > rho_out % droplet
        sigma = 1.0/(R0*R0*rho_in/((n - 1)*(2*n + 1)*mu_in));
    else % bubble
        sigma = 1.0/(R0*R0*rho_out/((n + 2)*(2*n + 1)*mu_out));
    end
    sigma = sigma + 1i*sqrt(n*(n + 1)*(n - 1)*(n + 2)*gamma/((n + 1)*rho_in + n*rho_out)*R0*R0*R0);

    return
end
