function post_processing(datafile, rho_in, rho_out, mu_in, mu_out, gamma, R0, eps_0)
    if(nargin < 2)
        rho_in = 1;
        rho_out = 0.001;
        mu_in = 0.02;
        mu_out = 0.00002;
        gamma = 0.5;
        R0 = 1;
        eps_0 = 0.01;
    end
    sigma_lamb          = lamb_prediction(rho_in, rho_out, mu_in, mu_out, gamma, R0);
    sigma_prosperetti   = prosperetti_prediction(rho_in, rho_out, mu_in, mu_out, gamma, R0);
    
    
    file_to_load = sprintf(datafile);
    fid = fopen(file_to_load, 'r');
    fgetl(fid);
    read_data   = fscanf(fid, '%f %f %f %f %f %f %f %f', [8 Inf]);
    time        = read_data(1, :)';
    R_var       = read_data(2, :)';
    R_0         = read_data(3, :)';
    a_2_var_R   = read_data(4, :)';
    a_2_R_0     = read_data(5, :)';
    fclose(fid);
    
    figure('Units','normalized','Position',[0.01 0.35 0.45 0.6]);
    hold on
    plot(time, R_var,   'b-', 'linewidth', 3)
    plot(time, R_0,     'r-', 'linewidth', 3)
    xlabel('\fontsize{18}{0}\selectfont $t$', 'Interpreter', 'Latex');
    ylabel('\fontsize{18}{0}\selectfont $R$', 'Interpreter', 'Latex', 'rotation', 0, 'Units', 'Normalized', 'position', [-0.08, 0.93]);
    legend('variable $R$', ...
        '$R = R_0$', ...
        'Interpreter', 'Latex', 'fontsize', 24, 'position', [0.75 0.63 0.1 0.2])
    grid on
    hold off
    
    figure('Units','normalized','Position',[0.51 0.35 0.45 0.6]);
    hold on
    plot(time, a_2_var_R,   'b-', 'linewidth', 3)
    plot(time, a_2_R_0,     'r-', 'linewidth', 3)
    plot(time, eps_0*exp(-real(sigma_prosperetti)*time).*cos(imag(sigma_prosperetti)*time), 'k-.', 'linewidth', 3)
    plot(time, eps_0*exp(-real(sigma_lamb)*time).*cos(imag(sigma_lamb)*time), '-.', 'Color', [0.2 0.6 0.2], 'linewidth', 3)
    xlabel('\fontsize{18}{0}\selectfont $t$', 'Interpreter', 'Latex');
    ylabel('\fontsize{18}{0}\selectfont $a_{2}$', 'Interpreter', 'Latex', 'rotation', 0, 'Units', 'Normalized', 'position', [-0.08, 0.93]);
    legend('variable $R$', ...
        '$R = R_0$', ...
        'Prosperetti', ...
        'Lamb', ...
        'Interpreter', 'Latex', 'fontsize', 24, 'position', [0.75 0.63 0.1 0.2])
    grid on
    hold off
    
end