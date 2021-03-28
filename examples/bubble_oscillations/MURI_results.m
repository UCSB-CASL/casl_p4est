function MURI_results(ratio_rho, ratio_mu, Re, niter)
    if(nargin < 1)
        ratio_rho = 0.001;
        ratio_mu = 0.001;
        Re = 35.5;
        niter = 1;
    end
    R0 = 1;
    eps_0 = 0.01;
    rho_plus = 0.001;
    rho_minus = rho_plus/ratio_rho;
    mu_plus = 0.00002;
    mu_minus = mu_plus/ratio_mu;
    gamma = (mu_minus*Re)^2/(rho_minus*R0);
    path = "/home/regan/workspace/projects/two_phase_flow/bubble_oscillations/3D/";
    if(niter == 1)
        path = strcat(path, "one_iter/");
    elseif (niter == 3)
        path = strcat(path, "three_iter/");
    else
        return
    end
    path = strcat(path, "mu_ratio_");
    if(ratio_mu == 0.1)
        path = strcat(path, "0.1");
    elseif (ratio_mu == 0.001)
        path = strcat(path, "0.001");
    else
        return
    end
    
    path = strcat(path, "_rho_ratio_");
    if(ratio_rho == 0.01)
        path = strcat(path, "0.01");
    elseif (ratio_rho == 0.001)
        path = strcat(path, "0.001");
    else
        return
    end
    
    path = strcat(path, "_Reynolds_");
    if(Re == 30)
        path = strcat(path, "30");
    elseif (Re == 35.5)
        path = strcat(path, "35.5");
    else
        return
    end
    
    path = strcat(path, "/");
    sigma_lamb          = lamb_prediction(rho_minus, rho_plus, mu_minus, mu_plus, gamma, R0);
    sigma_prosperetti   = prosperetti_prediction(rho_minus, rho_plus, mu_minus, mu_plus, gamma, R0);
    
    figure('Units','normalized','Position',[0.51 0.35 0.45 0.6]);
    hold on
    time_exact = linspace(0, 2.1*(2.0*pi/imag(sigma_lamb)), 2000);
    plot(time_exact, eps_0*exp(-real(sigma_prosperetti)*time_exact).*cos(imag(sigma_prosperetti)*time_exact), 'k-.', 'linewidth', 3)
    LGD = [{'Prosperetti'}];
    plot(time_exact, eps_0*exp(-real(sigma_lamb)*time_exact).*cos(imag(sigma_lamb)*time_exact), 'b-.', 'linewidth', 3)
    LGD = [LGD {'Lamb'}];
    if(ratio_rho == 0.001 && ratio_mu == 0.001 && Re == 35.5 && niter == 1)
        datafile_4_6        = strcat(path, "lmin_4_lmax_6/results/monitoring_results.dat");
        [t_4_6, a_2_4_6, a_2_R0_4_6, R_var_4_6, R0_4_6] = read_results(datafile_4_6);
        plot(t_4_6, a_2_4_6, 'r-', 'linewidth', 3)
        LGD = [LGD {'$\ell_{\min}/\ell_{\max} = 4/6$'}];
    end
    datafile_4_7        = strcat(path, "lmin_4_lmax_7/results/monitoring_results.dat");
    [t_4_7, a_2_4_7, a_2_R0_4_7, R_var_4_7, R0_4_7] = read_results(datafile_4_7);
    plot(t_4_7, a_2_4_7, 'm-', 'linewidth', 3)
    LGD = [LGD {'$\ell_{\min}/\ell_{\max} = 4/7$'}];
    
    datafile_5_8        = strcat(path, "lmin_5_lmax_8/results/monitoring_results.dat");
    [t_5_8, a_2_5_8, a_2_R0_5_8, R_var_5_8, R0_5_8] = read_results(datafile_5_8);
    plot(t_5_8, a_2_5_8, '-', 'Color', [0.85 0.325 0.098], 'linewidth', 3)
    LGD = [LGD {'$\ell_{\min}/\ell_{\max} = 5/8$'}];
    
    datafile_6_9        = strcat(path, "lmin_6_lmax_9/results/monitoring_results.dat");
    [t_6_9, a_2_6_9, a_2_R0_6_9, R_var_6_9, R0_6_9] =  read_results(datafile_6_9);
    plot(t_6_9, a_2_6_9, '-', 'Color', [0.2 0.6 0.2], 'linewidth', 3)
    LGD = [LGD {'$\ell_{\min}/\ell_{\max} = 6/9$'}];

    legend(LGD, 'Interpreter', 'Latex', 'fontsize', 24, 'location', 'SouthEast')
    xlabel('\fontsize{14}{0}\selectfont $t$', 'Interpreter', 'Latex');
    ylabel('\fontsize{14}{0}\selectfont $a_{2}$', 'Interpreter', 'Latex', 'rotation', 0, 'Units', 'Normalized', 'position', [-0.12, 0.93]);
    TITRE = ['\fontsize{22}{0}\selectfont $a_{2}\left(t\right)$ with ' int2str(niter) ' iteration'];
    if(niter > 1)
        TITRE = [TITRE 's'];
    end
    title(TITRE, 'Interpreter', 'Latex');
    set(gca, 'FontSize', 24)
    grid on
    hold off
    
end

function [time, a_2, a_2_R_0, R_var, R_0] = read_results(datafile)
    file_to_load = sprintf(datafile);
    fid = fopen(file_to_load, 'r');
    fgetl(fid);
    read_data   = fscanf(fid, '%f %f %f %f %f %f %f %f', [8 Inf]);
    time        = read_data(1, :)';
    R_var       = read_data(2, :)';
    R_0         = read_data(3, :)';
    a_2         = read_data(4, :)';
    a_2_R_0     = read_data(5, :)';
    fclose(fid);
end