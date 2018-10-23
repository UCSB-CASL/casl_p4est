clear;

dimensions = 2;

subplotted = 1;

num_err_ticks = 4;
num_res_ticks = 4;

plot_detailed_convergence = 1;
plot_condensed_convergence = 1;
plot_detailed_cond_num = 0;
plot_condensed_cond_num = 0;

plot_slope_sl = 2;
plot_slope_gr = 2;
plot_slope_cn = 2;

% use_n_points = 0.25;
use_n_points = 0.8;

% -----------------------------
% smooth solutions
% -----------------------------
figure;

% mu_m = mu_p = 1
% s_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_02/fdm';
% n_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_02/fvm';

% mu_m = 10, mu_p = 1
% s_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_03/fdm';
% n_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_03/fvm';

% mu_m != mu_p variable 
% s_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_04/fdm';
% n_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_04/fvm';

% mu_m = 10^5, mu_p = 1
% s_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_05/fdm';
% n_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_05/fvm';

% mu_m = mu_p variable 
s_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_06/fdm';
n_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_06/fvm';

% mu_m = mu_p = 10 
% s_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_07/fdm';
% n_out_dir = '/home/dbochkov/Outputs/poisson_jump_nodes_mls/case_07/fvm';

h = importdata(strcat(s_out_dir,'/h_arr.txt'));

% fit = 1:length(h);
fit = 1+round((1-use_n_points)*length(h)):length(h);

lvl = importdata(strcat(s_out_dir,'/lvl.txt'));
N = (2.^lvl + ones(size(lvl))).^dimensions;

s_error_sl_all = max( [importdata(strcat(s_out_dir,'/error_m_sl_all.txt')); importdata(strcat(s_out_dir,'/error_p_sl_all.txt'))] );
s_error_gr_all = max( [importdata(strcat(s_out_dir,'/error_m_gr_all.txt')); importdata(strcat(s_out_dir,'/error_p_gr_all.txt'))] );
s_error_sl_max = max( [importdata(strcat(s_out_dir,'/error_m_sl_max.txt')); importdata(strcat(s_out_dir,'/error_p_sl_max.txt'))] );
s_error_gr_max = max( [importdata(strcat(s_out_dir,'/error_m_gr_max.txt')); importdata(strcat(s_out_dir,'/error_p_gr_max.txt'))] );

s_cond_num_all = importdata(strcat(s_out_dir,'/cond_num_all.txt'));
s_cond_num_max = importdata(strcat(s_out_dir,'/cond_num_max.txt'));

n_error_sl_all = max( [importdata(strcat(n_out_dir,'/error_m_sl_all.txt')); importdata(strcat(n_out_dir,'/error_p_sl_all.txt'))] );
n_error_gr_all = max( [importdata(strcat(n_out_dir,'/error_m_gr_all.txt')); importdata(strcat(n_out_dir,'/error_p_gr_all.txt'))] );
n_error_sl_max = max( [importdata(strcat(n_out_dir,'/error_m_sl_max.txt')); importdata(strcat(n_out_dir,'/error_p_sl_max.txt'))] );
n_error_gr_max = max( [importdata(strcat(n_out_dir,'/error_m_gr_max.txt')); importdata(strcat(n_out_dir,'/error_p_gr_max.txt'))] );

n_cond_num_all = importdata(strcat(n_out_dir,'/cond_num_all.txt'));
n_cond_num_max = importdata(strcat(n_out_dir,'/cond_num_max.txt'));

num_resolutions = length(h);
num_shifts = length(s_error_sl_all)/num_resolutions;

% detailed convergence
if (plot_detailed_convergence == 1)
    
    s_error_sl_all_max = s_error_sl_all;
    s_error_gr_all_max = s_error_gr_all;
    n_error_sl_all_max = n_error_sl_all;
    n_error_gr_all_max = n_error_gr_all;
    
    for i=1:num_resolutions
        s_error_sl_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = s_error_sl_max(:,i)*ones(1,num_shifts);
        s_error_gr_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = s_error_gr_max(:,i)*ones(1,num_shifts);
        n_error_sl_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_sl_max(:,i)*ones(1,num_shifts);
        n_error_gr_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_gr_max(:,i)*ones(1,num_shifts);
    end
    
    % solution error
    
    if (subplotted == 1)
        subplot(2,3,[1,2]);
    else
        figure;
    end
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 0.5, 2, 'auto'};
    
    n = 0:length(s_error_sl_all)-1;
    
    L = semilogy(n, s_error_sl_all); set(L, PropName, PropValue);
    hold on
    L = semilogy(n, n_error_sl_all); set(L, PropName, PropValue);
    
    set(gca, 'ColorOrderIndex', 1);
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 2, 2, 'auto'};
    
    L = semilogy(n, s_error_sl_all_max); set(L, PropName, PropValue);
    L = semilogy(n, n_error_sl_all_max); set(L, PropName, PropValue);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Solution Error');
    
    % legend
    L = legend('FDM', 'FVM', 'FDM (max)', 'FVM (max)', 'Location', 'best');
    set(L, 'interpreter', 'latex');
    
    % figure size
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 8 2.5];
    end
    
    % gradient error
    
    if (subplotted == 1)
        subplot(2,3,[4,5]);
    else
        figure;
    end
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 0.5, 2, 'auto'};
    
    n = 0:length(s_error_sl_all)-1;
    
    L = semilogy(n, s_error_gr_all); set(L, PropName, PropValue);
    hold on
    L = semilogy(n, n_error_gr_all); set(L, PropName, PropValue);
    
    set(gca, 'ColorOrderIndex', 1);
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 2, 2, 'auto'};
    
    L = semilogy(n, s_error_gr_all_max); set(L, PropName, PropValue);
    L = semilogy(n, n_error_gr_all_max); set(L, PropName, PropValue);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Gradient Error');
    
    % legend
    L = legend('FDM', 'FVM', 'FDM (max)', 'FVM (max)', 'Location', 'best');
    set(L, 'interpreter', 'latex');
    
    % figure size
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 8 2.5];
    end
    
end

% condensed convergence
if (plot_condensed_convergence == 1)
    
    % solution error
    
    if (subplotted == 1)
        subplot(2,3,3);
    else
        figure;
    end
    
    % error plots
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'LineStyle'};
    PropValue = {'o', 1, 4, 'auto', 'none'};
    
    L = loglog(h, s_error_sl_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', 'w');
    
    hold on
    
    L = loglog(h, n_error_sl_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', get(L, 'Color'));
    
    % guide lines
    
    slope_s = polyfit(log(h(fit)), log(s_error_sl_max(fit)),1);
    slope_n = polyfit(log(h(fit)), log(n_error_sl_max(fit)),1);
   
    if (plot_slope_sl > 0) loglog(h, exp(slope_s(2))*h.^slope_s(1), ':k', 'LineWidth', 1); end
    if (plot_slope_sl > 1) loglog(h, exp(slope_n(2))*h.^slope_n(1), '-k', 'LineWidth', 1); end
%     a = mean(sqrt(s_error_sl_max.*n_error_sl_max)./h.^2);
%     
%     PropName  = {'LineWidth', 'Color'};
%     PropValue = {2, 'k'};
%     
%     L = loglog(h, a*h.^2); set(L, PropName, PropValue);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Grid resolution');
    ylabel('Solution Error');
    
    % legend, 
    L = legend('FDM', 'FVM', ['Slope: ', num2str(slope_s(1),3)], ['Slope: ', num2str(slope_n(1),3)], 'Location', 'best');
    set(L, 'interpreter', 'latex');
    
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 3.1 2.5];
    end
    
    % gradient error
    
    if (subplotted == 1)
        subplot(2,3,6);
    else
        figure;
    end
    
    % error plots
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'LineStyle'};
    PropValue = {'o', 1, 4, 'auto', 'none'};
    
    L = loglog(h, s_error_gr_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', 'w');
    
    hold on
    
    L = loglog(h, n_error_gr_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', get(L, 'Color'));
    
    % guide lines
    s_a = max(s_error_gr_max./h.^1)*(max(s_error_gr_max)/min(s_error_gr_max))^0.1;
    n_a = max(n_error_gr_max./h.^2)*(max(n_error_gr_max)/min(n_error_gr_max))^0.1;
    
    slope_s = polyfit(log(h(fit)), log(s_error_gr_max(fit)),1);
    slope_n = polyfit(log(h(fit)), log(n_error_gr_max(fit)),1);
    
    if (plot_slope_sl > 0) loglog(h, exp(slope_s(2))*h.^slope_s(1), ':k', 'LineWidth', 1); end
    if (plot_slope_sl > 1) loglog(h, exp(slope_n(2))*h.^slope_n(1), '-k', 'LineWidth', 1); end
%     loglog(h, s_a*h.^1, '-k', 'LineWidth', 1);
%     loglog(h, n_a*h.^2, '-k', 'LineWidth', 2);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Grid resolution');
    ylabel('Gradient Error');
    
    % legend
    L = legend('FDM', 'FVM', ['Slope: ', num2str(slope_s(1),3)], ['Slope: ', num2str(slope_n(1),3)], 'Location', 'best');
    set(L, 'interpreter', 'latex');
    
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 3.1 2.5];
    end
    
end

fig = gcf;
fig.Units = 'inches';
fig.Position = [10 10 12 7];

% detailed condition number 
if (plot_detailed_cond_num == 1)
    
    figure;
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 0.5, 2, 'auto'};
    
    n = 0:length(s_cond_num_all)-1;
    
    L = semilogy(n, s_cond_num_all); set(L, PropName, PropValue);
    hold on
    L = semilogy(n, n_cond_num_all); set(L, PropName, PropValue);
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 2, 2, 'auto'};
    set(gca, 'ColorOrderIndex', 1);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Condition number');
    
    % legend
    L = legend('FDM', 'FVM', 'Location', 'best');
    set(L, 'interpreter', 'latex');
    
    % figure size
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 8 2.5];
    
end

% condensed condition number
if (plot_condensed_cond_num == 1)
    
    idx = 1:length(s_cond_num_max);
    
    figure;
    
    % error plots
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'o', 1, 4, 'auto'};
    
    L = loglog(h(idx), s_cond_num_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', 'w');
    
    hold on
    
    L = loglog(h(idx), n_cond_num_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', get(L, 'Color'));
    
    % guide lines
    a = mean(sqrt(s_cond_num_max.*n_cond_num_max).*h(idx).^2);
    
    PropName  = {'LineWidth', 'Color'};
    PropValue = {2, 'k'};
    
    L = loglog(h(idx), a./h(idx).^2); set(L, PropName, PropValue);
    
    hold off
    
    grid on
    
    % axes
    L = xlabel('Grid resolution'); %set(L, 'interpreter', 'latex');
    L = ylabel('Max condition number'); %set(L, 'interpreter', 'latex');
    
    % legend
    L = legend('FDM', 'FVM', '$\sim h^2$', 'Location', 'best');
    set(L, 'interpreter', 'latex');
    
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 3.1 2.5];
    
end


