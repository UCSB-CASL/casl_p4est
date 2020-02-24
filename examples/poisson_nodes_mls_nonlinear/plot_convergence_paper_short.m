clear;

error_ticks_num = 3;
resolution_ticks_num = 5;

plot_detailed_convergence = 0;
plot_condensed_convergence = 1;

% % data
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_2nd_order_b/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/union/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/union/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/difference/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/difference/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/three_flowers/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/three_flowers/gradients_2nd_order_c/convergence';

% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/triangle/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/triangle/gradients_2nd_order_b/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/union/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/union/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/difference/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/difference/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/three_flowers/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/three_flowers/gradients_2nd_order_c/convergence';

% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/triangle/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/triangle/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/union/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/union/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/difference/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/difference/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/three_flowers/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/three_flowers/gradients_2nd_order_c/convergence';

% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/triangle/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/triangle/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/union/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/union/gradients_2nd_order_c/convergence';
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/difference/gradients_1st_order/convergence';
% n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/difference/gradients_2nd_order_c/convergence';
s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/three_flowers/gradients_1st_order/convergence';
n_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/three_flowers/gradients_2nd_order_c/convergence';


h = importdata(strcat(s_out_dir,'/h.txt'));
    
s_error_sl_all = importdata(strcat(s_out_dir,'/error_sl_all.txt'));
s_error_gr_all = importdata(strcat(s_out_dir,'/error_gr_all.txt'));
s_error_sl_max = importdata(strcat(s_out_dir,'/error_sl_max.txt'));
s_error_gr_max = importdata(strcat(s_out_dir,'/error_gr_max.txt'));

n_error_sl_all = importdata(strcat(n_out_dir,'/error_sl_all.txt'));
n_error_gr_all = importdata(strcat(n_out_dir,'/error_gr_all.txt'));
n_error_sl_max = importdata(strcat(n_out_dir,'/error_sl_max.txt'));
n_error_gr_max = importdata(strcat(n_out_dir,'/error_gr_max.txt'));

num_resolutions = length(h);
num_shifts = length(s_error_sl_all)/num_resolutions;

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

% detailed convergence
if (plot_detailed_convergence == 1)
    
    % solution error
    
    figure;
    
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
    
    xmin = 0;
    xmax = n(end);
    
    ymin = min(min([s_error_sl_all_max; n_error_sl_all]));
    ymax = max(max([s_error_sl_all_max; n_error_sl_all]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Symmetric', 'Superconvergent', 'Symmetric (max)', 'Superconvergent (max)');
    set(L, 'interpreter', 'latex');
    
    % figure size
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 8 2.5];
    
    % gradient error
    
    figure;
    
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
    
    xmin = 0;
    xmax = n(end);
    
    ymin = min(min([s_error_gr_all; n_error_gr_all]));
    ymax = max(max([s_error_gr_all; n_error_gr_all]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Symmetric', 'Superconvergent', 'Symmetric (max)', 'Superconvergent (max)');
    set(L, 'interpreter', 'latex');
    
    % figure size
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 8 2.5];
    
end

% condensed convergence
if (plot_condensed_convergence == 1)
    
    % solution error
    
    figure;
    
    % error plots
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'o', 1, 4, 'auto'};
    
    L = loglog(h, s_error_sl_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', 'w');
    
    hold on
    
    L = loglog(h, n_error_sl_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', get(L, 'Color'));
    
    % guide lines
    % a = mean(mean([s_error_sl_max; n_error_sl_max])./h.^2);
    a = mean(sqrt(s_error_sl_max.*n_error_sl_max)./h.^2);
    
    PropName  = {'LineWidth', 'Color'};
    PropValue = {2, 'k'};
    
    L = loglog(h, a*h.^2); set(L, PropName, PropValue);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Grid resolution');
    ylabel('Solution Error');
    
    xmin = min(h);
    xmax = max(h);
    
    xrel = xmax/xmin;
    
    xmin = xmin/xrel^0.1;
    xmax = xmax*xrel^0.1;
    
    ymin = min(min([s_error_sl_max; n_error_sl_max]));
    ymax = max(max([s_error_sl_max; n_error_sl_max]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    xticks(round(10.^linspace(log10(xmin), log10(xmax), resolution_ticks_num), 1, 'significant'));
%     yticks(round(10.^linspace(log10(ymin), log10(ymax), error_ticks_num), 1, 'significant'));
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Symmetric', 'Superconvergent', '2nd order');
    set(L, 'interpreter', 'latex');
    
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 3 2.5];
    
    % gradient error
    
    figure;
    
    % error plots
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'o', 1, 4, 'auto'};
    
    L = loglog(h, s_error_gr_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', 'w');
    
    hold on
    
    L = loglog(h, n_error_gr_max); set(L, PropName, PropValue); set(L, 'MarkerFaceColor', get(L, 'Color'));
    
    % guide lines
    s_a = max(s_error_gr_max./h.^1)*(max(s_error_gr_max)/min(s_error_gr_max))^0.1;
    n_a = max(n_error_gr_max./h.^2)*(max(n_error_gr_max)/min(n_error_gr_max))^0.1;
    
    loglog(h, s_a*h.^1, '-k', 'LineWidth', 1);
    loglog(h, n_a*h.^2, '-k', 'LineWidth', 2);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Grid resolution');
    ylabel('Gradient Error');
    
    xmin = min(h);
    xmax = max(h);
    
    xrel = xmax/xmin;
    
    xmin = xmin/xrel^0.1;
    xmax = xmax*xrel^0.1;
    
    ymin = min(min([s_error_gr_max; n_error_gr_max]));
    ymax = max(max([s_error_gr_max; n_error_gr_max]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    xticks(round(10.^linspace(log10(xmin), log10(xmax), resolution_ticks_num), 1, 'significant'));
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Symmetric', 'Superconvergent', '1st order', '2nd order');
    set(L, 'interpreter', 'latex');
    
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 3 2.5];
    
end


