clear;

error_ticks_num = 5;
resolution_ticks_num = 4;

plot_detailed_convergence = 1;
plot_condensed_convergence = 1;

colors = lines(8);
colors(3,:) = colors(5, :);

% set(groot,'defaultAxesColorOrder',colors);
set(groot,'defaultAxesColorOrder',[colors; colors]);

% data
% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_1st_order/convergence';
% n_out_dir_a = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_2nd_order_a/convergence';
% n_out_dir_b = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_2nd_order_b/convergence';
% n_out_dir_c = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_2nd_order_c/convergence';

% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/triangle/gradients_1st_order/convergence';
% n_out_dir_a = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/triangle/gradients_2nd_order_a/convergence';
% n_out_dir_b = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/triangle/gradients_2nd_order_b/convergence';
% n_out_dir_c = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d_test/triangle/gradients_2nd_order_c/convergence';

% s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/triangle/gradients_1st_order/convergence';
% n_out_dir_a = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/triangle/gradients_2nd_order_a/convergence';
% n_out_dir_b = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/triangle/gradients_2nd_order_b/convergence';
% n_out_dir_c = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d/triangle/gradients_2nd_order_c/convergence';

s_out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/triangle/gradients_1st_order/convergence';
n_out_dir_a = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/triangle/gradients_2nd_order_a/convergence';
n_out_dir_b = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/triangle/gradients_2nd_order_b/convergence';
n_out_dir_c = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/3d_test/triangle/gradients_2nd_order_c/convergence';

h = importdata(strcat(s_out_dir,'/h.txt'));
    
s_error_sl_all = importdata(strcat(s_out_dir,'/error_sl_all.txt'));
s_error_gr_all = importdata(strcat(s_out_dir,'/error_gr_all.txt'));
s_error_sl_max = importdata(strcat(s_out_dir,'/error_sl_max.txt'));
s_error_gr_max = importdata(strcat(s_out_dir,'/error_gr_max.txt'));

n_error_sl_all_a = importdata(strcat(n_out_dir_a,'/error_sl_all.txt'));
n_error_gr_all_a = importdata(strcat(n_out_dir_a,'/error_gr_all.txt'));
n_error_sl_max_a = importdata(strcat(n_out_dir_a,'/error_sl_max.txt'));
n_error_gr_max_a = importdata(strcat(n_out_dir_a,'/error_gr_max.txt'));

n_error_sl_all_b = importdata(strcat(n_out_dir_b,'/error_sl_all.txt'));
n_error_gr_all_b = importdata(strcat(n_out_dir_b,'/error_gr_all.txt'));
n_error_sl_max_b = importdata(strcat(n_out_dir_b,'/error_sl_max.txt'));
n_error_gr_max_b = importdata(strcat(n_out_dir_b,'/error_gr_max.txt'));

n_error_sl_all_c = importdata(strcat(n_out_dir_c,'/error_sl_all.txt'));
n_error_gr_all_c = importdata(strcat(n_out_dir_c,'/error_gr_all.txt'));
n_error_sl_max_c = importdata(strcat(n_out_dir_c,'/error_sl_max.txt'));
n_error_gr_max_c = importdata(strcat(n_out_dir_c,'/error_gr_max.txt'));

num_resolutions = length(h);
num_shifts = length(s_error_sl_all)/num_resolutions;

s_error_sl_all_max = s_error_sl_all;
s_error_gr_all_max = s_error_gr_all;

n_error_sl_all_max_a = n_error_sl_all_a;
n_error_gr_all_max_a = n_error_gr_all_a;
n_error_sl_all_max_b = n_error_sl_all_b;
n_error_gr_all_max_b = n_error_gr_all_b;
n_error_sl_all_max_c = n_error_sl_all_c;
n_error_gr_all_max_c = n_error_gr_all_c;

for i=1:num_resolutions
    s_error_sl_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = s_error_sl_max(:,i)*ones(1,num_shifts);
    s_error_gr_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = s_error_gr_max(:,i)*ones(1,num_shifts);
    
    n_error_sl_all_max_a(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_sl_max_a(:,i)*ones(1,num_shifts);
    n_error_gr_all_max_a(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_gr_max_a(:,i)*ones(1,num_shifts);
    n_error_sl_all_max_b(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_sl_max_b(:,i)*ones(1,num_shifts);
    n_error_gr_all_max_b(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_gr_max_b(:,i)*ones(1,num_shifts);
    n_error_sl_all_max_c(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_sl_max_c(:,i)*ones(1,num_shifts);
    n_error_gr_all_max_c(:, (i-1)*num_shifts+1:i*num_shifts) = n_error_gr_max_c(:,i)*ones(1,num_shifts);
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
    L = semilogy(n, n_error_sl_all_a); set(L, PropName, PropValue);
    L = semilogy(n, n_error_sl_all_b); set(L, PropName, PropValue);
    L = semilogy(n, n_error_sl_all_c); set(L, PropName, PropValue);
    
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 2, 2, 'auto'};
    set(gca, 'ColorOrderIndex', 1);
    
    L = semilogy(n, s_error_sl_all_max); set(L, PropName, PropValue);
    L = semilogy(n, n_error_sl_all_max_a); set(L, PropName, PropValue);
    L = semilogy(n, n_error_sl_all_max_b); set(L, PropName, PropValue);
    L = semilogy(n, n_error_sl_all_max_c); set(L, PropName, PropValue);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Solution Error');
    
    xmin = 0;
    xmax = n(end);
    
    ymin = min(min([s_error_sl_all_max; n_error_sl_all_max_a; n_error_sl_all_max_b; n_error_sl_all_max_c]));
    ymax = max(max([s_error_sl_all_max; n_error_sl_all_max_a; n_error_sl_all_max_b; n_error_sl_all_max_c]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Sym', 'SC 1', 'SC 2', 'SC 3', 'Sym (max)', 'SC 1 (max)', 'SC 2 (max)', 'SC 3 (max)');
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
    L = semilogy(n, n_error_gr_all_a); set(L, PropName, PropValue);
    L = semilogy(n, n_error_gr_all_b); set(L, PropName, PropValue);
    L = semilogy(n, n_error_gr_all_c); set(L, PropName, PropValue);
    
    set(gca, 'ColorOrderIndex', 1);
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 2, 2, 'auto'};
    
    L = semilogy(n, s_error_gr_all_max); set(L, PropName, PropValue);   
    L = semilogy(n, n_error_gr_all_max_a); set(L, PropName, PropValue); 
    L = semilogy(n, n_error_gr_all_max_b); set(L, PropName, PropValue);
    L = semilogy(n, n_error_gr_all_max_c); set(L, PropName, PropValue);
    
    hold off
    
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Gradient Error');
    
    xmin = 0;
    xmax = n(end);
    
    ymin = min(min([s_error_gr_all_max; n_error_gr_all_max_a; n_error_gr_all_max_b; n_error_gr_all_max_c]));
    ymax = max(max([s_error_gr_all_max; n_error_gr_all_max_a; n_error_gr_all_max_b; n_error_gr_all_max_c]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Sym', 'SC 1', 'SC 2', 'SC 3', 'Sym (max)', 'SC 1 (max)', 'SC 2 (max)', 'SC 3 (max)');
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
    
    L = loglog(h, s_error_sl_max); set(L, PropName, PropValue); set(L, 'Marker', 'o');
    
    hold on
    
    L = loglog(h, n_error_sl_max_a); set(L, PropName, PropValue); set(L, 'Marker', 'd'); set(L, 'MarkerSize', 6);
    L = loglog(h, n_error_sl_max_b); set(L, PropName, PropValue); set(L, 'Marker', '*'); set(L, 'MarkerSize', 6);
    L = loglog(h, n_error_sl_max_c); set(L, PropName, PropValue); set(L, 'Marker', 's');
    
    % guide lines
    a = mean(sqrt(s_error_sl_max.*n_error_sl_max_c)./h.^2);
    
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
    
    ymin = min(min([s_error_sl_max; n_error_sl_max_a; n_error_sl_max_b; n_error_sl_max_c]));
    ymax = max(max([s_error_sl_max; n_error_sl_max_a; n_error_sl_max_b; n_error_sl_max_c]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    xticks(round(10.^linspace(log10(xmin), log10(xmax), resolution_ticks_num), 1, 'significant'));
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Sym', 'SC 1', 'SC 2', 'SC 3', '2nd order');
    set(L, 'interpreter', 'latex');
    
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 3 2.5];
    
    % gradient error
    
    figure;
    
    % error plots
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'o', 1, 4, 'auto'};
    
    L = loglog(h, s_error_gr_max); set(L, PropName, PropValue);   set(L, 'Marker', 'o');
    
    hold on
    
    L = loglog(h, n_error_gr_max_a); set(L, PropName, PropValue); set(L, 'Marker', 'd'); set(L, 'MarkerSize', 6);
    L = loglog(h, n_error_gr_max_b); set(L, PropName, PropValue); set(L, 'Marker', '*'); set(L, 'MarkerSize', 6);
    L = loglog(h, n_error_gr_max_c); set(L, PropName, PropValue); set(L, 'Marker', 's');
    
    % guide lines
    s_a = max(s_error_gr_max./h.^1)*(max(s_error_gr_max)/min(s_error_gr_max))^0.1;
    n_a = max(n_error_gr_max_c./h.^2)*(max(n_error_gr_max_c)/min(n_error_gr_max_c))^0.1;
    
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
    
    ymin = min(min([s_error_gr_max; n_error_gr_max_a; n_error_gr_max_b; n_error_gr_max_c]));
    ymax = max(max([s_error_gr_max; n_error_gr_max_a; n_error_gr_max_b; n_error_gr_max_c]));
    
    yrel = ymax/ymin;
    
    ymin = ymin/yrel^0.1;
    ymax = ymax*yrel^0.1;
    
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    
    xticks(round(10.^linspace(log10(xmin), log10(xmax), resolution_ticks_num), 1, 'significant'));
    yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/error_ticks_num):ceil(log10(ymax))]);
    
    % legend
    L = legend('Sym', 'SC 1', 'SC 2', 'SC 3', '1st order', '2nd order');
    set(L, 'interpreter', 'latex');
    
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 3 2.5];
    
end