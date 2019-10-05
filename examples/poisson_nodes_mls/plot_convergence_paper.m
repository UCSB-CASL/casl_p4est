clear;
plot_intersections = 0;


out_dir = {'/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_1st_order/convergence', ...
           '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_2nd_order_a/convergence', ...
           '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_2nd_order_b/convergence', ...
           '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/poisson/2d/triangle/gradients_2nd_order_c/convergence'};
       
legend_str = {'Symmetric', 'Non-symmetric 1', 'Non-symmetric 2', 'Non-symmetric 3'};

S0_error_sl_all = importdata(strcat(out_dir{1},'/error_sl_all.txt'));
S0_error_gr_all = importdata(strcat(out_dir{1},'/error_gr_all.txt'));
S0_error_sl_max = importdata(strcat(out_dir{1},'/error_sl_max.txt'));
S0_error_gr_max = importdata(strcat(out_dir{1},'/error_gr_max.txt'));

PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue = {'none', 0.5, 2, 'auto'};
colors = lines(8);
figure
set(gca, 'ColorOrderIndex', 2);
for j = 1:length(out_dir)
    h = importdata(strcat(out_dir{j},'/h.txt'));
    
    error_sl_all = importdata(strcat(out_dir{j},'/error_sl_all.txt'));
    error_gr_all = importdata(strcat(out_dir{j},'/error_gr_all.txt'));
    
    error_sl_max = importdata(strcat(out_dir{j},'/error_sl_max.txt'));
    error_gr_max = importdata(strcat(out_dir{j},'/error_gr_max.txt'));
    
    num_resolutions = length(h);
    num_shifts = length(error_sl_all)/num_resolutions;
    
    error_sl_all_avg = error_sl_all;
    error_gr_all_avg = error_gr_all;
    
    error_sl_all_max = error_sl_all;
    error_gr_all_max = error_gr_all;
    
    for i=1:num_resolutions        
        error_sl_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_sl_max(:,i)*ones(1,num_shifts);
        error_gr_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_gr_max(:,i)*ones(1,num_shifts);
    end
    
    n = 0:length(error_sl_all)-1;
    
    L = semilogy(n, error_gr_all, 'Color', colors(j+1,:));
    set(L, PropName, PropValue);
    
    if j == 1
        hold on
    end
    
end

set(gca, 'ColorOrderIndex', 2);
PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue = {'none', 2, 2, 'auto'};
    
for j = 1:length(out_dir)
    h = importdata(strcat(out_dir{j},'/h.txt'));
    
    error_sl_all = importdata(strcat(out_dir{j},'/error_sl_all.txt'));
    error_gr_all = importdata(strcat(out_dir{j},'/error_gr_all.txt'));
    
    error_sl_max = importdata(strcat(out_dir{j},'/error_sl_max.txt'));
    error_gr_max = importdata(strcat(out_dir{j},'/error_gr_max.txt'));
    
    num_resolutions = length(h);
    num_shifts = length(error_sl_all)/num_resolutions;
    
    error_sl_all_avg = error_sl_all;
    error_gr_all_avg = error_gr_all;
    
    error_sl_all_max = error_sl_all;
    error_gr_all_max = error_gr_all;
    
    for i=1:num_resolutions        
        error_sl_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_sl_max(:,i)*ones(1,num_shifts);
        error_gr_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_gr_max(:,i)*ones(1,num_shifts);
    end
    
    n = 0:length(error_sl_all)-1;
    
    L = semilogy(n, error_gr_all_max);
    set(L, PropName, PropValue);
    
end
hold off

xlim([0,n(end)]);
grid on
xlabel('Case no.');
ylabel('Gradient Error');
fig = gcf;
fig.Units = 'inches';
fig.Position = [10 10 8 2.5];
    
% figure
% num_subplots = 5;
% n = 0:length(error_sl_all)-1;
% 
% subplot(num_subplots, 3, 1:2);
% 
% semilogy(n, error_sl_all);
% hold on
% semilogy(n, error_sl_all_avg);
% semilogy(n, error_sl_all_max);
% hold off
% 
% subplot(num_subplots, 3, 4:5);
% 
% semilogy(n, error_gr_all);
% hold on
% semilogy(n, error_gr_all_avg);
% semilogy(n, error_gr_all_max);
% hold off
% 
% subplot(num_subplots, 3, 7:8);
% 
% semilogy(n, error_dd_all);
% hold on
% semilogy(n, error_dd_all_avg);
% semilogy(n, error_dd_all_max);
% hold off
% 
% subplot(num_subplots, 3, 10:11);
% 
% semilogy(n, error_tr_all);
% hold on
% semilogy(n, error_tr_all_avg);
% semilogy(n, error_tr_all_max);
% hold off
% 
% subplot(num_subplots, 3, 13:14);
% 
% semilogy(n, error_ex_all);
% hold on
% semilogy(n, error_ex_all_avg);
% semilogy(n, error_ex_all_max);
% hold off
% 
% subplot(num_subplots, 3, 3);
% 
% loglog(h, error_sl_avg);
% hold on
% loglog(h, error_sl_max);
% loglog(h, error_sl_one);
% a = 1*max(([error_sl_max])./h.^2);
% loglog(h, a*h.^2, '-k', 'LineWidth', 2);
% hold off
% 
% xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
% 
% subplot(num_subplots, 3, 6);
% 
% loglog(h, error_gr_avg);
% hold on
% loglog(h, error_gr_max);
% loglog(h, error_gr_one);
% a = 1*max(([error_gr_max])./h.^2);
% loglog(h, a*h.^2, '-k', 'LineWidth', 2);
% hold off
% 
% xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
% 
% subplot(num_subplots, 3, 9);
% 
% loglog(h, error_dd_avg);
% hold on
% loglog(h, error_dd_max);
% loglog(h, error_dd_one);
% a = 1*max(([error_dd_max])./h.^1);
% loglog(h, a*h.^1, '-k', 'LineWidth', 2);
% hold off
% 
% xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
% 
% subplot(num_subplots, 3, 12);
% 
% loglog(h, error_tr_avg);
% hold on
% loglog(h, error_tr_max);
% loglog(h, error_tr_one);
% a = 1*max(([error_tr_max])./h.^1);
% loglog(h, a*h.^1, '-k', 'LineWidth', 2);
% hold off
% 
% xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
% 
% subplot(num_subplots, 3, 15);
% 
% loglog(h, error_ex_avg);
% hold on
% loglog(h, error_ex_max);
% loglog(h, error_ex_one);
% a = 1*max(([error_ex_max])./h.^2);
% loglog(h, a*h.^2, '-k', 'LineWidth', 2);
% hold off
% 
% xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
