clear;
plot_intersections = 0;

out_dir = '/home/dbochkov/Outputs/nodes_mls/convergence';

h = importdata(strcat(out_dir,'/h.txt'));

error_sl_all = importdata(strcat(out_dir,'/error_sl_all.txt'));
error_gr_all = importdata(strcat(out_dir,'/error_gr_all.txt'));
error_dd_all = importdata(strcat(out_dir,'/error_dd_all.txt'));
error_tr_all = importdata(strcat(out_dir,'/error_tr_all.txt'));
error_ex_all = importdata(strcat(out_dir,'/error_ex_all.txt'));

error_sl_max = importdata(strcat(out_dir,'/error_sl_max.txt'));
error_gr_max = importdata(strcat(out_dir,'/error_gr_max.txt'));
error_dd_max = importdata(strcat(out_dir,'/error_dd_max.txt'));
error_tr_max = importdata(strcat(out_dir,'/error_tr_max.txt'));
error_ex_max = importdata(strcat(out_dir,'/error_ex_max.txt'));

error_sl_one = importdata(strcat(out_dir,'/error_sl_one.txt'));
error_gr_one = importdata(strcat(out_dir,'/error_gr_one.txt'));
error_dd_one = importdata(strcat(out_dir,'/error_dd_one.txt'));
error_tr_one = importdata(strcat(out_dir,'/error_tr_one.txt'));
error_ex_one = importdata(strcat(out_dir,'/error_ex_one.txt'));

error_sl_avg = importdata(strcat(out_dir,'/error_sl_avg.txt'));
error_gr_avg = importdata(strcat(out_dir,'/error_gr_avg.txt'));
error_dd_avg = importdata(strcat(out_dir,'/error_dd_avg.txt'));
error_tr_avg = importdata(strcat(out_dir,'/error_tr_avg.txt'));
error_ex_avg = importdata(strcat(out_dir,'/error_ex_avg.txt'));

num_resolutions = length(h);
num_shifts = length(error_sl_all)/num_resolutions;

error_sl_all_avg = error_sl_all;
error_gr_all_avg = error_gr_all;
error_dd_all_avg = error_dd_all;
error_tr_all_avg = error_tr_all;
error_ex_all_avg = error_ex_all;

error_sl_all_max = error_sl_all;
error_gr_all_max = error_gr_all;
error_dd_all_max = error_dd_all;
error_tr_all_max = error_tr_all;
error_ex_all_max = error_ex_all;

for i=1:num_resolutions
    error_sl_all_avg(:, (i-1)*num_shifts+1:i*num_shifts) = error_sl_avg(:,i)*ones(1,num_shifts);
    error_gr_all_avg(:, (i-1)*num_shifts+1:i*num_shifts) = error_gr_avg(:,i)*ones(1,num_shifts);
    error_dd_all_avg(:, (i-1)*num_shifts+1:i*num_shifts) = error_dd_avg(:,i)*ones(1,num_shifts);
    error_tr_all_avg(:, (i-1)*num_shifts+1:i*num_shifts) = error_tr_avg(:,i)*ones(1,num_shifts);
    error_ex_all_avg(:, (i-1)*num_shifts+1:i*num_shifts) = error_ex_avg(:,i)*ones(1,num_shifts);
    
    error_sl_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_sl_max(:,i)*ones(1,num_shifts);
    error_gr_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_gr_max(:,i)*ones(1,num_shifts);
    error_dd_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_dd_max(:,i)*ones(1,num_shifts);
    error_tr_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_tr_max(:,i)*ones(1,num_shifts);
    error_ex_all_max(:, (i-1)*num_shifts+1:i*num_shifts) = error_ex_max(:,i)*ones(1,num_shifts);
end

figure
num_subplots = 5;
n = 0:length(error_sl_all)-1;

subplot(num_subplots, 3, 1:2);

semilogy(n, error_sl_all);
hold on
semilogy(n, error_sl_all_avg);
semilogy(n, error_sl_all_max);
hold off

subplot(num_subplots, 3, 4:5);

semilogy(n, error_gr_all);
hold on
semilogy(n, error_gr_all_avg);
semilogy(n, error_gr_all_max);
hold off

subplot(num_subplots, 3, 7:8);

semilogy(n, error_dd_all);
hold on
semilogy(n, error_dd_all_avg);
semilogy(n, error_dd_all_max);
hold off

subplot(num_subplots, 3, 10:11);

semilogy(n, error_tr_all);
hold on
semilogy(n, error_tr_all_avg);
semilogy(n, error_tr_all_max);
hold off

subplot(num_subplots, 3, 13:14);

semilogy(n, error_ex_all);
hold on
semilogy(n, error_ex_all_avg);
semilogy(n, error_ex_all_max);
hold off

subplot(num_subplots, 3, 3);

loglog(h, error_sl_avg);
hold on
loglog(h, error_sl_max);
loglog(h, error_sl_one);
a = 1*max(([error_sl_max])./h.^2);
loglog(h, a*h.^2, '-k', 'LineWidth', 2);
hold off

xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);

subplot(num_subplots, 3, 6);

loglog(h, error_gr_avg);
hold on
loglog(h, error_gr_max);
loglog(h, error_gr_one);
a = 1*max(([error_gr_max])./h.^2);
loglog(h, a*h.^2, '-k', 'LineWidth', 2);
hold off

xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);

subplot(num_subplots, 3, 9);

loglog(h, error_dd_avg);
hold on
loglog(h, error_dd_max);
loglog(h, error_dd_one);
a = 1*max(([error_dd_max])./h.^1);
loglog(h, a*h.^1, '-k', 'LineWidth', 2);
hold off

xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);

subplot(num_subplots, 3, 12);

loglog(h, error_tr_avg);
hold on
loglog(h, error_tr_max);
loglog(h, error_tr_one);
a = 1*max(([error_tr_max])./h.^1);
loglog(h, a*h.^1, '-k', 'LineWidth', 2);
hold off

xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);

subplot(num_subplots, 3, 15);

loglog(h, error_ex_avg);
hold on
loglog(h, error_ex_max);
loglog(h, error_ex_one);
a = 1*max(([error_ex_max])./h.^2);
loglog(h, a*h.^2, '-k', 'LineWidth', 2);
hold off

xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);

% for i=1:n_subs
%     subplot(num_subplots, 3, i*3+1:i*3+2);
%     
%     semilogy(l_all_isb(i,:), '.-', 'linewidth', 2);
%     hold on
%     semilogy(q_all_isb(i,:), '.-', 'linewidth', 2);
%     semilogy(q_all_max_isb(i,:), 'linewidth', 2);
%     semilogy(q_all_avg_isb(i,:), 'linewidth', 2);
%     hold off
% end
% 
% if plot_intersections == 1
% for i=1:n_Xs
%     subplot(num_subplots, 3, (i+n_subs)*3+1:(i+n_subs)*3+2);
%     
%     semilogy(l_all_ix(i,:), '.-', 'linewidth', 2);
%     hold on
%     semilogy(q_all_ix(i,:), '.-', 'linewidth', 2);
%     semilogy(q_all_max_ix(i,:), 'linewidth', 2);
%     semilogy(q_all_avg_ix(i,:), 'linewidth', 2);
%     hold off
% end
% end
% 
% 
% subplot(num_subplots, 3, 3);
% loglog(h, l_avg_id, 'linewidth', 2);
% hold on
% loglog(h, l_one_id, 'linewidth', 2);
% loglog(h, l_max_id, 'linewidth', 2);
% loglog(h, q_avg_id, '-k', 'markersize', 4, 'linewidth', 2);
% loglog(h, q_one_id, '-sr', 'markersize', 4, 'linewidth', 1);
% loglog(h, q_max_id, 'ob', 'markersize', 4, 'linewidth', 2);
% hold off
% xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
% 
% for i=1:n_subs
%     subplot(num_subplots, 3, (i+1)*3);
%     loglog(h, l_avg_isb(i,:), 'linewidth', 2);
%     hold on
%     loglog(h, l_one_isb(i,:), 'linewidth', 2);
%     loglog(h, l_max_isb(i,:), 'linewidth', 2);
%     loglog(h, q_avg_isb(i,:), '-k', 'markersize', 4, 'linewidth', 2);
%     loglog(h, q_one_isb(i,:), '-sr', 'markersize', 4, 'linewidth', 1);
%     loglog(h, q_max_isb(i,:), 'ob', 'markersize', 4, 'linewidth', 2);
%     hold off
%     xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
% end
% 
% if plot_intersections == 1
% for i=1:n_Xs
%     subplot(num_subplots, 3, (i+1+n_subs)*3);
%     loglog(h, l_avg_ix(i,:), 'linewidth', 2);
%     hold on
%     loglog(h, l_one_ix(i,:), 'linewidth', 2);
%     loglog(h, l_max_ix(i,:), 'linewidth', 2);
%     loglog(h, q_avg_ix(i,:), '-k', 'markersize', 4, 'linewidth', 2);
%     loglog(h, q_one_ix(i,:), '-sr', 'markersize', 4, 'linewidth', 1);
%     loglog(h, q_max_ix(i,:), 'ob', 'markersize', 4, 'linewidth', 2);
%     hold off
%     xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
% end
% end
% 
% c = lines(7);
% % subplot(1, 1, 1);
% 
% 
% 
% PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
% PropValue = {'o', 2, 4, 'auto'};
% figure;
% L = loglog(h, l_max_id);
% set(L, PropName, PropValue);
% hold on
% for i=1:n_subs
%     L = loglog(h, l_max_isb(i,:));
%     set(L, PropName, PropValue);
% end
% 
% if plot_intersections == 1
% for i=1:n_Xs
%     L = loglog(h, l_max_ix(i,:));
%     set(L, PropName, PropValue);
% end
% end
% 
% PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
% PropValue = {'s', 2, 4, 'auto'};
% set(gca, 'ColorOrderIndex', 1);
% loglog(h, q_max_id, 'linewidth', 2);
% for i=1:n_subs
%     L = loglog(h, q_max_isb(i,:));
%     set(L, PropName, PropValue);
% end
% 
% if plot_intersections == 1
% for i=1:n_Xs
%     L = loglog(h, q_max_ix(i,:));
%     set(L, PropName, PropValue);
% end
% end
% 
% % find coefficient for second order line
% a2 = 1*max(max([l_max_id; l_max_isb])./h.^2);
% a3 = 1*min(mean([q_max_isb])./h.^4);
% a4 = 1*min(([q_max_id])./h.^4);
% 
% loglog(h, a2*h.^2, '-k', 'LineWidth', 1);
% loglog(h, a3*h.^4, '-k', 'LineWidth', 2);
% loglog(h, a4*h.^4, '-k', 'LineWidth', 3);
% hold off
% xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
