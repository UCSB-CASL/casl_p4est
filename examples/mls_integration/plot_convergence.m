clear;

out_dir = '/home/dbochkov/Outputs/integration_mls/convergence';

h = importdata(strcat(out_dir,'/h.txt'));

l_all_id = importdata(strcat(out_dir,'/l_all_id.txt'));
q_all_id = importdata(strcat(out_dir,'/q_all_id.txt'));
l_avg_id = importdata(strcat(out_dir,'/l_avg_id.txt'));
q_avg_id = importdata(strcat(out_dir,'/q_avg_id.txt'));
l_dev_id = importdata(strcat(out_dir,'/l_dev_id.txt'));
q_dev_id = importdata(strcat(out_dir,'/q_dev_id.txt'));
l_one_id = importdata(strcat(out_dir,'/l_one_id.txt'));
q_one_id = importdata(strcat(out_dir,'/q_one_id.txt'));
l_max_id = importdata(strcat(out_dir,'/l_max_id.txt'));
q_max_id = importdata(strcat(out_dir,'/q_max_id.txt'));


l_all_isb = importdata(strcat(out_dir,'/l_all_isb.txt'));
q_all_isb = importdata(strcat(out_dir,'/q_all_isb.txt'));
l_avg_isb = importdata(strcat(out_dir,'/l_avg_isb.txt'));
q_avg_isb = importdata(strcat(out_dir,'/q_avg_isb.txt'));
l_dev_isb = importdata(strcat(out_dir,'/l_dev_isb.txt'));
q_dev_isb = importdata(strcat(out_dir,'/q_dev_isb.txt'));
l_one_isb = importdata(strcat(out_dir,'/l_one_isb.txt'));
q_one_isb = importdata(strcat(out_dir,'/q_one_isb.txt'));
l_max_isb = importdata(strcat(out_dir,'/l_max_isb.txt'));
q_max_isb = importdata(strcat(out_dir,'/q_max_isb.txt'));

num_resolutions = length(h)
num_shifts = length(l_all_id)/num_resolutions

q_all_max_id = q_all_id;
q_all_avg_id = q_all_id;
q_all_max_isb = q_all_isb;
q_all_avg_isb = q_all_isb;

s = size(l_all_isb);
n_subs = s(1);

n_Xs = 0;

for i=1:num_resolutions
    q_all_max_id(:, (i-1)*num_shifts+1:i*num_shifts) = q_max_id(:,i)*ones(1,num_shifts);
    q_all_avg_id(:, (i-1)*num_shifts+1:i*num_shifts) = q_avg_id(:,i)*ones(1,num_shifts);
    q_all_max_isb(:, (i-1)*num_shifts+1:i*num_shifts) = q_max_isb(:,i)*ones(1,num_shifts);
    q_all_avg_isb(:, (i-1)*num_shifts+1:i*num_shifts) = q_avg_isb(:,i)*ones(1,num_shifts);
end

num_subplots = 1 + n_subs + n_Xs;

subplot(num_subplots, 3, 1:2);
semilogy(l_all_id, '.-', 'linewidth', 2);
hold on
semilogy(q_all_id, '.-', 'linewidth', 2);
semilogy(q_all_max_id, 'linewidth', 2);
semilogy(q_all_avg_id, 'linewidth', 2);
hold off

for i=1:n_subs
    subplot(num_subplots, 3, i*3+1:i*3+2);
    
    semilogy(l_all_isb(i,:), '.-', 'linewidth', 2);
    hold on
    semilogy(q_all_isb(i,:), '.-', 'linewidth', 2);
    semilogy(q_all_max_isb(i,:), 'linewidth', 2);
    semilogy(q_all_avg_isb(i,:), 'linewidth', 2);
    hold off
end


subplot(num_subplots, 3, 3);
loglog(h, l_avg_id, 'linewidth', 2);
hold on
loglog(h, l_one_id, 'linewidth', 2);
loglog(h, l_max_id, 'linewidth', 2);
loglog(h, q_avg_id, '-k', 'markersize', 4, 'linewidth', 2);
loglog(h, q_one_id, '-sr', 'markersize', 4, 'linewidth', 1);
loglog(h, q_max_id, 'ob', 'markersize', 4, 'linewidth', 2);
hold off
xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);

for i=1:n_subs
    subplot(num_subplots, 3, (i+1)*3);
    loglog(h, l_avg_isb(i,:), 'linewidth', 2);
    hold on
    loglog(h, l_one_isb(i,:), 'linewidth', 2);
    loglog(h, l_max_isb(i,:), 'linewidth', 2);
    loglog(h, q_avg_isb(i,:), '-k', 'markersize', 4, 'linewidth', 2);
    loglog(h, q_one_isb(i,:), '-sr', 'markersize', 4, 'linewidth', 1);
    loglog(h, q_max_isb(i,:), 'ob', 'markersize', 4, 'linewidth', 2);
    hold off
    xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
end

c = lines(7);
% subplot(1, 1, 1);



PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue = {'o', 2, 4, 'auto'};
figure;
L = loglog(h, l_max_id);
set(L, PropName, PropValue);
hold on
for i=1:n_subs
    L = loglog(h, l_max_isb(i,:));
    set(L, PropName, PropValue);
end
set(gca, 'ColorOrderIndex', 1);
loglog(h, q_max_id, 'linewidth', 2);
for i=1:n_subs
    loglog(h, q_max_isb(i,:), 'linewidth', 2);
end

% find coefficient for second order line
a2 = 1*max(max([l_max_id; l_max_isb])./h.^2);
a3 = 1*max(max([q_max_isb])./h.^3);
a4 = 1*min(([q_max_id])./h.^3);

loglog(h, a2*h.^2, '-k', 'LineWidth', 1);
loglog(h, a3*h.^3, '-k', 'LineWidth', 2);
loglog(h, a4*h.^3, '-k', 'LineWidth', 3);
hold off
xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
