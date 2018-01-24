clear;
plot_intersections = 0;

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


if plot_intersections == 1
l_all_ix = importdata(strcat(out_dir,'/l_all_ix.txt'));
q_all_ix = importdata(strcat(out_dir,'/q_all_ix.txt'));
l_avg_ix = importdata(strcat(out_dir,'/l_avg_ix.txt'));
q_avg_ix = importdata(strcat(out_dir,'/q_avg_ix.txt'));
l_dev_ix = importdata(strcat(out_dir,'/l_dev_ix.txt'));
q_dev_ix = importdata(strcat(out_dir,'/q_dev_ix.txt'));
l_one_ix = importdata(strcat(out_dir,'/l_one_ix.txt'));
q_one_ix = importdata(strcat(out_dir,'/q_one_ix.txt'));
l_max_ix = importdata(strcat(out_dir,'/l_max_ix.txt'));
q_max_ix = importdata(strcat(out_dir,'/q_max_ix.txt'));
end

num_resolutions = length(h)
num_shifts = length(l_all_id)/num_resolutions

q_all_max_id = q_all_id;
q_all_avg_id = q_all_id;
q_all_max_isb = q_all_isb;
q_all_avg_isb = q_all_isb;
if plot_intersections == 1
q_all_max_ix = q_all_ix;
q_all_avg_ix = q_all_ix;
end

s = size(l_all_isb);
n_subs = s(1);

n_Xs = 0;
if plot_intersections == 1
s = size(l_all_ix);
n_Xs = s(1);
end

for i=1:num_resolutions
    q_all_max_id(:, (i-1)*num_shifts+1:i*num_shifts) = q_max_id(:,i)*ones(1,num_shifts);
    q_all_avg_id(:, (i-1)*num_shifts+1:i*num_shifts) = q_avg_id(:,i)*ones(1,num_shifts);
    q_all_max_isb(:, (i-1)*num_shifts+1:i*num_shifts) = q_max_isb(:,i)*ones(1,num_shifts);
    q_all_avg_isb(:, (i-1)*num_shifts+1:i*num_shifts) = q_avg_isb(:,i)*ones(1,num_shifts);
end

if plot_intersections == 1
for i=1:num_resolutions
    q_all_max_ix(:, (i-1)*num_shifts+1:i*num_shifts) = q_max_ix(:,i)*ones(1,num_shifts);
    q_all_avg_ix(:, (i-1)*num_shifts+1:i*num_shifts) = q_avg_ix(:,i)*ones(1,num_shifts);
end
end

figure
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

if plot_intersections == 1
for i=1:n_Xs
    subplot(num_subplots, 3, (i+n_subs)*3+1:(i+n_subs)*3+2);
    
    semilogy(l_all_ix(i,:), '.-', 'linewidth', 2);
    hold on
    semilogy(q_all_ix(i,:), '.-', 'linewidth', 2);
    semilogy(q_all_max_ix(i,:), 'linewidth', 2);
    semilogy(q_all_avg_ix(i,:), 'linewidth', 2);
    hold off
end
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

if plot_intersections == 1
for i=1:n_Xs
    subplot(num_subplots, 3, (i+1+n_subs)*3);
    loglog(h, l_avg_ix(i,:), 'linewidth', 2);
    hold on
    loglog(h, l_one_ix(i,:), 'linewidth', 2);
    loglog(h, l_max_ix(i,:), 'linewidth', 2);
    loglog(h, q_avg_ix(i,:), '-k', 'markersize', 4, 'linewidth', 2);
    loglog(h, q_one_ix(i,:), '-sr', 'markersize', 4, 'linewidth', 1);
    loglog(h, q_max_ix(i,:), 'ob', 'markersize', 4, 'linewidth', 2);
    hold off
    xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
end
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

if plot_intersections == 1
for i=1:n_Xs
    L = loglog(h, l_max_ix(i,:));
    set(L, PropName, PropValue);
end
end

PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue = {'s', 2, 4, 'auto'};
set(gca, 'ColorOrderIndex', 1);
loglog(h, q_max_id, 'linewidth', 2);
for i=1:n_subs
    L = loglog(h, q_max_isb(i,:));
    set(L, PropName, PropValue);
end

if plot_intersections == 1
for i=1:n_Xs
    L = loglog(h, q_max_ix(i,:));
    set(L, PropName, PropValue);
end
end

% find coefficient for second order line
a2 = 1*max(max([l_max_id; l_max_isb])./h.^2);
a3 = 1*min(mean([q_max_isb])./h.^3);
a4 = 1*min(([q_max_id])./h.^4);

loglog(h, a2*h.^2, '-k', 'LineWidth', 1);
loglog(h, a3*h.^3, '-k', 'LineWidth', 2);
loglog(h, a4*h.^4, '-k', 'LineWidth', 3);
hold off
xlim([min(h)/((max(h)/min(h))^(0.07)), max(h)*((max(h)/min(h))^(0.07)) ]);
