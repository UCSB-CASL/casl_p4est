clear;

% out_dir = '/home/dbochkov/Outputs/integration_mls/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/circle/convergence';

% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f0/2d/union/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f0/2d/difference/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f0/3d/union/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f0/3d/difference/convergence';

% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f1/2d/union/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f1/2d/difference/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f1/3d/union/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/f1/3d/difference/convergence';


out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/without_all/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/without_one/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/with_all/convergence';
% out_dir = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/with_one/convergence';

h = importdata(strcat(out_dir,'/h.txt'));

order_L = 2;
order_Q = 3;
order_Q2 = 4;
error_spacing = 2;

plot_intersections = 0;
plot_detailed_convergence = 0;
plot_condensed_convergence = 1;
plot_detailed_convergence2 = 0;

legend_fontsize = 10;


num_xticks = 3;
num_yticks = 5;

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

if (plot_detailed_convergence == 1)
    
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
    % xlim([h_min, h_max]);
    
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
        %     xlim([h_min, h_max]);
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
            %     xlim([h_min, h_max]);
        end
    end
    
end

% condensed convergence plots
if (plot_condensed_convergence == 1)
    
    figure;
    
    e_rat = max(max([l_max_id; l_max_isb; q_max_id; q_max_isb]))/min(min([l_max_id; l_max_isb; q_max_id; q_max_isb]));
    
%     if (plot_intersections == 1)
%         aL = max(max([l_max_id; l_max_isb; l_max_ix])./h.^order_L)*e_rat^0.03;
%         aQ = max(max([q_max_id; q_max_isb; q_max_ix])./h.^order_Q)*e_rat^0.03;
%         aQ2 = min(min([q_max_id; q_max_isb; q_max_ix])./h.^order_Q2);
%     else
        aL = max(max([l_max_id; l_max_isb])./h.^order_L)*e_rat^0.03;
        aQ = max(max([q_max_id; q_max_isb])./h.^order_Q)*e_rat^0.03;
        aQ2 = min(min([q_max_id; q_max_isb])./h.^order_Q2);
%     end
    
    PropName  = {'Marker','LineStyle', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'d','-', 1, 2, 'auto'};
    
    L = loglog(h, l_max_id);
    legend_str = {'$I_\Omega^L$'};
    
    set(L, PropName, PropValue);
    
    hold on
    for i=1:n_subs
        L = loglog(h, l_max_isb(i,:));
        set(L, PropName, PropValue);
        legend_str = [legend_str, strcat('$I_{\Gamma_',num2str(i),'}^L$') ];
    end
    
    if plot_intersections == 1
        for i=1:n_Xs
            L = loglog(h, l_max_ix(i,:));
            set(L, PropName, PropValue);
            legend_str = [legend_str, strcat('$I_{X_',num2str(i),'}^L$') ];
        end
    end
    
    loglog(h, aL*h.^order_L, '-k', 'LineWidth', 1);
    legend_str = [legend_str, strcat('$\sim h^',num2str(order_L),'$') ];
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'s', 1, 2, 'auto'};
    
    set(gca, 'ColorOrderIndex', 1);
    
    L = loglog(h, q_max_id);
    legend_str = [legend_str, '$I_\Omega^Q$' ];
    
    set(L, PropName, PropValue);
    
    for i=1:n_subs
        L = loglog(h, q_max_isb(i,:));
        set(L, PropName, PropValue);
        legend_str = [legend_str, strcat('$I_{\Gamma_',num2str(i),'}^Q$') ];
    end
    
    if plot_intersections == 1
        for i=1:n_Xs
            L = loglog(h, q_max_ix(i,:));
            set(L, PropName, PropValue);
            legend_str = [legend_str, strcat('$I_{X_',num2str(i),'}^Q$') ];
        end
    end
    
    loglog(h, aQ*h.^order_Q, '-k', 'LineWidth', 2);
    legend_str = [legend_str, strcat('$\sim h^',num2str(order_Q),'$')];
        
    loglog(h, aQ2*h.^order_Q2, ':k', 'LineWidth', 2);
    legend_str = [legend_str, strcat('$\sim h^',num2str(order_Q2),'$')];
    
    hold off
    
    grid on
    
    L = legend(legend_str);
    set(L, 'interpreter', 'latex');    
    set(L, 'FontSize', legend_fontsize);
    
    % axes
    h_min = min(h)/((max(h)/min(h))^(0.07));
    h_max = max(h)*((max(h)/min(h))^(0.4));
    
    e_max = max(max([l_max_id; l_max_isb; q_max_id; q_max_isb]))*e_rat^0.1;
    e_min = min(min([l_max_id; l_max_isb; q_max_id; q_max_isb]))/e_rat^0.1;
    
    xlim([h_min, h_max]);
    ylim([e_min, e_max]);
    
    xlabel('Grid resolution');
    ylabel('Integration error');
    
    xticks(round(10.^linspace(log10(min(h)), log10(max(h)), num_xticks), 1, 'significant'));
    yticks(10.^[floor(log10(e_min)):round((log10(e_max)-log10(e_min))/num_yticks):ceil(log10(e_max))]);
    
    % figure size
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 3 2.5];
    
end

if (plot_detailed_convergence2 == 1)
    
    figure
    
%     subplot(1, 3, 1:2);
    
    PropName  = {'Marker', 'LineStyle','LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', '-', 1, 4, 'none'};
    
    L = semilogy(q_all_id);
    set(L, PropName, PropValue);
    
    legend_str = {'$I_\Omega^Q$'};
    hold on
    for i=1:n_subs
        L = semilogy(q_all_isb(i,:));
        set(L, PropName, PropValue);
        legend_str = [ legend_str, strcat('$I_{\Gamma_',num2str(i),'}^Q$') ];
    end
    
    if plot_intersections == 1
        for i=1:n_Xs
            L = semilogy(q_all_ix(i,:));
            set(L, PropName, PropValue);
            legend_str = [ legend_str, strcat('$I_{X_',num2str(i),'}^Q$') ];
        end
    end
    
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'none', 2, 2, 'auto'};
    
    set(gca, 'ColorOrderIndex', 1);
    L = semilogy(q_all_max_id);
    set(L, PropName, PropValue);
    legend_str = [ legend_str, '$\max(I_\Omega^Q)$' ];
    
    for i=1:n_subs
        L = semilogy(q_all_max_isb(i,:));
        set(L, PropName, PropValue);
        legend_str = [ legend_str, strcat('$\max(I_{\Gamma_',num2str(i),'}^Q)$') ];
    end
    
    if plot_intersections == 1
        for i=1:n_Xs
            L = semilogy(q_all_max_ix(i,:));
            set(L, PropName, PropValue);
            legend_str = [ legend_str, strcat('$\max(I_{X_',num2str(i),'}^Q)$') ];
        end
    end
    
    hold off
    
    xlabel('Case no.');
    ylabel('Integration error');
    
    A = [q_all_id; q_all_isb];
    e_rat = max(max(A))/min(min(A(A>0)));

    e_max = max(max(A(A>0)))*e_rat^0.1;
    e_min = min(min(A(A>0)))/e_rat^0.1;
    
    yticks(10.^[floor(log10(e_min)):round((log10(e_max)-log10(e_min))/num_yticks):ceil(log10(e_max))]);
    
    xlims = [0, length(q_all_id)];
    ylims = [e_min, e_max];
    
    xlim(xlims);
    ylim(ylims);
    
    grid on
    L = legend(legend_str);
    set(L, 'interpreter', 'latex');    
    set(L, 'FontSize', legend_fontsize);
    
    
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 8 2.5];
    
    
%     subplot(1, 3, 3);
    
    figure;
        
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
    PropValue = {'s', 2, 4, 'auto'};
    
    L = loglog(h, q_max_id);
    set(L, PropName, PropValue);
    
    legend_str = {'$\max(I_\Omega^Q)$' };
    hold on
    
    
    for i=1:n_subs
        L = loglog(h, q_max_isb(i,:));
        set(L, PropName, PropValue);
        legend_str = [ legend_str, strcat('$\max(I_{\Gamma_',num2str(i),'}^Q)$') ];
    end
    
    if plot_intersections == 1
        for i=1:n_Xs
            L = loglog(h, q_max_ix(i,:));
            set(L, PropName, PropValue);
            legend_str = [ legend_str, strcat('$\max(I_{X_',num2str(i),'}^Q)$') ];
        end
    end
    
    aQ = max(max([q_max_id; q_max_isb])./h.^order_Q);
    aQ2 = min(min([q_max_id; q_max_isb])./h.^order_Q2);
    
    loglog(h, aQ*h.^order_Q, '-k', 'LineWidth', 2);
    legend_str = [legend_str, strcat('$\sim h^',num2str(order_Q),'$')];
        
    loglog(h, aQ2*h.^order_Q2, ':k', 'LineWidth', 2);
    legend_str = [legend_str, strcat('$\sim h^',num2str(order_Q2),'$')];
    hold off

    grid on
    
    xlabel('Grid resolution');
    ylabel('Integration error');
    
    % axes
    h_min = min(h)/((max(h)/min(h))^(0.07));
    h_max = max(h)*((max(h)/min(h))^(0.07));
    
    e_rat = max(max([q_max_id; q_max_isb]))/min(min([q_max_id; q_max_isb]));

    e_max = max(max([q_max_id; q_max_isb]))*e_rat^0.1;
    e_min = min(min([q_max_id; q_max_isb]))/e_rat^0.1;
    
    xlim([h_min, h_max]);
    ylim([e_min, e_max]);
    
    xticks(round(10.^linspace(log10(min(h)), log10(max(h)), num_xticks), 1, 'significant'));
    yticks(10.^[floor(log10(e_min)):round((log10(e_max)-log10(e_min))/num_yticks):ceil(log10(e_max))]);
    
    L = legend(legend_str);
    set(L, 'interpreter', 'latex');    
    set(L, 'FontSize', legend_fontsize);
    
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 3 2.5];
end
