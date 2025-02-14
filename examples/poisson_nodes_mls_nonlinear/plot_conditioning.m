clear;

subplotted = 0;

plot_detailed_convergence = 0;
plot_condensed_convergence = 1;
plot_detailed_cond_num = 0;
plot_condensed_cond_num = 1;

plot_slope_sl = 1;
plot_slope_gr = 1;
plot_slope_cn = 1;

use_n_points = 1;

PropName_all  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue_all = {'none', 1, 2, 'auto'};
    
PropName_all_max  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue_all_max = {'none', 2, 2, 'auto'};
    
PropName_max  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'LineStyle'};
PropValue_max = {'o', 1, 4, 'auto', '-'};

PropName_guide  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'LineStyle', 'Color'};
PropValue_guide = {'none', 1, 2, 'auto', '-', 'k'};

% -----------------------------
% smooth solutions
% -----------------------------
figure; 

% dirs = {}; titles = {};
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/v0.1/2d/conditioning/slow/convergence'; titles{end+1} = 'Slow';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/v0.1/2d/conditioning/fast/convergence'; titles{end+1} = 'Fast';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/v0.1/2d/conditioning/neut/convergence'; titles{end+1} = 'Random';

% dirs = {}; titles = {};
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/2d/single/conditioning/slow/convergence'; titles{end+1} = 'Slow';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/2d/single/conditioning/fast/convergence'; titles{end+1} = 'Fast';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/2d/single/conditioning/neut/convergence'; titles{end+1} = 'Random';

% dirs = {}; titles = {};
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/2d/multiple/conditioning/slow/convergence'; titles{end+1} = 'Bias Slow';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/2d/multiple/conditioning/fast/convergence'; titles{end+1} = 'Bias Fast';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/2d/multiple/conditioning/neut/convergence'; titles{end+1} = 'Random';

% dirs = {}; titles = {};
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/3d/single/conditioning/slow/convergence'; titles{end+1} = 'Slow';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/3d/single/conditioning/fast/convergence'; titles{end+1} = 'Fast';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/3d/single/conditioning/neut/convergence'; titles{end+1} = 'Random';

dirs = {}; titles = {};
dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/3d/multiple/conditioning/slow/convergence'; titles{end+1} = 'Bias Slow';
dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/3d/multiple/conditioning/fast/convergence'; titles{end+1} = 'Bias Fast';
dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/3d/multiple/conditioning/neut/convergence'; titles{end+1} = 'Random';

% dirs = {}; titles = {};
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/v0.1/voro/2d/conditioning/slow/convergence'; titles{end+1} = 'Slow';

% dirs = {}; titles = {};
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Presentations/13_socal_fluids_2019_jump_solver/data/results/v1.0/2d/conditioning/fvm/slow/convergence'; titles{end+1} = 'Slow';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Presentations/13_socal_fluids_2019_jump_solver/data/results/v1.0/2d/conditioning/fvm/fast/convergence'; titles{end+1} = 'Fast';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Presentations/13_socal_fluids_2019_jump_solver/data/results/v1.0/2d/conditioning/fvm/neut/convergence'; titles{end+1} = 'Random';

% dirs = {}; titles = {};
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Presentations/13_socal_fluids_2019_jump_solver/data/results/v1.0/3d/conditioning/fvm/slow/convergence'; titles{end+1} = 'Slow';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Presentations/13_socal_fluids_2019_jump_solver/data/results/v1.0/3d/conditioning/fvm/fast/convergence'; titles{end+1} = 'Fast';
% dirs{end+1} = '/home/dbochkov/Dropbox/Docs/Presentations/13_socal_fluids_2019_jump_solver/data/results/v1.0/3d/conditioning/fvm/neut/convergence'; titles{end+1} = 'Random';

h = importdata(strcat(dirs{1},'/h_arr.txt'));
mu = importdata(strcat(dirs{1},'/mu_arr.txt'));

% fit = 1:length(h);
fit = 1+round((1-use_n_points)*length(h)):length(h);

lvl = importdata(strcat(dirs{1},'/lvl.txt'));

limit = 1;

for i = 1:length(dirs)
    error_sl_all{i} = max( [importdata(strcat(dirs{i},'/error_m_sl_all.txt')); importdata(strcat(dirs{i},'/error_p_sl_all.txt'))] );
    error_gr_all{i} = max( [importdata(strcat(dirs{i},'/error_m_gr_all.txt')); importdata(strcat(dirs{i},'/error_p_gr_all.txt'))] );
    error_sl_max{i} = max( [importdata(strcat(dirs{i},'/error_m_sl_max.txt')); importdata(strcat(dirs{i},'/error_p_sl_max.txt'))] );
    error_gr_max{i} = max( [importdata(strcat(dirs{i},'/error_m_gr_max.txt')); importdata(strcat(dirs{i},'/error_p_gr_max.txt'))] );
    cond_num_all{i} = importdata(strcat(dirs{i},'/cond_num_all.txt'));
    cond_num_max{i} = importdata(strcat(dirs{i},'/cond_num_max.txt'));
    
    for j = 1:length(error_sl_max{i})
        if (error_sl_max{i}(j) > limit)
            error_sl_max{i}(j) = nan;
        end
        if (error_gr_max{i}(j) > limit)
            error_gr_max{i}(j) = nan;
        end
    end
end

num_resolutions = length(h);
num_shifts = length(error_sl_all{1})/num_resolutions;

% detailed convergence
if (plot_detailed_convergence == 1)
    
    for j = 1:length(dirs)
        error_sl_all_max{j} = error_sl_all{j};
        error_gr_all_max{j} = error_gr_all{j};
        
        for i=1:num_resolutions
            error_sl_all_max{j}(:, (i-1)*num_shifts+1:i*num_shifts) = error_sl_max{j}(:,i)*ones(1,num_shifts);
            error_gr_all_max{j}(:, (i-1)*num_shifts+1:i*num_shifts) = error_gr_max{j}(:,i)*ones(1,num_shifts);
        end
    end
    
    % solution error
    
    if (subplotted == 1)
        subplot(3,3,[1,2]);
    else
        figure;
    end
    
    n = 0:length(error_sl_all{1})-1;
    
    for i = 1:length(dirs)
        L = semilogy(n, error_sl_all{i}); set(L, PropName_all, PropValue_all); set(L, 'DisplayName', titles{i});
        if (i==1) hold on; end
    end
    
    set(gca, 'ColorOrderIndex', 1);
    
    for i = 1:length(dirs)
        L = semilogy(n, error_sl_all_max{i}); set(L, PropName_all_max, PropValue_all_max); set(L, 'DisplayName', strcat(titles{i}, ' (max)'));
    end
    
    hold off
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Solution Error');
    
    % legend
    L = legend;
    L = legend('Location', 'best');
    set(L, 'interpreter', 'latex');
    
    % figure size
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 8 2.5];
    end
    
    % gradient error
    
    if (subplotted == 1)
        subplot(3,3,[4,5]);
    else
        figure;
    end
    
    n = 0:length(error_gr_all{1})-1;
    
    for i = 1:length(dirs)
        L = semilogy(n, error_gr_all{i}); set(L, PropName_all, PropValue_all); set(L, 'DisplayName', titles{i});
        if (i==1) hold on; end
    end
    
    set(gca, 'ColorOrderIndex', 1);
    
    for i = 1:length(dirs)
        L = semilogy(n, error_gr_all_max{i}); set(L, PropName_all_max, PropValue_all_max); set(L, 'DisplayName', strcat(titles{i}, ' (max)'));
    end
    
    hold off
    
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Gradient Error');
    
    % legend
    L = legend('Location', 'best');
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
        subplot(3,3,3);
    else
        figure;
    end
    
    % error plots    
    for i = 1:length(dirs)
        L = loglog(mu, error_sl_max{i}); set(L, PropName_max, PropValue_max); set(L, 'MarkerFaceColor', get(L, 'Color')); set(L, 'DisplayName', titles{i});
        if (i==1); hold on; end
    end
    
    hold off
    grid on
    
    % axes
    L = xlabel('Ratio \mu^{−}/\mu^+'); %set(L, 'interpreter', 'latex');
    ylabel('Solution Error');
    
    % legend, 
    L = legend('Location', 'best');
    set(L, 'interpreter', 'latex');
    
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 3.1 2.5];
    end
    
    % gradient error
    
    if (subplotted == 1)
        subplot(3,3,6);
    else
        figure;
    end
    
    % error plots    
    for i = 1:length(dirs)
        L = loglog(mu, error_gr_max{i}); set(L, PropName_max, PropValue_max); set(L, 'MarkerFaceColor', get(L, 'Color')); set(L, 'DisplayName', titles{i});
        if (i==1); hold on; end
    end
    
    hold off
    grid on
    
    % axes
    L = xlabel('Ratio \mu^{−}/\mu^+'); %set(L, 'interpreter', 'latex');
    ylabel('Gradient Error');
    
    % legend
    L = legend('Location', 'best');
    set(L, 'interpreter', 'latex');
    
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 3.1 2.5];
    end
    
end

% detailed condition number 
if (plot_detailed_cond_num == 1)
    
    if (subplotted == 1)
        subplot(3,3,[7,8]);
    else
        figure;
    end
    
    for j = 1:length(dirs)
        cond_num_all_max{j} = cond_num_all{j};
        
        for i=1:num_resolutions
            cond_num_all_max{j}(:, (i-1)*num_shifts+1:i*num_shifts) = cond_num_max{j}(:,i)*ones(1,num_shifts);
        end
    end
        
    for i = 1:length(dirs)
        n = 0:length(cond_num_all{i})-1;
        L = semilogy(n, cond_num_all{i}); set(L, PropName_all, PropValue_all); set(L, 'DisplayName', titles{i});
        if (i == 1) hold on; end
    end
    
    set(gca, 'ColorOrderIndex', 1);
    
    for i = 1:length(dirs)
        L = semilogy(n, cond_num_all_max{i}); set(L, PropName_all_max, PropValue_all_max); set(L, 'DisplayName', strcat(titles{i}, ' (max)'));
    end
    
    hold off
    grid on
    
    % axes
    xlabel('Case no.');
    ylabel('Condition number');
    
    % legend
    L = legend('Location', 'best');
    set(L, 'interpreter', 'latex');
    
    % figure size
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 8 2.5];
    end
    
end

% condensed condition number
if (plot_condensed_cond_num == 1)
    
    if (subplotted == 1)
        subplot(3,3,9);
    else
        figure;
    end
    
    % plots    
    for i = 1:length(dirs)
        L = loglog(mu, cond_num_max{i}); set(L, PropName_max, PropValue_max); set(L, 'MarkerFaceColor', get(L, 'Color')); set(L, 'DisplayName', titles{i});
        if (i==1); hold on; end
    end    
    hold off
    grid on
    
    % axes
    L = xlabel('Ratio \mu^{−}/\mu^+'); %set(L, 'interpreter', 'latex');
    L = ylabel('Condition number'); %set(L, 'interpreter', 'latex');
    
    % legend
    L = legend('Location', 'best');
    set(L, 'interpreter', 'latex');
    
    if (subplotted ~= 1)
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 3.1 2.5];
    end
    
end

if (subplotted == 1)
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 12 9];
end


