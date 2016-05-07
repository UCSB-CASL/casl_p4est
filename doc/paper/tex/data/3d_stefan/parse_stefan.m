set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')

folder = 'small';
files = {
    'stdout.n_16';
    'stdout.n_32';
    'stdout.n_64';
    'stdout.n_128';
    'stdout.n_256';
    'stdout.n_512';
%     'stdout.n_1024';
%     'stdout.n_2048';
%     'stdout.n_4096';
    };

modes = {'ro-', 'bs-', 'k>-', 'md-', 'ro--', 'bs--', 'k>--', 'md--'};
faces = {'r', 'b', 'k', 'm', 'r', 'b', 'k', 'm'};
events = {
    'stefan_main_loop'
    'Semilagrangian::update_p4est_second_order_last_grid_Vec'
    'my_p4est_level_set::reinit_1st_time_2nd_space'  
    'my_p4est_level_set::extend_from_interface_TVD'
    'my_p4est_level_set::extend_over_interface_TVD'
    'PoissonSolverNodeBased::matrix_preallocation'
    'PoissonSolverNodeBased::rhsvec_setup'
    'PoissonSolverNodeBased::matrix_setup'
    'PoissonSolverNodeBased::solve'
    'my_p4est_new'
    'my_p4est_ghost_new'
    'my_p4est_copy'
    'my_p4est_refine'
    'my_p4est_coarsen'    
    'my_p4est_partition'
    'my_p4est_nodes_new'
};

%% Total time:
t = zeros(1, length(files));
p = zeros(1, length(files));

log_begin = '---------------------------------------------- PETSc Performance Summary: ----------------------------------------------';
flag = 0;

for i=1:length(files)
    fid = fopen(strcat(folder, '/', files{i}));
    
    while ~feof(fid)
        line = fgetl(fid);
        if flag
            line = fgetl(fid);
            p(i) = sscanf(line, '%*s %*s %*s %*s %*s %*s %*s %d %*[^\n]');
            flag = 0;
        end
        if strfind(line, log_begin)
            flag = 1;
        end
        if strfind(line, 'Time (sec):');
            t(i) = sscanf(line, '%*s %*s %e %*[^\n]');
            break
        end
    end
    
    fclose(fid);
    
end

%% separate events
t = zeros(length(events), length(files));
p = zeros(1, length(files));

for j=1:length(events)
    event = events{j};    
    log_begin = '---------------------------------------------- PETSc Performance Summary: ----------------------------------------------';
    flag = 0;
    
    for i=1:length(files)
        fid = fopen(strcat(folder, '/', files{i}));
        
        while ~feof(fid)
            line = fgetl(fid);
            if flag
                line = fgetl(fid);
                p(i) = sscanf(line, '%*s %*s %*s %*s %*s %*s %*s %d %*[^\n]');
                flag = 0;
            end
            if strfind(line, log_begin)
                flag = 1;
            end
            if strfind(line, event);
                t(j,i) = sscanf(line, '%*s %*d %*f %e %*[^\n]');
                break
            end
        end
        
        fclose(fid);
        
    end
    
    s  = t(j,1)./t(j,:);
    ti = t(1,1)*p(1)./p;
    disp(event);
    disp('s = '); disp(s);
    disp('t = '); disp(t);
    disp('p = '); disp(p);
    disp('e = '); disp(s*p(1)./p);
   
end

fs = 15;
ls = 13;

close all;
hfig = figure(1);
% Total time
j = 1; loglog(p, t(1,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;
% SemiLagrangian
j = 2; loglog(p, t(2,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;
% Reinitialization
j = 3; loglog(p, t(3,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k' , 'linewidth', 1); hold on;
% Extensions
j = 4; loglog(p, sum(t(4:5,:)), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;
% LinearSystem Setup
j = 5; loglog(p, sum(t(6:8,:)), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;
% LinearSystem Solve
j = 6; loglog(p, t(9,:) - sum(t(6:8,:)), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;
% p4est
j = 7; loglog(p, sum(t(10:end,:)), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;

eta = t(1,1)./t(1,:)*p(1)./p;
disp('eff = ');
fprintf('%1.2f ', eta);
fprintf('\n');

xlabel('$P$', 'fontsize', fs, 'interpreter', 'latex');
ylabel('$T_{\max} \:(s)$', 'fontsize', fs, 'interpreter', 'latex');
shg;

legend_titles = {'Total', 'Semi-Lagrangian', 'Reinitialization', 'Solution Extension', ...
                 'Linear System Setup', 'Linear System Solve', 'p4est'};
h = legend(legend_titles, 'fontsize', ls);
set(h, 'interpreter', 'latex');
set(h,'DefaultTextFontname', 'CMU Serif')

legend('boxoff')
loglog(p, 2.5*t(1,1)*p(1)./p, 'k--', 'linewidth', 2); 
xlim([p(1) p(end)])

set(gca, 'fontsize', fs);
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca, 'color', 'none');
set(gca, 'XTick', p, 'XTickLabel', strsplit(num2str(p)), 'FontName', 'Times');

hold off

axis square
export_fig(gcf, strcat('Stefan_', folder, '.pdf'), '-transparent'); % download from http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
% print('-depsc2', '-r300', '-f1', strcat('Stefan_', folder));
