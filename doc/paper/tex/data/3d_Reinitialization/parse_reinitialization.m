set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
   
folder = 'Large';
files = {
%     'stdout.n_16';
%     'stdout.n_32';
%     'stdout.n_64';
    'stdout.n_128';
    'stdout.n_256';
    'stdout.n_512';
    'stdout.n_1024';
    'stdout.n_2048';
    'stdout.n_4096';
    };

modes = {'ro-', 'bs-', 'k>-', 'md-', 'ro--', 'bs--', 'k>--', 'md--'};
faces = {'r', 'b', 'k', 'm', 'r', 'b', 'k', 'm'};
events = {
    'my_p4est_level_set::reinit_2nd_order'    
    'my_p4est_level_set::reinit_1_iter_2nd_order'  
    'my_p4est_node_neighbors_t::2nd_derivatives_cent'
    'my_p4est_nodes_new'
    'my_p4est_refine'
    'my_p4est_partition'
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
    
    if j==2 || j==3
        t(j,:) = t(j,:)*2./3.;
    end    
    
    s  = t(j,1)./t(j,:);
    ti = t(1,1)*p(1)./p;
    disp(event);
    disp('s = '); disp(s);
    disp('t = '); disp(t);
    disp('p = '); disp(p);
    disp('e = '); disp(s*p(1)./p);
   
end
t(1,:) = 0;
for j=2:length(events)
    t(1,:) = t(1,:) + t(j,:);
end

fs = 14;
ls = 14;

close all;
hfig = figure(1);
j = 1; loglog(p, t(1,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;
j = 2; loglog(p, t(2,:) + t(3,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;
j = 3; loglog(p, t(4,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k','linewidth', 1 ); hold on;
j = 4; loglog(p, t(5,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k','linewidth', 1 ); hold on;
j = 5; loglog(p, t(6,:), modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k','linewidth', 1 ); hold on;

eta = t(1,1)./t(1,:)*p(1)./p;
disp('eff = ');
fprintf('%1.2f ', eta);
fprintf('\n');
% loglog(p, 5*ti, 'k--', 'linewidth', 1); hold off
% set(gca, 'fontsize', 16);
% xlabel('# of processors', 'fontsize', 16);
% ylabel('Wall time(s)', 'fontsize', 16);
% shg;

xlabel('$P$', 'fontsize', fs, 'interpreter', 'latex');
ylabel('$T_{\max} \:(s)$', 'fontsize', fs, 'interpreter', 'latex');
shg;

legend_titles = {'Total', 'Reinitialization', 'p4est\_nodes\_new', 'p4est\_refine', 'p4est\_partition'};
h = legend(legend_titles, 'fontsize', ls);
set(h, 'interpreter', 'latex');
set(h,'DefaultTextFontname', 'CMU Serif')

legend('boxoff')
loglog(p, 2*t(1,1)*p(1)./p, 'k--', 'linewidth', 2); 
xlim([p(1) p(end)])

set(gca, 'fontsize', fs);
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca, 'color', 'none');
set(gca, 'XTick', p, 'XTickLabel', strsplit(num2str(p)), 'FontName', 'Times');

hold off

axis square
export_fig(gcf, strcat('Reinit_', folder, '.pdf'), '-transparent'); % download from http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
% print('-depsc2', '-r300', '-f1', strcat('Reinit_', folder));
% set(gca,'position',[0 0 1 1],'units','normalized')
% saveas(hfig, strcat('Reinit_', folder, '.eps'));
