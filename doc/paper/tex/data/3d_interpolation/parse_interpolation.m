set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')

type = 'host';
run  = 'super_large';
alpha = 50;

folder = strcat(type, '/', run, '/alpha_0.50');%, num2str(alpha/100));
files = {
%     'stdout.n_16';
%     'stdout.n_32';
%     'stdout.n_64';
%     'stdout.n_128';
%     'stdout.n_256';
    'stdout.n_512';
    'stdout.n_1024';
    'stdout.n_2048';
    'stdout.n_4096';
    };

modes = {'ro-', 'bs-', 'k>-', 'md-', 'ro--', 'bs--', 'k>--', 'md--'};
faces = {'r', 'b', 'k', 'm', 'r', 'b', 'k', 'm'};
events = {
    'log_interpolation_all'
    'log_interpolation_add_points'
    'InterpolatingFunctionHost::process_local'  
    'InterpolatingFunctionHost::process_queries'
    'InterpolatingFunctionHost::process_replies'
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
% p = zeros(1, length(files));
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
%                 p(i) = sscanf(line, '%*s %*s %*s %*s %*s %*s %*s %d %*[^\n]');
                flag = 0;
            end
            if strfind(line, log_begin)
                flag = 1;
            end
            if strfind(line, event);
%                 line
                t(j, i) = sscanf(line, '%*s %*d %*f %e %*[^\n]') / 10;
                break
            end
        end
        
        fclose(fid);
        
    end
    
%     s  = t(1)./t;
%     ti(j,:) = t(1)*p(1)./p;
%     disp(event);
%     disp('s = '); disp(s);
%     disp('t = '); disp(t);
%     disp('p = '); disp(p);
%     disp('e = '); disp(s*p(1)./p);
%     loglog(p, t/10, modes{j}, 'markersize', 7, 'MarkerfaceColor',faces{j}, 'MarkerEdgeColor','k' ); hold on;       
%     
%     set(gca, 'fontsize', 16);
%     xlabel('# of processors', 'fontsize', 16);
%     ylabel('Wall time(s)', 'fontsize', 16);
%     shg;
%         
end
%%
% [~, idx] = sort(t(:,1), 'descend');
idx = 1:length(events);
close all;
hfig = figure(1);
fs = 16;
ls = 15;
t(:,:) = 2*t(:,:);
for i=1:length(events)
    loglog(p, t(idx(i),:), modes{i}, 'markersize', 7, 'MarkerfaceColor',faces{i}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;           
    set(gca, 'fontsize', fs);
end

eta = t(1,1)./t(1,:)*p(1)./p;
disp('eff = ');
fprintf('%1.2f ', eta);
fprintf('\n');

xlabel('$P$', 'fontsize', fs, 'interpreter', 'latex');
ylabel('$T_{\max} \:(s)$', 'fontsize', fs, 'interpreter', 'latex');
shg;

legend_titles = {'Total', 'buffer', 'local', 'queries', 'replies'};
legend(legend_titles{idx}, 'fontsize', ls);
legend('boxoff')
loglog(p, 2*t(1,1)*p(1)./p, 'k--', 'linewidth', 2); 
xlim([p(1) p(end)])
set(gca, 'XTick', p, 'XTickLabel', strsplit(num2str(p)));

hold off

axis square
filename = strcat(type, '_Interpolation_', run, '_alpha_', num2str(alpha));
export_fig(gcf, strcat(filename,'.pdf'), '-transparent'); % download from http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
% print('-depsc2', '-r300', '-f1', strcat(folder, '/Interpolation'));
