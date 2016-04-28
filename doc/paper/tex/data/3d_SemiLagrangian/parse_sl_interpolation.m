set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')

type = 'host';
run  = 'small';
cfl  = 100;

folder = strcat(type, '/', run, '/', num2str(cfl));
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
    'InterpolatingFunctionHost::interpolate'
    'InterpolatingFunctionHost::process_local'  
    'InterpolatingFunctionHost::process_queries'
    'InterpolatingFunctionHost::process_replies'
};

%% Total time:
t = zeros(1, length(files));
p = zeros(1, length(files));


if strcmp(run, 'small')
    if cfl == 1
        it = [2 2 3 3 3 3];
    elseif cfl == 10
        it = [3 3 3 3 3 3];
    elseif cfl == 100
        it = [6 6 6 6 6 6];
    end
elseif strcmp(run, 'large')
    if cfl == 1
        it = [3 3 3 3 3 3];
    elseif cfl == 10
        it = [3 3 4 4 4 4];
    else
        it = [6 6 6 6 6 7];
    end
end

% large - 1, 10, 100
% it = [3 3 3 3 3 3];
% it = [3 3 4 4 4 4];
% it = [6 6 6 6 6 7];

% small - 1, 10, 100
% it = [2 2 3 3 3 3];
% it = [3 3 3 3 3 3];
% it = [6 6 6 6 6 6];

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
                t(j, i) = sscanf(line, '%*s %*d %*f %e %*[^\n]') / it(i);
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
% title('Interpolation', 'Interpreter', 'none', 'fontsize', 16);
% [~, idx] = sort(t(:,1), 'descend');
fs = 16;
ls = 16;

close all;
hfig = figure(1);

for i=1:length(events)
    loglog(p, t(i,:), modes{i}, 'markersize', 8, 'linewidth', 1, 'MarkerfaceColor',faces{i}, 'MarkerEdgeColor','k', 'linewidth', 1 ); hold on;               
end

eta = t(1,1)./t(1,:)*p(1)./p;
disp('eff = ');
fprintf('%1.2f ', eta);
fprintf('\n');

xlabel('$P$', 'fontsize', fs, 'interpreter', 'latex');
ylabel('$T_{\max} \:(s)$', 'fontsize', fs, 'interpreter', 'latex');
shg;

legend_titles = {'Total', 'local', 'queries', 'replies'};
h = legend(legend_titles, 'fontsize', ls);
set(h, 'interpreter', 'latex');
set(h,'DefaultTextFontname', 'CMU Serif')

legend('boxoff')
loglog(p, 3*t(1,1)*(p(1)./p).^1.0, 'k--', 'linewidth', 2); 
% loglog(p, t(1,end)*(p(end)./p).^0.5, 'k-.', 'linewidth', 2); 
xlim([p(1) p(end)])

set(gca, 'fontsize', fs);
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca, 'color', 'none');
set(gca, 'XTick', p, 'XTickLabel', strsplit(num2str(p)), 'FontName', 'Times');

hold off

axis square
filename = strcat(type, '_SemiLagrangian_Interpolation_', run, '_CFL_', num2str(cfl));
export_fig(gcf, strcat(filename,'.pdf'), '-transparent'); % download from http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
% print('-depsc2', '-r300', '-f1', strcat(type, '_SemiLagrangian_Interpolation_', run, '_CFL_', num2str(cfl)));
