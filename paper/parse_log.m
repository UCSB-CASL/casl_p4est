folder = '3d_sl_st-cfl_1p0';
files = {
%     'stdout.n_16';
%     'stdout.n_32';
    'stdout.n_64';
    'stdout.n_128';
    'stdout.n_256';
    'stdout.n_512';
    'stdout.n_1024';
%     'stdout.n_2048';
%     'stdout.n_4096';
    };

events = {
    'my_p4est_level_set::reinit_1st_time_2nd_space';
    'Semilagrangian::update_p4est_second_order_CF2';
    'my_p4est_node_neighbors_t::2nd_derivatives_cent';
    'my_p4est_nodes_new';
    'InterpolatingFunction::interpolate';
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

s  = t(1)./t;
ti = t(1)*p(1)./p;
disp('Total time');
disp('s = '); disp(s);
disp('t = '); disp(t);
disp('p = '); disp(p);
disp('e = '); disp(s*p(1)./p);
loglog(p, t, 'o', 'markersize', 8); hold on;
loglog(p, ti, 'k--', 'linewidth', 1); hold off;
title('Total time', 'Interpreter', 'none', 'fontsize', 16);
set(gca, 'fontsize', 16);
xlabel('# of processors', 'fontsize', 16);
ylabel('Wall time(s)', 'fontsize', 16);
ylim(10.^[floor(min(log10(t(end)), log10(ti(end)))), ceil(log10(t(1)))]);
shg;

print('-depsc2', '-r300', '-f1', strcat(folder, '/', 'Total time'));
%% separate events
for j=1:length(events)
    event = events{j};
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
            if strfind(line, event);
                t(i) = sscanf(line, '%*s %*d %*f %e %*[^\n]');
                break
            end
        end
        
        fclose(fid);
        
    end
    
    s  = t(1)./t;
    ti = t(1)*p(1)./p;
    disp(event);
    disp('s = '); disp(s);
    disp('t = '); disp(t);
    disp('p = '); disp(p);
    disp('e = '); disp(s*p(1)./p);
    loglog(p, t, 'o', 'markersize', 8); hold on;
    loglog(p, ti, 'k--', 'linewidth', 1); hold off;
    title(event, 'Interpreter', 'none', 'fontsize', 16);
    set(gca, 'fontsize', 16);
    xlabel('# of processors', 'fontsize', 16);
    ylabel('Wall time(s)', 'fontsize', 16);
    ylim(10.^[floor(min(log10(t(end)), log10(ti(end)))), ceil(log10(t(1)))]);
    shg;
    
    print('-depsc2', '-r300', '-f1', strcat(folder, '/', event));
end

