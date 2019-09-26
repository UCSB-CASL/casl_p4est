function plot_results(nprocs, min_lvl, max_lvl, n_fields, exec_times)
    if(size(exec_times, 1) ~= 3 || size(exec_times, 2) ~= length(n_fields))
        error('The array of execution times must be of size (3, length(n_fields))');
    end
    FONTSIZE = 32;
    figure('Units','normalized','Position',[0.01 0.35 0.45 0.6]);
    hold on
    plot(n_fields, exec_times(1, :), 'bo', 'linewidth', 3, 'Markersize', 20)
    plot(n_fields, exec_times(2, :), 'ro', 'linewidth', 3, 'Markersize', 20)
    plot(n_fields, exec_times(3, :), 'ko', 'linewidth', 3, 'Markersize', 20)
    xlabel('Number of fields', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    ylabel('Execution time', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    grid on
    set(gca,'fontsize',FONTSIZE);
    title_ = sprintf('Calculation of the derivatives on a %d/%d grid using %d processors', min_lvl, max_lvl, nprocs);
    title(title_, 'fontsize', FONTSIZE, 'Interpreter', 'Latex')
    legend('original capability', ...
        'serialized strategy', ...
        'block-structured strategy', ...
        'Interpreter', 'Latex', 'fontsize', 24, 'location', 'NorthWest')
    hold off
    
    
    figure('Units','normalized','Position',[0.51 0.35 0.45 0.6]);
    hold on
    plot(n_fields, 100*(1.0-exec_times(2, :)./exec_times(1, :)), 'ro', 'linewidth', 3, 'Markersize', 20)
    plot(n_fields, 100*(1.0-exec_times(3, :)./exec_times(1, :)), 'ko', 'linewidth', 3, 'Markersize', 20)
    xlabel('Number of fields', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    ylabel('Time saved (\%)', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    grid on
    set(gca,'fontsize',FONTSIZE);
    title_ = sprintf('Calculation of the derivatives on a %d/%d grid using %d processors', min_lvl, max_lvl, nprocs);
    title(title_, 'fontsize', FONTSIZE, 'Interpreter', 'Latex')
    legend('serialized strategy', ...
        'block-structured strategy', ...
        'Interpreter', 'Latex', 'fontsize', 24, 'location', 'NorthWest')
    axis([0 max(n_fields) -10 100])
    hold off

end

