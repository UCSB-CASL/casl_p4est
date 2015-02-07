clear all;
close all;

folder = { 
    '.';
    };

p = [512; 4096];
xticks = [0, 120, 240, 360, 480;
          0, 1000, 2000, 3000, 4000];
it = [6 7;
      3 4;
      3 3];
modes = {'o', 's', 'd'};
faces = {'b', 'r', 'k'};
edges = faces;

fs = 16;
ms = [4 5 5] - 1;
for i=1:length(p)
    close all;
    for k =1:length(folder)    
%         comm = zeros(p(i), 5);
%         for j=1:it(k, i)
%             comm = comm + load(strcat(folder{k}, '/n_', num2str(p(i)), '/interpolation_log_0_', num2str(j-1), '.dat'));
%         end
%         comm = comm / it(k, i);

%         comm = load(strcat(folder{k}, '/n_', num2str(p(i)), '/interpolation_log_0_', num2str(it(k,i)-1), '.dat')); postfix='_last_';
        comm = load(strcat(folder{k}, '/n_', num2str(p(i)), '/interpolation_log_0.dat')); postfix='_first_';

        num_points   = comm(:,1) + comm(:,4);
        num_messages = (comm(:,3) + comm(:,5));
        volume       = (4*comm(:,2) + 3*comm(:,4)) * 8 / 1024^2; % recv: 3 doubles (xyz of point) -- send: 4 doubles (1 func + 3 derv)

        h1 = figure(1); hold on
        plot(num_points, modes{k}, 'markersize', ms(k), 'markeredgecolor', edges{k}, 'markerfacecolor', faces{k});
        set(gca,'DefaultTextFontname', 'CMU Serif', 'fontsize', fs);
%         h = legend('$\text{CFL} = 100$', '$\text{CFL} = 10$', '$\text{CFL} = 1$');
%         set(h, 'fontsize', fs, 'interpreter', 'latex');
        xlabel('$p$', 'fontsize', fs, 'interpreter', 'latex');
        ylabel('$N_p$', 'fontsize', fs, 'interpreter', 'latex');    
        xlim([0 p(i)]); box on;
        set(gca, 'fontsize', fs);
        set(gca,'DefaultTextFontname', 'CMU Serif')
        set(gca, 'XTick', xticks(i,:))
        set(gca, 'color', 'none');
        
        if i == 1
            ylim([1.5 3]*1e6);
        else
            ylim([0 5]*1e5);
        end
        axis square

        h2 = figure(2); hold on
        plot(num_messages, modes{k}, 'markersize', ms(k), 'markeredgecolor', edges{k}, 'markerfacecolor', faces{k});
        set(gca,'DefaultTextFontname', 'CMU Serif', 'fontsize', fs);
%         h = legend('$\text{CFL} = 100$', '$\text{CFL} = 10$', '$\text{CFL} = 1$');
%         set(h, 'fontsize', fs, 'interpreter', 'latex');
        xlabel('$p$', 'fontsize', fs, 'interpreter', 'latex');    
        ylabel('$N_m$', 'fontsize', fs, 'interpreter', 'latex');    
        xlim([0 p(i)]); box on
        set(gca, 'fontsize', fs);
        set(gca,'DefaultTextFontname', 'CMU Serif')
        set(gca, 'XTick', xticks(i,:))
        set(gca, 'color', 'none');
        
        if i == 1
            ylim([0 150]);
        else
            ylim([0 450]);
        end
        axis square

        h3 = figure(3); hold on
        plot(volume, modes{k}, 'markersize', ms(k), 'markeredgecolor', edges{k}, 'markerfacecolor', faces{k});
        set(gca,'DefaultTextFontname', 'CMU Serif', 'fontsize', fs);
%         h = legend('$\text{CFL} = 100$', '$\text{CFL} = 10$', '$\text{CFL} = 1$');
%         set(h, 'fontsize', fs, 'interpreter', 'latex');
        xlabel('$p$', 'fontsize', fs, 'interpreter', 'latex');
        ylabel('$V_m$ (MB)', 'fontsize', fs, 'interpreter', 'latex');            
        xlim([0 p(i)]);box on
        set(gca, 'fontsize', fs);
        set(gca,'DefaultTextFontname', 'CMU Serif')
        set(gca, 'XTick', xticks(i,:))
        set(gca, 'color', 'none');
        
        if i == 1
            ylim([0 70]);
        else
            ylim([0 12]);
        end
        axis square
    
        fprintf('p = %d, folder =  %s, mode = %s (min, max, avg, std) \n', p(i), folder{k}, postfix);
        fprintf('point  : %1.2e, %1.2e, %1.2e, %1.2e\n', min(num_points), max(num_points), mean(num_points), std(num_points));
        fprintf('message: %d, %d, %1.2f, %1.2f\n', min(num_messages), max(num_messages), mean(num_messages), std(num_messages));
        fprintf('volume : %1.2e, %1.2e, %1.2e, %1.2e\n\n\n', min(volume), max(volume), mean(volume), std(volume));
    end   
    
    saveas(h1, strcat('host_large_point',   postfix, num2str(p(i)), '.png')); % download from http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
    saveas(h2, strcat('host_large_message', postfix, num2str(p(i)), '.png')); % download from http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
    saveas(h3, strcat('host_large_volume',  postfix, num2str(p(i)), '.png')); % download from http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
end