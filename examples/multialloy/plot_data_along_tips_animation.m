clear;

x_start = 0.1;
x_max = 4;
x_plot = 4;

file_start = 0;
file_iter  = 1;
file_final = 20;

% dir_all = '/home/dbochkov/Dendrites/4/data_dendrites_all';
dir_all = '/home/dbochkov/Dendrites/5/data_dendrites_all';

folder = dir(dir_all);

num_cases = length(folder)-2;

Tm = 1996;

m0 =-874;
m1 =-1378;
k0 = 0.848;
k1 = 0.848;

liq_rez = 100;

N = 1;

for case_idx = 1:num_cases
    dir_case = [dir_all, '/', folder(2+case_idx).name, '/dendrites'];
    file_final = length(dir(dir_case))-3;
    
    % get maximum values
    dir_cur = sprintf('%s/%05d', dir_case, file_final);
    
    phi   = importdata(strcat(dir_cur,'/phi.txt'));
    c0    = importdata(strcat(dir_cur,'/c0.txt'));
    c1    = importdata(strcat(dir_cur,'/c1.txt'));
    t     = importdata(strcat(dir_cur,'/t.txt'));
    vn    = importdata(strcat(dir_cur,'/vn.txt'));
    c0s   = importdata(strcat(dir_cur,'/c0s.txt'));
    c1s   = importdata(strcat(dir_cur,'/c1s.txt'));
    tf    = importdata(strcat(dir_cur,'/tf.txt'));
    kappa = importdata(strcat(dir_cur,'/kappa.txt'));
    velo  = importdata(strcat(dir_cur,'/velo.txt'));
    
    solid  = phi > 0;
    liquid = phi < 0;
    
    c0_common = c0.*liquid + c0s.*solid/k0;
    c1_common = c1.*liquid + c1s.*solid/k1;
    
    t_common = tf.*solid + t.*liquid;
    
    sorting = sortrows([sum(solid'); 1:length(solid(:,1))]', 'descend');
    
    c0_min = min(c0_common(sorting(1,2),:));
    c0_max = max(c0_common(sorting(1,2),:));
    
    c1_min = min(c1_common(sorting(1,2),:));
    c1_max = max(c1_common(sorting(1,2),:));
    
    t_min = min(t_common(sorting(1,2),:));
    t_max = max(t_common(sorting(1,2),:));
    
    extend = 0.1;
    
    c0_gap = c0_max - c0_min;
    c0_max = c0_max + extend*c0_gap;
    c0_min = c0_min - extend*c0_gap;
    
    c1_gap = c1_max - c1_min;
    c1_max = c1_max + extend*c1_gap;
    c1_min = c1_min - extend*c1_gap;
    
    t_gap = t_max - t_min;
    t_max = t_max + extend*t_gap;
    t_min = t_min - extend*t_gap;
    
    
    colors = lines(16);
    markers = {'o','+','*','.','x','s','d','^','v','>','<','p','h'};
    
    outputname = ['video_',num2str(case_idx)];
    % vidfile = VideoWriter([dir_all,'/../../d4.avi'],'Motion JPEG AVI');
    vidfile = VideoWriter([dir_all,'/',outputname,'.avi'],'Uncompressed AVI');
    % vidfile.Quality = 98;
    vidfile.FrameRate = 15;
    open(vidfile)
    
    for file_idx = file_start:file_iter:file_final
        
        dir_cur = sprintf('%s/%05d', dir_case, file_idx);
        
        phi   = importdata(strcat(dir_cur,'/phi.txt'));
        c0    = importdata(strcat(dir_cur,'/c0.txt'));
        c1    = importdata(strcat(dir_cur,'/c1.txt'));
        t     = importdata(strcat(dir_cur,'/t.txt'));
        vn    = importdata(strcat(dir_cur,'/vn.txt'));
        c0s   = importdata(strcat(dir_cur,'/c0s.txt'));
        c1s   = importdata(strcat(dir_cur,'/c1s.txt'));
        tf    = importdata(strcat(dir_cur,'/tf.txt'));
        kappa = importdata(strcat(dir_cur,'/kappa.txt'));
        velo  = importdata(strcat(dir_cur,'/velo.txt'));
        
        nx = length(phi(1,:));
        
        n = min(N,length(phi(:,1)));
        
        x = linspace(0, x_max, nx);
        
        solid  = phi > 0;
        liquid = phi < 0;
        
        sorting = sortrows([sum(solid'); 1:length(solid(:,1))]', 'descend');
        
        c0_common = c0.*liquid + c0s.*solid/k0;
        c1_common = c1.*liquid + c1s.*solid/k1;
        
        t_common = tf.*solid + t.*liquid;
        
        c0 = c0./liquid;
        c1 = c1./liquid;
        
        c0s = c0s./solid;
        c1s = c1s./solid;
        
        tf  = tf./solid;
        
        t = t./liquid;
        %     figure;
        clf;
        subplot(2,2,1);
        hold on
        area(x, max([c0_max, c1_max])*solid(sorting(1,2),:),'LineStyle','none','FaceAlpha',0.25);
        for i = 1:n
            PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
            %         PropValue = {markers{i}, 1, 3, 'auto', colors(i,:), 'none'};
            PropValue = {'none', 2, 3, 'auto', colors(i,:), '-'};
            L = plot(x, c0_common(sorting(i,2),:), 'DisplayName', ['Dendrite ', num2str(i)]); set(L, PropName, PropValue);
            L = plot(x, c1_common(sorting(i,2),:),'HandleVisibility','off'); set(L, PropName, PropValue);
        end
        hold off
        %     legend
        xlabel('Distance');
        ylabel('Concentration');
        xlim([0, x_plot]);
        ylim([min([c0_min, c1_min]), max([c0_max, c1_max])]);
        grid on
        
        %     figure;
        subplot(2,2,3);
        area(x, t_max*solid(sorting(1,2),:),'LineStyle','none','FaceAlpha',0.25);
        hold on
        for i = 1:n
            PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
            %         PropValue = {markers{i}, 1, 4, 'auto', colors(i,:), '-'};
            PropValue = {'none', 2, 3, 'auto', colors(i,:), '-'};
            L = plot(x, t_common(sorting(i,2),:), 'DisplayName', ['Dendrite ', num2str(i)]); set(L, PropName, PropValue);
        end
        hold off
        xlabel('Distance');
        ylabel('Temperature');
        xlim([0, x_plot]);
        ylim([t_min, t_max]);
        %     legend
        grid on
        
        
        subplot(2,2, [2,4]);
        hold on
        
        for i = 1:n
            
            d = sorting(i,2);
            plot(c0s(d,:)/k0, c1s(d,:)/k1, '-', 'MarkerSize', 2, 'LineWidth', 2, 'Marker', markers{i}, 'DisplayName', ['Dendrite ', num2str(i)])
            %         plot(c0s(d,:)/k0, c1s(d,:)/k1, '-', 'MarkerSize', 2, 'LineWidth', 2, 'Marker', 'none', 'DisplayName', ['Dendrite ', num2str(i)])
            
        end
        
        hold off
        
        xlim([c0_min, c0_max]);
        ylim([c1_min, c1_max]);
        grid on
        %     legend
        xlabel('C_0');
        ylabel('C_1');
        
        % figure size
        fig = gcf;
        fig.Units = 'inches';
        fig.Position = [10 10 15 8];
        
        im = getframe(gcf);
        
        writeVideo(vidfile, im);
        %     drawnow;
    end
    
    close(vidfile)
    
    system(['avconv -i ',dir_all,'/',outputname,'.avi -c:v libx264 -c:a copy ',dir_all,'/',outputname,'.mp4']);
    system(['rm ',dir_all,'/',outputname,'.avi']);
end