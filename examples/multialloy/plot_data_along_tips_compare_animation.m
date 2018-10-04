clear;

% all dendritic
% dir_all = {'/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00001_G_00500.1972666', ...  
%            '/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00002_G_00500.1972671', ...  
%            '/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00004_G_00500.1972676', ...  
%            '/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00008_G_00500.1972679', ...  
%            '/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00016_G_00500.1972680', ...  
%            '/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00032_G_00500.1972681'};
%        
% outputname = '/home/dbochkov/Dendrites/dendritic_all';
% 
% titles = {'D_1/D_0 = 1', ...
%           'D_1/D_0 = 2', ...
%           'D_1/D_0 = 4', ...
%           'D_1/D_0 = 8', ...
%           'D_1/D_0 = 16', ...
%           'D_1/D_0 = 32' };
       

% all planar
% dir_all = {'/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00001_G_00500.2173295', ... 
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00002_G_00500.2173296', ... 
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00004_G_00500.2173297', ... 
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00008_G_00500.2173298', ... 
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00016_G_00500.2173299', ... 
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00032_G_00500.2173301'};
%       
% outputname = '/home/dbochkov/Dendrites/planar_all';
%        
% titles = {'D_1/D_0 = 1', ...
%           'D_1/D_0 = 2', ...
%           'D_1/D_0 = 4', ...
%           'D_1/D_0 = 8', ...
%           'D_1/D_0 = 16', ...
%           'D_1/D_0 = 32' };
%        
% dir_all = {'/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00001_G_00500.1972666', ...
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00001_G_00500.2173295'};
%        
% outputname = '/home/dbochkov/Dendrites/dendrite_vs_planar_1';
%        
% dir_all = {'/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00002_G_00500.1972671', ...
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00002_G_00500.2173296'};
%        
% outputname = '/home/dbochkov/Dendrites/dendrite_vs_planar_2';
% 
% dir_all = {'/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00004_G_00500.1972676', ...
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00004_G_00500.2173297'};
%        
% outputname = '/home/dbochkov/Dendrites/dendrite_vs_planar_4';
%        
dir_all = {'/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00008_G_00500.1972679', ...
           '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00008_G_00500.2173298'};
       
outputname = '/home/dbochkov/Dendrites/dendrite_vs_planar_8';
% 
% dir_all = {'/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00016_G_00500.1972680', ...
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00016_G_00500.2173299'};
%        
% outputname = '/home/dbochkov/Dendrites/dendrite_vs_planar_16';
% 
%        
% dir_all = {'/home/dbochkov/Dendrites/4/data_dendrites_all/D1_0.00032_G_00500.1972681', ...
%            '/home/dbochkov/Dendrites/5/data_dendrites_all/D1_0.00032_G_00500.2173301'};
%        
% outputname = '/home/dbochkov/Dendrites/dendrite_vs_planar_32';
% 
titles = {'Dendritic', ...
          'Planar' };

save_video = 1;
show_legend = 1;

alpha = 1-0.75^(1/length(dir_all));

x_start = 0.1;
x_max = 4;
x_plot = 4;

file_start = 0;
file_iter  = 1;
file_final = 20;

Tm = 1996;

m0 =-874;
m1 =-1378;
k0 = 0.848;
k1 = 0.848;

liq_rez = 100;

colors = lines(16);
markers = {'o','+','*','.','x','s','d','^','v','>','<','p','h'};

N = 1;

dir_case = {};

for dir_idx = 1:length(dir_all)
    dir_case(dir_idx) = {[dir_all{dir_idx}, '/dendrites']};
    file_final(dir_idx) = length(dir(dir_case{dir_idx}))-3;
    
    % get maximum values
    dir_cur = sprintf('%s/%05d', dir_case{dir_idx}, file_final(dir_idx));
    
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
    
    c0_min(dir_idx) = min(c0_common(sorting(1,2),:));
    c0_max(dir_idx) = max(c0_common(sorting(1,2),:));
    
    c1_min(dir_idx) = min(c1_common(sorting(1,2),:));
    c1_max(dir_idx) = max(c1_common(sorting(1,2),:));
    
    t_min(dir_idx) = min(t_common(sorting(1,2),:));
    t_max(dir_idx) = max(t_common(sorting(1,2),:));
    
    extend = 0.1;
    
    c0_gap = c0_max(dir_idx) - c0_min(dir_idx);
    c0_max(dir_idx) = c0_max(dir_idx) + extend*c0_gap;
    c0_min(dir_idx) = c0_min(dir_idx) - extend*c0_gap;
    
    c1_gap = c1_max(dir_idx) - c1_min(dir_idx);
    c1_max(dir_idx) = c1_max(dir_idx) + extend*c1_gap;
    c1_min(dir_idx) = c1_min(dir_idx) - extend*c1_gap;
    
    t_gap = t_max(dir_idx) - t_min(dir_idx);
    t_max(dir_idx) = t_max(dir_idx) + extend*t_gap;
    t_min(dir_idx) = t_min(dir_idx) - extend*t_gap;
end

c0_max_gl = max(c0_max);
c0_min_gl = min(c0_min);

c1_max_gl = max(c1_max);
c1_min_gl = min(c1_min);

t_max_gl = max(t_max);
t_min_gl = min(t_min);
    
if (save_video == 1)
    vidfile = VideoWriter([outputname,'.avi'],'Uncompressed AVI');
    vidfile.FrameRate = 15;
    open(vidfile);
end

for file_idx = file_start:file_iter:max(file_final)
    
    clf;
    
    for dir_idx = 1:length(dir_all)
        
        dir_cur = sprintf('%s/%05d', dir_case{dir_idx}, min([file_idx, file_final(dir_idx)]));
        
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
        t   = t./liquid;
        
        subplot(2,2,1);
        
        hold on
        area(x, max([c0_max_gl, c1_max_gl])*solid(sorting(1,2),:),'LineStyle','none','FaceAlpha',alpha,'FaceColor',colors(dir_idx,:), 'HandleVisibility', 'off');
%         area(x, max([c0_max_gl, c1_max_gl])*solid(sorting(1,2),:),'LineStyle','-','FaceAlpha',0.0,'EdgeColor',colors(dir_idx,:), 'HandleVisibility', 'off');
        for i = 1:n
            PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
            %PropValue = {markers{i}, 1, 3, 'auto', colors(i,:), 'none'};
            PropValue = {'none', 2, 3, 'auto', colors(dir_idx,:), '-'};
            L = plot(x, c0_common(sorting(i,2),:), 'DisplayName', titles{dir_idx}); set(L, PropName, PropValue);
            L = plot(x, c1_common(sorting(i,2),:), 'HandleVisibility', 'off');      set(L, PropName, PropValue);
        end
        hold off
        
        subplot(2,2,3);
        
        hold on
        area(x, t_max_gl*solid(sorting(1,2),:),'LineStyle','none','FaceAlpha',alpha,'FaceColor',colors(dir_idx,:), 'HandleVisibility', 'off');
        for i = 1:n
            PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
            % PropValue = {markers{i}, 1, 4, 'auto', colors(i,:), '-'};
            PropValue = {'none', 2, 3, 'auto', colors(dir_idx,:), '-'};
            L = plot(x, t_common(sorting(i,2),:), 'DisplayName', titles{dir_idx}); set(L, PropName, PropValue);
        end
        hold off
        
        subplot(2,2, [2,4]);
        
        hold on
        for i = 1:n
            d = sorting(i,2);
            plot(c0s(d,:)/k0, c1s(d,:)/k1, '-', 'Color', colors(dir_idx,:),'MarkerSize', 2, 'LineWidth', 1, 'Marker', markers{i}, 'DisplayName', titles{dir_idx});  
            plot(c0s(d,sum(solid(d,:)))/k0, c1s(d,sum(solid(d,:)))/k1, '-', 'Color', colors(dir_idx,:),'MarkerSize', 5, 'LineWidth', 2, 'Marker', markers{i}, 'HandleVisibility', 'off');       
        end
        hold off
    end
        
    subplot(2,2,1);
    
    xlabel('Distance');
    ylabel('Concentration');
    xlim([0, x_plot]);
    ylim([min([c0_min_gl, c1_min_gl]), max([c0_max_gl, c1_max_gl])]);
    grid on
    
%     if (show_legend == 1)
%         legend
%     end
    
    subplot(2,2,3);
    
    xlabel('Distance');
    ylabel('Temperature');
    xlim([0, x_plot]);
    ylim([t_min_gl, t_max_gl]);
    grid on
    
%     if (show_legend == 1)
%         legend
%     end
    
    subplot(2,2,[2,4]);
    
    xlabel('C_0');
    ylabel('C_1');
    xlim([c0_min_gl, c0_max_gl]);
    ylim([c1_min_gl, c1_max_gl]);
    grid on
    
    if (show_legend == 1)
        legend('Location','southoutside');
        box off
    end
    
    % figure size
    fig = gcf;
    fig.Units = 'inches';
    fig.Position = [10 10 15 8];
    
    drawnow;
    
    if (save_video == 1)
        im = getframe(gcf);
        writeVideo(vidfile, im);
    end
end

if (save_video == 1)
    close(vidfile)
    system(['avconv -i ',outputname,'.avi -c:v libx264 -c:a copy ',outputname,'.mp4']);
    system(['rm ',outputname,'.avi']);
end